"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os

import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt

from . import datasets, decoder, network, show, transforms, visualizer, __version__

LOG = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.predict',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    network.cli(parser)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.1, seed_threshold=0.5)
    show.cli(parser)
    visualizer.cli(parser)
    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('--show', default=False, action='store_true',
                        help='show image of output overlay')
    parser.add_argument('--image-output', nargs='?', const=True,
                        help='image output file or directory')
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='json output file or directory')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='processing batch size')
    parser.add_argument('--long-edge', default=None, type=int,
                        help='apply preprocessing to batch images')
    parser.add_argument('--loader-workers', default=None, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--line-width', default=6, type=int,
                        help='line width for skeleton')
    parser.add_argument('--monocolor-connections', default=False, action='store_true')
    parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
    parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
    group = parser.add_argument_group('logging')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    group.add_argument('--debug-images', default=False, action='store_true',
                       help='print debug messages and enable all debug images')
    args = parser.parse_args()

    if args.debug_images:
        args.debug = True

    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig()
    logging.getLogger('openpifpaf').setLevel(log_level)
    LOG.setLevel(log_level)

    network.configure(args)
    show.configure(args)
    visualizer.configure(args)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    return args


def processor_factory(args):
    # load model
    model_cpu, _ = network.factory_from_args(args)
    model = model_cpu.to(args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.base_net = model_cpu.base_net
        model.head_nets = model_cpu.head_nets
    processor = decoder.factory_from_args(args, model)
    return processor, model


def preprocess_factory(args):
    preprocess = [transforms.NormalizeAnnotations()]
    if args.long_edge:
        preprocess.append(transforms.RescaleAbsolute(args.long_edge))
    if args.batch_size > 1:
        assert args.long_edge, '--long-edge must be provided for batch size > 1'
        preprocess.append(transforms.CenterPad(args.long_edge))
    else:
        preprocess.append(transforms.CenterPadTight(16))
    return transforms.Compose(preprocess + [transforms.EVAL_TRANSFORM])


def out_name(arg, in_name, default_extension):
    """Determine an output name from args, input name and extension.

    arg can be:
    - none: return none (e.g. show image but don't store it)
    - True: activate this output and determine a default name
    - string:
        - not a directory: use this as the output file name
        - is a directory: use directory name and input name to form an output
    """
    if arg is None:
        return None

    if arg is True:
        return in_name + default_extension

    if os.path.isdir(arg):
        return os.path.join(
            arg,
            os.path.basename(in_name)
        ) + default_extension

    return arg


def freihand_multi_predict():
    args = cli()

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList_Freihand(args.images[0], mode='evaluation', preprocess=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_anns_meta)

    # visualizers
    keypoint_painter = show.KeypointPainter(
        color_connections=not args.monocolor_connections,
        linewidth=args.line_width,
    )
    annotation_painter = show.AnnotationPainter(keypoint_painter=keypoint_painter)
    B = 0
    b = 0
    pred_array = []
    gt_array = []

    for batch_i, (image_tensors_batch, gt_batch, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(model, image_tensors_batch, device=args.device)

        # unbatch
        for pred, gt, meta in zip(pred_batch, gt_batch, meta_batch):
            b += 1
            # print('b={}'.format(b))
            percent_completed=b/data.__len__()*100
            print('progress = {:.2f}'.format(percent_completed))
            # bar.next()
            LOG.info('batch %d: %s', batch_i, meta['file_name'])

            # load the original image if necessary
            cpu_image = None
            if args.debug or args.show or args.image_output is not None:
                with open(meta['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

            visualizer.BaseVisualizer.image(cpu_image)
            if preprocess is not None:
                pred = preprocess.annotations_inverse(pred, meta)

            # error = pred[0].data - gt[0]['keypoints']
            try:
                if pred[0].data.shape==(21,3) and gt[0]['keypoints'].shape==(21,3):
                    pred_array.append(pred[0].data)
                    gt_array.append(gt[0]['keypoints'])
            except:
                pass

    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/pred_array.npy', pred_array)
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/gt_array.npy', gt_array)


def PCK_plot():
    checkpoint_name = 'shufflenetv2k16w-200723-003131-cif-caf-caf25-edge280.pkl.epoch118'
    eval_dataset = 'rhd'
    pred_array = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/pred_array_{}.npy'.format(eval_dataset, checkpoint_name))
    gt_array = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/gt_array_{}.npy'.format(eval_dataset, checkpoint_name))



    def PCK(PCK_thresh, pred_score_thresh = 0.1, gt_conf_thresh = 0):
        total_conted_data = 0
        total_correct_data = 0
        for data_id in range (0, pred_array.shape[0]):
            bool_acceptable_data = (pred_array[data_id, :, 2] > pred_score_thresh) * (gt_array[data_id, :, 2] > gt_conf_thresh)
            _errors = pred_array[data_id, bool_acceptable_data, :2] - gt_array[data_id, bool_acceptable_data, :2]
            _norms = np.linalg.norm(_errors, axis=1)
            if sum(_norms<PCK_thresh)==0:
                print('data_id=',data_id)
                print('np.argwhere(pred_array[data_id, :, 2]==0)=',np.argwhere(pred_array[data_id, :, 2]==0))
                print('np.argwhere(gt_array[data_id, :, 2]==0)=',np.argwhere(gt_array[data_id, :, 2]==0))

            total_correct_data += sum(_norms<PCK_thresh)
            total_conted_data += _norms.shape[0]

        PCK_value = total_correct_data/total_conted_data
        return PCK_value

    num_intervals = 100
    max_error = 300
    PCK_thresh = np.linspace(0, max_error, num_intervals)
    # PCK_thresh = np.geomspace(0.5, max_error, num_intervals)


    y=[]
    for iter in range(0,num_intervals):
        PCK_value = PCK(PCK_thresh[iter])
        y.append(PCK_value)
        print('PCK_thresh[iter] = {:.2f}: PCK_value = {:.2f}; progress = {:.2f} %'.format(PCK_thresh[iter], PCK_value, iter/num_intervals*100))

    # attention_paper_2DPCK_PXLs_freihand = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/attention_network_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')
    # MobilePose_paper_2DPCK_PXLs_freihand = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/MobilePose_network_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')
    # EfficientDet_paper_2DPCK_PXLs_freihand = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/EfficientDet_network_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')


    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}_2DPCKvsPXLs.npy'.format(eval_dataset, checkpoint_name), np.vstack((PCK_thresh, np.asarray(y))))
    # handPifPaf_paper_2DPCK_PXLs_freihand = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}_2DPCKvsPXLs.npy'.format(eval_dataset, checkpoint_name))
    # PCK_thresh = handPifPaf_paper_2DPCK_PXLs_freihand[0, :]
    # y = handPifPaf_paper_2DPCK_PXLs_freihand[1, :]

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    axes.plot(PCK_thresh, np.asarray(y), label='handPifPaf', c='b')
    # axes.plot(attention_paper_2DPCK_PXLs_freihand[:, 0], attention_paper_2DPCK_PXLs_freihand[:, 1], label='Attention', c='g')
    # axes.plot(MobilePose_paper_2DPCK_PXLs_freihand[:, 0], MobilePose_paper_2DPCK_PXLs_freihand[:, 1], label='MobilePose224V2', c='m')
    # axes.plot(EfficientDet_paper_2DPCK_PXLs_freihand[:, 0], EfficientDet_paper_2DPCK_PXLs_freihand[:, 1], label='EfficientDet224', c='brown')
    axes.set_xlabel('Error Thresholds [px]')
    axes.set_ylabel('2D PCK')
    axes.set_title('2D PCK vs error threshold in pixels')
    axes.grid(True)
    axes.legend()
    axes.set_ylim([0, 1])
    axes.set_xlim([0, max_error])
    plt.savefig('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/2DPCK_{}.png'.format(eval_dataset, checkpoint_name), format='png')
    plt.show()

def rhd_multi_predict():
    checkpoint_name = 'shufflenetv2k16w-200723-003131-cif-caf-caf25-edge280.pkl.epoch118'
    args = cli()

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList_RHD(args.images[0], mode='evaluation', preprocess=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_anns_meta)

    # visualizers
    keypoint_painter = show.KeypointPainter(
        color_connections=not args.monocolor_connections,
        linewidth=args.line_width,
    )
    annotation_painter = show.AnnotationPainter(keypoint_painter=keypoint_painter)
    b = 0
    pred_array = []
    gt_array = []

    for batch_i, (image_tensors_batch, gt_batch, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(model, image_tensors_batch, device=args.device)

        # unbatch
        for pred, gt, meta in zip(pred_batch, gt_batch, meta_batch):
            b += 1
            # print('b={}'.format(b))
            percent_completed=b/data.__len__()*100
            print('progress = {:.2f}'.format(percent_completed))
            # bar.next()
            LOG.info('batch %d: %s', batch_i, meta['file_name'])

            # load the original image if necessary
            cpu_image = None
            if args.debug or args.show or args.image_output is not None:
                with open(meta['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

            visualizer.BaseVisualizer.image(cpu_image)
            if preprocess is not None:
                pred = preprocess.annotations_inverse(pred, meta)

            # error = pred[0].data - gt[0]['keypoints']
            try:
                if pred[0].data.shape==(42,3) and gt[0]['keypoints'].shape==(42,3):
                    pred_array.append(pred[0].data)
                    gt_array.append(gt[0]['keypoints'])
            except:
                pass

    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/rhd/predict_output/pred_array_{}.npy'.format(checkpoint_name),pred_array)
    np.save(
        '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/rhd/predict_output/gt_array_{}.npy'.format(checkpoint_name), gt_array)


if __name__ == '__main__':
    # freihand_multi_predict()
    # rhd_multi_predict()
    PCK_plot()

# time CUDA_VISIBLE_DEVICES=0,1 python3 -m openpifpaf.predict_multi /home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/Freihand_pub_v2/ --image-output /home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/  --checkpoint=/home/mahdi/HVR/git_repos/openpifpaf/outputs/shufflenetv2k16w-200720-202350-cif-caf-caf25-edge200.pkl.epoch052  --json-output=/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/  --batch-size=16  --long-edge=224