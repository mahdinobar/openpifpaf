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
import matplotlib
# uncomment for server
matplotlib.use('Agg')
import copy

from . import datasets, decoder, network, show, transforms, visualizer, __version__

from .datasets.freihand_utils import *
from .google_pose_estimator.predict_joints import google_predict_only
from PIL import Image, ImageDraw
import scipy.ndimage
from .datasets.freihand_constants import FREIHAND_KEYPOINTS

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


def freihand_multi_predict(checkpoint_name, eval_dataset):
    args = cli()

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList_Freihand(args.images[0], mode='test', preprocess=preprocess)
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

    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/pred_array.npy'.format(eval_dataset, checkpoint_name), pred_array)
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/gt_array.npy'.format(eval_dataset, checkpoint_name), gt_array)


def panoptic_multi_predict(checkpoint_name, eval_dataset):
    args = cli()

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList_Panoptic(image_dir=args.images[0], mode='test', preprocess=preprocess)
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
            if args.debug or args.show is not False:
                with open(meta['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')
                    gt_before_preprocess = meta['gt_before_preprocess']
                    # crop image
                    max_gt_bbx = max(max(gt_before_preprocess[:, 0]) - min(gt_before_preprocess[:, 0]),
                                     max(gt_before_preprocess[:, 1]) - min(gt_before_preprocess[:, 1]))
                    bbx_factor = 2.2
                    bbx = bbx_factor * max_gt_bbx
                    x_offset = (max(gt_before_preprocess[:, 0]) + min(gt_before_preprocess[:, 0])) / 2 - bbx / 2
                    y_offset = (max(gt_before_preprocess[:, 1]) + min(gt_before_preprocess[:, 1])) / 2 - bbx / 2
                    ltrb = (x_offset, y_offset, x_offset + bbx, y_offset + bbx)
                    img = cpu_image.crop(ltrb)

                    # import matplotlib.pyplot as plt
                    # fig, ax = plt.subplots(1, 2, figsize=(12, 12))
                    # ax[0].imshow(np.asarray(cpu_image))
                    # ax[1].imshow(np.asarray(img))
                    # plt.show()

                    # rescale image
                    order = 1  # order of resize interpolation; 1 means linear interpolation
                    w, h = img.size
                    # keep aspect ratio the same
                    reference_edge = 224
                    target_max_edge = reference_edge
                    max_edge = max(h, w)
                    ratio_factor = target_max_edge / max_edge
                    target_h = int(ratio_factor * h)
                    target_w = int(ratio_factor * w)
                    im_np = np.asarray(img)
                    im_np = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1), order=order)
                    img = PIL.Image.fromarray(im_np)
                    LOG.debug('input raw image before resize = (%f, %f), after = %s', w, h, img.size)
                    assert img.size[0] == target_w
                    assert img.size[1] == target_h

                    # pad frames
                    img = np.asarray(img)
                    pad_up = (reference_edge - img.shape[0]) // 2
                    pad_down = (reference_edge - img.shape[0]) // 2
                    pad_left = (reference_edge - img.shape[1]) // 2
                    pad_right = (reference_edge - img.shape[1]) // 2
                    img = np.pad(img, pad_width=((pad_up, pad_down), (pad_left, pad_right), (0, 0)), mode='symmetric')
                    cpu_image = Image.fromarray(img.astype('uint8'), 'RGB')




            visualizer.BaseVisualizer.image(cpu_image)
            if preprocess is not None:
                pred = preprocess.annotations_inverse(pred, meta)

            # uncomment to save prediction vs gt
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            # ax.imshow(np.asarray(cpu_image))
            # bool_annotated_joints_1 = gt[0]['keypoints'][:, 2] == 2
            # ax.plot(gt[0]['keypoints'][bool_annotated_joints_1, 0], gt[0]['keypoints'][bool_annotated_joints_1, 1],
            #            'go', label='ground truth')
            # bool_annotated_joints_1 = pred[0].data[:, 2] > 0.1
            # ax.plot(pred[0].data[bool_annotated_joints_1, 0], pred[0].data[bool_annotated_joints_1, 1],
            #            'ro', label='prediction')
            # ax.legend()
            # plt.savefig('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/panoptic/shufflenetv2k16w-200809-021328-cif-caf-caf25-edge200-o10s.pkl.epoch187/prediction_{}.png'.format(meta['file_name'][-12:-4]))
            # plt.show()


            # error = pred[0].data - gt[0]['keypoints']
            try:
                if pred[0].data.shape==(21,3) and gt[0]['keypoints'].shape==(21,3):
                    pred_array.append(pred[0].data)
                    gt_array.append(gt[0]['keypoints'])
            except:
                pass

    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/pred_array.npy'.format(eval_dataset, checkpoint_name), pred_array)
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/gt_array.npy'.format(eval_dataset, checkpoint_name), gt_array)



def posedataset_multi_predict_hvr(checkpoint_name, eval_dataset):
    args = cli()

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList_PoseDataset_hvr(args.images[0], preprocess=preprocess)
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
    hvr_pred_array = []
    hvr_name_array = []
    error_counter = 0
    for batch_i, (image_tensors_batch, gt_batch, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(model, image_tensors_batch, device=args.device)

        # unbatch
        for pred, gt, meta in zip(pred_batch, gt_batch, meta_batch):
            b += 1
            try:
                # # print('b={}'.format(b))
                # percent_completed=b/data.__len__()*100
                # print('progress = {:.2f}'.format(percent_completed))
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

                pred_hvr = np.copy(pred[0].data)



                # # rescale image
                # order = 1  # order of resize interpolation; 1 means linear interpolation
                # w, h = img.size
                # # keep aspect ratio the same
                # reference_edge = 224
                # target_max_edge = reference_edge
                # max_edge = max(h, w)
                # ratio_factor = target_max_edge / max_edge
                # target_h = int(ratio_factor * h)
                # target_w = int(ratio_factor * w)
                # im_np = np.asarray(img)
                # im_np = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1), order=order)
                # img = PIL.Image.fromarray(im_np)
                # assert img.size[0] == target_w
                # assert img.size[1] == target_h
                # # pad frames
                # img = np.asarray(img)
                # pad_up = (reference_edge - img.shape[0]) // 2
                # pad_down = (reference_edge - img.shape[0]) // 2
                # pad_left = (reference_edge - img.shape[1]) // 2
                # pad_right = (reference_edge - img.shape[1]) // 2
                # img = np.pad(img, pad_width=((pad_up, pad_down), (pad_left, pad_right), (0, 0)), mode='symmetric')

                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                # ax.imshow(np.asarray(cpu_image))
                # ax.set_title(meta['file_name'])
                # rescale back keypoints
                w=640
                h=480
                w_t=int(640*224/640)
                h_t=int(480*224/640)
                x_scale = (w_t) / (w)
                y_scale = (h_t) / (h)

                pred_hvr[:, 0]=pred_hvr[:, 0]/x_scale
                pred_hvr[:, 1]=pred_hvr[:, 1]/y_scale

                pad_up = (640 - h) // 2
                # pad_down = (reference_edge - img.shape[0]) // 2
                pad_left = (640 - w) // 2
                # pad_right = (reference_edge - img.shape[1]) // 2
                # img = np.pad(img, pad_width=((pad_up, pad_down), (pad_left, pad_right), (0, 0)), mode='symmetric')
                # img = Image.fromarray(img.astype('uint8'), 'RGB')
                # for ann in anns:
                #     modify for pad
                    # ann['keypoints'][:, 0] = ann['keypoints'][:, 0] + pad_left
                    # ann['keypoints'][:, 1] = ann['keypoints'][:, 1] + pad_up
                # pred_hvr[:, 0]=pred_hvr[:, 0]-pad_left
                # pred_hvr[:, 1]=pred_hvr[:, 1]-pad_up
                #
                # ax.plot(pred_hvr[:, 0], pred_hvr[:, 1], 'ro')
                # ax.plot(pred_hvr[14, 0], pred_hvr[14, 1], 'bx')
                # ax.plot(pred_hvr[13, 0], pred_hvr[13, 1], 'gx')
                #
                # plt.show()

                # error = pred[0].data - gt[0]['keypoints']

                if pred_hvr.shape==(21,3):
                    hvr_pred_array.append(pred_hvr)
                    hvr_name_array.append(meta['file_name'])
                else:
                    error_counter += 1
                    print('======================ERROR={} at meta["file_name"]={}, pred_hvr.shape==(21,3)={}'.format(error_counter, meta['file_name'], pred_hvr.shape==(21,3)))

            except:
                error_counter += 1
                print('======================ERROR counter={} at meta["file_name"]={}'.format(error_counter, meta['file_name']))
                continue

    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/hvr_pred_array.npy'.format(eval_dataset, checkpoint_name), hvr_pred_array)
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/hvr_name_array.npy'.format(eval_dataset, checkpoint_name), hvr_name_array)



def onehand10k_multi_predict(checkpoint_name, eval_dataset):
    args = cli()

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList_OneHand10K(args.images[0], mode='evaluation', preprocess=preprocess)
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

    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/pred_array_{}.npy'.format(eval_dataset, checkpoint_name), pred_array)
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/gt_array_{}.npy'.format(eval_dataset, checkpoint_name), gt_array)


def rhd_multi_predict(checkpoint_name, eval_dataset):
    args = cli()

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList_RHD(args.images[0], mode='evaluation', preprocess=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_anns_meta)

    # # visualizers
    # keypoint_painter = show.KeypointPainter(
    #     color_connections=not args.monocolor_connections,
    #     linewidth=args.line_width,
    # )
    # annotation_painter = show.AnnotationPainter(keypoint_painter=keypoint_painter)
    b = 0
    pred_array = []
    gt_array = []
    index_array=[]

    for batch_i, (image_tensors_batch, gt_batch, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(model, image_tensors_batch, device=args.device)

        # import pdb; pdb.set_trace()
        # unbatch
        for pred, gt, meta in zip(pred_batch, gt_batch, meta_batch):
            # bar.next()
            # LOG.info('batch %d: %s', batch_i, meta['file_name'])

            # # load the original image if necessary
            # cpu_image = None
            # if args.debug or args.show or args.image_output is not None:
            #     with open(meta['file_name'], 'rb') as f:
            #         cpu_image = PIL.Image.open(f).convert('RGB')
            #
            # visualizer.BaseVisualizer.image(cpu_image)
            if preprocess is not None:
                pred = preprocess.annotations_inverse(pred, meta)

            print('pred=', pred)

            failure_counter = 0
            # error = pred[0].data - gt[0]['keypoints']
            # pred[2].json_data()['score']
            try:
                # remove predictions with score less than a threshold
                pred_overall_score_threshod = 0.15
                wrong_annotated_instance = []
                for counter in range(0, pred.__len__()):
                    if pred[counter].json_data()['score'] < pred_overall_score_threshod:
                        wrong_annotated_instance.append(counter)
                for _index in sorted(wrong_annotated_instance, reverse=True):
                    del pred[_index]

                if gt.__len__() == 2 and pred.__len__() == 3:
                    wrong_annotated_instance = 0
                    for counter in range(0,3):
                        if pred[counter].json_data()['score']<pred[wrong_annotated_instance].json_data()['score']:
                            wrong_annotated_instance = counter
                    del pred[wrong_annotated_instance]
                    del gt[wrong_annotated_instance]

                if gt.__len__()==1 and pred.__len__()==1:
                    if pred[0].data.shape==(21,3) and gt[0]['keypoints'].shape==(21,3):
                        index_array.append(meta['dataset_index'])
                        pred_array.append(pred[0].data)
                        gt_array.append(gt[0]['keypoints'])
                elif gt.__len__()==2 and pred.__len__()==2:
                    left_pred_dist_from_left_gt = np.mean(
                        np.linalg.norm(pred[0].data[:, :2] - gt[0]['keypoints'][:, :2], axis=1))
                    left_pred_dist_from_right_gt = np.mean(
                        np.linalg.norm(pred[0].data[:, :2] - gt[1]['keypoints'][:, :2], axis=1))
                    if left_pred_dist_from_left_gt<left_pred_dist_from_right_gt:
                        if pred[0].data.shape==(21,3) and gt[0]['keypoints'].shape==(21,3):
                            index_array.append(meta['dataset_index'])
                            pred_array.append(pred[0].data)
                            gt_array.append(gt[0]['keypoints'])
                        if pred[1].data.shape == (21, 3) and gt[1]['keypoints'].shape == (21, 3):
                            index_array.append(meta['dataset_index'])
                            pred_array.append(pred[1].data)
                            gt_array.append(gt[1]['keypoints'])
                    if left_pred_dist_from_left_gt>left_pred_dist_from_right_gt:
                        if pred[0].data.shape==(21,3) and gt[1]['keypoints'].shape==(21,3):
                            index_array.append(meta['dataset_index'])
                            pred_array.append(pred[0].data)
                            gt_array.append(gt[1]['keypoints'])
                        if pred[1].data.shape == (21, 3) and gt[0]['keypoints'].shape == (21, 3):
                            index_array.append(meta['dataset_index'])
                            pred_array.append(pred[1].data)
                            gt_array.append(gt[0]['keypoints'])

                elif gt.__len__()==2 and pred.__len__()==1:
                    pred_dist_from_left_gt = np.mean(np.linalg.norm(pred[0].data[:,:2]-gt[0]['keypoints'][:,:2], axis=1))
                    pred_dist_from_right_gt = np.mean(np.linalg.norm(pred[0].data[:, :2] - gt[1]['keypoints'][:, :2], axis=1))
                    if pred_dist_from_left_gt<pred_dist_from_right_gt:
                        if pred[0].data.shape==(21,3) and gt[0]['keypoints'].shape==(21,3):
                            index_array.append(meta['dataset_index'])
                            pred_array.append(pred[0].data)
                            gt_array.append(gt[0]['keypoints'])
                    elif pred_dist_from_left_gt>pred_dist_from_right_gt:
                        if pred[0].data.shape==(21,3) and gt[1]['keypoints'].shape==(21,3):
                            index_array.append(meta['dataset_index'])
                            pred_array.append(pred[0].data)
                            gt_array.append(gt[1]['keypoints'])

                elif gt.__len__()==1 and pred.__len__()==2:
                    gt_dist_from_left_pred = np.mean(np.linalg.norm(pred[0].data[:,:2]-gt[0]['keypoints'][:,:2], axis=1))
                    gt_dist_from_right_pred = np.mean(np.linalg.norm(pred[1].data[:, :2] - gt[0]['keypoints'][:, :2], axis=1))
                    if gt_dist_from_left_pred<gt_dist_from_right_pred:
                        if pred[0].data.shape==(21,3) and gt[0]['keypoints'].shape==(21,3):
                            index_array.append(meta['dataset_index'])
                            pred_array.append(pred[0].data)
                            gt_array.append(gt[0]['keypoints'])
                    elif gt_dist_from_left_pred>gt_dist_from_right_pred:
                        if pred[1].data.shape==(21,3) and gt[0]['keypoints'].shape==(21,3):
                            index_array.append(meta['dataset_index'])
                            pred_array.append(pred[1].data)
                            gt_array.append(gt[0]['keypoints'])

                    pass

            except:
                failure_counter += 1
                pass
            b += 1
            # print('b={}'.format(b))
            percent_completed=b/data.__len__()*100
            print('progress = {:.2f}'.format(percent_completed))
            print('failure_counter = {}'.format(failure_counter))
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/index_array.npy'.format(eval_dataset,checkpoint_name),index_array)
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/pred_array.npy'.format(eval_dataset, checkpoint_name),pred_array)
    np.save(
        '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/gt_array.npy'.format(eval_dataset,checkpoint_name), gt_array)



def posedataset_multi_predict(checkpoint_name, eval_dataset):
    args = cli()

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList_PoseDataset_multi(image_dir=args.images[0], mode='evaluation', preprocess=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_anns_meta)

    # # visualizers
    # keypoint_painter = show.KeypointPainter(
    #     color_connections=not args.monocolor_connections,
    #     linewidth=args.line_width,
    # )
    # annotation_painter = show.AnnotationPainter(keypoint_painter=keypoint_painter)
    b = 0
    pred_array = []
    gt_array = []
    pred_names= []
    for batch_i, (image_tensors_batch, gt_batch, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(model, image_tensors_batch, device=args.device)

        # import pdb; pdb.set_trace()
        # unbatch
        for pred, gt, meta in zip(pred_batch, gt_batch, meta_batch):
            failure_counter = 0
            try:
                if preprocess is not None:
                    pred = preprocess.annotations_inverse(pred, meta)
                if b>185:
                    if pred[0].data.shape == (21, 3) and gt[0]['keypoints'].shape == (21, 3):
                        # with open(meta['file_name'], 'rb') as f:
                        #     cpu_image = PIL.Image.open(f).convert('RGB')
                        #
                        #     # uncomment for PoseDataset
                        #     img = cpu_image
                        #     order = 1  # order of resize interpolation; 1 means linear interpolation
                        #     w, h = img.size
                        #     # keep aspect ratio the same
                        #     reference_edge = 224
                        #     target_max_edge = reference_edge
                        #     max_edge = max(h, w)
                        #     ratio_factor = target_max_edge / max_edge
                        #     target_h = int(ratio_factor * h)
                        #     target_w = int(ratio_factor * w)
                        #     im_np = np.asarray(img)
                        #     im_np = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1), order=order)
                        #     img = PIL.Image.fromarray(im_np)
                        #     assert img.size[0] == target_w
                        #     assert img.size[1] == target_h
                        #     # pad frames
                        #     img = np.asarray(img)
                        #     pad_up = (reference_edge - img.shape[0]) // 2
                        #     pad_down = (reference_edge - img.shape[0]) // 2
                        #     pad_left = (reference_edge - img.shape[1]) // 2
                        #     pad_right = (reference_edge - img.shape[1]) // 2
                        #     img = np.pad(img, pad_width=((pad_up, pad_down), (pad_left, pad_right), (0, 0)),
                        #                  mode='symmetric')
                        #     cpu_image = Image.fromarray(img.astype('uint8'), 'RGB')
                            # import matplotlib.pyplot as plt
                            # fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                            # ax.imshow(np.asarray(img))
                            # # bool_annotated_joints_1 = _anns[0][0]['keypoints'][:, 2] == 2
                            # ax.plot(gt[0]['keypoints'][:, 0],
                            #            gt[0]['keypoints'][:, 1], 'ro')
                            # ax.plot(pred[0].data[:, 0],
                            #            pred[0].data[:, 1], 'bx')
                            # for txt in range(0, 21):
                            #     ax.annotate(txt, (gt[0]['keypoints'][txt, 0]-3, gt[0]['keypoints'][txt, 1]-3), c='r')
                            #     ax.annotate(txt, (pred[0].data[txt, 0]+3, pred[0].data[txt, 1]+3), c='b')
                            # plt.title('b={}'.format(b))
                            # plt.show(block=True)

                        pred_names.append(meta['file_name'])
                        pred_array.append(pred[0].data)
                        gt_array.append(gt[0]['keypoints'])
            except:
                failure_counter += 1
                pass

            b += 1
            # print('b={}'.format(b))
            percent_completed=b/data.__len__()*100
            print('progress = {:.2f}; failure_counter = {}'.format(percent_completed,failure_counter))
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/pred_array.npy'.format(eval_dataset, checkpoint_name),pred_array)
    np.save(
        '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/gt_array.npy'.format(eval_dataset, checkpoint_name), gt_array)
    np.save(
        '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/pred_names.npy'.format(eval_dataset, checkpoint_name), pred_names)



def onehand10k_multi_predict(checkpoint_name, eval_dataset):
    args = cli()

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList_OneHand10K(args.images[0], mode='evaluation', preprocess=preprocess)
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

            if preprocess is not None:
                pred = preprocess.annotations_inverse(pred, meta)

            # error = pred[0].data - gt[0]['keypoints']
            try:
                if pred[0].data.shape==(21,3) and gt[0]['keypoints'].shape==(21,3):
                    pred_array.append(pred[0].data)
                    gt_array.append(gt[0]['keypoints'])
            except:
                pass

    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/pred_array_{}.npy'.format(eval_dataset, checkpoint_name), pred_array)
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/gt_array_{}.npy'.format(eval_dataset, checkpoint_name), gt_array)

def freihand_multi_predict_google(checkpoint_name, eval_dataset, mode = 'evaluation'):
    number_unique_imgs = db_size('training')
    K_list, xyz_list = load_db_annotation('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/Freihand_pub_v2', 'training')

    if mode == 'evaluation':
        number_version = 1
    else:
        raise AssertionError('number_version not defined!')
    if mode == 'evaluation':
        version = sample_version.auto
    else:
        raise AssertionError('version not defined!')
    b = 0
    error_counter = 0
    google_pred_array = []
    google_gt_array = []
    for index in range(34, number_unique_imgs*number_version):
        b += 1
        img = read_img(index%number_unique_imgs, '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/Freihand_pub_v2', 'training', version)
        img = Image.fromarray(img.astype('uint8'), 'RGB')

        # fig, ax = plt.subplots(1, 3, figsize=(12, 12))
        # ax[0].imshow(img)
        # rescale image
        order = 1  # order of resize interpolation; 1 means linear interpolation
        w, h = img.size

        target_h = 480
        target_w = 480
        im_np = np.asarray(img)
        img = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1), order=order)
        # ax[1].imshow(img)
        pad_up = (480-img.shape[0])//2
        pad_down = (480-img.shape[0])//2
        pad_left = (640-img.shape[1])//2
        pad_right = (640-img.shape[1])//2
        img = np.pad(img, pad_width=((pad_up, pad_down), (pad_left, pad_right), (0, 0)), mode='symmetric')
        # ax[2].imshow(img)

        try:
            percent_completed = b / (number_unique_imgs * number_version) * 100
            print('progress = {:.2f}'.format(percent_completed))
            pred = google_predict_only(img)
            assert pred.shape == (21, 2)
        except:
            error_counter += 1
            google_pred_array.append(np.zeros((21,3)))
            # annotation for this frame
            K, xyz = K_list[index], xyz_list[index]
            K, xyz = [np.array(x) for x in [K, xyz]]
            uv = projectPoints(xyz, K)  # 2D gt keypoints
            visibility_flag = 2
            uv = np.hstack((uv, visibility_flag * np.ones((uv.shape[0], 1))))
            google_gt_array.append(uv)
            print('error in index = {}; total error = {:.2f} %'.format(index, error_counter / (number_unique_imgs * number_version) * 100))
            continue


        # ax[2].plot(pred[:, 0], pred[:, 1], 'ro')
        # n = [21]
        # for txt in range (0, 21):
        #     ax[2].annotate(txt, (pred[txt, 0], pred[txt, 1]), c='w')

        # rescale back predictions
        # x_scale = (img.shape[1] - 1) / (w - 1)
        # y_scale = (img.shape[0] - 1) / (h - 1)
        # pred[:, 0] = pred[:, 0] / x_scale
        # pred[:, 1] = pred[:, 1] / y_scale
        # return back pad effect on annotations to prepare for rescale back
        pred[:, 0] = pred[:, 0] - pad_left
        pred[:, 1] = pred[:, 1] - pad_up
        # rescale back predictions
        x_scale = (target_w - 1) / (w - 1)
        y_scale = (target_h - 1) / (h - 1)
        pred[:, 0] = pred[:, 0] / x_scale
        pred[:, 1] = pred[:, 1] / y_scale

        # ax[0].plot(pred[:, 0], pred[:, 1], 'ro')
        # for txt in range(0, 21):
        #     ax[0].annotate(txt, (pred[txt, 0], pred[txt, 1]), c='w')
        # plt.show()

        conf_flag = 1
        pred = np.hstack((pred, conf_flag*np.ones((pred.shape[0], 1))))

        # annotation for this frame
        K, xyz = K_list[index], xyz_list[index]
        K, xyz = [np.array(x) for x in [K, xyz]]
        uv = projectPoints(xyz, K) # 2D gt keypoints
        visibility_flag = 2
        uv = np.hstack((uv, visibility_flag*np.ones((uv.shape[0], 1))))

        if pred.shape == (21, 3):
            google_pred_array.append(pred)
            google_gt_array.append(uv)

    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/google_pred_array_{}.npy'.format(
        eval_dataset, checkpoint_name), google_pred_array)
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/google_gt_array_{}.npy'.format(
        eval_dataset, checkpoint_name), google_gt_array)


def PCK_plot(checkpoint_name, eval_dataset):
    pred_array = np.load(
        '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/pred_array.npy'.format(
            eval_dataset, checkpoint_name))
    gt_array = np.load(
        '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/gt_array.npy'.format(eval_dataset,
                                                                                                           checkpoint_name))
    # pred_names = np.load(
    #     '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/pred_names.npy'.format(
    #         eval_dataset, checkpoint_name))

    # index_array = np.load(
    #     '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/index_array.npy'.format(eval_dataset,
    #                                                                                                        checkpoint_name))

    # # TODO uncomment for rhd? and?
    # pred_array_correct = np.copy(pred_array)
    # pred_array_correct[:, 4, :] = pred_array[:, 20, :]
    # pred_array_correct[:, 3, :] = pred_array[:, 19, :]
    # pred_array_correct[:, 2, :] = pred_array[:, 18, :]
    # pred_array_correct[:, 1, :] = pred_array[:, 17, :]
    # pred_array_correct[:, 5, :] = pred_array[:, 13, :]
    # pred_array_correct[:, 6, :] = pred_array[:, 14, :]
    # pred_array_correct[:, 7, :] = pred_array[:, 15, :]
    # pred_array_correct[:, 8, :] = pred_array[:, 16, :]
    #
    # pred_array_correct[:, 20, :] = pred_array[:, 4, :]
    # pred_array_correct[:, 19, :] = pred_array[:, 3, :]
    # pred_array_correct[:, 18, :] = pred_array[:, 2, :]
    # pred_array_correct[:, 17, :] = pred_array[:, 1, :]
    # pred_array_correct[:, 13, :] = pred_array[:, 5, :]
    # pred_array_correct[:, 14, :] = pred_array[:, 6, :]
    # pred_array_correct[:, 15, :] = pred_array[:, 7, :]
    # pred_array_correct[:, 16, :] = pred_array[:, 8, :]
    # pred_array = np.copy(pred_array_correct)


    def PCK(PCK_thresh, pred_score_thresh=0.15, gt_conf_thresh=0):
        total_counted_data = 0
        total_correct_data = 0
        total_counted_data_fingers = np.zeros(21)
        total_correct_data_fingers = np.zeros(21)

        correction_factor = 1 # (320 / 224) for rhd_resize
        for data_id in range(0, pred_array.shape[0]):

            # # guided head
            # # exchange thumb and index with little and ring if thumb_mcp is further than little_mcp to the palm
            # # _palm = pred_array[data_id, 0, :2]
            # # _thumb_mcp = pred_array[data_id, 1, :2]
            # # _little_mcp = pred_array[data_id, 17, :2]
            # if all([pred_array.shape[1]==21, pred_array[data_id, 0, 2]>pred_score_thresh, pred_array[data_id, 1, 2]>pred_score_thresh, pred_array[data_id, 17, 2]>pred_score_thresh, np.linalg.norm(pred_array[data_id, 0, :2] - pred_array[data_id, 17, :2], axis=0)<np.linalg.norm(pred_array[data_id, 0, :2] - pred_array[data_id, 1, :2], axis=0)]):
            #     correct_pred = np.zeros_like(pred_array[data_id, :, :])
            #
            #     correct_pred[1, :] = pred_array[data_id, 17, :]
            #     correct_pred[2, :] = pred_array[data_id, 18, :]
            #     correct_pred[3, :] = pred_array[data_id, 19, :]
            #     correct_pred[4, :] = pred_array[data_id, 20, :]
            #     correct_pred[17, :] = pred_array[data_id, 1, :]
            #     correct_pred[18, :] = pred_array[data_id, 2, :]
            #     correct_pred[19, :] = pred_array[data_id, 3, :]
            #     correct_pred[20, :] = pred_array[data_id, 4, :]
            #     correct_pred[5, :] = pred_array[data_id, 13, :]
            #     correct_pred[6, :] = pred_array[data_id, 14, :]
            #     correct_pred[7, :] = pred_array[data_id, 15, :]
            #     correct_pred[8, :] = pred_array[data_id, 16, :]
            #     correct_pred[13, :] = pred_array[data_id, 5, :]
            #     correct_pred[14, :] = pred_array[data_id, 6, :]
            #     correct_pred[15, :] = pred_array[data_id, 7, :]
            #     correct_pred[16, :] = pred_array[data_id, 8, :]
            #     pred_array[data_id, :, :] = correct_pred
            #     del correct_pred
            #     print('!!!!!!!!guided_head!!!!!!!PCK_thresh={};data_id={}'.format(PCK_thresh,
            #                                                                                   data_id))

            # print(index_array[data_id])
            # bool_gt_acceptable_data = (gt_array[data_id, :, 2] > gt_conf_thresh)
            bool_acceptable_data = (pred_array[data_id, :, 2] > pred_score_thresh) * (
                        gt_array[data_id, :, 2] > gt_conf_thresh)
            _errors = pred_array[data_id, bool_acceptable_data, :2] - gt_array[data_id, bool_acceptable_data, :2]
            _norms = np.linalg.norm(_errors, axis=1)*correction_factor

# # ---------------------------------------------------------------------------
#             with open('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/RHD_published_v2/evaluation/color/%.5d.png' %
#                       index_array[data_id], 'rb') as f:
#                 cpu_image = PIL.Image.open(f).convert('RGB')
#                 # uncomment for rhd
#                 img = cpu_image
#                 # rescale image
#                 order = 1  # order of resize interpolation; 1 means linear interpolation
#                 w, h = img.size
#                 # keep aspect ratio the same
#                 reference_edge = 224
#                 target_max_edge = reference_edge
#                 max_edge = max(h, w)
#                 ratio_factor = target_max_edge / max_edge
#                 target_h = int(ratio_factor * h)
#                 target_w = int(ratio_factor * w)
#                 im_np = np.asarray(img)
#                 im_np = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1), order=order)
#                 img = PIL.Image.fromarray(im_np)
#                 assert img.size[0] == target_w
#                 assert img.size[1] == target_h
#                 # pad frames
#                 img = np.asarray(img)
#                 pad_up = (reference_edge - img.shape[0]) // 2
#                 pad_down = (reference_edge - img.shape[0]) // 2
#                 pad_left = (reference_edge - img.shape[1]) // 2
#                 pad_right = (reference_edge - img.shape[1]) // 2
#                 img = np.pad(img, pad_width=((pad_up, pad_down), (pad_left, pad_right), (0, 0)), mode='symmetric')
#                 import matplotlib.pyplot as plt
#                 fig, ax = plt.subplots(1, 1, figsize=(12, 12))
#                 ax.imshow(np.asarray(img))
#                 ax.plot(gt_array[data_id, :, 0],
#                         gt_array[data_id, :, 1], 'ro')
#                 ax.plot(pred_array[data_id, :, 0],
#                         pred_array[data_id, :, 1], 'bx')
#
#                 for txt in range(0, 21):
#                     ax.annotate(txt, (gt_array[data_id, txt, 0] - 1, gt_array[data_id, txt, 1] - 1),
#                                 c='r')
#                     ax.annotate(txt, (pred_array[data_id, txt, 0] + 1, pred_array[data_id, txt, 1] + 1), c='b')
#                 plt.show()
# # ---------------------------------------------------------------------------

            for joint_id in range(0, 21):
                if bool_acceptable_data[joint_id] == True:
                    _error = pred_array[data_id, joint_id, :2] - gt_array[data_id, joint_id, :2]
                    _norm = np.linalg.norm(_error, axis=0)*correction_factor
                    total_correct_data_fingers[joint_id] = total_correct_data_fingers[joint_id] + sum(
                        [_norm] < PCK_thresh)
                    total_counted_data_fingers[joint_id] = total_counted_data_fingers[joint_id] + 1
            # if sum(_norms < PCK_thresh)!=_norms.shape[0]:
            #     print('index at pred_array={}, no. data out thresh={}'.format(data_id,_norms.shape[0]-sum(_norms < PCK_thresh)))
            total_correct_data += sum(_norms < PCK_thresh)
            total_counted_data += _norms.shape[0]

            # # for debug: sho names of wrong frames
            # if PCK_thresh == np.linspace(0, 30, 60)[30]:
            #     if sum(_norms > PCK_thresh)>4:
            #         print('***large error***PCK_thresh={};pred_names[data_id]={};_norms={}'.format(PCK_thresh,pred_names[data_id][-7:],_norms))

            # # modified definition: count failures
            # total_counted_data += sum(bool_gt_acceptable_data)

        PCK_value = total_correct_data / total_counted_data

        PCK_value_fingers = total_correct_data_fingers / total_counted_data_fingers

        return PCK_value, PCK_value_fingers

    num_intervals = 60
    max_error = 30
    PCK_thresh = np.linspace(0, max_error, num_intervals)
    # PCK_thresh = np.geomspace(0.5, max_error, num_intervals)

    y = []
    y_joints = np.zeros((21, num_intervals))
    for iter in range(0, num_intervals):
        PCK_value, PCK_value_fingers = PCK(PCK_thresh[iter])
        y_joints[:, iter] = PCK_value_fingers
        y.append(PCK_value)
        print('PCK_thresh[iter] = {:.2f}: PCK_value = {:.2f}; progress = {:.2f} %'.format(PCK_thresh[iter], PCK_value,
                                                                                          iter / num_intervals * 100))

    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/2DPCKvsPXLs.npy'.format(
        eval_dataset, checkpoint_name), np.vstack((PCK_thresh, np.asarray(y))))
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/2DPCK_fingers.npy'.format(
        eval_dataset, checkpoint_name), y_joints)
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/PCK_thresh.npy'.format(
        eval_dataset, checkpoint_name), PCK_thresh)
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/y.npy'.format(eval_dataset,
                                                                                                        checkpoint_name),
            y)

    # y_joints = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/2DPCK_fingers.npy'.format(
    #     eval_dataset, checkpoint_name))
    # PCK_thresh = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/PCK_thresh.npy'.format(
    #     eval_dataset, checkpoint_name))
    # y = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/y.npy'.format(eval_dataset,
    #                                                                                                     checkpoint_name))

    # attention_paper_2DPCK_PXLs_freihand = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/attention_network_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')
    # MobilePose_paper_2DPCK_PXLs_freihand = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/MobilePose_network_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')
    # EfficientDet_paper_2DPCK_PXLs_freihand = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/EfficientDet_network_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')

    # CPM_2DPCKvsPXLs = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/CPM_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')
    # CPM_2DPCKvsPXLs = np.vstack((np.sort(CPM_2DPCKvsPXLs[:, 0]), np.sort(CPM_2DPCKvsPXLs[:, 1]))).T
    # CPM_gt_2DPCKvsPXLs = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/CPM_gt_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')
    # CPM_gt_2DPCKvsPXLs = np.vstack((np.sort(CPM_gt_2DPCKvsPXLs[:, 0]), np.sort(CPM_gt_2DPCKvsPXLs[:, 1]))).T
    # CPMAtt_2DPCKvsPXLs = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/CPMAtt_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')
    # CPMAtt_2DPCKvsPXLs = np.vstack((np.sort(CPMAtt_2DPCKvsPXLs[:, 0]), np.sort(CPMAtt_2DPCKvsPXLs[:, 1]))).T
    # CPMAtt_gt_2DPCKvsPXLs = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/CPMAtt_gt_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')
    # CPMAtt_gt_2DPCKvsPXLs = np.vstack((np.sort(CPMAtt_gt_2DPCKvsPXLs[:, 0]), np.sort(CPMAtt_gt_2DPCKvsPXLs[:, 1]))).T
    # RGBPI_2DPCKvsPXLs = np.genfromtxt(
    #     '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/RGBPI_2DPCKvsPXLs.csv'.format(
    #         eval_dataset), delimiter=',')
    # RGBPI_2DPCKvsPXLs = np.vstack((np.sort(RGBPI_2DPCKvsPXLs[:, 0]), np.sort(RGBPI_2DPCKvsPXLs[:, 1]))).T
    # Wang1_2DPCKvsPXLs = np.genfromtxt(
    #     '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/Wang1_2DPCKvsPXLs.csv'.format(
    #         eval_dataset), delimiter=',')
    # Wang1_2DPCKvsPXLs = np.vstack((np.sort(Wang1_2DPCKvsPXLs[:, 0]), np.sort(Wang1_2DPCKvsPXLs[:, 1]))).T
    # Wang2_2DPCKvsPXLs = np.genfromtxt(
    #     '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/Wang2_2DPCKvsPXLs.csv'.format(
    #         eval_dataset), delimiter=',')
    # Wang2_2DPCKvsPXLs = np.vstack((np.sort(Wang2_2DPCKvsPXLs[:, 0]), np.sort(Wang2_2DPCKvsPXLs[:, 1]))).T



    # handPifPaf_paper_2DPCK_PXLs_freihand = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}_2DPCKvsPXLs.npy'.format(eval_dataset, checkpoint_name))
    # PCK_thresh = handPifPaf_paper_2DPCK_PXLs_freihand[0, :]
    # y = handPifPaf_paper_2DPCK_PXLs_freihand[1, :]



    from sklearn.metrics import auc
    AUC = auc(PCK_thresh, np.asarray(y))/max_error
    # AUC_attention = auc(attention_paper_2DPCK_PXLs_freihand[:,0], attention_paper_2DPCK_PXLs_freihand[:,1])/max_error
    # AUC_mobilepose = auc(MobilePose_paper_2DPCK_PXLs_freihand[:, 0], MobilePose_paper_2DPCK_PXLs_freihand[:, 1])/max_error
    # AUC_efficientdet = auc(EfficientDet_paper_2DPCK_PXLs_freihand[:, 0], EfficientDet_paper_2DPCK_PXLs_freihand[:, 1])/max_error

    # AUC2 = auc(RGBPI_2DPCKvsPXLs[:, 0], RGBPI_2DPCKvsPXLs[:, 1])/(max(RGBPI_2DPCKvsPXLs[:, 0])-min(RGBPI_2DPCKvsPXLs[:,0]))

    print('AUC of 2PCKvsPX is =',AUC)
    # print('AUC_attention of 2PCKvsPX is =',AUC_attention)
    # print('AUC_mobilepose of 2PCKvsPX is =',AUC_mobilepose)
    # print('AUC_efficientdet of 2PCKvsPX is =',AUC_efficientdet)


    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    axes.plot(PCK_thresh, np.asarray(y), label='handPifPaf', c='b')
    # axes.plot(attention_paper_2DPCK_PXLs_freihand[:, 0], attention_paper_2DPCK_PXLs_freihand[:, 1], label='Attention', c='m')
    # axes.plot(MobilePose_paper_2DPCK_PXLs_freihand[:, 0], MobilePose_paper_2DPCK_PXLs_freihand[:, 1], label='MobilePose224V2', c='g')
    # axes.plot(EfficientDet_paper_2DPCK_PXLs_freihand[:, 0], EfficientDet_paper_2DPCK_PXLs_freihand[:, 1], label='EfficientDet224', c='brown')
    # axes.plot(CPM_2DPCKvsPXLs[:, 0], CPM_2DPCKvsPXLs[:, 1], label='CPM', c='green')
    # axes.plot(CPM_gt_2DPCKvsPXLs[:, 0], CPM_gt_2DPCKvsPXLs[:, 1], label='CPM_gt', c='orange')
    # axes.plot(CPMAtt_2DPCKvsPXLs[:, 0], CPMAtt_2DPCKvsPXLs[:, 1], label='CPMAtt', c='olive')
    # axes.plot(CPMAtt_gt_2DPCKvsPXLs[:, 0], CPMAtt_gt_2DPCKvsPXLs[:, 1], label='CPMAtt_gt', c='m')
    # axes.plot(RGBPI_2DPCKvsPXLs[:, 0], RGBPI_2DPCKvsPXLs[:, 1], label='RGB+PI', c='brown')
    # axes.plot(Wang1_2DPCKvsPXLs[:, 0], Wang1_2DPCKvsPXLs[:, 1], label='Wang1', c='red')
    # axes.plot(Wang2_2DPCKvsPXLs[:, 0], Wang2_2DPCKvsPXLs[:, 1], label='Wang2', c='black')



    axes.set_xlabel('Error Threshold [px]')
    axes.set_ylabel('2D PCK')
    axes.set_title('Percentage of Correct Key-points vs Error Threshold')
    axes.grid(True)
    axes.legend()
    axes.set_ylim([0, 1])
    axes.set_xlim([0, max_error])
    plt.savefig(
        '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/2DPCK.png'.format(eval_dataset,
                                                                                                        checkpoint_name),
        format='png')
    # plt.show()

    fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    axes2.plot(PCK_thresh, y_joints[0, :], label='{}'.format(FREIHAND_KEYPOINTS[0]), c='k')
    cm = plt.get_cmap('tab20')
    axes2.set_prop_cycle(color=[cm(1. * i / 21) for i in range(21)])
    for joint_id in range(1, 21):
        axes2.plot(PCK_thresh, y_joints[joint_id, :], label='{}'.format(FREIHAND_KEYPOINTS[joint_id]))

    axes2.set_xlabel('Error Threshold [px]')
    axes2.set_ylabel('2D PCK')
    axes2.set_title('Percentage of Correct Key-points vs Error Threshold per Joint')
    axes2.grid(True)
    axes2.legend()
    axes2.set_ylim([0, 1])
    axes2.set_xlim([0, max_error])
    plt.savefig('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/2DPCK_joints.png'.format(
        eval_dataset, checkpoint_name), format='png')
    # plt.show()
    print('PCK_plot Successfully Ended!')


def PCK_normalized_plot(checkpoint_name, eval_dataset):
    pred_array = np.load(
        '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/pred_array.npy'.format(
            eval_dataset, checkpoint_name))
    gt_array = np.load(
        '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/gt_array.npy'.format(eval_dataset,
                                                                                                           checkpoint_name))
    # index_array = np.load(
    #     '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/index_array.npy'.format(eval_dataset,
    #                                                                                                        checkpoint_name))
    assert pred_array.shape[0] == gt_array.shape[0]
    # assert index_array.shape[0] == gt_array.shape[0]

    # # TODO are these hflip correction necessary?
    # pred_array_correct = np.copy(pred_array)
    # pred_array_correct[:, 4, :] = pred_array[:, 20, :]
    # pred_array_correct[:, 3, :] = pred_array[:, 19, :]
    # pred_array_correct[:, 2, :] = pred_array[:, 18, :]
    # pred_array_correct[:, 1, :] = pred_array[:, 17, :]
    # pred_array_correct[:, 5, :] = pred_array[:, 13, :]
    # pred_array_correct[:, 6, :] = pred_array[:, 14, :]
    # pred_array_correct[:, 7, :] = pred_array[:, 15, :]
    # pred_array_correct[:, 8, :] = pred_array[:, 16, :]
    #
    # pred_array_correct[:, 20, :] = pred_array[:, 4, :]
    # pred_array_correct[:, 19, :] = pred_array[:, 3, :]
    # pred_array_correct[:, 18, :] = pred_array[:, 2, :]
    # pred_array_correct[:, 17, :] = pred_array[:, 1, :]
    # pred_array_correct[:, 13, :] = pred_array[:, 5, :]
    # pred_array_correct[:, 14, :] = pred_array[:, 6, :]
    # pred_array_correct[:, 15, :] = pred_array[:, 7, :]
    # pred_array_correct[:, 16, :] = pred_array[:, 8, :]
    # pred_array = np.copy(pred_array_correct)


    def PCK(PCK_thresh, pred_score_thresh=0.15, gt_conf_thresh=0):
        total_counted_data = 0
        total_correct_data = 0
        total_counted_data_fingers = np.zeros(21)
        total_correct_data_fingers = np.zeros(21)
        bbx_factor=2.2
        for data_id in range(0, pred_array.shape[0]):
            tightest_edge_bbox = bbx_factor*max(abs(max(gt_array[data_id, :, 0])-min(gt_array[data_id, :, 0])), abs(max(gt_array[data_id, :, 1])-min(gt_array[data_id, :, 1])))
            # print(index_array[data_id])
            # bool_gt_acceptable_data = (gt_array[data_id, :, 2] > gt_conf_thresh)
            bool_acceptable_data = (pred_array[data_id, :, 2] > pred_score_thresh) * (
                        gt_array[data_id, :, 2] > gt_conf_thresh)
            _errors = pred_array[data_id, bool_acceptable_data, :2] - gt_array[data_id, bool_acceptable_data, :2]
            _norms = np.linalg.norm(_errors, axis=1)

# # ---------------------------------------------------------------------------
#             with open('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/RHD_published_v2/evaluation/color/%.5d.png' %
#                       index_array[data_id], 'rb') as f:
#                 cpu_image = PIL.Image.open(f).convert('RGB')
#                 # uncomment for rhd
#                 img = cpu_image
#                 # rescale image
#                 order = 1  # order of resize interpolation; 1 means linear interpolation
#                 w, h = img.size
#                 # keep aspect ratio the same
#                 reference_edge = 224
#                 target_max_edge = reference_edge
#                 max_edge = max(h, w)
#                 ratio_factor = target_max_edge / max_edge
#                 target_h = int(ratio_factor * h)
#                 target_w = int(ratio_factor * w)
#                 im_np = np.asarray(img)
#                 im_np = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1), order=order)
#                 img = PIL.Image.fromarray(im_np)
#                 assert img.size[0] == target_w
#                 assert img.size[1] == target_h
#                 # pad frames
#                 img = np.asarray(img)
#                 pad_up = (reference_edge - img.shape[0]) // 2
#                 pad_down = (reference_edge - img.shape[0]) // 2
#                 pad_left = (reference_edge - img.shape[1]) // 2
#                 pad_right = (reference_edge - img.shape[1]) // 2
#                 img = np.pad(img, pad_width=((pad_up, pad_down), (pad_left, pad_right), (0, 0)), mode='symmetric')
#                 import matplotlib.pyplot as plt
#                 fig, ax = plt.subplots(1, 1, figsize=(12, 12))
#                 ax.imshow(np.asarray(img))
#                 ax.plot(gt_array[data_id, :, 0],
#                         gt_array[data_id, :, 1], 'ro')
#                 ax.plot(pred_array[data_id, :, 0],
#                         pred_array[data_id, :, 1], 'bx')
#
#                 for txt in range(0, 21):
#                     ax.annotate(txt, (gt_array[data_id, txt, 0] - 1, gt_array[data_id, txt, 1] - 1),
#                                 c='r')
#                     ax.annotate(txt, (pred_array[data_id, txt, 0] + 1, pred_array[data_id, txt, 1] + 1), c='b')
#                 plt.show()
# # ---------------------------------------------------------------------------

            for joint_id in range(0, 21):
                if bool_acceptable_data[joint_id] == True:
                    _error = pred_array[data_id, joint_id, :2] - gt_array[data_id, joint_id, :2]
                    _norm = np.linalg.norm(_error, axis=0)
                    total_correct_data_fingers[joint_id] = total_correct_data_fingers[joint_id] + sum(
                        [_norm/tightest_edge_bbox] < PCK_thresh)
                    total_counted_data_fingers[joint_id] = total_counted_data_fingers[joint_id] + 1

            total_correct_data += sum(_norms/tightest_edge_bbox < PCK_thresh)
            total_counted_data += _norms.shape[0]
            # # modified definition: count failures
            # total_counted_data += sum(bool_gt_acceptable_data)

        PCK_value = total_correct_data / total_counted_data

        PCK_value_fingers = total_correct_data_fingers / total_counted_data_fingers

        return PCK_value, PCK_value_fingers

    num_intervals = 40
    max_error = 0.20
    PCK_thresh = np.linspace(0.01, max_error, num_intervals)
    # PCK_thresh = np.geomspace(0.5, max_error, num_intervals)

    y = []
    y_joints = np.zeros((21, num_intervals))
    for iter in range(0, num_intervals):
        PCK_value, PCK_value_fingers = PCK(PCK_thresh[iter])
        y_joints[:, iter] = PCK_value_fingers
        y.append(PCK_value)
        print('PCK_thresh[iter] = {:.2f}: PCK_value = {:.2f}; progress = {:.2f} %'.format(PCK_thresh[iter], PCK_value,
                                                                                          iter / num_intervals * 100))

    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/normalized_2DPCKvsPXLs.npy'.format(
        eval_dataset, checkpoint_name), np.vstack((PCK_thresh, np.asarray(y))))
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/normalized_2DPCK_fingers.npy'.format(
        eval_dataset, checkpoint_name), y_joints)
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/normalized_PCK_thresh.npy'.format(
        eval_dataset, checkpoint_name), PCK_thresh)
    np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/normalized_y.npy'.format(eval_dataset,
                                                                                                        checkpoint_name),
            y)

    # y_joints = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/normalized_2DPCK_fingers.npy'.format(
    #     eval_dataset, checkpoint_name))
    # PCK_thresh = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/normalized_PCK_thresh.npy'.format(
    #     eval_dataset, checkpoint_name))
    # y = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/normalized_y.npy'.format(eval_dataset,
    #                                                                                                     checkpoint_name))

    # y_joints = np.load(
    #     '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/normalized_2DPCK_fingers_with_bbx_factor.npy'.format(
    #         eval_dataset, checkpoint_name))
    # PCK_thresh = np.load(
    #     '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/normalized_PCK_thresh_with_bbx_factor.npy'.format(
    #         eval_dataset, checkpoint_name))
    # y = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/normalized_y_with_bbx_factor.npy'.format(
    #     eval_dataset,
    #     checkpoint_name))

    # attention_paper_2DPCK_PXLs_freihand = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/attention_network_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')
    # MobilePose_paper_2DPCK_PXLs_freihand = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/MobilePose_network_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')
    # EfficientDet_paper_2DPCK_PXLs_freihand = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/EfficientDet_network_2DPCKvsPXLs.csv'.format(eval_dataset), delimiter=',')

    AGMN_jointly_fine_tuned = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/AGMN_jointly_fine_tuned.csv'.format(eval_dataset), delimiter=',')
    AGMN_separately_trained = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/AGMN_separately_trained.csv'.format(eval_dataset), delimiter=',')
    AGMN_with_gt_relative_positions = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/AGMN_with_gt_relative_positions.csv'.format(eval_dataset), delimiter=',')
    AGMN_CPM = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/AGMN_CPM.csv'.format(eval_dataset), delimiter=',')
    # NSRM_CPM = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/NSRM_CPM.csv'.format(eval_dataset), delimiter=',')
    # NSRM_LDM_G1 = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/NSRM_LDM_G1.csv'.format(eval_dataset), delimiter=',')
    # NSRM_LDM_G1and6 = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/NSRM_LDM_G1and6.csv'.format(eval_dataset), delimiter=',')
    # NSRM_LPM_G1 = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/NSRM_LPM_G1.csv'.format(eval_dataset), delimiter=',')
    # NSRM_LPM_G1and6 = np.genfromtxt('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/NSRM_LPM_G1and6.csv'.format(eval_dataset), delimiter=',')

    # handPifPaf_paper_2DPCK_PXLs_freihand = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}_2DPCKvsPXLs.npy'.format(eval_dataset, checkpoint_name))
    # PCK_thresh = handPifPaf_paper_2DPCK_PXLs_freihand[0, :]
    # y = handPifPaf_paper_2DPCK_PXLs_freihand[1, :]

    from sklearn.metrics import auc
    AUC = auc(PCK_thresh, np.asarray(y))/max_error

    AUC_AGMN_jointly_fine_tuned = auc(AGMN_jointly_fine_tuned[:, 0], AGMN_jointly_fine_tuned[:, 1]) / max_error
    AUCAGMN_separately_trained = auc(AGMN_separately_trained[:, 0], AGMN_separately_trained[:, 1])/max_error
    AUC_AGMN_with_gt_relative_positions = auc(AGMN_with_gt_relative_positions[:, 0], AGMN_with_gt_relative_positions[:, 1])/max_error
    AUC_AGMN_CPM = auc(AGMN_CPM[:, 0], AGMN_CPM[:, 1])/max_error


    # AUC_NSRM_LDM_G1 = auc(NSRM_LDM_G1[:-1, 0], NSRM_LDM_G1[:-1, 1])/max_error
    # AUC_NSRM_LDM_G1and6 = auc(NSRM_LDM_G1and6[:, 0], NSRM_LDM_G1and6[:, 1])/max_error
    # AUC_NSRM_LPM_G1 = auc(NSRM_LPM_G1[:, 0], NSRM_LPM_G1[:, 1])/max_error
    # AUC_NSRM_LPM_G1and6 = auc(NSRM_LPM_G1and6[:, 0], NSRM_LPM_G1and6[:, 1])/max_error


    print('handPifpaf AUC of normalized 2PCK is =',AUC)

    print('handPifpaf AUC_AGMN_jointly_fine_tuned of normalized 2PCK is =',AUC_AGMN_jointly_fine_tuned)
    print('handPifpaf AUCAGMN_separately_trained of normalized 2PCK is =',AUCAGMN_separately_trained)
    print('handPifpaf AUC_AGMN_with_gt_relative_positions of normalized 2PCK is =',AUC_AGMN_with_gt_relative_positions)
    print('handPifpaf AUC_AGMN_CPM of normalized 2PCK is =',AUC_AGMN_CPM)

    # print('handPifpaf AUC_NSRM_LDM_G1 of normalized 2PCK is =',AUC_NSRM_LDM_G1)
    # print('handPifpaf AUC_NSRM_LDM_G1and6 of normalized 2PCK is =',AUC_NSRM_LDM_G1and6)
    # print('handPifpaf AUC_NSRM_LPM_G1 of normalized 2PCK is =',AUC_NSRM_LPM_G1)
    # print('handPifpaf AUC_NSRM_LPM_G1and6 of normalized 2PCK is =',AUC_NSRM_LPM_G1and6)



    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    axes.plot(PCK_thresh, np.asarray(y), label='handPifPaf', c='b')
    # axes.plot(attention_paper_2DPCK_PXLs_freihand[:, 0], attention_paper_2DPCK_PXLs_freihand[:, 1], label='Attention', c='m')
    # axes.plot(MobilePose_paper_2DPCK_PXLs_freihand[:, 0], MobilePose_paper_2DPCK_PXLs_freihand[:, 1], label='MobilePose224V2', c='g')
    # axes.plot(EfficientDet_paper_2DPCK_PXLs_freihand[:, 0], EfficientDet_paper_2DPCK_PXLs_freihand[:, 1], label='EfficientDet224', c='brown')

    # panoptic SOA
    axes.plot(AGMN_jointly_fine_tuned[:, 0], AGMN_jointly_fine_tuned[:, 1], label='jointly_fine_tuned', c='tab:orange')
    axes.plot(AGMN_separately_trained[:, 0], AGMN_separately_trained[:, 1], label='separately_trained', c='tab:green')
    axes.plot(AGMN_with_gt_relative_positions[:, 0], AGMN_with_gt_relative_positions[:, 1], label='with_gt_relative_positions', c='tab:red')
    axes.plot(AGMN_CPM[:, 0], AGMN_CPM[:, 1], label='CPM', c='tab:purple')
    # axes.plot(NSRM_CPM[:, 0], NSRM_CPM[:, 1], label='CPM', c='tab:brown')
    # axes.plot(NSRM_LDM_G1[:, 0], NSRM_LDM_G1[:, 1], label='LDM_G1', c='tab:green')
    # axes.plot(NSRM_LDM_G1and6[:, 0], NSRM_LDM_G1and6[:, 1], label='LDM_G1and6', c='black')
    # axes.plot(NSRM_LPM_G1[:, 0], NSRM_LPM_G1[:, 1], label='LPM_G1', c='tab:orange')
    # axes.plot(NSRM_LPM_G1and6[:, 0], NSRM_LPM_G1and6[:, 1], label='LPM_G1and6', c='tab:red')


    axes.set_xlabel('Normalized Threshold')
    axes.set_ylabel('2D PCK')
    axes.set_title('Percentage of Correct Key-points vs Normalized Threshold')
    axes.grid(True)
    axes.legend()
    axes.yaxis.set_ticks(np.arange(0.2, 1, 0.05))
    axes.set_ylim([0.2, 1])

    axes.xaxis.set_ticks(np.arange(0.01, 0.16, 0.05))
    axes.set_xlim([0.01, 0.2])

    plt.savefig(
        '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/normalized_2DPCK_vs_AGMN.png'.format(eval_dataset,
                                                                                                        checkpoint_name),
        format='png')
    plt.show()

    fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    axes2.plot(PCK_thresh, y_joints[0, :], label='{}'.format(FREIHAND_KEYPOINTS[0]), c='k')
    cm = plt.get_cmap('tab20')
    axes2.set_prop_cycle(color=[cm(1. * i / 21) for i in range(21)])
    for joint_id in range(1, 21):
        axes2.plot(PCK_thresh, y_joints[joint_id, :], label='{}'.format(FREIHAND_KEYPOINTS[joint_id]))

    axes2.set_xlabel('Normalized Threshold')
    axes2.set_ylabel('2D PCK')
    axes2.set_title('Percentage of Correct Key-points vs Normalized Threshold per Joint')
    axes2.grid(True)
    axes2.legend()
    axes2.set_ylim([0, 1])
    axes2.set_xlim([0, max_error])
    plt.savefig('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/{}/{}/normalized_2DPCK_joints_vs_AGMN.png'.format(
        eval_dataset, checkpoint_name), format='png')
    plt.show()
    print('PCK_plot Successfully Ended!')


def pose_dataset_multi_predict_google():
    import logging
    logger = logging.getLogger('ftpuploader')
    b = 0
    error_counter = 0
    # fail_flag = -1
    successful_flag = 2
    google_annot_array = []
    google_annot_names = []
    index_fail_google_annotation = []
    total_raw_frames = 5118
    for index in range(42489, 42489+total_raw_frames):
    # total_raw_frames = 4
    # for index in range(3000, 3004):
        try:
            b += 1
            with open('/home/mahdi/HVR/hvr/data/iPad/2_pose_dataset/images/{:07}'.format(index), 'rb') as f:
                img = PIL.Image.open(f).convert('RGB')

            img = np.asarray(img)
            # fig, ax = plt.subplots(1, 2, figsize=(12, 12))
            # ax[0].imshow(img)
            # plt.show()
            pred = google_predict_only(img)
            assert pred.shape == (21, 2)
            pred = np.hstack((pred, successful_flag * np.ones((pred.shape[0], 1))))
            google_annot_array.append(pred)
            google_annot_names.append('{:07}'.format(index))
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            draw = ImageDraw.Draw(img)
            for annot_id in range(0,21):
                draw.ellipse((pred[annot_id,0]-4, pred[annot_id,1]-4, pred[annot_id,0]+4, pred[annot_id,1]+4), fill='red', outline='red')
            # draw.ellipse((20, 20, 30, 30), fill = 'red', outline ='red')
            img.save('/home/mahdi/HVR/hvr/data/iPad/2_pose_dataset/annotations/preview_annotated_images/{:07}.png'.format(index))
            percent_completed = b / (total_raw_frames) * 100
            print('progress = {:.2f}, total error={}'.format(percent_completed, error_counter))

        except Exception as e:  # work on python 3.x
            logger.error('Failed to upload to ftp: ' + str(e))
            error_counter += 1
            percent_completed = b / (total_raw_frames) * 100
            print('ERROR OCCURED!! progress = {:.2f}; total error={}'.format(percent_completed, error_counter))
            index_fail_google_annotation.append(index)
            # pred = fail_flag*np.ones((21,3))
            # google_annot_array.append(pred)
            continue


        # ax[0].plot(pred[:, 0], pred[:, 1], 'ro')
        # for txt in range(0, 21):
        #     ax[0].annotate(txt, (pred[txt, 0], pred[txt, 1]), c='w')
        # plt.show()

    np.save('/home/mahdi/HVR/hvr/data/iPad/2_pose_dataset/annotations/google_annot_array.npy', google_annot_array)
    np.save('/home/mahdi/HVR/hvr/data/iPad/2_pose_dataset/annotations/google_annot_names.npy', google_annot_names)


def pose_dataset_multi_predict_google_confirmation():
    all_annoted_frames_names = np.load('/home/mahdi/HVR/hvr/data/iPad/pose_dataset/annotations/all_annoted_frames_names_joined.npy')
    all_annoted_frames_annot = np.load('/home/mahdi/HVR/hvr/data/iPad/pose_dataset/annotations/all_annoted_frames_annot_joined.npy')
    counter=0
    error_counter = 0
    unannotated_counter = 0
    for index in range(0, all_annoted_frames_annot.shape[0]):
        counter+=1
        pred = all_annoted_frames_annot[index,:,:]
        if pred[0,0]==-1:
            unannotated_counter += 1
            continue
        try:
            with open('/home/mahdi/HVR/hvr/data/iPad/pose_dataset/images/{}'.format(all_annoted_frames_names[index]), 'rb') as f:
                img = PIL.Image.open(f).convert('RGB')

            draw = ImageDraw.Draw(img)
            for annot_id in range(0, 21):
                draw.ellipse((pred[annot_id, 0] - 4, pred[annot_id, 1] - 4, pred[annot_id, 0] + 4, pred[annot_id, 1] + 4),
                             fill='green', outline='green')
        except:
            error_counter+=1
            continue
        img.save('/home/mahdi/HVR/hvr/data/iPad/pose_dataset_annotations_confirmation/{}.png'.format(all_annoted_frames_names[index]))
        percent_completed = counter / ( all_annoted_frames_annot.shape[0]) * 100
        print('progress = {:.2f}, total error={}'.format(percent_completed, error_counter))

    print('ended')

if __name__ == '__main__':
    # checkpoint_name = 'shufflenetv2k16w-200724-004154-cif-caf-caf25-edge200.pkl.epoch172'
    # eval_dataset = 'freihand'
    # checkpoint_name = 'shufflenetv2k16w-200725-113056-cif-caf-caf25-edge200.pkl.epoch320'
    # eval_dataset = 'onehand10k'

    # checkpoint_name = 'shufflenetv2k16w-200803-140030-cif-caf-caf25-edge200-o10s.pkl.epoch277'
    # checkpoint_name = 'shufflenetv2k16w-200804-203646-cif-caf-caf25-edge200-o10s.pkl.epoch050'
    # checkpoint_name = 'shufflenetv2k16w-200809-021328-cif-caf-caf25-edge200-o10s.pkl.epoch187'
    # checkpoint_name = 'shufflenetv2k16w-200810-234209-cif-caf-caf25-edge200-o10s.pkl.epoch320'
    # checkpoint_name = 'shufflenetv2k16w-200812-092552-cif-caf-edge200-o10s.pkl.epoch023'
    # checkpoint_name = 'shufflenetv2k16w-200811-215425-cif-caf-edge200-o10s.pkl.epoch006'
    # eval_dataset = 'pose_dataset'


    # posedataset_multi_predict_hvr(checkpoint_name, eval_dataset)

    # checkpoint_name = 'shufflenetv2k16w-200730-095321-cif-caf-caf25-edge200-o10s.pkl.epoch067'
    # checkpoint_name = 'shufflenetv2k16w-200730-200536-cif-caf-caf25-edge200-o10s.pkl.epoch094'
    # checkpoint_name = 'shufflenetv2k16w-200730-200536-cif-caf-caf25-edge200-o10s.pkl.epoch240'
    # checkpoint_name = 'shufflenetv2k16w-200731-220146-cif-caf-caf25-edge200-o10.pkl.epoch117'
    # checkpoint_name = 'shufflenetv2k16w-200731-220146-cif-caf-caf25-edge200-o10.pkl.epoch146'
    # checkpoint_name = 'shufflenetv2k16w-200731-220146-cif-caf-caf25-edge200-o10.pkl.epoch212'
    # checkpoint_name = 'shufflenetv2k16w-200731-220146-cif-caf-caf25-edge200-o10.pkl.epoch320'
    # checkpoint_name = 'shufflenetv2k16w-200721-232112-cif-caf-caf25-edge200.pkl.epoch148'
    # checkpoint_name = 'test_on_panoptic/shufflenetv2k16w-200721-232112-cif-caf-caf25-edge200.pkl.epoch148'
    # checkpoint_name = 'test_on_rhd/shufflenetv2k16w-200721-232112-cif-caf-caf25-edge200.pkl.epoch148'
    # checkpoint_name = 'test_on_freihand/shufflenetv2k16w-200809-021328-cif-caf-caf25-edge200-o10s.pkl.epoch263'
    # checkpoint_name = 'shufflenetv2k16w-200809-021328-cif-caf-caf25-edge200-o10s.pkl.epoch263'

    # checkpoint_name = 'test_on_correct_freihand/shufflenetv2k16w-200731-220146-cif-caf-caf25-edge200-o10.pkl.epoch212'
    # checkpoint_name = 'shufflenetv2k16w-200816-140011-cif-caf25-edge200-o50s.pkl.epoch040_guided_head'
    # checkpoint_name = 'shufflenetv2k16w-200816-140011-cif-caf25-edge200-o50s.pkl.epoch014'
    # checkpoint_name = 'shufflenetv2k16w-200815-173928-cif-caf-edge200-o50s.pkl.epoch036_guided_head'
    # checkpoint_name = 'shufflenetv2k16w-200814-212724-cif-caf-edge200-o10s.pkl.epoch022_guided_head'
    # eval_dataset = 'rhd'

    # panoptic_multi_predict(checkpoint_name, eval_dataset)
    # freihand_multi_predict(checkpoint_name, eval_dataset)
    # rhd_multi_predict(checkpoint_name, eval_dataset)

    # posedataset_multi_predict(checkpoint_name, eval_dataset)

    # PCK_plot(checkpoint_name, eval_dataset)

    # PCK_normalized_plot(checkpoint_name, eval_dataset)

    # onehand10k_multi_predict(checkpoint_name, eval_dataset)

    # freihand_multi_predict_google(checkpoint_name, eval_dataset)

    pose_dataset_multi_predict_google()
    # pose_dataset_multi_predict_google_confirmation()


# the best model handPifPaf name: shufflenetv2k16w-200724-004154-cif-caf-caf25-edge200.pkl.epoch172
# time CUDA_VISIBLE_DEVICES=0,1 python3 -m openpifpaf.predict_multi /home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/Freihand_pub_v2/ --image-output /home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/  --checkpoint=/home/mahdi/HVR/git_repos/openpifpaf/outputs/shufflenetv2k16w-200720-202350-cif-caf-caf25-edge200.pkl.epoch052  --json-output=/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/results/predict_output/  --batch-size=16  --long-edge=224  --quiet

# /home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/RHD_published_v2
# --image-output
# /home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/tmp/predict_output/rhd
# --checkpoint=/home/mahdi/HVR/git_repos/openpifpaf/outputs/shufflenetv2k16w-200730-095321-cif-caf-caf25-edge200-o10s.pkl.epoch067
# --json-output=/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/tmp/predict_output/rhd
# --batch-size=16
# --long-edge=200
# --quiet
# --loader-workers=0