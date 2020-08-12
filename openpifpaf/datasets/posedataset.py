import logging
import torch.utils.data
from PIL import Image
from .. import transforms, utils
import os
import pickle
import numpy as np
import scipy.ndimage
import PIL


LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))

class Posedataset(torch.utils.data.Dataset):
    """
        Pose Dataset
    """

    def __init__(self, *, image_dir, mode, target_transforms, preprocess):
        """
        mode = 'training' or 'evaluation'
        """
        self.image_dir = image_dir
        self.mode = mode # 'evaluation' or 'training'
        self.target_transforms = target_transforms
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

        _all_names = np.load('{}/annotations/all_annoted_frames_names.npy'.format(self.image_dir))
        _all_annots = np.load('{}/annotations/all_annoted_frames_annot.npy'.format(self.image_dir))

        # remove repeated frames 1
        _wrong_names = np.genfromtxt('{}/annotations/wrong_correct_pose_dataset.csv'.format(self.image_dir), delimiter=',')[1:, 0].astype(int)
        _wrong_names_str = []
        for w in range(0, _wrong_names.__len__()):
            _wrong_names_str.append("{:07}".format(_wrong_names[w]))

        _wrong_names_str = np.asarray(_wrong_names_str)

        for ww in range(0, _wrong_names.__len__()):
            arg_wrong = np.argwhere(_all_names == _wrong_names_str[ww])
            if arg_wrong!=0:
                _all_names = np.delete(_all_names, arg_wrong, 0)
                _all_annots =  np.delete(_all_annots, arg_wrong, 0)

        # remove repeated frames 2
        _wrong_names = np.genfromtxt('{}/annotations/wrong_correct_pose_dataset_2.csv'.format(self.image_dir), delimiter=',')[1:, 0].astype(int)
        _wrong_names_str = []
        for w in range(0, _wrong_names.__len__()):
            _wrong_names_str.append("{:07}".format(_wrong_names[w]))

        _wrong_names_str = np.asarray(_wrong_names_str)

        for ww in range(0, _wrong_names.__len__()):
            arg_wrong = np.argwhere(_all_names == _wrong_names_str[ww])
            if arg_wrong!=0:
                _all_names = np.delete(_all_names, arg_wrong, 0)
                _all_annots =  np.delete(_all_annots, arg_wrong, 0)

        # remove repeated frames 3
        _wrong_names =  np.load('{}/annotations/wrong_correct_pose_dataset_3.npy'.format(self.image_dir))
        _wrong_names_str = []
        for w in range(0, _wrong_names.__len__()):
            _wrong_names_str.append("{:07}".format(_wrong_names[w]))

        _wrong_names_str = np.asarray(_wrong_names_str)

        for ww in range(0, _wrong_names.__len__()):
            arg_wrong = np.argwhere(_all_names == _wrong_names_str[ww])
            if arg_wrong!=0:
                _all_names = np.delete(_all_names, arg_wrong, 0)
                _all_annots =  np.delete(_all_annots, arg_wrong, 0)

        # random split data
        # from sklearn.model_selection import train_test_split
        # annots_train, annots_test, names_train, names_test = train_test_split(_all_annots, _all_names, test_size = 0.20, shuffle=True)
        #
        # if self.mode=='training':
        #     self.all_names = names_train
        #     self.all_annots = annots_train
        # elif self.mode=='evaluation':
        #     self.all_names = names_test
        #     self.all_annots = annots_test
        # else:
        #     raise AssertionError('dataset mode is not defined!')

        # # train with 2 subjects(~82% data; 23817 data; all correctly annotated frames from name 0000001 to 0026584) and test with the rest 13 subjects(~18% data; 4991 data)
        # if self.mode=='training':
        #     self.all_names = _all_names[:np.argwhere(_all_names == '0026584')[0][0]]
        #
        #     self.all_annots = _all_annots[:np.argwhere(_all_names == '0026584')[0][0], : ,:]
        # elif self.mode=='evaluation':
        #     self.all_names = _all_names[np.argwhere(_all_names == '0026584')[0][0]:]
        #     self.all_annots = _all_annots[np.argwhere(_all_names == '0026584')[0][0]:, :, :]
        # else:
        #     raise AssertionError('dataset mode is not defined!')

        # train with 1 subjects(~% data;  data; all correctly annotated frames from name 0000001 to 0025788) and test with the validation dataset wtih 15 subjects(~% data;  data)
        if self.mode=='training':
            self.all_names = _all_names[:np.argwhere(_all_names == '0025788')[0][0]]

            self.all_annots = _all_annots[:np.argwhere(_all_names == '0025788')[0][0], : ,:]
        elif self.mode=='evaluation':
            self.all_names = _all_names[np.argwhere(_all_names == '0025788')[0][0]:]
            self.all_annots = _all_annots[np.argwhere(_all_names == '0025788')[0][0]:, :, :]
        else:
            raise AssertionError('dataset mode is not defined!')

        # annotation for this frame
        visibility_flag = 2
        self.all_annots[:, :, 2] = visibility_flag * self.all_annots[:, :, 2]

    def __getitem__(self, index):
        # load image
        with open('{}/images/{}'.format(self.image_dir, self.all_names[index]), 'rb') as f:
            img = Image.open(f).convert('RGB')

        anns = [{'keypoints': self.all_annots[index, :, :]}]
        anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
        anns[0].update({'iscrowd': 0})

        # for i in range(0, 41258):
        #     if sum(self.anno_all[i]['uv_vis'][:, 2])==0:
        #         print('found unannotated frame!!! i={}'.format(i))
        # uncomment to combine left and right hand to one type

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2, figsize=(12, 12))
        # ax[0].imshow(np.asarray(img))
        # bool_annotated_joints_1 = anns[0]['keypoints'][:, 2]==2
        # ax[0].plot(anns[0]['keypoints'][bool_annotated_joints_1, 0], anns[0]['keypoints'][bool_annotated_joints_1, 1], 'ro')
        # bool_annotated_joints_2 = anns[1]['keypoints'][:, 2] == 2
        # ax[0].plot(anns[1]['keypoints'][bool_annotated_joints_2, 0], anns[1]['keypoints'][bool_annotated_joints_2, 1], 'go')

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

        # rescale keypoints
        x_scale = (img.size[0]) / (w)
        y_scale = (img.size[1]) / (h)
        # anns2 = copy.deepcopy(anns)
        for ann in anns:
            ann['keypoints'][:, 0] = ann['keypoints'][:, 0] * x_scale
            ann['keypoints'][:, 1] = ann['keypoints'][:, 1] * y_scale
            ann['bbox'][0] *= x_scale
            ann['bbox'][1] *= y_scale
            ann['bbox'][2] *= x_scale
            ann['bbox'][3] *= y_scale

        # pad frames
        img = np.asarray(img)
        pad_up = (reference_edge - img.shape[0]) // 2
        pad_down = (reference_edge - img.shape[0]) // 2
        pad_left = (reference_edge - img.shape[1]) // 2
        pad_right = (reference_edge - img.shape[1]) // 2
        img = np.pad(img, pad_width=((pad_up, pad_down), (pad_left, pad_right), (0, 0)), mode='symmetric')
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        for ann in anns:
            # modify for pad
            ann['keypoints'][:, 0] = ann['keypoints'][:, 0] + pad_left
            ann['keypoints'][:, 1] = ann['keypoints'][:, 1] + pad_up
            ann['bbox'][0] += pad_left
            ann['bbox'][1] += pad_up
            ann['bbox'][2] += pad_left
            ann['bbox'][3] += pad_up

        # ax[1].imshow(np.asarray(img))
        # bool_annotated_joints_1 = anns[0]['keypoints'][:, 2] == 2
        # ax[1].plot(anns[0]['keypoints'][bool_annotated_joints_1, 0], anns[0]['keypoints'][bool_annotated_joints_1, 1],
        #            'ro')
        # bool_annotated_joints_2 = anns[1]['keypoints'][:, 2] == 2
        # ax[1].plot(anns[1]['keypoints'][bool_annotated_joints_2, 0], anns[1]['keypoints'][bool_annotated_joints_2, 1],
        #            'go')
            # plt.show()

        meta = None

        # preprocess image and annotations
        img, anns, meta = self.preprocess(img, anns, meta)

        # mask valid TODO still necessary?
        valid_area = meta['valid_area']
        utils.mask_valid_area(img, valid_area)
        LOG.debug(meta)

        # log stats
        for ann in anns:
            if getattr(ann, 'iscrowd', False):
                continue
            if not np.any(ann['keypoints'][:, 2] > 0.0):
                continue
            STAT_LOG.debug({'bbox': [int(v) for v in ann['bbox']]})

        # transform targets
        if self.target_transforms is not None:
            anns = [t(img, anns, meta) for t in self.target_transforms]

        return img, anns, meta

    def __len__(self):
        # print('self.mode={}; self.all_names.shape[0]={}'.format(self.mode, self.all_names.shape[0]))
        return self.all_names.shape[0]