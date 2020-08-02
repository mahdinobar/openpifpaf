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
        if self.mode=='training':
            self.all_names = np.load('{}/annotations/all_annoted_frames_names.npy'.format(self.image_dir))
            self.all_annots = np.load('{}/annotations/all_annoted_frames_annot.npy'.format(self.image_dir))
        else:
            raise AssertionError('dataset mode is not defined!')
        # annotation for this frame
        visibility_flag = 2
        self.all_annots[:, :, 2] = visibility_flag * self.all_annots[:, :, 2]

    def __getitem__(self, index):
        # load image
        with open('{}/images/{}.png'.format(self.image_dir, self.all_names[index])) as f:
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
        return self.all_names.shape[0]