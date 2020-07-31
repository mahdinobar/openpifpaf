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


class Rhd(torch.utils.data.Dataset):
    """
        RHD Dataset
    """

    def __init__(self, *, image_dir, mode, target_transforms, preprocess):
        """
        mode = 'training' or 'evaluation'
        """
        self.image_dir = image_dir
        self.mode = mode # 'evaluation' or 'training'
        self.target_transforms = target_transforms
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        # load all annotations of this mode
        with open(os.path.join(self.image_dir, self.mode, 'anno_%s.pickle' % self.mode), 'rb') as fi:
            self.anno_all = pickle.load(fi)


    def __getitem__(self, index):
        # load image
        with open(os.path.join(self.image_dir, self.mode, 'color', '%.5d.png' % index), 'rb') as f:
            img = Image.open(f).convert('RGB')

        # annotation for this frame
        visibility_flag = 2
        self.anno_all[index]['uv_vis'][:, 2] = visibility_flag*self.anno_all[index]['uv_vis'][:,2]

        # for debug
        # unannotated_frame = []
        # self.anno_all[index]['uv_vis'][:, 2] = visibility_flag*self.anno_all[index]['uv_vis'][:,2]
        # for k in range(0, self.anno_all.__len__()):
        #     if sum(self.anno_all[k]['uv_vis'][:,2]==0)==40:
        #         unannotated_frame.append(k)
        #         print ('FOUND UNANNOTATED!!')


        # # uncomment to separate left and right hand TODO removal of unannotated annotations
        # anns = [{'keypoints':  self.anno_all[index]['uv_vis']}]
        # anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
        # anns[0].update({'iscrowd': 0})
        # anns[1].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
        # anns[1].update({'iscrowd': 0})

        # for i in range(0, 41258):
        #     if sum(self.anno_all[i]['uv_vis'][:, 2])==0:
        #         print('found unannotated frame!!! i={}'.format(i))
        # uncomment to combine left and right hand to one type
        if np.any(self.anno_all[index]['uv_vis'][:21, 2]) and np.any(self.anno_all[index]['uv_vis'][21:, 2]):
            anns_left_hand = [{'keypoints': self.anno_all[index]['uv_vis'][:21, :]}]
            anns_right_hand = [{'keypoints': self.anno_all[index]['uv_vis'][21:, :]}]
            anns = list([anns_left_hand[0], anns_right_hand[0]])
            anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
            anns[0].update({'iscrowd': 0})
            anns[1].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
            anns[1].update({'iscrowd': 0})
        elif np.any(self.anno_all[index]['uv_vis'][:21, 2]):
            anns_left_hand = [{'keypoints': self.anno_all[index]['uv_vis'][:21, :]}]
            anns = list([anns_left_hand[0]])
            anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
            anns[0].update({'iscrowd': 0})
        elif np.any(self.anno_all[index]['uv_vis'][21:, 2]):
            anns_right_hand = [{'keypoints': self.anno_all[index]['uv_vis'][21:, :]}]
            anns = list([anns_right_hand[0]])
            anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
            anns[0].update({'iscrowd': 0})
        else:
            raise AssertionError('frame index={} has no annotation!'.format(index))

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
        return len(self.anno_all)