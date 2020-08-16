import PIL
import torch


from .. import transforms

from .freihand_utils import *
from PIL import Image
import pickle
import scipy.ndimage
import scipy.io as scio


class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None):
        self.image_paths = image_paths
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = PIL.Image.open(f).convert('RGB')


        # rescale image
        order = 1  # order of resize interpolation; 1 means linear interpolation
        w, h = image.size
        # keep aspect ratio the same
        reference_edge = 224
        target_max_edge = reference_edge
        max_edge = max(h, w)
        ratio_factor = target_max_edge / max_edge
        target_h = int(ratio_factor * h)
        target_w = int(ratio_factor * w)
        im_np = np.asarray(image)
        im_np = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1), order=order)
        img = PIL.Image.fromarray(im_np)
        assert img.size[0] == target_w
        assert img.size[1] == target_h
        # pad frames
        img = np.asarray(img)
        pad_up = (reference_edge - img.shape[0]) // 2
        pad_down = (reference_edge - img.shape[0]) // 2
        pad_left = (reference_edge - img.shape[1]) // 2
        pad_right = (reference_edge - img.shape[1]) // 2
        img = np.pad(img, pad_width=((pad_up, pad_down), (pad_left, pad_right), (0, 0)), mode='symmetric')
        image = Image.fromarray(img.astype('uint8'), 'RGB')


        anns = []
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update({
            'dataset_index': index,
            'file_name': image_path,
        })

        return image, anns, meta

    def __len__(self):
        return len(self.image_paths)


class ImageList_PoseDataset_multi(torch.utils.data.Dataset):
    """
        Panoptic Dataset
    """

    def __init__(self, *, image_dir, mode, preprocess):
        """
        mode = 'training' or 'evaluation' or 'test'
        """
        self.image_dir = image_dir
        self.mode = mode # 'evaluation' or 'training' or 'test'
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

        _all_names = np.load('{}/annotations/all_annoted_frames_names.npy'.format(self.image_dir))
        _all_annots = np.load('{}/annotations/all_annoted_frames_annot.npy'.format(self.image_dir))

        # array_2 = np.load('/home/mahdi/HVR/hvr/data/iPad/2_pose_dataset/annotations/google_annot_array.npy')
        # names_2 = np.load('/home/mahdi/HVR/hvr/data/iPad/2_pose_dataset/annotations/google_annot_names.npy')
        #
        # np.save('{}/annotations/all_annoted_frames_names.npy'.format(self.image_dir), np.hstack((_all_names, names_2)))
        # np.save('{}/annotations/all_annoted_frames_annot.npy'.format(self.image_dir), np.concatenate((_all_annots, array_2)))

        # remove repeated frames 1
        _wrong_names = np.genfromtxt('{}/annotations/wrong_correct_pose_dataset.csv'.format(self.image_dir),
                                     delimiter=',')[1:, 0].astype(int)
        _wrong_names_str = []
        for w in range(0, _wrong_names.__len__()):
            _wrong_names_str.append("{:07}".format(_wrong_names[w]))

        _wrong_names_str = np.asarray(_wrong_names_str)

        for ww in range(0, _wrong_names.__len__()):
            arg_wrong = np.argwhere(_all_names == _wrong_names_str[ww])
            if arg_wrong != 0:
                _all_names = np.delete(_all_names, arg_wrong, 0)
                _all_annots = np.delete(_all_annots, arg_wrong, 0)

        # remove repeated frames 2
        _wrong_names = np.genfromtxt('{}/annotations/wrong_correct_pose_dataset_2.csv'.format(self.image_dir),
                                     delimiter=',')[1:, 0].astype(int)
        _wrong_names_str = []
        for w in range(0, _wrong_names.__len__()):
            _wrong_names_str.append("{:07}".format(_wrong_names[w]))

        _wrong_names_str = np.asarray(_wrong_names_str)

        for ww in range(0, _wrong_names.__len__()):
            arg_wrong = np.argwhere(_all_names == _wrong_names_str[ww])
            if arg_wrong != 0:
                _all_names = np.delete(_all_names, arg_wrong, 0)
                _all_annots = np.delete(_all_annots, arg_wrong, 0)

        # remove repeated frames 3
        _wrong_names = np.load('{}/annotations/wrong_correct_pose_dataset_3.npy'.format(self.image_dir))
        _wrong_names_str = []
        for w in range(0, _wrong_names.__len__()):
            _wrong_names_str.append("{:07}".format(_wrong_names[w]))

        _wrong_names_str = np.asarray(_wrong_names_str)

        for ww in range(0, _wrong_names.__len__()):
            arg_wrong = np.argwhere(_all_names == _wrong_names_str[ww])
            if arg_wrong != 0:
                _all_names = np.delete(_all_names, arg_wrong, 0)
                _all_annots = np.delete(_all_annots, arg_wrong, 0)

        # remove repeated frames 4
        _wrong_names = np.genfromtxt('{}/annotations/wrong_correct_pose_dataset_4.csv'.format(self.image_dir),
                                     delimiter=',')[1:].astype(int)
        _wrong_names_str = []
        for w in range(0, _wrong_names.__len__()):
            _wrong_names_str.append("{:07}".format(_wrong_names[w]))

        _wrong_names_str = np.asarray(_wrong_names_str)

        for ww in range(0, _wrong_names.__len__()):
            arg_wrong = np.argwhere(_all_names == _wrong_names_str[ww])
            if arg_wrong != 0:
                _all_names = np.delete(_all_names, arg_wrong, 0)
                _all_annots = np.delete(_all_annots, arg_wrong, 0)

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
        if self.mode == 'training':
            # self.all_names = _all_names[:np.argwhere(_all_names == '0025791')[0][0]]

            # self.all_names = np.concatenate((_all_names[:np.argwhere(_all_names == '0025791')[0][0]], _all_names[np.argwhere(_all_names == '0031792')[0][0]:]))

            # self.all_annots = _all_annots[:np.argwhere(_all_names == '0025791')[0][0], : ,:]
            # self.all_annots = np.concatenate((_all_annots[:np.argwhere(_all_names == '0025791')[0][0], : ,:], _all_annots[np.argwhere(_all_names == '0031792')[0][0]:, : ,:]))

            # select only number_random_select left hand images for training between 0000000 to 0025791
            number_random_select = 3000
            selected_id = np.random.choice(np.argwhere(_all_names == '0025791')[0][0], number_random_select,
                                           replace=False)
            self.all_names = np.concatenate((_all_names[selected_id],
                                             _all_names[np.argwhere(_all_names == '0031792')[0][0]:]))
            self.all_annots = np.concatenate(
                (_all_annots[selected_id], _all_annots[np.argwhere(_all_names == '0031792')[0][0]:, :, :]))


        elif self.mode == 'evaluation':
            # self.all_names = _all_names[np.argwhere(_all_names == '0025791')[0][0]:]
            self.all_names = _all_names[
                             np.argwhere(_all_names == '0025791')[0][0]:np.argwhere(_all_names == '0031792')[0][0]]
            # self.all_annots = _all_annots[np.argwhere(_all_names == '0025791')[0][0]:, :, :]
            self.all_annots = _all_annots[
                              np.argwhere(_all_names == '0025791')[0][0]:np.argwhere(_all_names == '0031792')[0][0], :,
                              :]
        else:
            raise AssertionError('dataset mode is not defined!')

        # annotation for this frame
        visibility_flag = 2
        self.all_annots[:, :, 2] = visibility_flag * self.all_annots[:, :, 2]


    def __getitem__(self, index):
        # load image
        with open('{}/images/{}'.format(self.image_dir, self.all_names[index]), 'rb') as f:
            img = Image.open(f).convert('RGB')

        # annotation for this frame
        visibility_flag = 2
        self.all_annots[index, :, 2] = visibility_flag*self.all_annots[index, :, 2]
        gt_before_preprocess = np.copy(self.all_annots[index, :, :])

        anns = [{'keypoints': self.all_annots[index, :, :]}]
        anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
        anns[0].update({'iscrowd': 0})

        # # # TODO uncomment for old trained rhd evaluation
        # # for matching rhd annotation when combining with posedataset/google/freihand/panoptic: this also matches with rhd_constants.py
        # if anns.__len__() == 2:
        #     anns_coorect_matching_posedataset = np.zeros_like(anns[0]['keypoints'])
        #     anns_coorect_matching_posedataset[1, :] = anns[0]['keypoints'][4, :]
        #     anns_coorect_matching_posedataset[2, :] = anns[0]['keypoints'][3, :]
        #     anns_coorect_matching_posedataset[3, :] = anns[0]['keypoints'][2, :]
        #     anns_coorect_matching_posedataset[4, :] = anns[0]['keypoints'][1, :]
        #     anns_coorect_matching_posedataset[5, :] = anns[0]['keypoints'][8, :]
        #     anns_coorect_matching_posedataset[6, :] = anns[0]['keypoints'][7, :]
        #     anns_coorect_matching_posedataset[7, :] = anns[0]['keypoints'][6, :]
        #     anns_coorect_matching_posedataset[8, :] = anns[0]['keypoints'][5, :]
        #     anns_coorect_matching_posedataset[13, :] = anns[0]['keypoints'][16, :]
        #     anns_coorect_matching_posedataset[14, :] = anns[0]['keypoints'][15, :]
        #     anns_coorect_matching_posedataset[15, :] = anns[0]['keypoints'][14, :]
        #     anns_coorect_matching_posedataset[16, :] = anns[0]['keypoints'][13, :]
        #     anns_coorect_matching_posedataset[17, :] = anns[0]['keypoints'][20, :]
        #     anns_coorect_matching_posedataset[18, :] = anns[0]['keypoints'][19, :]
        #     anns_coorect_matching_posedataset[19, :] = anns[0]['keypoints'][18, :]
        #     anns_coorect_matching_posedataset[20, :] = anns[0]['keypoints'][17, :]
        #     anns[0]['keypoints'] = np.copy(anns_coorect_matching_posedataset)
        #
        #     anns_coorect_matching_posedataset_2 = np.zeros_like(anns[1]['keypoints'])
        #     anns_coorect_matching_posedataset_2[1, :] = anns[1]['keypoints'][4, :]
        #     anns_coorect_matching_posedataset_2[2, :] = anns[1]['keypoints'][3, :]
        #     anns_coorect_matching_posedataset_2[3, :] = anns[1]['keypoints'][2, :]
        #     anns_coorect_matching_posedataset_2[4, :] = anns[1]['keypoints'][1, :]
        #     anns_coorect_matching_posedataset_2[5, :] = anns[1]['keypoints'][8, :]
        #     anns_coorect_matching_posedataset_2[6, :] = anns[1]['keypoints'][7, :]
        #     anns_coorect_matching_posedataset_2[7, :] = anns[1]['keypoints'][6, :]
        #     anns_coorect_matching_posedataset_2[8, :] = anns[1]['keypoints'][5, :]
        #     anns_coorect_matching_posedataset_2[13, :] = anns[1]['keypoints'][16, :]
        #     anns_coorect_matching_posedataset_2[14, :] = anns[1]['keypoints'][15, :]
        #     anns_coorect_matching_posedataset_2[15, :] = anns[1]['keypoints'][14, :]
        #     anns_coorect_matching_posedataset_2[16, :] = anns[1]['keypoints'][13, :]
        #     anns_coorect_matching_posedataset_2[17, :] = anns[1]['keypoints'][20, :]
        #     anns_coorect_matching_posedataset_2[18, :] = anns[1]['keypoints'][19, :]
        #     anns_coorect_matching_posedataset_2[19, :] = anns[1]['keypoints'][18, :]
        #     anns_coorect_matching_posedataset_2[20, :] = anns[1]['keypoints'][17, :]
        #     anns[1]['keypoints'] = np.copy(anns_coorect_matching_posedataset_2)
        #
        # elif anns.__len__() == 1:
        #     anns_coorect_matching_posedataset = np.zeros_like(anns[0]['keypoints'])
        #     anns_coorect_matching_posedataset[1, :] = anns[0]['keypoints'][4, :]
        #     anns_coorect_matching_posedataset[2, :] = anns[0]['keypoints'][3, :]
        #     anns_coorect_matching_posedataset[3, :] = anns[0]['keypoints'][2, :]
        #     anns_coorect_matching_posedataset[4, :] = anns[0]['keypoints'][1, :]
        #     anns_coorect_matching_posedataset[5, :] = anns[0]['keypoints'][8, :]
        #     anns_coorect_matching_posedataset[6, :] = anns[0]['keypoints'][7, :]
        #     anns_coorect_matching_posedataset[7, :] = anns[0]['keypoints'][6, :]
        #     anns_coorect_matching_posedataset[8, :] = anns[0]['keypoints'][5, :]
        #     anns_coorect_matching_posedataset[13, :] = anns[0]['keypoints'][16, :]
        #     anns_coorect_matching_posedataset[14, :] = anns[0]['keypoints'][15, :]
        #     anns_coorect_matching_posedataset[15, :] = anns[0]['keypoints'][14, :]
        #     anns_coorect_matching_posedataset[16, :] = anns[0]['keypoints'][13, :]
        #     anns_coorect_matching_posedataset[17, :] = anns[0]['keypoints'][20, :]
        #     anns_coorect_matching_posedataset[18, :] = anns[0]['keypoints'][19, :]
        #     anns_coorect_matching_posedataset[19, :] = anns[0]['keypoints'][18, :]
        #     anns_coorect_matching_posedataset[20, :] = anns[0]['keypoints'][17, :]
        #     anns[0]['keypoints'] = np.copy(anns_coorect_matching_posedataset)
        #
        # else:
        #     raise AssertionError('frame index={} has no annotation!'.format(index))

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2, figsize=(12, 12))
        # ax[0].imshow(np.asarray(img))
        # bool_annotated_joints_1 = anns[0]['keypoints'][:, 2]==2
        # ax[0].plot(anns[0]['keypoints'][bool_annotated_joints_1, 0], anns[0]['keypoints'][bool_annotated_joints_1, 1], 'r.')

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
        # plt.show()

        meta = None

        # preprocess image and annotations
        img, anns, meta = self.preprocess(img, anns, meta)
        meta.update({
            'dataset_index': index,
            'file_name': '{}/images/{}'.format(self.image_dir, self.all_names[index]),
            'gt_before_preprocess': gt_before_preprocess,
        })


        return img, anns, meta

    def __len__(self):
        return self.all_names.__len__()



class ImageList_Panoptic(torch.utils.data.Dataset):
    """
        Panoptic Dataset
    """

    def __init__(self, *, image_dir, mode, preprocess):
        """
        mode = 'training' or 'evaluation' or 'test'
        """
        self.image_dir = image_dir
        self.mode = mode # 'evaluation' or 'training' or 'test'
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

        if self.mode == 'test':
            self.all_names = np.load(
                '{}/names_test.npy'.format(self.image_dir))
            self.all_annots = np.load(
                '{}/annots_test.npy'.format(self.image_dir))
        else:
            raise AssertionError('mode is not defined!')



    def __getitem__(self, index):
        # load image
        with open(os.path.join(self.image_dir, self.all_names[index]), 'rb') as f:
            img = Image.open(f).convert('RGB')

        # annotation for this frame
        visibility_flag = 2
        self.all_annots[index, :, 2] = visibility_flag*self.all_annots[index, :, 2]
        gt_before_preprocess = np.copy(self.all_annots[index, :, :])

        anns = [{'keypoints': self.all_annots[index, :, :]}]
        anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
        anns[0].update({'iscrowd': 0})

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2, figsize=(12, 12))
        # ax[0].imshow(np.asarray(img))
        # bool_annotated_joints_1 = anns[0]['keypoints'][:, 2]==2
        # ax[0].plot(anns[0]['keypoints'][bool_annotated_joints_1, 0], anns[0]['keypoints'][bool_annotated_joints_1, 1], 'r.')


        # crop image
        max_gt_bbx = max(max(self.all_annots[index, :, 0]) - min(self.all_annots[index, :, 0]), max(self.all_annots[index, :, 1])-min(self.all_annots[index, :, 1]))
        bbx_factor = 2.2
        bbx = bbx_factor*max_gt_bbx
        x_offset = (max(self.all_annots[index, :, 0]) + min(self.all_annots[index, :, 0])) / 2 - bbx/2
        y_offset = (max(self.all_annots[index, :, 1]) + min(self.all_annots[index, :, 1])) / 2 - bbx/2
        ltrb = (x_offset, y_offset, x_offset + bbx, y_offset + bbx)
        img = img.crop(ltrb)
        # crop keypoints
        for ann in anns:
            ann['keypoints'][:, 0] -= x_offset
            ann['keypoints'][:, 1] -= y_offset
            ann['bbox'][0] -= x_offset
            ann['bbox'][1] -= y_offset

        # # # TODO uncomment for old trained rhd evaluation
        # # for matching rhd annotation when combining with posedataset/google/freihand/panoptic: this also matches with rhd_constants.py
        # if anns.__len__() == 2:
        #     anns_coorect_matching_posedataset = np.zeros_like(anns[0]['keypoints'])
        #     anns_coorect_matching_posedataset[1, :] = anns[0]['keypoints'][4, :]
        #     anns_coorect_matching_posedataset[2, :] = anns[0]['keypoints'][3, :]
        #     anns_coorect_matching_posedataset[3, :] = anns[0]['keypoints'][2, :]
        #     anns_coorect_matching_posedataset[4, :] = anns[0]['keypoints'][1, :]
        #     anns_coorect_matching_posedataset[5, :] = anns[0]['keypoints'][8, :]
        #     anns_coorect_matching_posedataset[6, :] = anns[0]['keypoints'][7, :]
        #     anns_coorect_matching_posedataset[7, :] = anns[0]['keypoints'][6, :]
        #     anns_coorect_matching_posedataset[8, :] = anns[0]['keypoints'][5, :]
        #     anns_coorect_matching_posedataset[13, :] = anns[0]['keypoints'][16, :]
        #     anns_coorect_matching_posedataset[14, :] = anns[0]['keypoints'][15, :]
        #     anns_coorect_matching_posedataset[15, :] = anns[0]['keypoints'][14, :]
        #     anns_coorect_matching_posedataset[16, :] = anns[0]['keypoints'][13, :]
        #     anns_coorect_matching_posedataset[17, :] = anns[0]['keypoints'][20, :]
        #     anns_coorect_matching_posedataset[18, :] = anns[0]['keypoints'][19, :]
        #     anns_coorect_matching_posedataset[19, :] = anns[0]['keypoints'][18, :]
        #     anns_coorect_matching_posedataset[20, :] = anns[0]['keypoints'][17, :]
        #     anns[0]['keypoints'] = np.copy(anns_coorect_matching_posedataset)
        #
        #     anns_coorect_matching_posedataset_2 = np.zeros_like(anns[1]['keypoints'])
        #     anns_coorect_matching_posedataset_2[1, :] = anns[1]['keypoints'][4, :]
        #     anns_coorect_matching_posedataset_2[2, :] = anns[1]['keypoints'][3, :]
        #     anns_coorect_matching_posedataset_2[3, :] = anns[1]['keypoints'][2, :]
        #     anns_coorect_matching_posedataset_2[4, :] = anns[1]['keypoints'][1, :]
        #     anns_coorect_matching_posedataset_2[5, :] = anns[1]['keypoints'][8, :]
        #     anns_coorect_matching_posedataset_2[6, :] = anns[1]['keypoints'][7, :]
        #     anns_coorect_matching_posedataset_2[7, :] = anns[1]['keypoints'][6, :]
        #     anns_coorect_matching_posedataset_2[8, :] = anns[1]['keypoints'][5, :]
        #     anns_coorect_matching_posedataset_2[13, :] = anns[1]['keypoints'][16, :]
        #     anns_coorect_matching_posedataset_2[14, :] = anns[1]['keypoints'][15, :]
        #     anns_coorect_matching_posedataset_2[15, :] = anns[1]['keypoints'][14, :]
        #     anns_coorect_matching_posedataset_2[16, :] = anns[1]['keypoints'][13, :]
        #     anns_coorect_matching_posedataset_2[17, :] = anns[1]['keypoints'][20, :]
        #     anns_coorect_matching_posedataset_2[18, :] = anns[1]['keypoints'][19, :]
        #     anns_coorect_matching_posedataset_2[19, :] = anns[1]['keypoints'][18, :]
        #     anns_coorect_matching_posedataset_2[20, :] = anns[1]['keypoints'][17, :]
        #     anns[1]['keypoints'] = np.copy(anns_coorect_matching_posedataset_2)
        #
        # elif anns.__len__() == 1:
        #     anns_coorect_matching_posedataset = np.zeros_like(anns[0]['keypoints'])
        #     anns_coorect_matching_posedataset[1, :] = anns[0]['keypoints'][4, :]
        #     anns_coorect_matching_posedataset[2, :] = anns[0]['keypoints'][3, :]
        #     anns_coorect_matching_posedataset[3, :] = anns[0]['keypoints'][2, :]
        #     anns_coorect_matching_posedataset[4, :] = anns[0]['keypoints'][1, :]
        #     anns_coorect_matching_posedataset[5, :] = anns[0]['keypoints'][8, :]
        #     anns_coorect_matching_posedataset[6, :] = anns[0]['keypoints'][7, :]
        #     anns_coorect_matching_posedataset[7, :] = anns[0]['keypoints'][6, :]
        #     anns_coorect_matching_posedataset[8, :] = anns[0]['keypoints'][5, :]
        #     anns_coorect_matching_posedataset[13, :] = anns[0]['keypoints'][16, :]
        #     anns_coorect_matching_posedataset[14, :] = anns[0]['keypoints'][15, :]
        #     anns_coorect_matching_posedataset[15, :] = anns[0]['keypoints'][14, :]
        #     anns_coorect_matching_posedataset[16, :] = anns[0]['keypoints'][13, :]
        #     anns_coorect_matching_posedataset[17, :] = anns[0]['keypoints'][20, :]
        #     anns_coorect_matching_posedataset[18, :] = anns[0]['keypoints'][19, :]
        #     anns_coorect_matching_posedataset[19, :] = anns[0]['keypoints'][18, :]
        #     anns_coorect_matching_posedataset[20, :] = anns[0]['keypoints'][17, :]
        #     anns[0]['keypoints'] = np.copy(anns_coorect_matching_posedataset)
        #
        # else:
        #     raise AssertionError('frame index={} has no annotation!'.format(index))

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
        # plt.show()

        meta = None

        # preprocess image and annotations
        img, anns, meta = self.preprocess(img, anns, meta)
        meta.update({
            'dataset_index': index,
            'file_name': os.path.join(self.image_dir, self.all_names[index]),
            'gt_before_preprocess': gt_before_preprocess,
        })


        return img, anns, meta

    def __len__(self):
        return self.all_names.__len__()

class ImageList_PoseDataset_hvr(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None):
        self.image_paths = image_paths
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.all_RGB_names=np.load('{}/RGB_names.npy'.format(self.image_paths))

    def __getitem__(self, index):
        image_path = '{}'.format(self.all_RGB_names[index])
        with open(image_path, 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')

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
        assert img.size[0] == target_w
        assert img.size[1] == target_h
        # pad frames
        img = np.asarray(img)
        pad_up = (reference_edge - img.shape[0]) // 2
        pad_down = (reference_edge - img.shape[0]) // 2
        pad_left = (reference_edge - img.shape[1]) // 2
        pad_right = (reference_edge - img.shape[1]) // 2
        img = np.pad(img, pad_width=((pad_up, pad_down), (pad_left, pad_right), (0, 0)), mode='symmetric')
        img = Image.fromarray(img.astype('uint8'), 'RGB')


        anns = []
        img, anns, meta = self.preprocess(img, anns, None)
        meta.update({
            'dataset_index': index,
            'file_name': '{}'.format(self.all_RGB_names[index]),
        })

        return img, anns, meta

    def __len__(self):
        return len(self.all_RGB_names)


class PilImageList(torch.utils.data.Dataset):
    def __init__(self, images, preprocess=None):
        self.images = images
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

    def __getitem__(self, index):
        image = self.images[index].copy().convert('RGB')

        anns = []
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update({
            'dataset_index': index,
            'file_name': 'pilimage{}'.format(index),
        })

        return image, anns, meta

    def __len__(self):
        return len(self.all_RGB_names)



class ImageList_Freihand(torch.utils.data.Dataset):
    def __init__(self, image_dir, mode, preprocess=None):
        self.image_dir = image_dir
        self.K_list, self.xyz_list = load_db_annotation(image_dir, 'training')
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.mode = mode
        self.number_unique_imgs = db_size('training')
        # self.number_unique_imgs = 5
        if self.mode == 'evaluation':
            self.number_version = 1
        else:
            raise AssertionError('number_version not defined!')


    def __getitem__(self, index):
        if self.mode == 'evaluation':
            version = sample_version.auto
        else:
            raise AssertionError ('version not defined!')
        # load image and mask
        img = read_img(index%self.number_unique_imgs, self.image_dir, 'training', version)

        # annotation for this frame
        K, xyz = self.K_list[index%self.number_unique_imgs], self.xyz_list[index%self.number_unique_imgs]
        K, xyz = [np.array(x) for x in [K, xyz]]
        uv = projectPoints(xyz, K) # 2D gt keypoints
        meta = None

        img = Image.fromarray(img.astype('uint8'), 'RGB')
        visibility_flag = 2 # 0 not labeled and not visible, 1 labeled but not visible and 2 means labeled and visible
        anns = [{'keypoints': np.hstack((uv, visibility_flag*np.ones((uv.shape[0], 1))))}]
        anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
        anns[0].update({'iscrowd': 0})


        # # TODO uncomment for old trained rhd evaluation
        # # for matching rhd annotation when combining with posedataset/google/freihand/panoptic: this also matches with rhd_constants.py
        # if anns.__len__() == 2:
        #     anns_coorect_matching_posedataset = np.zeros_like(anns[0]['keypoints'])
        #     anns_coorect_matching_posedataset[1, :] = anns[0]['keypoints'][4, :]
        #     anns_coorect_matching_posedataset[2, :] = anns[0]['keypoints'][3, :]
        #     anns_coorect_matching_posedataset[3, :] = anns[0]['keypoints'][2, :]
        #     anns_coorect_matching_posedataset[4, :] = anns[0]['keypoints'][1, :]
        #     anns_coorect_matching_posedataset[5, :] = anns[0]['keypoints'][8, :]
        #     anns_coorect_matching_posedataset[6, :] = anns[0]['keypoints'][7, :]
        #     anns_coorect_matching_posedataset[7, :] = anns[0]['keypoints'][6, :]
        #     anns_coorect_matching_posedataset[8, :] = anns[0]['keypoints'][5, :]
        #     anns_coorect_matching_posedataset[13, :] = anns[0]['keypoints'][16, :]
        #     anns_coorect_matching_posedataset[14, :] = anns[0]['keypoints'][15, :]
        #     anns_coorect_matching_posedataset[15, :] = anns[0]['keypoints'][14, :]
        #     anns_coorect_matching_posedataset[16, :] = anns[0]['keypoints'][13, :]
        #     anns_coorect_matching_posedataset[17, :] = anns[0]['keypoints'][20, :]
        #     anns_coorect_matching_posedataset[18, :] = anns[0]['keypoints'][19, :]
        #     anns_coorect_matching_posedataset[19, :] = anns[0]['keypoints'][18, :]
        #     anns_coorect_matching_posedataset[20, :] = anns[0]['keypoints'][17, :]
        #     anns[0]['keypoints'] = np.copy(anns_coorect_matching_posedataset)
        #
        #     anns_coorect_matching_posedataset_2 = np.zeros_like(anns[1]['keypoints'])
        #     anns_coorect_matching_posedataset_2[1, :] = anns[1]['keypoints'][4, :]
        #     anns_coorect_matching_posedataset_2[2, :] = anns[1]['keypoints'][3, :]
        #     anns_coorect_matching_posedataset_2[3, :] = anns[1]['keypoints'][2, :]
        #     anns_coorect_matching_posedataset_2[4, :] = anns[1]['keypoints'][1, :]
        #     anns_coorect_matching_posedataset_2[5, :] = anns[1]['keypoints'][8, :]
        #     anns_coorect_matching_posedataset_2[6, :] = anns[1]['keypoints'][7, :]
        #     anns_coorect_matching_posedataset_2[7, :] = anns[1]['keypoints'][6, :]
        #     anns_coorect_matching_posedataset_2[8, :] = anns[1]['keypoints'][5, :]
        #     anns_coorect_matching_posedataset_2[13, :] = anns[1]['keypoints'][16, :]
        #     anns_coorect_matching_posedataset_2[14, :] = anns[1]['keypoints'][15, :]
        #     anns_coorect_matching_posedataset_2[15, :] = anns[1]['keypoints'][14, :]
        #     anns_coorect_matching_posedataset_2[16, :] = anns[1]['keypoints'][13, :]
        #     anns_coorect_matching_posedataset_2[17, :] = anns[1]['keypoints'][20, :]
        #     anns_coorect_matching_posedataset_2[18, :] = anns[1]['keypoints'][19, :]
        #     anns_coorect_matching_posedataset_2[19, :] = anns[1]['keypoints'][18, :]
        #     anns_coorect_matching_posedataset_2[20, :] = anns[1]['keypoints'][17, :]
        #     anns[1]['keypoints'] = np.copy(anns_coorect_matching_posedataset_2)
        #
        # elif anns.__len__() == 1:
        #     anns_coorect_matching_posedataset = np.zeros_like(anns[0]['keypoints'])
        #     anns_coorect_matching_posedataset[1, :] = anns[0]['keypoints'][4, :]
        #     anns_coorect_matching_posedataset[2, :] = anns[0]['keypoints'][3, :]
        #     anns_coorect_matching_posedataset[3, :] = anns[0]['keypoints'][2, :]
        #     anns_coorect_matching_posedataset[4, :] = anns[0]['keypoints'][1, :]
        #     anns_coorect_matching_posedataset[5, :] = anns[0]['keypoints'][8, :]
        #     anns_coorect_matching_posedataset[6, :] = anns[0]['keypoints'][7, :]
        #     anns_coorect_matching_posedataset[7, :] = anns[0]['keypoints'][6, :]
        #     anns_coorect_matching_posedataset[8, :] = anns[0]['keypoints'][5, :]
        #     anns_coorect_matching_posedataset[13, :] = anns[0]['keypoints'][16, :]
        #     anns_coorect_matching_posedataset[14, :] = anns[0]['keypoints'][15, :]
        #     anns_coorect_matching_posedataset[15, :] = anns[0]['keypoints'][14, :]
        #     anns_coorect_matching_posedataset[16, :] = anns[0]['keypoints'][13, :]
        #     anns_coorect_matching_posedataset[17, :] = anns[0]['keypoints'][20, :]
        #     anns_coorect_matching_posedataset[18, :] = anns[0]['keypoints'][19, :]
        #     anns_coorect_matching_posedataset[19, :] = anns[0]['keypoints'][18, :]
        #     anns_coorect_matching_posedataset[20, :] = anns[0]['keypoints'][17, :]
        #     anns[0]['keypoints'] = np.copy(anns_coorect_matching_posedataset)
        #
        # else:
        #     raise AssertionError('frame index={} has no annotation!'.format(index))

        # preprocess image and annotations
        img, anns, meta = self.preprocess(img, anns, meta)

        meta.update({
            'dataset_index': index,
            'file_name': self.image_dir+'training'+'/rgb/'+'%08d.jpg' % sample_version.map_id(index%self.number_unique_imgs, version),
        })

        return img, anns, meta

    def __len__(self):
        return self.number_unique_imgs*self.number_version





# class ImageList_Nyu(torch.utils.data.Dataset):
#     def __init__(self, image_dir, mode, preprocess=None):
#         self.camera_number = 1
#         self.image_dir = image_dir
#         self.preprocess = preprocess or transforms.EVAL_TRANSFORM
#         if mode == 'training':
#             self.mode = 'train'
#         elif mode == 'evaluation':
#             self.mode = 'test'
#         else:
#             raise AssertionError('mode not defined!')
#         self.keypointsUV = scio.loadmat('{}/{}/joint_data.mat'.format(self.image_dir, self.mode))['joint_uvd'][:,:,:32,:2]
#
#
#     def __getitem__(self, index):
#         filename = self.image_dir + '/' + self.mode + '/' + 'rgb_{}_{:07d}'.format(self.camera_number, index+1) + '.png'
#         img = Image.open(filename).convert('RGB')
#         uv = self.keypointsUV[self.camera_number - 1, index, :, :]
#
#         meta = None
#
#         img = Image.fromarray(img.astype('uint8'), 'RGB')
#         visibility_flag = 2 # 0 not labeled and not visible, 1 labeled but not visible and 2 means labeled and visible
#         anns = [{'keypoints': np.hstack((uv, visibility_flag*np.ones((uv.shape[0], 1))))}]
#         anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
#         anns[0].update({'iscrowd': 0})
#
#         # preprocess image and annotations
#         img, anns, meta = self.preprocess(img, anns, meta)
#
#         meta.update({
#             'dataset_index': index,
#             'file_name': filename,
#         })
#
#         return img, anns, meta
#
#     def __len__(self):
#         return self.keypointsUV.shape[1]
class ImageList_Nyu(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None):
        self.image_paths = image_paths
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')

        anns = []
        img, anns, meta = self.preprocess(img, anns, None)
        meta.update({
            'dataset_index': index,
            'file_name': image_path,
        })

        return img, anns, meta

    def __len__(self):
        return len(self.image_paths)

class ImageList_OneHand10K(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None):
        self.image_paths = image_paths
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')

        anns = []

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

        image, anns, meta = self.preprocess(image, anns, None)
        meta.update({
            'dataset_index': index,
            'file_name': image_path,
        })

        return image, anns, meta

    def __len__(self):
        return len(self.image_paths)


class ImageList_RHD(torch.utils.data.Dataset):
    def __init__(self, image_paths, mode='predict_list', preprocess=None):
        self.image_paths = image_paths
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

        if mode == 'predict_list':
            self.mode = mode
            # TODO modify
            with open('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/RHD_published_v2/evaluation/anno_evaluation.pickle', 'rb') as fi:
                self.anno_all = pickle.load(fi)
        elif mode == 'evaluation':
            self.mode = mode
            # load all annotations of this mode
            with open(os.path.join(self.image_paths, self.mode, 'anno_%s.pickle' % self.mode), 'rb') as fi:
                self.anno_all = pickle.load(fi)
        else:
            raise AssertionError('mode not defined!')
        self.index = []

    def __getitem__(self, index):
        # index=3

        if self.mode== 'predict_list':
            image_path = self.image_paths[index]
            # image_path = self.image_paths[0]
            with open(image_path, 'rb') as f:
                img = PIL.Image.open(f).convert('RGB')
        elif self.mode== 'evaluation':
            # load image
            with open(os.path.join(self.image_paths, self.mode, 'color', '%.5d.png' % index), 'rb') as f:
                img = Image.open(f).convert('RGB')

        # # uncomment for mode predict_list
        # index=3

        # annotation for this frame
        visibility_flag = 2
        self.anno_all[index]['uv_vis'][:, 2] = visibility_flag*self.anno_all[index]['uv_vis'][:,2]
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


        # for matching rhd annotation when combining with posedataset/google/freihand/panoptic: this also matches with rhd_constants.py
        if anns.__len__()==2:
            anns_coorect_matching_posedataset = np.zeros_like(anns[0]['keypoints'])
            anns_coorect_matching_posedataset[1,:] = anns[0]['keypoints'][4, :]
            anns_coorect_matching_posedataset[2, :] = anns[0]['keypoints'][3, :]
            anns_coorect_matching_posedataset[3, :] = anns[0]['keypoints'][2, :]
            anns_coorect_matching_posedataset[4, :] = anns[0]['keypoints'][1, :]
            anns_coorect_matching_posedataset[5, :] = anns[0]['keypoints'][8, :]
            anns_coorect_matching_posedataset[6, :] = anns[0]['keypoints'][7, :]
            anns_coorect_matching_posedataset[7, :] = anns[0]['keypoints'][6, :]
            anns_coorect_matching_posedataset[8, :] = anns[0]['keypoints'][5, :]
            anns_coorect_matching_posedataset[13, :] = anns[0]['keypoints'][16, :]
            anns_coorect_matching_posedataset[14, :] = anns[0]['keypoints'][15, :]
            anns_coorect_matching_posedataset[15, :] = anns[0]['keypoints'][14, :]
            anns_coorect_matching_posedataset[16, :] = anns[0]['keypoints'][13, :]
            anns_coorect_matching_posedataset[17, :] = anns[0]['keypoints'][20, :]
            anns_coorect_matching_posedataset[18, :] = anns[0]['keypoints'][19, :]
            anns_coorect_matching_posedataset[19, :] = anns[0]['keypoints'][18, :]
            anns_coorect_matching_posedataset[20, :] = anns[0]['keypoints'][17, :]
            anns[0]['keypoints'] = np.copy(anns_coorect_matching_posedataset)

            anns_coorect_matching_posedataset_2 = np.zeros_like(anns[1]['keypoints'])
            anns_coorect_matching_posedataset_2[1,:] = anns[1]['keypoints'][4, :]
            anns_coorect_matching_posedataset_2[2, :] = anns[1]['keypoints'][3, :]
            anns_coorect_matching_posedataset_2[3, :] = anns[1]['keypoints'][2, :]
            anns_coorect_matching_posedataset_2[4, :] = anns[1]['keypoints'][1, :]
            anns_coorect_matching_posedataset_2[5, :] = anns[1]['keypoints'][8, :]
            anns_coorect_matching_posedataset_2[6, :] = anns[1]['keypoints'][7, :]
            anns_coorect_matching_posedataset_2[7, :] = anns[1]['keypoints'][6, :]
            anns_coorect_matching_posedataset_2[8, :] = anns[1]['keypoints'][5, :]
            anns_coorect_matching_posedataset_2[13, :] = anns[1]['keypoints'][16, :]
            anns_coorect_matching_posedataset_2[14, :] = anns[1]['keypoints'][15, :]
            anns_coorect_matching_posedataset_2[15, :] = anns[1]['keypoints'][14, :]
            anns_coorect_matching_posedataset_2[16, :] = anns[1]['keypoints'][13, :]
            anns_coorect_matching_posedataset_2[17, :] = anns[1]['keypoints'][20, :]
            anns_coorect_matching_posedataset_2[18, :] = anns[1]['keypoints'][19, :]
            anns_coorect_matching_posedataset_2[19, :] = anns[1]['keypoints'][18, :]
            anns_coorect_matching_posedataset_2[20, :] = anns[1]['keypoints'][17, :]
            anns[1]['keypoints'] = np.copy(anns_coorect_matching_posedataset_2)

        elif anns.__len__()==1:
            anns_coorect_matching_posedataset = np.zeros_like(anns[0]['keypoints'])
            anns_coorect_matching_posedataset[1,:] = anns[0]['keypoints'][4, :]
            anns_coorect_matching_posedataset[2, :] = anns[0]['keypoints'][3, :]
            anns_coorect_matching_posedataset[3, :] = anns[0]['keypoints'][2, :]
            anns_coorect_matching_posedataset[4, :] = anns[0]['keypoints'][1, :]
            anns_coorect_matching_posedataset[5, :] = anns[0]['keypoints'][8, :]
            anns_coorect_matching_posedataset[6, :] = anns[0]['keypoints'][7, :]
            anns_coorect_matching_posedataset[7, :] = anns[0]['keypoints'][6, :]
            anns_coorect_matching_posedataset[8, :] = anns[0]['keypoints'][5, :]
            anns_coorect_matching_posedataset[13, :] = anns[0]['keypoints'][16, :]
            anns_coorect_matching_posedataset[14, :] = anns[0]['keypoints'][15, :]
            anns_coorect_matching_posedataset[15, :] = anns[0]['keypoints'][14, :]
            anns_coorect_matching_posedataset[16, :] = anns[0]['keypoints'][13, :]
            anns_coorect_matching_posedataset[17, :] = anns[0]['keypoints'][20, :]
            anns_coorect_matching_posedataset[18, :] = anns[0]['keypoints'][19, :]
            anns_coorect_matching_posedataset[19, :] = anns[0]['keypoints'][18, :]
            anns_coorect_matching_posedataset[20, :] = anns[0]['keypoints'][17, :]
            anns[0]['keypoints'] = np.copy(anns_coorect_matching_posedataset)

        else:
            raise AssertionError('frame index={} has no annotation!'.format(index))

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2, figsize=(12, 12))
        # ax[0].imshow(np.asarray(img))
        # bool_annotated_joints_1 = anns[0]['keypoints'][:, 2]==2
        # ax[0].plot(anns[0]['keypoints'][bool_annotated_joints_1, 0], anns[0]['keypoints'][bool_annotated_joints_1, 1], 'ro')
        # # bool_annotated_joints_2 = anns[1]['keypoints'][:, 2] == 2
        # # ax[0].plot(anns[1]['keypoints'][bool_annotated_joints_2, 0], anns[1]['keypoints'][bool_annotated_joints_2, 1], 'go')

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
        # # bool_annotated_joints_2 = anns[1]['keypoints'][:, 2] == 2
        # # ax[1].plot(anns[1]['keypoints'][bool_annotated_joints_2, 0], anns[1]['keypoints'][bool_annotated_joints_2, 1],
        # #            'go')
        # plt.show()


        img, anns, meta = self.preprocess(img, anns, None)

        if self.mode == 'evaluation':
            meta.update({
                'dataset_index': index,
                'file_name': os.path.join(self.image_paths, self.mode, 'color', '%.5d.png' % index),
            })
        elif self.mode == 'predict_list':
            meta.update({
                'dataset_index': index,
                'file_name': self.image_paths[0],
            })


        return img, anns, meta

    def __len__(self):
        if self.mode == 'predict_list':
            return len(self.image_paths)
        elif self.mode == 'evaluation':
            return self.anno_all.__len__()
        else:
            raise AssertionError('mode not defined!')




class ImageList_Freihand_google():
    def __init__(self, image_dir, mode, preprocess=None):
        self.image_dir = image_dir
        self.K_list, self.xyz_list = load_db_annotation(image_dir, 'training')
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.mode = mode
        self.number_unique_imgs = db_size('training')
        # self.number_unique_imgs = 5
        if self.mode == 'evaluation':
            self.number_version = 1
        else:
            raise AssertionError('number_version not defined!')


    def __getitem__(self, index):
        if self.mode == 'evaluation':
            version = sample_version.auto
        else:
            raise AssertionError ('version not defined!')
        # load image and mask
        img = read_img(index%self.number_unique_imgs, self.image_dir, 'training', version)

        # annotation for this frame
        K, xyz = self.K_list[index%self.number_unique_imgs], self.xyz_list[index%self.number_unique_imgs]
        K, xyz = [np.array(x) for x in [K, xyz]]
        uv = projectPoints(xyz, K) # 2D gt keypoints
        meta = None

        img = Image.fromarray(img.astype('uint8'), 'RGB')
        visibility_flag = 2 # 0 not labeled and not visible, 1 labeled but not visible and 2 means labeled and visible
        anns = [{'keypoints': np.hstack((uv, visibility_flag*np.ones((uv.shape[0], 1))))}]
        anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
        anns[0].update({'iscrowd': 0})

        # preprocess image and annotations
        img, anns, meta = self.preprocess(img, anns, meta)

        meta.update({
            'dataset_index': index,
            'file_name': self.image_dir+'training'+'/rgb/'+'%08d.jpg' % sample_version.map_id(index%self.number_unique_imgs, version),
        })

        return img, anns, meta

    def __len__(self):
        return self.number_unique_imgs*self.number_version