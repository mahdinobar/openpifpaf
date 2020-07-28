import logging
import torch.utils.data
from PIL import Image
from .. import transforms, utils
import numpy as np
import scipy.io as scio



LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class Nyu(torch.utils.data.Dataset):
    """
        Nyu Dataset
    """

    def __init__(self, *, image_dir, mode, target_transforms, preprocess):
        """
        mode = 'training' or 'evaluation'
        self.image_dir = '/media/mahdi/276bbbd7-a9de-4030-857d-34c615ff4e0a/aleix-data/NYU/dataset'
        """
        
        self.camera_number = 1
        self.image_dir = image_dir
        self.target_transforms = target_transforms
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        if mode == 'training':
            self.mode = 'train'
        elif mode == 'evaluation':
            self.mode = 'test'
        else:
            raise AssertionError('mode not defined!')
        self.keypointsUV = scio.loadmat('{}/{}/joint_data.mat'.format(self.image_dir, self.mode))['joint_uvd'][:,:,:32,:2]


    def __getitem__(self, index):
        # print('********** __getitem__ index now is = ', index)
        filename = self.image_dir + self.mode + '/' + 'rgb_{}_{:07d}'.format(self.camera_number , index+1) + '.png'
        img = Image.open(filename).convert('RGB')
        uv = self.keypointsUV[self.camera_number - 1, index, :, :]

        meta = None

        visibility_flag = 2. # 0 not labeled and not visible, 1 labeled but not visible and 2 means labeled and visible
        anns = [{'keypoints': np.hstack((uv, visibility_flag*np.ones((uv.shape[0], 1))))}]
        anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
        anns[0].update({'iscrowd': 0})

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
        return self.keypointsUV.shape[1]

# import scipy.io as scio
# keypointsUVD_test = scio.loadmat(keypoint_file)['keypoints3D'].astype(np.float32)  #shape (K, num_joints, 3)

# def loadDepthMap(filename):
#     """
#     Read a depth-map from png raw data of NYU
#     :param filename: file name to load
#     :return: image data of depth image
#     """
#     img = Image.open(filename)
#     # top 8 bits of depth are packed into green channel and lower 8 bits into blue
#     assert len(img.getbands()) == 3
#     r, g, b = img.split()
#     r = np.asarray(r, np.int32)
#     g = np.asarray(g, np.int32)
#     b = np.asarray(b, np.int32)
#     dpt = np.bitwise_or(np.left_shift(g, 8), b)
#     imgdata = np.asarray(dpt, np.float32)
#     return imgdata
# depth = loadDepthMap(self.ImgDir + 'depth_1_{:07d}'.format(index+1) + '.png')

if __name__ == '__main__':
    def loadDepthMap(filename):
        """
        Read a depth-map from png raw data of NYU
        :param filename: file name to load
        :return: image data of depth image
        """
        # img = Image.open(filename).convert('RGB')
        # # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        # assert len(img.getbands()) == 3
        # r, g, b = img.split()
        # r = np.asarray(r, np.int32)
        # g = np.asarray(g, np.int32)
        # b = np.asarray(b, np.int32)
        # dpt = np.bitwise_or(np.left_shift(g, 8), b)
        # imgdata = np.asarray(dpt, np.float32)
        # return imgdata


    index = 1
    for index in range(1,10000,500):
        camera_number = 1
        filename = '/home/mahdi/HVR/hvr/A2J/data/nyu/train/' + 'rgb_{}_{:07d}'.format(camera_number, index) + '.png'
        img = Image.open(filename).convert('RGB')

        keypoint_file = '/home/mahdi/HVR/hvr/A2J/data/nyu/nyu_keypointsUVD_train.mat'  # shape: (K, num_joints, 3)

        keypointsUV = scio.loadmat('/home/mahdi/HVR/hvr/A2J/data/nyu/train/joint_data.mat')['joint_uvd'][camera_number-1, index, :, :2]

        import matplotlib.pyplot as plt
        # import matplotlib
        # matplotlib.use('qt5agg')

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(img)
        ax.plot(keypointsUV[:32, 0], keypointsUV[:32, 1], 'ro')
        for txt in range(0,  32):#keypointsUV.shape[0]):
            ax.annotate(txt, (keypointsUV[txt, 0]+7, keypointsUV[txt, 1]+7), c='w', bbox=dict(fc=(.2, .2, .2), lw=0, pad=1))
        plt.show()

    print('end')
