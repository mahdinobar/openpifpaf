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
            target_min_edge = 224
            min_edge = min(h, w)
            ratio_factor = target_min_edge / min_edge
            target_h = int(ratio_factor * h)
            target_w = int(ratio_factor * w)
            im_np = np.asarray(image)
            im_np = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1), order=order)
            image = PIL.Image.fromarray(im_np)
            assert image.size[0] == target_w
            assert image.size[1] == target_h
        # # rescale image
        # order = 1  # order of resize interpolation; 1 means linear interpolation
        # w, h = image.size
        # target_max_edge = 320
        # max_edge = max(h, w)
        # ratio_factor = target_max_edge / max_edge
        # target_h = int(ratio_factor * h)
        # target_w = int(ratio_factor * w)
        # im_np = np.asarray(image)
        # image = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1), order=order)
        # # ax[1].imshow(image)
        # pad_up = (320-image.shape[0])//2
        # pad_down = (320-image.shape[0])//2
        # pad_left = (320-image.shape[1])//2
        # pad_right = (320-image.shape[1])//2
        # image = np.pad(image, pad_width=((pad_up, pad_down), (pad_left, pad_right), (0, 0)), mode='symmetric')
        # image = Image.fromarray(image.astype('uint8'), 'RGB')
        # # ax[2].imshow(image)


        anns = []
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update({
            'dataset_index': index,
            'file_name': image_path,
        })

        return image, anns, meta

    def __len__(self):
        return len(self.image_paths)


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
        return len(self.images)



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

        # preprocess image and annotations
        img, anns, meta = self.preprocess(img, anns, meta)

        meta.update({
            'dataset_index': index,
            'file_name': self.image_dir+'training'+'/rgb/'+'%08d.jpg' % sample_version.map_id(index%self.number_unique_imgs, version),
        })

        return img, anns, meta

    def __len__(self):
        return self.number_unique_imgs*self.number_version


class ImageList_Nyu(torch.utils.data.Dataset):
    def __init__(self, image_dir, mode, preprocess=None):
        self.camera_number = 1
        self.image_dir = image_dir
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        if mode == 'training':
            self.mode = 'train'
        elif mode == 'evaluation':
            self.mode = 'test'
        else:
            raise AssertionError('mode not defined!')
        self.keypointsUV = scio.loadmat('{}/{}/joint_data.mat'.format(self.image_dir, self.mode))['joint_uvd'][:,:,:32,:2]


    def __getitem__(self, index):
        filename = self.image_dir + '/' + self.mode + '/' + 'rgb_{}_{:07d}'.format(self.camera_number, index+1) + '.png'
        img = Image.open(filename).convert('RGB')
        uv = self.keypointsUV[self.camera_number - 1, index, :, :]

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
            'file_name': filename,
        })

        return img, anns, meta

    def __len__(self):
        return self.keypointsUV.shape[1]


class ImageList_OneHand10K(torch.utils.data.Dataset):
    def __init__(self, image_dir, mode, preprocess=None):
        self.image_dir = image_dir
        if mode == 'evaluation':
            self.mode = 'Test'
        else:
            raise AssertionError('evaluation mode not defined!')
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        # load all annotations, widths, heights
        self.names = np.genfromtxt('{}/{}/label_joint.txt'.format(self.image_dir, self.mode), delimiter=',', usecols=0,
                                   dtype=str)
        keypoints = np.genfromtxt('{}/{}/label_joint.txt'.format(self.image_dir, self.mode), delimiter=',')[:, 4:]
        self.keypoints = keypoints.reshape(keypoints.shape[0], int(keypoints.shape[1] / 2), 2)

    def __getitem__(self, index):
        # load image
        with open('/home/mahdi/HVR/git_repos/openpifpaf/OneHand10K/{}/source/{}'.format(self.mode, self.names[index]),
                  'rb') as f:
            img = Image.open(f).convert('RGB')
        # annotation for this frame
        bool_invisible_keypoints = np.logical_or(self.keypoints[index, :, 0] == -1, self.keypoints[index, :, 1] == -1)
        visibility_flag = 2.
        anns = [{'keypoints': np.hstack((self.keypoints[index, :, :],
                                         np.expand_dims(visibility_flag * (np.invert(bool_invisible_keypoints)),
                                                        axis=1)))}]
        anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
        anns[0].update({'iscrowd': 0})
        # rescale image
        order = 1  # order of resize interpolation; 1 means linear interpolation
        w, h = img.size
        # keep aspect ratio the same
        target_min_edge = 224
        min_edge = min(h, w)
        ratio_factor = target_min_edge / min_edge
        target_h = int(ratio_factor * h)
        target_w = int(ratio_factor * w)
        im_np = np.asarray(img)
        im_np = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1), order=order)
        img = PIL.Image.fromarray(im_np)
        assert img.size[0] == target_w
        assert img.size[1] == target_h
        # rescale keypoints
        x_scale = (img.size[0] - 1) / (w - 1)
        y_scale = (img.size[1] - 1) / (h - 1)
        for ann in anns:
            ann['keypoints'][:, 0] = ann['keypoints'][:, 0] * x_scale
            ann['keypoints'][:, 1] = ann['keypoints'][:, 1] * y_scale
            ann['bbox'][0] *= x_scale
            ann['bbox'][1] *= y_scale
            ann['bbox'][2] *= x_scale
            ann['bbox'][3] *= y_scale
        meta = None

        # preprocess image and annotations
        img, anns, meta = self.preprocess(img, anns, meta)
        meta.update({
            'dataset_index': index,
            'file_name': '/home/mahdi/HVR/git_repos/openpifpaf/OneHand10K/{}/source/{}'.format(self.mode, self.names[index]),
        })
        return img, anns, meta

    def __len__(self):
        return len(self.names)

class ImageList_RHD(torch.utils.data.Dataset):
    def __init__(self, image_dir, mode, preprocess=None):
        self.image_dir = image_dir
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        if mode == 'evaluation':
            self.mode = mode  # 'evaluation' or 'training'
        else:
            raise AssertionError('for prediction, mode must be evaluation!')
        # load all annotations of this mode
        with open(os.path.join(self.image_dir, self.mode, 'anno_%s.pickle' % self.mode), 'rb') as fi:
            self.anno_all = pickle.load(fi)

    def __getitem__(self, index):
        # load image
        with open(os.path.join(self.image_dir, self.mode, 'color', '%.5d.png' % index), 'rb') as f:
            img = Image.open(f).convert('RGB')

        # annotation for this frame
        anns = [{'keypoints':  self.anno_all[index]['uv_vis']}]
        anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
        anns[0].update({'iscrowd': 0})

        meta = None

        # preprocess image and annotations
        img, anns, meta = self.preprocess(img, anns, meta)

        meta.update({
            'dataset_index': index,
            'file_name': (self.image_dir + '/' + self.mode + '/color/' + '%.5d.png' % index),
        })

        return img, anns, meta

    def __len__(self):
        return len(self.anno_all)




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