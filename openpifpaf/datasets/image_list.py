import PIL
import torch


from .. import transforms

from .freihand_utils import *
from PIL import Image
import pickle


class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None):
        self.image_paths = image_paths
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = PIL.Image.open(f).convert('RGB')

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
