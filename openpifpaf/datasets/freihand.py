import logging
import torch.utils.data
from .freihand_utils import *
from .. import transforms
from PIL import Image



LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class Freihand(torch.utils.data.Dataset):
    """
        Freihand Dataset
    """

    def __init__(self, *, image_dir, mode, target_transforms, preprocess):
        """
        mode = 'training' or 'evaluation'
        """

        self.K_list, self.xyz_list = load_db_annotation(image_dir, 'training')
        self.image_dir = image_dir
        self.mode = mode
        self.number_unique_imgs = db_size('training')
        if self.mode == 'training':
            self.number_version = 3
        elif self.mode == 'evaluation':
            self.number_version = 1

        self.target_transforms = target_transforms
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM


    def __getitem__(self, index):
        # print('********** __getitem__ index now is = ', index)
        if self.mode == 'training':
            if index//self.number_unique_imgs == 0:
                version = sample_version.gs # green background
            elif index // self.number_unique_imgs == 1:
                version = sample_version.hom
            elif index // self.number_unique_imgs == 2:
                version = sample_version.sample
            else:
                raise AssertionError('index out of allowed range!')
        elif self.mode == 'evaluation':
            version = sample_version.auto
        else:
            raise AssertionError('mode not defined!')

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

        # transform targets
        if self.target_transforms is not None:
            anns = [t(img, anns, meta) for t in self.target_transforms]


        return img, anns, meta

    def __len__(self):
        return self.number_unique_imgs*self.number_version