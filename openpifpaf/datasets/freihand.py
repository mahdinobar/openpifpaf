from collections import defaultdict
import copy
import logging
import os

import numpy as np
import torch.utils.data
from PIL import Image

from .. import transforms, utils

from .freihand_utils import *


LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class Freihand(torch.utils.data.Dataset):
    """
        Freihand Dataset
    """

    def __init__(self, *, image_dir, mode):
        """
        mode = 'training' or 'evaluation'
        """

        self.K_list, self.xyz_list = load_db_annotation(image_dir, mode)
        self.image_dir = image_dir
        self.mode = mode
        self.number_unique_imgs = db_size(mode)
        self.number_version = 4

    def __getitem__(self, index):
        print('********** __getitem__ index now is = ', index)

        if index//self.number_unique_imgs == 0:
            version = sample_version.gs # green background
        elif index // self.number_unique_imgs == 1:
            version = sample_version.hom
        elif index // self.number_unique_imgs == 2:
            version = sample_version.sample
        elif index // self.number_unique_imgs == 3:
            version = sample_version.auto
        else:
            raise AssertionError('index out of allowed range!')

        # load image and mask
        img = read_img(index, self.image_dir, self.mode, version)

        # annotation for this frame
        K, xyz = self.K_list[index], self.xyz_list[index]
        K, xyz = [np.array(x) for x in [K, xyz]]
        uv = projectPoints(xyz, K) # 2D gt keypoints

        return img, uv

    def __len__(self):
        return self.number_unique_imgs*self.number_version
