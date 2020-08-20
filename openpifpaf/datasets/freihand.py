import logging
import torch.utils.data
from .freihand_utils import *
from PIL import Image
from .. import transforms, utils




LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class Freihand(torch.utils.data.Dataset):
    """
        Freihand Dataset
    """

    def __init__(self, *, image_dir, mode, target_transforms, preprocess, even_dataset_fusion=False):
        """
        mode = 'training' or 'evaluation' or 'test'
        """

        self.K_list, self.xyz_list = load_db_annotation(image_dir, 'training')
        self.image_dir = image_dir
        self.mode = mode
        self.number_unique_imgs = db_size('training')

        if self.mode == 'training':
            # rand_id = np.random.randint(0,self.number_unique_imgs,self.number_unique_imgs)
            # self.data_names_id = rand_id[:int(rand_id.size*0.80)]
            # np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/Freihand_pub_v2/data_names_train.npy',self.data_names_id)
            # self.data_names = np.random.choice(self.number_unique_imgs, int(self.number_unique_imgs * 0.80),
            #                                    replace=False)
            # self.data_names_id = rand_id[int(rand_id.size*0.80):int(rand_id.size*0.90)]
            # np.save('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/Freihand_pub_v2/data_names_eval.npy', self.data_names_id)
            #
            # self.data_names_id = rand_id[int(rand_id.size * 0.90):]

            self.data_names_id = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/Freihand_pub_v2/data_names_train.npy')

            if even_dataset_fusion == True:
                # uncomment to make equal train data number with posedataset for fusion
                number_random_select = 2765
                selected_id=np.random.choice(self.data_names_id.size, number_random_select, replace=False)
                self.data_names_id = self.data_names_id[selected_id]

        elif self.mode == 'evaluation':
            self.data_names_id = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/Freihand_pub_v2/data_names_eval.npy')


        elif self.mode == 'test':
            self.data_names_id = np.load('/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/Freihand_pub_v2/data_names_test.npy')

        self.target_transforms = target_transforms
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM


    def __getitem__(self, index):
        # print('********** __getitem__ index now is = ', index)
        if index%4 == 0:
            version = sample_version.gs # green background
        elif index % 4 == 1:
            version = sample_version.hom
        elif index % 4 == 2:
            version = sample_version.sample
        elif index % 4 == 3:
            version = sample_version.auto
        else:
            raise AssertionError('index out of allowed range!')

        # load image and mask
        img = read_img(self.data_names_id[index%self.data_names_id.size], self.image_dir, 'training', version)

        # annotation for this frame
        K, xyz = self.K_list[self.data_names_id[index%self.data_names_id.size]], self.xyz_list[self.data_names_id[index%self.data_names_id.size]]
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
        return self.data_names_id.size*4