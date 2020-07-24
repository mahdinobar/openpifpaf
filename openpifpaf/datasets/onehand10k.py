import logging
import torch.utils.data
from PIL import Image
from .. import transforms, utils
import os
import pickle
import numpy as np






LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class OneHand10K(torch.utils.data.Dataset):
   """
       OneHand10K Dataset
   """

   def __init__(self, *, image_dir, mode, target_transforms, preprocess):
       """
       mode = 'training' or 'evaluation'
       """
       self.image_dir = image_dir
       if mode == 'training':
           self.mode = 'Train'
       elif mode == 'evaluation':
           self.mode = 'Test'
       else:
           raise AssertionError('mode not defined!')
       self.target_transforms = target_transforms
       self.preprocess = preprocess or transforms.EVAL_TRANSFORM
       # load all annotations, widths, heights
       self.names = np.genfromtxt('{}/{}/label_joint.txt'.format(self.image_dir, self.mode), delimiter=',', usecols=0,
                             dtype=str)
       # self.widths = np.genfromtxt('{}/{}/label_joint.txt'.format(self.image_dir, self.mode), delimiter=',')[:, 1]
       # self.heights = np.genfromtxt('{}/{}/label_joint.txt'.format(self.image_dir, self.mode), delimiter=',')[:, 2]
       # self.hand_number_per_frame = np.genfromtxt('{}/{}/label_joint.txt'.format(self.image_dir, self.mode),
       #                                       delimiter=',')[:, 3]
       keypoints = np.genfromtxt('{}/{}/label_joint.txt'.format(self.image_dir, self.mode), delimiter=',')[:, 4:]
       self.keypoints = keypoints.reshape(keypoints.shape[0], int(keypoints.shape[1] / 2), 2)


   def __getitem__(self, index):
       # load image
       with open('/home/mahdi/HVR/git_repos/openpifpaf/OneHand10K/{}/source/{}'.format(self.mode, self.names[index]), 'rb') as f:
           img = Image.open(f).convert('RGB')

       # annotation for this frame
       bool_invisible_keypoints = np.logical_or(self.keypoints[index, :, 0]==-1, self.keypoints[index, :, 1]==-1)
       visibility_flag = 2.
       anns = [{'keypoints':  np.hstack((self.keypoints[index,:,:], np.expand_dims(visibility_flag*(np.invert(bool_invisible_keypoints)),axis=1)))}]
       anns[0].update({'bbox': np.array([0, 0, img.size[0], img.size[1]])})
       anns[0].update({'iscrowd': 0})

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
       return len(self.names)

# def main():
#     names = np.genfromtxt('/home/mahdi/HVR/original_datasets/OneHand10K/Train/label_joint.txt', delimiter=',', usecols=0, dtype=str)
#     widths = np.genfromtxt('/home/mahdi/HVR/original_datasets/OneHand10K/Train/label_joint.txt', delimiter=',')[:,1]
#     heights = np.genfromtxt('/home/mahdi/HVR/original_datasets/OneHand10K/Train/label_joint.txt', delimiter=',')[:,2]
#     hand_number_per_frame = np.genfromtxt('/home/mahdi/HVR/original_datasets/OneHand10K/Train/label_joint.txt', delimiter=',')[:,3]
#     keypoints = np.genfromtxt('/home/mahdi/HVR/original_datasets/OneHand10K/Train/label_joint.txt', delimiter=',')[:,4:]
#     keypoints = keypoints.reshape(keypoints.shape[0], int(keypoints.shape[1]/2), 2)
#     index = 0
#     for index in range (0, 100):
#         with open('/home/mahdi/HVR/original_datasets/OneHand10K/Train/source/{}'.format(names[index]),'rb') as f:
#             img = Image.open(f).convert('RGB')
#
#         # Visualize data
#         import matplotlib.pyplot as plt
#         fig = plt.figure(figsize=(20,20))
#         ax1 = fig.add_subplot('111')
#         ax1.imshow(img)
#         bool_invisible_keypoints = np.logical_or(keypoints[index, :, 0] == -1, keypoints[index, :, 1] == -1)
#         ax1.plot(keypoints[index, np.invert(bool_invisible_keypoints), 0], keypoints[index, np.invert(bool_invisible_keypoints), 1], 'ro')
#
#         n = list(np.argwhere(np.invert(bool_invisible_keypoints) == True).squeeze())
#         for i, txt in enumerate(n):
#             ax1.annotate(txt, (keypoints[index, txt, 0], keypoints[index, txt, 1]), c='w')
#         plt.show()
#     print('here!')
#
# if __name__ == '__main__':
#    main()

