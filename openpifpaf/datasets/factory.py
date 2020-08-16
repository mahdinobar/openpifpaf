import torch

from .coco import Coco
from .collate import collate_images_targets_meta
from .constants import COCO_KEYPOINTS, HFLIP
from .. import transforms

from .freihand import Freihand
from .freihand_constants import FREIHAND_KEYPOINTS, FREIHAND_HFLIP

from .panoptic import Panoptic
from .panoptic_constants import PANOPTIC_KEYPOINTS, PANOPTIC_HFLIP

from .posedataset import Posedataset
from .posedataset_constants import POSEDATASET_KEYPOINTS, POSEDATASET_HFLIP

from .rhd import Rhd
from .rhd_constants import RHD_KEYPOINTS, RHD_HFLIP

from .onehand10k import OneHand10K
from .onehand10k_constants import ONEHAND10K_KEYPOINTS, ONEHAND10K_HFLIP

from .nyu import Nyu
from .nyu_constants import NYU_KEYPOINTS, NYU_HFLIP


COCOKP_ANNOTATIONS_TRAIN = 'data-mscoco/annotations/person_keypoints_train2017.json'
COCOKP_ANNOTATIONS_VAL = 'data-mscoco/annotations/person_keypoints_val2017.json'
COCODET_ANNOTATIONS_TRAIN = 'data-mscoco/annotations/instances_train2017.json'
COCODET_ANNOTATIONS_VAL = 'data-mscoco/annotations/instances_val2017.json'
COCO_IMAGE_DIR_TRAIN = 'data-mscoco/images/train2017/'
COCO_IMAGE_DIR_VAL = 'data-mscoco/images/val2017/'

FREIHAND_IMAGE_DIR_TRAIN = 'Freihand_pub_v2/'
POSEDATASET_IMAGE_DIR_TRAIN = 'pose_dataset'
RHD_IMAGE_DIR_TRAIN = 'RHD_published_v2/'
ONEHAND10K_IMAGE_DIR_TRAIN = 'OneHand10K'
NYU_IMAGE_DIR_TRAIN = 'nyu/'
PANOPTIC_IMAGE_DIR_TRAIN = '/home/mahdi/HVR/git_repos/openpifpaf/openpifpaf/Panoptic/hand143_panopticdb/'






def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--cocokp-train-annotations', default=COCOKP_ANNOTATIONS_TRAIN)
    group.add_argument('--cocodet-train-annotations', default=COCODET_ANNOTATIONS_TRAIN)
    group.add_argument('--cocokp-val-annotations', default=COCOKP_ANNOTATIONS_VAL)
    group.add_argument('--cocodet-val-annotations', default=COCODET_ANNOTATIONS_VAL)
    group.add_argument('--coco-train-image-dir', default=COCO_IMAGE_DIR_TRAIN)
    group.add_argument('--coco-val-image-dir', default=COCO_IMAGE_DIR_VAL)
    group.add_argument('--dataset', default='cocokp')
    group.add_argument('--n-images', default=None, type=int,
                       help='number of images to sample')
    group.add_argument('--duplicate-data', default=None, type=int,
                       help='duplicate data')
    group.add_argument('--loader-workers', default=None, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=8, type=int,
                       help='batch size')
    group_aug = parser.add_argument_group('augmentations')
    group_aug.add_argument('--square-edge', default=385, type=int,
                           help='square edge of input images')
    group_aug.add_argument('--extended-scale', default=False, action='store_true',
                           help='augment with an extended scale range')
    group_aug.add_argument('--orientation-invariant', default=0.0, type=float,
                           help='augment with random orientations')
    group_aug.add_argument('--no-augmentation', dest='augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')

    group.add_argument('--freihand-train-image-dir', default=FREIHAND_IMAGE_DIR_TRAIN)
    group.add_argument('--panoptic-train-image-dir', default=PANOPTIC_IMAGE_DIR_TRAIN)

    group.add_argument('--posedataset-train-image-dir', default=POSEDATASET_IMAGE_DIR_TRAIN)
    group.add_argument('--rhd-train-image-dir', default=RHD_IMAGE_DIR_TRAIN)
    group.add_argument('--onehand10k-train-image-dir', default=ONEHAND10K_IMAGE_DIR_TRAIN)
    group.add_argument('--nyu-train-image-dir', default=NYU_IMAGE_DIR_TRAIN)


    group.add_argument('--concatenate-with-dataset', default=None, nargs='+',
                       help='name of dataset to concatenate for training; None means no other dataset; [rhd, freihand, panoptic]')
    group.add_argument('--concatenate-with-dataset-image-dir', default=None , nargs='+',
                       help='directory of name of dataset to concatenate for training; None means no other dataset; [rhd_dir, freihand_dir, panoptic_dir]')






def train_configure(_):
    pass


def train_cocokp_preprocess_factory(
        *,
        square_edge,
        augmentation=True,
        extended_scale=False,
        orientation_invariant=0.0,
        rescale_images=1.0,
):
    if not augmentation:
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(square_edge),
            transforms.CenterPad(square_edge),
            transforms.EVAL_TRANSFORM,
        ])

    if extended_scale:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.25 * rescale_images, 2.0 * rescale_images),
            power_law=True)
    else:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.4 * rescale_images, 2.0 * rescale_images),
            power_law=True)

    orientation_t = None
    if orientation_invariant:
        orientation_t = transforms.RandomApply(transforms.RotateBy90(), orientation_invariant)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.AnnotationJitter(),
        transforms.RandomApply(transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
        rescale_t,
        transforms.Crop(square_edge, use_area_of_interest=True),
        transforms.CenterPad(square_edge),
        orientation_t,
        transforms.TRAIN_TRANSFORM,
    ])

def train_freihand_preprocess_factory(
        *,
        square_edge,
        augmentation=True,
        extended_scale=False,
        orientation_invariant=0.0,
        rescale_images=1.0,
):
    if not augmentation:
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(square_edge),
            transforms.CenterPad(square_edge),
            transforms.EVAL_TRANSFORM,
        ])

    if extended_scale:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.25 * rescale_images, 2.0 * rescale_images),
            power_law=True)
    else:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.4 * rescale_images, 2.0 * rescale_images),
            power_law=True)

    orientation_t = None
    if orientation_invariant:
        orientation_t = transforms.RandomApply(transforms.RotateBy90(), orientation_invariant)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.AnnotationJitter(),
        transforms.RandomApply(transforms.HFlip(FREIHAND_KEYPOINTS, FREIHAND_HFLIP), 0.50),
        rescale_t,
        transforms.Crop(square_edge, use_area_of_interest=True),
        transforms.CenterPad(square_edge),
        orientation_t,
        transforms.TRAIN_TRANSFORM,
    ])

def train_panoptic_preprocess_factory(
        *,
        square_edge,
        augmentation=True,
        extended_scale=False,
        orientation_invariant=0.0,
        rescale_images=1.0,
):
    if not augmentation:
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(square_edge),
            transforms.CenterPad(square_edge),
            transforms.EVAL_TRANSFORM,
        ])

    if extended_scale:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.25 * rescale_images, 2.0 * rescale_images),
            power_law=True)
    else:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.4 * rescale_images, 2.0 * rescale_images),
            power_law=True)

    orientation_t = None
    if orientation_invariant:
        orientation_t = transforms.RandomApply(transforms.RotateBy90(), orientation_invariant)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.AnnotationJitter(),
        transforms.RandomApply(transforms.HFlip(PANOPTIC_KEYPOINTS, PANOPTIC_HFLIP), 0.5),
        rescale_t,
        transforms.Crop(square_edge, use_area_of_interest=True),
        transforms.CenterPad(square_edge),
        orientation_t,
        transforms.TRAIN_TRANSFORM,
    ])

def train_posedataset_preprocess_factory(
        *,
        square_edge,
        augmentation=True,
        extended_scale=False,
        orientation_invariant=0.0,
        rescale_images=1.0,
):
    if not augmentation:
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(square_edge),
            transforms.CenterPad(square_edge),
            transforms.EVAL_TRANSFORM,
        ])

    if extended_scale:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.25 * rescale_images, 2.0 * rescale_images),
            power_law=True)
    else:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.4 * rescale_images, 2.0 * rescale_images),
            power_law=True)

    orientation_t = None
    if orientation_invariant:
        orientation_t = transforms.RandomApply(transforms.RotateBy90(), orientation_invariant)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.AnnotationJitter(),
        transforms.RandomApply(transforms.HFlip(POSEDATASET_KEYPOINTS, POSEDATASET_HFLIP), 0.5),
        rescale_t,
        transforms.Crop(square_edge, use_area_of_interest=True),
        transforms.CenterPad(square_edge),
        orientation_t,
        transforms.TRAIN_TRANSFORM,
    ])


def train_nyu_preprocess_factory(
        *,
        square_edge,
        augmentation=True,
        extended_scale=False,
        orientation_invariant=0.0,
        rescale_images=1.0,
):
    if not augmentation:
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(square_edge),
            transforms.CenterPad(square_edge),
            transforms.EVAL_TRANSFORM,
        ])

    if extended_scale:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.25 * rescale_images, 2.0 * rescale_images),
            power_law=True)
    else:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.4 * rescale_images, 2.0 * rescale_images),
            power_law=True)

    orientation_t = None
    if orientation_invariant:
        orientation_t = transforms.RandomApply(transforms.RotateBy90(), orientation_invariant)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.AnnotationJitter(),
        transforms.RandomApply(transforms.HFlip(NYU_KEYPOINTS, NYU_HFLIP), 0.5),
        rescale_t,
        transforms.Crop(square_edge, use_area_of_interest=True),
        transforms.CenterPad(square_edge),
        orientation_t,
        transforms.TRAIN_TRANSFORM,
    ])


def train_onehand10k_preprocess_factory(
        *,
        square_edge,
        augmentation=True,
        extended_scale=False,
        orientation_invariant=0.0,
        rescale_images=1.0,
):
    if not augmentation:
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(square_edge),
            transforms.CenterPad(square_edge),
            transforms.EVAL_TRANSFORM,
        ])

    if extended_scale:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.25 * rescale_images, 2.0 * rescale_images),
            power_law=True)
    else:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.4 * rescale_images, 2.0 * rescale_images),
            power_law=True)

    orientation_t = None
    if orientation_invariant:
        orientation_t = transforms.RandomApply(transforms.RotateBy90(), orientation_invariant)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.AnnotationJitter(),
        transforms.RandomApply(transforms.HFlip(ONEHAND10K_KEYPOINTS, ONEHAND10K_HFLIP), 0.5),
        rescale_t,
        transforms.Crop(square_edge, use_area_of_interest=True),
        transforms.CenterPad(square_edge),
        orientation_t,
        transforms.TRAIN_TRANSFORM,
    ])


def train_rhd_preprocess_factory(
        *,
        square_edge,
        augmentation=True,
        extended_scale=False,
        orientation_invariant=0.0,
        rescale_images=1.0,
):
    if not augmentation:
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(square_edge),
            transforms.CenterPad(square_edge),
            transforms.EVAL_TRANSFORM,
        ])

    if extended_scale:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.25 * rescale_images, 2.0 * rescale_images),
            power_law=True)
    else:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.4 * rescale_images, 2.0 * rescale_images),
            power_law=True)

    orientation_t = None
    if orientation_invariant:
        orientation_t = transforms.RandomApply(transforms.RotateBy90(), orientation_invariant)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.AnnotationJitter(),
        transforms.RandomApply(transforms.HFlip(RHD_KEYPOINTS, RHD_HFLIP), 0.5),
        rescale_t,
        transforms.Crop(square_edge, use_area_of_interest=True),
        transforms.CenterPad(square_edge),
        orientation_t,
        transforms.TRAIN_TRANSFORM,
    ])


def train_cocodet_preprocess_factory(
        *,
        square_edge,
        augmentation=True,
        extended_scale=False,
        orientation_invariant=0.0,
        rescale_images=1.0,
):
    if not augmentation:
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(square_edge),
            transforms.CenterPad(square_edge),
            transforms.EVAL_TRANSFORM,
        ])

    if extended_scale:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.5 * rescale_images, 2.0 * rescale_images),
            power_law=True)
    else:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.7 * rescale_images, 1.5 * rescale_images),
            power_law=True)

    orientation_t = None
    if orientation_invariant:
        orientation_t = transforms.RandomApply(transforms.RotateBy90(), orientation_invariant)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.AnnotationJitter(),
        transforms.RandomApply(transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
        rescale_t,
        transforms.Crop(square_edge, use_area_of_interest=False),
        transforms.CenterPad(square_edge),
        orientation_t,
        transforms.MinSize(min_side=4.0),
        transforms.UnclippedArea(),
        transforms.UnclippedSides(),
        transforms.TRAIN_TRANSFORM,
    ])


def train_cocokp_factory(args, target_transforms):
    preprocess = train_cocokp_preprocess_factory(
        square_edge=args.square_edge,
        augmentation=args.augmentation,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
        rescale_images=args.rescale_images)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    train_data = Coco(
        image_dir=args.coco_train_image_dir,
        ann_file=args.cocokp_train_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images,
        image_filter='keypoint-annotations',
        category_ids=[1],
    )
    if args.duplicate_data:
        train_data = torch.utils.data.ConcatDataset(
            [train_data for _ in range(args.duplicate_data)])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    val_data = Coco(
        image_dir=args.coco_val_image_dir,
        ann_file=args.cocokp_val_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images,
        image_filter='keypoint-annotations',
        category_ids=[1],
    )
    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader


def train_freihand_factory(args, target_transforms):
    preprocess = train_freihand_preprocess_factory(
        square_edge=args.square_edge,
        augmentation=args.augmentation,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
        rescale_images=args.rescale_images)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    train_data = Freihand(image_dir=args.freihand_train_image_dir, mode='training', preprocess=preprocess,
        target_transforms=target_transforms)

    if args.duplicate_data:
        train_data = torch.utils.data.ConcatDataset(
            [train_data for _ in range(args.duplicate_data)])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    val_data = Freihand(image_dir=args.freihand_train_image_dir, mode='evaluation', preprocess=preprocess,
        target_transforms=target_transforms)

    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader


def train_panoptic_factory(args, target_transforms):
    preprocess = train_panoptic_preprocess_factory(
        square_edge=args.square_edge,
        augmentation=args.augmentation,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
        rescale_images=args.rescale_images)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    train_data = Panoptic(image_dir=args.panoptic_train_image_dir, mode='training', preprocess=preprocess,
        target_transforms=target_transforms)

    if args.duplicate_data:
        train_data = torch.utils.data.ConcatDataset(
            [train_data for _ in range(args.duplicate_data)])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    val_data = Panoptic(image_dir=args.panoptic_train_image_dir, mode='evaluation', preprocess=preprocess,
        target_transforms=target_transforms)

    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader


def train_posedataset_factory(args, target_transforms):
    preprocess = train_posedataset_preprocess_factory(
        square_edge=args.square_edge,
        augmentation=args.augmentation,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
        rescale_images=args.rescale_images)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size


    if args.concatenate_with_dataset is not None:
        train_data_posedataset = Posedataset(image_dir=args.posedataset_train_image_dir, mode='training', preprocess=preprocess,
                                 target_transforms=target_transforms)
        train_data=[]
        train_data.append(train_data_posedataset)
        if args.concatenate_with_dataset[0]=='rhd':
            train_data_rhd = Rhd(image_dir=args.concatenate_with_dataset_image_dir[0], mode='training', preprocess=preprocess,
                             target_transforms=target_transforms)
            train_data.append(train_data_rhd)

        if args.concatenate_with_dataset[1] == 'freihand':
            train_data_freihand = Freihand(image_dir=args.concatenate_with_dataset_image_dir[1], mode='training', preprocess=preprocess,
                                  target_transforms=target_transforms)
            train_data.append(train_data_freihand)

        if args.concatenate_with_dataset[2] == 'panoptic':
            train_data_panoptic = Panoptic(image_dir=args.concatenate_with_dataset_image_dir[2], mode='training',
                                 preprocess=preprocess,
                                 target_transforms=target_transforms)
            train_data.append(train_data_panoptic)
        train_data = torch.utils.data.ConcatDataset(train_data)

    else:
        train_data = Posedataset(image_dir=args.posedataset_train_image_dir, mode='training', preprocess=preprocess,
                                 target_transforms=target_transforms)


    if args.duplicate_data:
        train_data = torch.utils.data.ConcatDataset(
            [train_data for _ in range(args.duplicate_data)])


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    val_data = Posedataset(image_dir=args.posedataset_train_image_dir, mode='evaluation', preprocess=preprocess,
        target_transforms=target_transforms)

    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader



def train_nyu_factory(args, target_transforms):
    preprocess = train_nyu_preprocess_factory(
        square_edge=args.square_edge,
        augmentation=args.augmentation,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
        rescale_images=args.rescale_images)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    train_data = Nyu(image_dir=args.nyu_train_image_dir, mode='training', preprocess=preprocess,
        target_transforms=target_transforms)

    if args.duplicate_data:
        train_data = torch.utils.data.ConcatDataset(
            [train_data for _ in range(args.duplicate_data)])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    val_data = Nyu(image_dir=args.nyu_train_image_dir, mode='evaluation', preprocess=preprocess,
        target_transforms=target_transforms)

    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader

def train_onehand10k_factory(args, target_transforms):
    preprocess = train_onehand10k_preprocess_factory(
        square_edge=args.square_edge,
        augmentation=args.augmentation,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
        rescale_images=args.rescale_images)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    train_data = OneHand10K(image_dir=args.onehand10k_train_image_dir, mode='training', preprocess=preprocess,
        target_transforms=target_transforms)

    if args.duplicate_data:
        train_data = torch.utils.data.ConcatDataset(
            [train_data for _ in range(args.duplicate_data)])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    val_data = OneHand10K(image_dir=args.onehand10k_train_image_dir, mode='evaluation', preprocess=preprocess,
        target_transforms=target_transforms)

    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader


def train_rhd_factory(args, target_transforms):
    preprocess = train_rhd_preprocess_factory(
        square_edge=args.square_edge,
        augmentation=args.augmentation,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
        rescale_images=args.rescale_images)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    train_data = Rhd(image_dir=args.rhd_train_image_dir, mode='training', preprocess=preprocess,
        target_transforms=target_transforms)

    if args.duplicate_data:
        train_data = torch.utils.data.ConcatDataset(
            [train_data for _ in range(args.duplicate_data)])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    val_data = Rhd(image_dir=args.rhd_train_image_dir, mode='evaluation', preprocess=preprocess,
        target_transforms=target_transforms)

    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader




def train_cocodet_factory(args, target_transforms):
    preprocess = train_cocodet_preprocess_factory(
        square_edge=args.square_edge,
        augmentation=args.augmentation,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
        rescale_images=args.rescale_images)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    train_data = Coco(
        image_dir=args.coco_train_image_dir,
        ann_file=args.cocodet_train_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images,
        image_filter='annotated',
        category_ids=[],
    )
    if args.duplicate_data:
        train_data = torch.utils.data.ConcatDataset(
            [train_data for _ in range(args.duplicate_data)])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        sampler=torch.utils.data.WeightedRandomSampler(
            train_data.class_aware_sample_weights(), len(train_data), replacement=True),
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    val_data = Coco(
        image_dir=args.coco_val_image_dir,
        ann_file=args.cocodet_val_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images,
        image_filter='annotated',
        category_ids=[],
    )
    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        sampler=torch.utils.data.WeightedRandomSampler(
            val_data.class_aware_sample_weights(), len(val_data), replacement=True),
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader


def train_factory(args, target_transforms):
    if args.dataset in ('cocokp',):
        return train_cocokp_factory(args, target_transforms)
    elif args.dataset in ('cocodet',):
        return train_cocodet_factory(args, target_transforms)
    elif args.dataset in ('freihand',):
        return train_freihand_factory(args, target_transforms)
    elif args.dataset in ('panoptic',):
        return train_panoptic_factory(args, target_transforms)
    elif args.dataset in ('rhd',):
        return train_rhd_factory(args, target_transforms)
    elif args.dataset in ('onehand10k',):
        return train_onehand10k_factory(args, target_transforms)
    elif args.dataset in ('nyu',):
        return train_nyu_factory(args, target_transforms)
    elif args.dataset in ('posedataset',):
        return train_posedataset_factory(args, target_transforms)

    raise Exception('unknown dataset: {}'.format(args.dataset))
