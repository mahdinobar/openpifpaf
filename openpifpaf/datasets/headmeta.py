from ..network.heads import AssociationMeta, DetectionMeta, IntensityMeta
from .constants import (
    COCO_CATEGORIES,
    COCO_KEYPOINTS,
    COCO_PERSON_SKELETON,
    COCO_PERSON_SIGMAS,
    COCO_UPRIGHT_POSE,
    DENSER_COCO_PERSON_CONNECTIONS,
    KINEMATIC_TREE_SKELETON,
)

from .freihand_constants import (
    FREIHAND_CATEGORIES,
    FREIHAND_KEYPOINTS,
    FREIHAND_HAND_SKELETON,
    FREIHAND_HAND_SIGMAS,
    FREIHAND_UPRIGHT_POSE,
    DENSER_FREIHAND_HAND_CONNECTIONS,
    FREIHAND_KINEMATIC_TREE_SKELETON,
)

from .nyu_constants import (
    NYU_CATEGORIES,
    NYU_KEYPOINTS,
    NYU_HAND_SKELETON,
    NYU_HAND_SIGMAS,
    NYU_UPRIGHT_POSE,
    DENSER_NYU_HAND_CONNECTIONS,
    NYU_KINEMATIC_TREE_SKELETON,
)

from .onehand10k_constants import (
    ONEHAND10K_CATEGORIES,
    ONEHAND10K_KEYPOINTS,
    ONEHAND10K_HAND_SKELETON,
    ONEHAND10K_HAND_SIGMAS,
    ONEHAND10K_UPRIGHT_POSE,
    DENSER_ONEHAND10K_HAND_CONNECTIONS,
    ONEHAND10K_KINEMATIC_TREE_SKELETON,
)

from .rhd_constants import (
    RHD_CATEGORIES,
    RHD_KEYPOINTS,
    RHD_HAND_SKELETON,
    RHD_HAND_SIGMAS,
    RHD_UPRIGHT_POSE,
    DENSER_RHD_HAND_CONNECTIONS,
    RHD_KINEMATIC_TREE_SKELETON,
)

from .posedataset_constants import (
    POSEDATASET_CATEGORIES,
    POSEDATASET_KEYPOINTS,
    POSEDATASET_HAND_SKELETON,
    POSEDATASET_HAND_SIGMAS,
    POSEDATASET_UPRIGHT_POSE,
    DENSER_POSEDATASET_HAND_CONNECTIONS,
    POSEDATASET_KINEMATIC_TREE_SKELETON,
)


def factory(head_names):
    if head_names is None:
        return None
    # # uncomment for coco dataset
    # return [factory_single(hn) for hn in head_names]
    # # uncomment for freihand dataset
    # return [factory_single_freihand(hn) for hn in head_names]
    # # uncomment for rhd dataset
    # return [factory_single_rhd(hn) for hn in head_names]
    # # uncomment for onehand10k dataset
    # return [factory_single_onehand10k(hn) for hn in head_names]
    # # uncomment for nyu dataset
    # return [factory_single_nyu(hn) for hn in head_names]
    # uncomment for posedataset dataset
    return [factory_single_posedataset(hn) for hn in head_names]

def factory_single(head_name):
    if 'cifdet' in head_name:
        return DetectionMeta(head_name, COCO_CATEGORIES)
    if 'pif' in head_name or 'cif' in head_name:
        return IntensityMeta(head_name,
                             COCO_KEYPOINTS,
                             COCO_PERSON_SIGMAS,
                             COCO_UPRIGHT_POSE,
                             COCO_PERSON_SKELETON)
    if 'caf25' in head_name:
        return AssociationMeta(head_name,
                               COCO_KEYPOINTS,
                               COCO_PERSON_SIGMAS,
                               COCO_UPRIGHT_POSE,
                               DENSER_COCO_PERSON_CONNECTIONS,
                               sparse_skeleton=COCO_PERSON_SKELETON,
                               only_in_field_of_view=True)
    if 'caf16' in head_name:
        return AssociationMeta(head_name,
                               COCO_KEYPOINTS,
                               COCO_PERSON_SIGMAS,
                               COCO_UPRIGHT_POSE,
                               KINEMATIC_TREE_SKELETON)
    if head_name == 'caf':
        return AssociationMeta(head_name,
                               COCO_KEYPOINTS,
                               COCO_PERSON_SIGMAS,
                               COCO_UPRIGHT_POSE,
                               COCO_PERSON_SKELETON)
    raise NotImplementedError


def factory_single_freihand(head_name):
    if 'cifdet' in head_name:
        return DetectionMeta(head_name, FREIHAND_CATEGORIES)
    if 'pif' in head_name or 'cif' in head_name:
        return IntensityMeta(head_name,
                             FREIHAND_KEYPOINTS,
                             FREIHAND_HAND_SIGMAS,
                             FREIHAND_UPRIGHT_POSE,
                             FREIHAND_HAND_SKELETON)
    if 'caf25' in head_name:
        return AssociationMeta(head_name,
                               FREIHAND_KEYPOINTS,
                               FREIHAND_HAND_SIGMAS,
                               FREIHAND_UPRIGHT_POSE,
                               DENSER_FREIHAND_HAND_CONNECTIONS,
                               sparse_skeleton=FREIHAND_HAND_SKELETON,
                               only_in_field_of_view=True)
    if 'caf16' in head_name:
        return AssociationMeta(head_name,
                               FREIHAND_KEYPOINTS,
                               FREIHAND_HAND_SIGMAS,
                               FREIHAND_UPRIGHT_POSE,
                               FREIHAND_KINEMATIC_TREE_SKELETON)
    if head_name == 'caf':
        return AssociationMeta(head_name,
                               FREIHAND_KEYPOINTS,
                               FREIHAND_HAND_SIGMAS,
                               FREIHAND_UPRIGHT_POSE,
                               FREIHAND_HAND_SKELETON)
    raise NotImplementedError

def factory_single_posedataset(head_name):
    if 'cifdet' in head_name:
        return DetectionMeta(head_name, POSEDATASET_CATEGORIES)
    if 'pif' in head_name or 'cif' in head_name:
        return IntensityMeta(head_name,
                             POSEDATASET_KEYPOINTS,
                             POSEDATASET_HAND_SIGMAS,
                             POSEDATASET_UPRIGHT_POSE,
                             POSEDATASET_HAND_SKELETON)
    if 'caf25' in head_name:
        return AssociationMeta(head_name,
                               POSEDATASET_KEYPOINTS,
                               POSEDATASET_HAND_SIGMAS,
                               POSEDATASET_UPRIGHT_POSE,
                               DENSER_POSEDATASET_HAND_CONNECTIONS,
                               sparse_skeleton=POSEDATASET_HAND_SKELETON,
                               only_in_field_of_view=True)
    if 'caf16' in head_name:
        return AssociationMeta(head_name,
                               POSEDATASET_KEYPOINTS,
                               POSEDATASET_HAND_SIGMAS,
                               POSEDATASET_UPRIGHT_POSE,
                               POSEDATASET_KINEMATIC_TREE_SKELETON)
    if head_name == 'caf':
        return AssociationMeta(head_name,
                               POSEDATASET_KEYPOINTS,
                               POSEDATASET_HAND_SIGMAS,
                               POSEDATASET_UPRIGHT_POSE,
                               POSEDATASET_HAND_SKELETON)
    raise NotImplementedError


def factory_single_nyu(head_name):
    if 'cifdet' in head_name:
        return DetectionMeta(head_name, NYU_CATEGORIES)
    if 'pif' in head_name or 'cif' in head_name:
        return IntensityMeta(head_name,
                             NYU_KEYPOINTS,
                             NYU_HAND_SIGMAS,
                             NYU_UPRIGHT_POSE,
                             NYU_HAND_SKELETON)
    if 'caf25' in head_name:
        return AssociationMeta(head_name,
                               NYU_KEYPOINTS,
                               NYU_HAND_SIGMAS,
                               NYU_UPRIGHT_POSE,
                               DENSER_NYU_HAND_CONNECTIONS,
                               sparse_skeleton=NYU_HAND_SKELETON,
                               only_in_field_of_view=True)
    if 'caf16' in head_name:
        return AssociationMeta(head_name,
                               NYU_KEYPOINTS,
                               NYU_HAND_SIGMAS,
                               NYU_UPRIGHT_POSE,
                               NYU_KINEMATIC_TREE_SKELETON)
    if head_name == 'caf':
        return AssociationMeta(head_name,
                               NYU_KEYPOINTS,
                               NYU_HAND_SIGMAS,
                               NYU_UPRIGHT_POSE,
                               NYU_HAND_SKELETON)
    raise NotImplementedError


def factory_single_onehand10k(head_name):
    if 'cifdet' in head_name:
        return DetectionMeta(head_name, ONEHAND10K_CATEGORIES)
    if 'pif' in head_name or 'cif' in head_name:
        return IntensityMeta(head_name,
                             ONEHAND10K_KEYPOINTS,
                             ONEHAND10K_HAND_SIGMAS,
                             ONEHAND10K_UPRIGHT_POSE,
                             ONEHAND10K_HAND_SKELETON)
    if 'caf25' in head_name:
        return AssociationMeta(head_name,
                               ONEHAND10K_KEYPOINTS,
                               ONEHAND10K_HAND_SIGMAS,
                               ONEHAND10K_UPRIGHT_POSE,
                               DENSER_ONEHAND10K_HAND_CONNECTIONS,
                               sparse_skeleton=ONEHAND10K_HAND_SKELETON,
                               only_in_field_of_view=True)
    if 'caf16' in head_name:
        return AssociationMeta(head_name,
                               ONEHAND10K_KEYPOINTS,
                               ONEHAND10K_HAND_SIGMAS,
                               ONEHAND10K_UPRIGHT_POSE,
                               ONEHAND10K_KINEMATIC_TREE_SKELETON)
    if head_name == 'caf':
        return AssociationMeta(head_name,
                               ONEHAND10K_KEYPOINTS,
                               ONEHAND10K_HAND_SIGMAS,
                               ONEHAND10K_UPRIGHT_POSE,
                               ONEHAND10K_HAND_SKELETON)
    raise NotImplementedError


def factory_single_rhd(head_name):
    if 'cifdet' in head_name:
        return DetectionMeta(head_name, RHD_CATEGORIES)
    if 'pif' in head_name or 'cif' in head_name:
        return IntensityMeta(head_name,
                             RHD_KEYPOINTS,
                             RHD_HAND_SIGMAS,
                             RHD_UPRIGHT_POSE,
                             RHD_HAND_SKELETON)
    if 'caf25' in head_name:
        return AssociationMeta(head_name,
                               RHD_KEYPOINTS,
                               RHD_HAND_SIGMAS,
                               RHD_UPRIGHT_POSE,
                               DENSER_RHD_HAND_CONNECTIONS,
                               sparse_skeleton=RHD_HAND_SKELETON,
                               only_in_field_of_view=True)
    if 'caf16' in head_name:
        return AssociationMeta(head_name,
                               RHD_KEYPOINTS,
                               RHD_HAND_SIGMAS,
                               RHD_UPRIGHT_POSE,
                               RHD_KINEMATIC_TREE_SKELETON)
    if head_name == 'caf':
        return AssociationMeta(head_name,
                               RHD_KEYPOINTS,
                               RHD_HAND_SIGMAS,
                               RHD_UPRIGHT_POSE,
                               RHD_HAND_SKELETON)
    raise NotImplementedError
