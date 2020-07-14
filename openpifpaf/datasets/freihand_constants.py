import numpy as np

FREIHAND_HAND_SKELETON = [
    (1, 2), (2, 3), (3, 4), (4, 5),
    (1, 6), (6, 7), (7, 8), (8, 9),
    (1, 10), (10, 11), (11, 12), (12, 13),
    (1, 14), (14, 15), (15, 16), (16, 17),
    (1, 18), (18, 19), (19, 20), (20, 21),
]

FREIHAND_KINEMATIC_TREE_SKELETON = FREIHAND_HAND_SKELETON


FREIHAND_KEYPOINTS = [
    'palm',  # 1
    'thumb_mcp',  # 2
    'thumb_pip',  # 3
    'thumb_dip',  # 4
    'thumb_tip',  # 5
    'index_mcp',  # 6
    'index_pip',  # 7
    'index_dip',  # 8
    'index_tip',  # 9
    'middle_mcp',  # 10
    'middle_pip',  # 11
    'middle_dip',  # 12
    'middle_tip',  # 13
    'ring_mcp',  # 14
    'ring_pip',  # 15
    'ring_dip',  # 16
    'ring_tip',  # 17
    'little_mcp',  # 18
    'little_pip',  # 19
    'little_dip',  # 20
    'little_tip',  # 21
]


FREIHAND_UPRIGHT_POSE = np.array([
    [0 ,0 , 2.0], # 'palm',  # 1
    [0.33 ,0.4 , 2.0], #thumb_mcp',  # 2
    [0.45 ,0.6 , 2.0], #thumb_pip',  # 3
    [0.5 ,.8 , 2.0], #thumb_dip',  # 4
    [0.54 ,1. , 2.0], #thumb_tip',  # 5
    [0.25 ,0.75 , 2.0], #index_mcp',  # 6
    [0.25 ,1. , 2.0], #index_pip',  # 7
    [0.25 ,1.25 , 2.0], #index_dip',  # 8
    [0.25 ,1.5 , 2.0], #index_tip',  # 9
    [0. ,0.75 , 2.0], #middle_mcp',  # 10
    [0. ,1. , 2.0], #middle_pip',  # 11
    [0. ,1.25 , 2.0], #middle_dip',  # 12
    [0. ,1.5 , 2.0], #middle_tip',  # 13
    [-0.25 ,0.75 , 2.0], #ring_mcp',  # 14
    [-0.25 ,1. , 2.0], #ring_pip',  # 15
    [-0.25 ,1.25 , 2.0], #ring_dip',  # 16
    [-0.25 ,1.5 , 2.0], #ring_tip',  # 17
    [-0.5 ,0.75 , 2.0], #little_mcp',  # 18
    [-0.5 ,1. , 2.0], #little_pip',  # 19
    [-0.5 ,1.25 , 2.0], #little_dip',  # 20
    [-0.5 ,1.5 , 2.0], #little_tip',  # 21
])


HFLIP = {
    'left_eye': 'right_eye',
    'right_eye': 'left_eye',
    'left_ear': 'right_ear',
    'right_ear': 'left_ear',
    'left_shoulder': 'right_shoulder',
    'right_shoulder': 'left_shoulder',
    'left_elbow': 'right_elbow',
    'right_elbow': 'left_elbow',
    'left_wrist': 'right_wrist',
    'right_wrist': 'left_wrist',
    'left_hip': 'right_hip',
    'right_hip': 'left_hip',
    'left_knee': 'right_knee',
    'right_knee': 'left_knee',
    'left_ankle': 'right_ankle',
    'right_ankle': 'left_ankle',
}


DENSER_FREIHAND_HAND_SKELETON = [
    (1, 2), (2, 3), (3, 4), (4, 5),
    (1, 6), (6, 7), (7, 8), (8, 9),
    (1, 10), (10, 11), (11, 12), (12, 13),
    (1, 14), (14, 15), (15, 16), (16, 17),
    (1, 18), (18, 19), (19, 20), (20, 21),
    (6, 10), (10, 14), (14, 18),
]


DENSER_FREIHAND_HAND_CONNECTIONS = [
    c
    for c in DENSER_FREIHAND_HAND_SKELETON
    if c not in FREIHAND_HAND_SKELETON
]


FREIHAND_HAND_SIGMAS = [
    0.10, # 'palm',  # 1
    0.05, # 'thumb_mcp',  # 2
    0.05, # 'thumb_pip',  # 3
    0.05, # 'thumb_dip',  # 4
    0.05, # 'thumb_tip',  # 5
    0.05, # 'index_mcp',  # 6
    0.05, # 'index_pip',  # 7
    0.05, # 'index_dip',  # 8
    0.05, #'index_tip',  # 9
    0.05, #'middle_mcp',  # 10
    0.05, #'middle_pip',  # 11
    0.05, #'middle_dip',  # 12
    0.05, #'middle_tip',  # 13
    0.05, #'ring_mcp',  # 14
    0.05, #'ring_pip',  # 15
    0.05, #'ring_dip',  # 16
    0.05, #'ring_tip',  # 17
    0.05, #'little_mcp',  # 18
    0.05, #'little_pip',  # 19
    0.05, #'little_dip',  # 20
    0.05, #'little_tip', # 21
]


FREIHAND_CATEGORIES = [
    'hand',
]


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from .. import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose):
    from ..annotation import Annotation  # pylint: disable=import-outside-toplevel
    from .. import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter(color_connections=True, linewidth=6)

    ann = Annotation(keypoints=FREIHAND_KEYPOINTS, skeleton=FREIHAND_HAND_SKELETON)
    ann.set(pose, np.array(FREIHAND_HAND_SIGMAS) * scale)
    draw_ann(ann, filename='/home/mahdi/HVR/git_repos/openpifpaf/docs/freihand_references/skeleton_freihand.png', keypoint_painter=keypoint_painter)

    ann = Annotation(keypoints=FREIHAND_KEYPOINTS, skeleton=FREIHAND_KINEMATIC_TREE_SKELETON)
    ann.set(pose, np.array(FREIHAND_HAND_SIGMAS) * scale)
    draw_ann(ann, filename='/home/mahdi/HVR/git_repos/openpifpaf/docs/freihand_references/skeleton_kinematic_tree_freihand.png', keypoint_painter=keypoint_painter)

    ann = Annotation(keypoints=FREIHAND_KEYPOINTS, skeleton=DENSER_FREIHAND_HAND_SKELETON)
    ann.set(pose, np.array(FREIHAND_HAND_SIGMAS) * scale)
    draw_ann(ann, filename='/home/mahdi/HVR/git_repos/openpifpaf/docs/freihand_references/skeleton_dense_freihand.png', keypoint_painter=keypoint_painter)


def print_associations():
    for j1, j2 in FREIHAND_HAND_SKELETON:
        print(FREIHAND_KEYPOINTS[j1 - 1], '-', FREIHAND_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()

    # c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    # rotate = np.array(((c, -s), (s, c)))
    # rotated_pose = np.copy(COCO_DAVINCI_POSE)
    # rotated_pose[:, :2] = np.einsum('ij,kj->ki', rotate, rotated_pose[:, :2])
    draw_skeletons(FREIHAND_UPRIGHT_POSE)
