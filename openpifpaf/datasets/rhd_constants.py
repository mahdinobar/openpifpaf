import numpy as np

RHD_HAND_SKELETON = [
#     left hand
    (1,5), (5,4), (4,3), (3,2),
    (1,9), (9,8), (8,7), (7,6),
    (1,13), (13,12), (12,11), (11,10),
    (1,17), (17,16), (16,15), (15,14),
    (1,21), (21,20), (20,19), (19,18),
#     right hand
    (22,26), (26,25), (25,24), (24,23),
    (22,30), (30, 29), (29, 28), (28,27),
    (22,34), (34,33), (33,32), (32,31),
    (22,38), (38,37), (37,36), (36,35),
    (22,42), (42,41), (41,40), (40,39),
]

RHD_KINEMATIC_TREE_SKELETON = RHD_HAND_SKELETON


RHD_KEYPOINTS = [
    'left_palm',  # 1
    'left_thumb_tip',  # 2
    'left_thumb_dip',  # 3
    'left_thumb_pip',  # 4
    'left_thumb_mcp',  # 5
    'left_index_tip',  # 6
    'left_index_dip',  # 7
    'left_index_pip',  # 8
    'left_index_mcp',  # 9
    'left_middle_tip',  # 10
    'left_middle_dip',  # 11
    'left_middle_pip',  # 12
    'left_middle_mcp',  # 13
    'left_ring_tip',  # 14
    'left_ring_dip',  # 15
    'left_ring_pip',  # 16
    'left_ring_mcp',  # 17
    'left_little_tip',  # 18
    'left_little_dip',  # 19
    'left_little_pip',  # 20
    'left_little_mcp',  # 21

    'right_palm',  # 22
    'right_thumb_tip',  # 23
    'right_thumb_dip',  # 24
    'right_thumb_pip',  # 25
    'right_thumb_mcp',  # 26
    'right_index_tip',  # 27
    'right_index_dip',  # 28
    'right_index_pip',  # 29
    'right_index_mcp',  # 30
    'right_middle_tip',  # 31
    'right_middle_dip',  # 32
    'right_middle_pip',  # 33
    'right_middle_mcp',  # 34
    'right_ring_tip',  # 35
    'right_ring_dip',  # 36
    'right_ring_pip',  # 37
    'right_ring_mcp',  # 38
    'right_little_tip',  # 39
    'right_little_dip',  # 40
    'right_little_pip',  # 41
    'right_little_mcp',  # 42
]


RHD_UPRIGHT_POSE = np.array([
    # left hand
    [0, 0, 2.0],  # 'left_palm',  # 1
    [0.54, 1., 2.0],  # 'left_thumb_tip',  # 2
    [0.5, 0.8, 2.0],  # 'left_thumb_dip',  # 3
    [0.45, .6, 2.0],  # 'left_thumb_pip',  # 4
    [0.33, .4, 2.0],  # 'left_thumb_mcp',  # 5
    [0.25, 1.5, 2.0],  # 'left_index_tip',  # 6
    [0.25, 1.25, 2.0],  # 'left_index_dip',  # 7
    [0.25, 1., 2.0],  # 'left_index_pip',  # 8
    [0.25, .75, 2.0],  # 'left_index_mcp',  # 9
    [0., 1.5, 2.0],  # 'left_middle_tip',  # 10
    [0., 1.25, 2.0],  # 'left_middle_dip',  # 11
    [0., 1., 2.0],  # 'left_middle_pip',  # 12
    [0., .75, 2.0],  # 'left_middle_mcp',  # 13
    [-0.25, 1.5, 2.0],  # 'left_ring_tip',  # 14
    [-0.25, 1.25, 2.0],  # 'left_ring_dip',  # 15
    [-0.25, 1., 2.0],  # 'left_ring_pip',  # 16
    [-0.25, .75, 2.0],  # 'left_ring_mcp',  # 17
    [-0.5, 1.5, 2.0],  # 'left_little_tip',  # 18
    [-0.5, 1.25, 2.0],  # 'left_little_dip',  # 19
    [-0.5, 1., 2.0],  # 'left_little_pip',  # 20
    [-0.5, .75, 2.0],  # 'left_little_mcp',  # 21
    # right hand
    [2+0, 0, 2.0],  # 'right_palm',  # 22
    [2-0.54, 1., 2.0],  # 'right_thumb_tip',  # 23
    [2-0.5, 0.8, 2.0],  # 'right_thumb_dip',  # 24
    [2-0.45, .6, 2.0],  # 'right_thumb_pip',  # 25
    [2-0.33, .4, 2.0],  # 'right_thumb_mcp',  # 26
    [2-0.25, 1.5, 2.0],  # 'right_index_tip',  # 27
    [2-0.25, 1.25, 2.0],  # 'right_index_dip',  # 28
    [2-0.25, 1., 2.0],  # 'right_index_pip',  # 29
    [2-0.25, .75, 2.0],  # 'right_index_mcp',  # 30
    [2+0., 1.5, 2.0],  # 'right_middle_tip',  # 31
    [2+0., 1.25, 2.0],  # 'right_middle_dip',  # 32
    [2+0., 1., 2.0],  # 'right_middle_pip',  # 33
    [2+0., .75, 2.0],  # 'right_middle_mcp',  # 34
    [2+0.25, 1.5, 2.0],  # 'right_ring_tip',  # 35
    [2+0.25, 1.25, 2.0],  # 'right_ring_dip',  # 36
    [2+0.25, 1., 2.0],  # 'right_ring_pip',  # 37
    [2+0.25, .75, 2.0],  # 'right_ring_mcp',  # 38
    [2+0.5, 1.5, 2.0],  # 'right_little_tip',  # 39
    [2+0.5, 1.25, 2.0],  # 'right_little_dip',  # 40
    [2+0.5, 1., 2.0],  # 'right_little_pip',  # 41
    [2+0.5, .75, 2.0],  # 'right_little_mcp',  # 42
])


RHD_HFLIP = {
     'left_thumb_mcp': 'left_little_mcp',
     'left_thumb_pip': 'left_little_pip',
     'left_thumb_dip': 'left_little_dip',
     'left_thumb_tip': 'left_little_tip',
     'left_index_mcp': 'left_ring_mcp',
     'left_index_pip': 'left_ring_pip',
     'left_index_dip': 'left_ring_dip',
     'left_index_tip': 'left_ring_tip',
     'left_middle_mcp': 'left_middle_mcp',
     'left_middle_pip': 'left_middle_pip',
     'left_middle_dip': 'left_middle_dip',
     'left_middle_tip': 'left_middle_tip',
     'left_ring_mcp': 'left_index_mcp',
     'left_ring_pip': 'left_index_pip',
     'left_ring_dip': 'left_index_dip',
     'left_ring_tip': 'left_index_tip',
     'left_little_mcp': 'left_thumb_mcp',
     'left_little_pip': 'left_thumb_pip',
     'left_little_dip': 'left_thumb_dip',
     'left_little_tip': 'left_thumb_tip',
    'right_thumb_mcp': 'right_little_mcp',
    'right_thumb_pip': 'right_little_pip',
    'right_thumb_dip': 'right_little_dip',
    'right_thumb_tip': 'right_little_tip',
    'right_index_mcp': 'right_ring_mcp',
    'right_index_pip': 'right_ring_pip',
    'right_index_dip': 'right_ring_dip',
    'right_index_tip': 'right_ring_tip',
    'right_middle_mcp': 'right_middle_mcp',
    'right_middle_pip': 'right_middle_pip',
    'right_middle_dip': 'right_middle_dip',
    'right_middle_tip': 'right_middle_tip',
    'right_ring_mcp': 'right_index_mcp',
    'right_ring_pip': 'right_index_pip',
    'right_ring_dip': 'right_index_dip',
    'right_ring_tip': 'right_index_tip',
    'right_little_mcp': 'right_thumb_mcp',
    'right_little_pip': 'right_thumb_pip',
    'right_little_dip': 'right_thumb_dip',
    'right_little_tip': 'right_thumb_tip',
}


DENSER_RHD_HAND_SKELETON = [
    #     left hand
    (1, 5), (5, 4), (4, 3), (3, 2),
    (1, 9), (9, 8), (8, 7), (7, 6), (7,13),
    (1, 13), (13, 12), (12, 11), (11, 10), (13,17),
    (1, 17), (17, 16), (16, 15), (15, 14), (17,21),
    (1, 21), (21, 20), (20, 19), (19, 18),
    #     right hand
    (22, 26), (26, 25), (25, 24), (24, 23),
    (22, 30), (30, 29), (29, 28), (28, 27), (30,34),
    (22, 34), (34, 33), (33, 32), (32, 31), (34,38),
    (22, 38), (38, 37), (37, 36), (36, 35), (38,42),
    (22, 42), (42, 41), (41, 40), (40, 39),
]


DENSER_RHD_HAND_CONNECTIONS = [
    c
    for c in DENSER_RHD_HAND_SKELETON
    if c not in RHD_HAND_SKELETON
]


RHD_HAND_SIGMAS = [
    0.10, # 'left_palm',  # 1
    0.05, # 'left_thumb_mcp',  # 2
    0.05, # 'left_thumb_pip',  # 3
    0.05, # 'left_thumb_dip',  # 4
    0.05, # 'left_thumb_tip',  # 5
    0.05, # 'left_index_mcp',  # 6
    0.05, # 'left_index_pip',  # 7
    0.05, # 'left_index_dip',  # 8
    0.05, # 'left_index_tip',  # 9
    0.05, # 'left_middle_mcp',  # 10
    0.05, # 'left_middle_pip',  # 11
    0.05, # 'left_middle_dip',  # 12
    0.05, # 'left_middle_tip',  # 13
    0.05, # 'left_ring_mcp',  # 14
    0.05, # 'left_ring_pip',  # 15
    0.05, # 'left_ring_dip',  # 16
    0.05, # 'left_ring_tip',  # 17
    0.05, # 'left_little_mcp',  # 18
    0.05, # 'left_little_pip',  # 19
    0.05, # 'left_little_dip',  # 20
    0.05, # 'left_little_tip', # 21
    0.10,  # 'right_palm',  # 22
    0.05,  # 'right_thumb_mcp',  # 23
    0.05,  # 'right_thumb_pip',  # 24
    0.05,  # 'right_thumb_dip',  # 25
    0.05,  # 'right_thumb_tip',  # 26
    0.05,  # 'right_index_mcp',  # 27
    0.05,  # 'right_index_pip',  # 28
    0.05,  # 'right_index_dip',  # 29
    0.05,  # 'right_index_tip',  # 30
    0.05,  # 'right_middle_mcp',  # 31
    0.05,  # 'right_middle_pip',  # 32
    0.05,  # 'right_middle_dip',  # 33
    0.05,  # 'right_middle_tip',  # 34
    0.05,  # 'right_ring_mcp',  # 35
    0.05,  # 'right_ring_pip',  # 36
    0.05,  # 'right_ring_dip',  # 37
    0.05,  # 'right_ring_tip',  # 38
    0.05,  # 'right_little_mcp',  # 39
    0.05,  # 'right_little_pip',  # 40
    0.05,  # 'right_little_dip',  # 41
    0.05,  # 'right_little_tip', # 42
]


RHD_CATEGORIES = [
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

    ann = Annotation(keypoints=RHD_KEYPOINTS, skeleton=RHD_HAND_SKELETON)
    ann.set(pose, np.array(RHD_HAND_SIGMAS) * scale)
    draw_ann(ann, filename='/home/mahdi/HVR/git_repos/openpifpaf/docs/rhd_references/skeleton_rhd.png', keypoint_painter=keypoint_painter)

    ann = Annotation(keypoints=RHD_KEYPOINTS, skeleton=RHD_KINEMATIC_TREE_SKELETON)
    ann.set(pose, np.array(RHD_HAND_SIGMAS) * scale)
    draw_ann(ann, filename='/home/mahdi/HVR/git_repos/openpifpaf/docs/rhd_references/skeleton_kinematic_tree_rhd.png', keypoint_painter=keypoint_painter)

    ann = Annotation(keypoints=RHD_KEYPOINTS, skeleton=DENSER_RHD_HAND_SKELETON)
    ann.set(pose, np.array(RHD_HAND_SIGMAS) * scale)
    draw_ann(ann, filename='/home/mahdi/HVR/git_repos/openpifpaf/docs/rhd_references/skeleton_dense_rhd.png', keypoint_painter=keypoint_painter)


def print_associations():
    for j1, j2 in RHD_HAND_SKELETON:
        print(RHD_KEYPOINTS[j1 - 1], '-', RHD_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()

    # c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    # rotate = np.array(((c, -s), (s, c)))
    # rotated_pose = np.copy(COCO_DAVINCI_POSE)
    # rotated_pose[:, :2] = np.einsum('ij,kj->ki', rotate, rotated_pose[:, :2])
    draw_skeletons(RHD_UPRIGHT_POSE)
