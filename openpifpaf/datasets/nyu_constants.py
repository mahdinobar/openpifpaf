import numpy as np
import scipy.io as scio

NYU_HAND_SKELETON = [
    (30, 29), (29, 28), (28, 27), (27, 26), (26, 25),
    (24, 23), (23, 22), (22, 21), (21, 20), (20, 19),
    (18, 17), (17, 16), (16, 15), (15, 14), (14, 13),
    (12, 11), (11, 10), (10 ,9), (9, 8), (8, 7),
    (6, 5), (5, 4), (4, 3), (3, 2), (2, 1),
    (30, 24), (30, 18), (30, 12), (30, 6),
    (30, 31), (30, 32), (31, 32),
]

NYU_KINEMATIC_TREE_SKELETON = NYU_HAND_SKELETON

NYU_KEYPOINTS_list = list(scio.loadmat('/home/mahdi/HVR/hvr/A2J/data/nyu/train/joint_data.mat')['joint_names'][0,:32])

NYU_KEYPOINTS = [
    str(NYU_KEYPOINTS_list[0][0]),
    str(NYU_KEYPOINTS_list[1][0]),
    str(NYU_KEYPOINTS_list[2][0]),
    str(NYU_KEYPOINTS_list[3][0]),
    str(NYU_KEYPOINTS_list[4][0]),
    str(NYU_KEYPOINTS_list[5][0]),
    str(NYU_KEYPOINTS_list[6][0]),
    str(NYU_KEYPOINTS_list[7][0]),
    str(NYU_KEYPOINTS_list[8][0]),
    str(NYU_KEYPOINTS_list[9][0]),
    str(NYU_KEYPOINTS_list[10][0]),
    str(NYU_KEYPOINTS_list[11][0]),
    str(NYU_KEYPOINTS_list[12][0]),
    str(NYU_KEYPOINTS_list[13][0]),
    str(NYU_KEYPOINTS_list[14][0]),
    str(NYU_KEYPOINTS_list[15][0]),
    str(NYU_KEYPOINTS_list[16][0]),
    str(NYU_KEYPOINTS_list[17][0]),
    str(NYU_KEYPOINTS_list[18][0]),
    str(NYU_KEYPOINTS_list[19][0]),
    str(NYU_KEYPOINTS_list[20][0]),
    str(NYU_KEYPOINTS_list[21][0]),
    str(NYU_KEYPOINTS_list[22][0]),
    str(NYU_KEYPOINTS_list[23][0]),
    str(NYU_KEYPOINTS_list[24][0]),
    str(NYU_KEYPOINTS_list[25][0]),
    str(NYU_KEYPOINTS_list[26][0]),
    str(NYU_KEYPOINTS_list[27][0]),
    str(NYU_KEYPOINTS_list[28][0]),
    str(NYU_KEYPOINTS_list[29][0]),
    str(NYU_KEYPOINTS_list[30][0]),
    str(NYU_KEYPOINTS_list[31][0]),
]

NYU_UPRIGHT_POSE = np.array([
    [-0.5, 1.2, 2.0],
    [-0.5, 1.1, 2.0],
    [-0.5, 1., 2.0],
    [-0.5, .9, 2.0],
    [-0.5, .8, 2.0],
    [-0.5, .7, 2.0],  # little

    [-0.25, 1.2, 2.0],
    [-0.25, 1.1, 2.0],
    [-0.25, 1., 2.0],
    [-0.25, .9, 2.0],
    [-0.25, .8, 2.0],
    [-0.25, .7, 2.0],  # ring

    [0., 1.2, 2.0],
    [0., 1.1, 2.0],
    [0., 1., 2.0],
    [0., .9, 2.0],
    [0., .8, 2.0],
    [0., .7, 2.0],  # middle

    [0.25 ,1.2 , 2.0],
    [0.25 ,1.1 , 2.0],
    [0.25 ,1. , 2.0],
    [0.25 ,.9 , 2.0],
    [0.25, .8, 2.0],
    [0.25, .7, 2.0],  # index

    [0.43, 1.1, 2.0],
    [0.41, .9, 2.0],
    [0.38, .7, 2.0],
    [0.33, .5, 2.0],
    [0.27, .4, 2.0],
    [0.2, .3, 2.0],  # thumb

    [0.2, 0.1, 2.0],
    [0., 0.1, 2.0],
])



NYU_HFLIP = {
    str(NYU_KEYPOINTS_list[0][0]): str(NYU_KEYPOINTS_list[24][0]),
    str(NYU_KEYPOINTS_list[1][0]): str(NYU_KEYPOINTS_list[25][0]),
    str(NYU_KEYPOINTS_list[2][0]): str(NYU_KEYPOINTS_list[26][0]),
    str(NYU_KEYPOINTS_list[3][0]): str(NYU_KEYPOINTS_list[27][0]),
    str(NYU_KEYPOINTS_list[4][0]): str(NYU_KEYPOINTS_list[28][0]),
    str(NYU_KEYPOINTS_list[5][0]): str(NYU_KEYPOINTS_list[29][0]),
    str(NYU_KEYPOINTS_list[6][0]): str(NYU_KEYPOINTS_list[18][0]),
    str(NYU_KEYPOINTS_list[7][0]): str(NYU_KEYPOINTS_list[19][0]),
    str(NYU_KEYPOINTS_list[8][0]): str(NYU_KEYPOINTS_list[20][0]),
    str(NYU_KEYPOINTS_list[9][0]): str(NYU_KEYPOINTS_list[21][0]),
    str(NYU_KEYPOINTS_list[10][0]): str(NYU_KEYPOINTS_list[22][0]),
    str(NYU_KEYPOINTS_list[11][0]): str(NYU_KEYPOINTS_list[23][0]),
    str(NYU_KEYPOINTS_list[12][0]): str(NYU_KEYPOINTS_list[12][0]),
    str(NYU_KEYPOINTS_list[13][0]): str(NYU_KEYPOINTS_list[13][0]),
    str(NYU_KEYPOINTS_list[14][0]): str(NYU_KEYPOINTS_list[14][0]),
    str(NYU_KEYPOINTS_list[15][0]): str(NYU_KEYPOINTS_list[15][0]),
    str(NYU_KEYPOINTS_list[16][0]): str(NYU_KEYPOINTS_list[16][0]),
    str(NYU_KEYPOINTS_list[17][0]): str(NYU_KEYPOINTS_list[17][0]),
    str(NYU_KEYPOINTS_list[30][0]): str(NYU_KEYPOINTS_list[31][0]),
    str(NYU_KEYPOINTS_list[24][0]): str(NYU_KEYPOINTS_list[0][0]),
    str(NYU_KEYPOINTS_list[25][0]): str(NYU_KEYPOINTS_list[1][0]),
    str(NYU_KEYPOINTS_list[26][0]): str(NYU_KEYPOINTS_list[2][0]),
    str(NYU_KEYPOINTS_list[27][0]): str(NYU_KEYPOINTS_list[3][0]),
    str(NYU_KEYPOINTS_list[28][0]): str(NYU_KEYPOINTS_list[4][0]),
    str(NYU_KEYPOINTS_list[29][0]): str(NYU_KEYPOINTS_list[5][0]),
    str(NYU_KEYPOINTS_list[18][0]): str(NYU_KEYPOINTS_list[6][0]),
    str(NYU_KEYPOINTS_list[19][0]): str(NYU_KEYPOINTS_list[7][0]),
    str(NYU_KEYPOINTS_list[20][0]): str(NYU_KEYPOINTS_list[8][0]),
    str(NYU_KEYPOINTS_list[21][0]): str(NYU_KEYPOINTS_list[9][0]),
    str(NYU_KEYPOINTS_list[22][0]): str(NYU_KEYPOINTS_list[10][0]),
    str(NYU_KEYPOINTS_list[23][0]): str(NYU_KEYPOINTS_list[11][0]),
    str(NYU_KEYPOINTS_list[12][0]): str(NYU_KEYPOINTS_list[12][0]),
    str(NYU_KEYPOINTS_list[13][0]): str(NYU_KEYPOINTS_list[13][0]),
    str(NYU_KEYPOINTS_list[14][0]): str(NYU_KEYPOINTS_list[14][0]),
    str(NYU_KEYPOINTS_list[15][0]): str(NYU_KEYPOINTS_list[15][0]),
    str(NYU_KEYPOINTS_list[16][0]): str(NYU_KEYPOINTS_list[16][0]),
    str(NYU_KEYPOINTS_list[17][0]): str(NYU_KEYPOINTS_list[17][0]),
    str(NYU_KEYPOINTS_list[31][0]): str(NYU_KEYPOINTS_list[30][0]),
}


DENSER_NYU_HAND_SKELETON = [
    (30, 29), (29, 28), (28, 27), (27, 26), (26, 25),
    (24, 23), (23, 22), (22, 21), (21, 20), (20, 19),
    (18, 17), (17, 16), (16, 15), (15, 14), (14, 13),
    (12, 11), (11, 10), (10 ,9), (9, 8), (8, 7),
    (6, 5), (5, 4), (4, 3), (3, 2), (2, 1),
    (30, 24), (30, 18), (30, 12), (30, 6),
    (30, 31), (30, 32), (31, 32),
    (24, 18), (18, 12), (12, 6),
]


DENSER_NYU_HAND_CONNECTIONS = [
    c
    for c in DENSER_NYU_HAND_SKELETON
    if c not in NYU_HAND_SKELETON
]


NYU_HAND_SIGMAS = [
    0.03, # 'palm',  # 1
    0.03, # 'thumb_mcp',  # 2
    0.03, # 'thumb_pip',  # 3
    0.03, # 'thumb_dip',  # 4
    0.03, # 'thumb_tip',  # 5
    0.03, # 'index_mcp',  # 6
    0.03, # 'index_pip',  # 7
    0.03, # 'index_dip',  # 8
    0.03, #'index_tip',  # 9
    0.03, #'middle_mcp',  # 10
    0.03, #'middle_pip',  # 11
    0.03, #'middle_dip',  # 12
    0.03, #'middle_tip',  # 13
    0.03, #'ring_mcp',  # 14
    0.03, #'ring_pip',  # 15
    0.03, #'ring_dip',  # 16
    0.03, #'ring_tip',  # 17
    0.03, #'little_mcp',  # 18
    0.03, #'little_pip',  # 19
    0.03, #'little_dip',  # 20
    0.03, #'little_tip', # 21
    0.03, # 22
    0.03, # 23
    0.03,
    0.03,
    0.03,
    0.03,
    0.03,
    0.03,
    0.04,
    0.03,
    0.03,
]


NYU_CATEGORIES = [
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

    ann = Annotation(keypoints=NYU_KEYPOINTS, skeleton=NYU_HAND_SKELETON)
    ann.set(pose, np.array(NYU_HAND_SIGMAS) * scale)
    draw_ann(ann, filename='/home/mahdi/HVR/git_repos/openpifpaf/docs/nyu_references/skeleton_nyu.png', keypoint_painter=keypoint_painter)

    ann = Annotation(keypoints=NYU_KEYPOINTS, skeleton=NYU_KINEMATIC_TREE_SKELETON)
    ann.set(pose, np.array(NYU_HAND_SIGMAS) * scale)
    draw_ann(ann, filename='/home/mahdi/HVR/git_repos/openpifpaf/docs/nyu_references/skeleton_kinematic_tree_nyu.png', keypoint_painter=keypoint_painter)

    ann = Annotation(keypoints=NYU_KEYPOINTS, skeleton=DENSER_NYU_HAND_SKELETON)
    ann.set(pose, np.array(NYU_HAND_SIGMAS) * scale)
    draw_ann(ann, filename='/home/mahdi/HVR/git_repos/openpifpaf/docs/nyu_references/skeleton_dense_nyu.png', keypoint_painter=keypoint_painter)


def print_associations():
    for j1, j2 in NYU_HAND_SKELETON:
        print(NYU_KEYPOINTS[j1 - 1], '-', NYU_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()
    # c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    # rotate = np.array(((c, -s), (s, c)))
    # rotated_pose = np.copy(COCO_DAVINCI_POSE)
    # rotated_pose[:, :2] = np.einsum('ij,kj->ki', rotate, rotated_pose[:, :2])
    draw_skeletons(NYU_UPRIGHT_POSE)
