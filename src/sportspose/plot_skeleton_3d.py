import numpy as np
import matplotlib.pyplot as plt

SPORTSPOSE_LINECOLORS = [
    "pink",
    "pink",
    "grey",
    "grey",
    "red",
    "red",
    "red",
    "red",
    "green",
    "blue",
    "green",
    "blue",
    "purple",
    "purple",
    "orange",
    "orange",
    "orange",
    "orange",
]

SPORTSPOSE_KINEMATIC_TREE = np.array(
    [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [5, 3],
        [6, 4],
        [3, 1],
        [4, 2],
        [1, 0],
        [2, 0],
    ]
)

def get_coordinate(tree, pose_3d):
    """Given a tree connection and a skeleton pose
    in 3 dimensions will connect the two points specified in tree
    """
    assert len(tree) == 2
    coordinates = np.zeros((3, 2))
    coordinates[:, 0] = pose_3d[tree[0]]
    coordinates[:, 1] = pose_3d[tree[1]]
    return coordinates


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return None


def plot_skeleton_3d(
    skeleton_3d,
    kinematic_tree=SPORTSPOSE_KINEMATIC_TREE,
    limits=None,
    ax=None,
    order=[1,0,2],
    joint_color="blue",
    kt_color=SPORTSPOSE_LINECOLORS,
    elev=30,
    azim=-150,
    flip_z=True,
):
    """
    Plots a skeleton in a 3d maplotlib plot

    Parameters
    ----------
    skeleton_3d : np.ndarray
        Skeleton positions in (x,y,z) coordinates. Shape should be
        (N,3)
    kinematic_tree : list, optional
        List descriping all the connections in the skeleton.
    limits : list or np.ndarray, optional
        Upper and lower limits to the 3 axes. Shape should be (2,3).
        The default is None for which the limits are default.
        The default is None.
    ax : matplotlib.axes, optional
        matplotlib axis. The default is None.
    order : TYPE, optional
        In which order the (x,y,z) coordinates should be plotted.
        The default is [2,0,1] meaning(z,x,y).
    joint_color : string, optional
        Color of the joint points. The default is "blue".
    kt_color : string, optional
        Color of the joint connections if kinematic_tree is given.
        The default is 'red'.
    elev : scalar, optional
        Elevation angle on the 3d viewer. The default is 26.
    azim : scalar, optional
        Azimut angle on the 3d viewer. The default is 38.
    flip_z : boolean, optional
        Whether to flip the z-axis. Can be done if 3D joints appear mirrored.

    Returns
    -------
    f :
        matplotlib figure
    ax :
        matplotlib axes

    """
    return_fig_and_axis = False
    assert skeleton_3d.ndim == 2 and skeleton_3d.shape[-1] == 3
    if isinstance(limits, np.ndarray):
        assert limits.shape == (2, 3)
    elif isinstance(limits, list):
        assert len(limits) == 2 and len(limits[0]) == 3

    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(projection="3d")
        return_fig_and_axis = True
    # Add limits
    if limits is not None:
        ax.set_xlim3d([limits[0, order[0]], limits[1, order[0]]])
        ax.set_ylim3d([limits[0, order[1]], limits[1, order[1]]])
        ax.set_zlim3d([limits[0, order[2]], limits[1, order[2]]])
    if isinstance(kt_color, str):
        kt_color = len(kinematic_tree) * [kt_color]
    assert len(kt_color) == len(kinematic_tree)
    # If z-axis should be flipped
    if flip_z:
        skeleton_3d[:, order[2]] *= -1
    # Plot joints
    ax.scatter(
        skeleton_3d[:, order[0]],
        skeleton_3d[:, order[1]],
        skeleton_3d[:, order[2]],
        color=joint_color,
    )
    # Plot connections between joints
    if kinematic_tree is not None:
        for idx, tree in enumerate(kinematic_tree):
            coordinates = get_coordinate(tree, skeleton_3d)
            ax.plot(
                coordinates[order[0]],
                coordinates[order[1]],
                coordinates[order[2]],
                c=kt_color[idx],
            )  # ,linewidth=2
    set_axes_equal(ax)
    ax.view_init(elev=elev, azim=azim)

    if return_fig_and_axis:
        return f, ax
    else:
        return None