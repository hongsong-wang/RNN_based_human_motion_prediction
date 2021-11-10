import torch
import numpy as np

def kinematic_forward(rotations, root_positions, dataset='h36m'):
    # Todo, check values of offset and parent index
    if dataset=='h36m':
        from viz_h36m import _some_variables
    if dataset=='cmu':
        from viz_cmu import _some_variables
    parent, offset, _, _ = _some_variables()

    offset = torch.FloatTensor(offset).to(rotations.device)

    skt = Skeleton(offset, parent)
    xyz = skt.forward_kinematics(rotations, root_positions)
    return xyz[:,:,:,[0,2,1]]

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

class Skeleton:
    def __init__(self, offsets, parents, joints_left=None, joints_right=None):
        assert len(offsets) == len(parents)

        self._offsets = offsets
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()

    def num_joints(self):
        return self._offsets.shape[0]

    def offsets(self):
        return self._offsets

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def forward_kinematics(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(rotations.shape[0], rotations.shape[1],
                                                self._offsets.shape[0], self._offsets.shape[1])

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[self._parents[i]], expanded_offsets[:, :, i]) \
                                       + positions_world[self._parents[i]])
                if self._has_children[i]:
                    rotations_world.append(qmul(rotations_world[self._parents[i]], rotations[:, :, i]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

def test_visualize():
    from batch_sample import get_batch_srnn
    from human_36m import human_36m_dataset
    dataset = human_36m_dataset('/home/hust/data/Human_3.6M/h3.6m/dataset')
    test_set = dataset.load_data(dataset.get_test_subject_ids())
    # "walking", "eating", "smoking"
    source_tst, target_tst = get_batch_srnn(test_set, "walking", 50, 25, 3 * 32 + 3)

    source_tst = np.concatenate((source_tst, target_tst), axis=1)
    source_tst = torch.FloatTensor(source_tst[:, :, 3:])

    from torch_utils import expmap_to_quaternion
    # rotations = torch.rand(2, 10, 32, 4)
    rotations = source_tst.view(source_tst.shape[0], source_tst.shape[1], source_tst.shape[2] / 3, 3).contiguous()
    rotations = expmap_to_quaternion(rotations)

    root_positions = torch.zeros(rotations.shape[0], rotations.shape[1], 3)
    joints = kinematic_forward(rotations, root_positions, dataset='h36m')

    xyz_gt = joints[0].numpy()

    for kx in [0, 1, 2]:
        xyz_slc = xyz_gt[:,:,kx]
        print(np.amax(xyz_slc), np.amin(xyz_slc), np.mean(xyz_slc) )

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!
    import viz_h36m as viz
    # === Plot and animate ===
    fig = plt.figure()
    # Todo, set axis on, use default elev and azmin when make video
    ax = Axes3D(fig, elev=30, azim=10)
    ax.set_axis_off()
    ob = viz.Ax3DPose(ax)

    # Plot the conditioning ground truth
    for i in range(xyz_gt.shape[0]):
        ob.update(xyz_gt[i, :])
        # ax.quiver()
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.01)

if __name__ == '__main__':
    test_visualize()
    pass