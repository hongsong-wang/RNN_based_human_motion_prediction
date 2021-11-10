import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math

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

def expmap2rotmat(e):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    Args
      r: nx3 exponential map
    Returns
      R: nx3x3 rotation matrix
    """
    assert(len(e.size())==2)
    assert e.size(-1) == 3
    theta = torch.norm(e, p=2, dim=-1, keepdim=True).unsqueeze(dim=-1)
    r0 = F.normalize(e, p=2, dim=-1)
    r0x = torch.zeros(e.size(0), 3, 3).to(e.device)
    r0x[:, 0, 1] = -r0[:, 2]
    r0x[:, 0, 2] = r0[:, 1]
    r0x[:, 1, 2] = -r0[:, 0]
    r0x[:, 1, 0] = r0[:, 2]
    r0x[:, 2, 0] = -r0[:, 1]
    r0x[:, 2, 1] = r0[:, 0]
    R = torch.eye(3).unsqueeze(dim=0).to(e.device) + torch.sin(theta)*r0x + (1 - torch.cos(theta))*torch.bmm(r0x, r0x)
    return R

def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.size(-1) == 3
    theta = torch.norm(e, p=2, dim=-1, keepdim=True)
    w = torch.cos(0.5 * theta)
    xyz = torch.sin(0.5 * theta)/(theta + 1e-6) * e
    q = torch.cat([w, xyz], dim=-1)
    return q

def quaternion_to_expmap(q):
    """
      Converts an exponential map angle to a rotation matrix
      Matlab port to python for evaluation purposes
      I believe this is also called Rodrigues' formula
      https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
    q is (*, 4)
    return (*, 3)
    examples:
        e = torch.rand(1, 3, 3)
        q = expmap_to_quaternion(e)
        e2 = quaternion_to_expmap(q)
    """
    sinhalftheta = torch.index_select(q, dim=-1, index=torch.tensor([1,2,3]).to(q.device))
    coshalftheta = torch.index_select(q, dim=-1, index=torch.tensor([0]).to(q.device))

    norm_sin = torch.norm(sinhalftheta, p=2, dim=-1, keepdim=True)
    r0 = torch.div(sinhalftheta, norm_sin)

    theta = 2 * torch.atan2(norm_sin, coshalftheta)
    theta = torch.fmod(theta + 2 * np.pi, 2 * np.pi)

    theta = torch.where(theta > np.pi, 2 * np.pi - theta, theta)
    r0 = torch.where(theta > np.pi, -r0, r0)
    r = r0 * theta
    return r

def qeuler(q, order='zyx', epsilon=1e-6):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    q0 = torch.index_select(q, dim=-1, index=torch.tensor([0]).to(q.device) )
    q1 = torch.index_select(q, dim=-1, index=torch.tensor([1]).to(q.device))
    q2 = torch.index_select(q, dim=-1, index=torch.tensor([2]).to(q.device))
    q3 = torch.index_select(q, dim=-1, index=torch.tensor([3]).to(q.device))

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise('not defined')

    return torch.cat([x, y, z], dim=-1)

def expmap2rotmat_new(data):
    # data is (*, 3)
    # output is (n, 3, 3)
    theta = torch.norm(data, p=2, dim=-1, keepdim=True).unsqueeze(dim=-1)
    k = F.normalize(data, p=2, dim=-1)
    kx = torch.index_select(k, dim=-1, index=torch.tensor([0]).to(data.device))
    ky = torch.index_select(k, dim=-1, index=torch.tensor([1]).to(data.device))
    kz = torch.index_select(k, dim=-1, index=torch.tensor([2]).to(data.device))
    z0 = Variable(torch.zeros(kx.shape).to(data.device), requires_grad=False)
    km = torch.cat([z0, -1*kz, ky,   kz,z0,-1*kx,   -1*ky,kx, z0], dim=-1)
    km = km.view(-1, 3, 3)
    rm = torch.eye(3).unsqueeze(dim=0).to(data.device) + torch.sin(theta) * km + (1 - torch.cos(theta)) * torch.bmm(km, km)
    return rm

def tensor_expmap_to_euler(data):
    # data is (*, feature_dim), feature_dim is multiple of 3
    ori_shp = data.size()
    eul = qeuler(expmap_to_quaternion(data.contiguous().view(-1, 3)) )
    return eul.view(ori_shp)

def tensor_expmap_to_quaternion(data, channel_first=False):
    ori_shp = list(data.shape)
    new_shp = ori_shp[0:-1] + [ori_shp[-1] / 3 * 4]
    qu = expmap_to_quaternion(data.contiguous().view(-1, 3))

    if channel_first:
        ori_shp = ori_shp[0:-1] + [ori_shp[-1] / 3, 4]
        qu = qu.view(ori_shp).transpose(-1, -2).contiguous()
    return qu.view(new_shp)

def tensor_split_quaternion(data):
    # split 4 quaternion values from tensor from last dimension
    ori_shp = list(data.shape)
    assert(ori_shp[-1] % 4 == 0)
    ori_shp = ori_shp[0:-1] + [4, ori_shp[-1] / 4]
    return data.view(ori_shp).transpose(-1,-2).contiguous()

def tensor_quaternion_to_euler(data, channel_first=False):
    ori_shp = list(data.shape)
    if channel_first:
        eul = qeuler( tensor_split_quaternion(data) )
    else:
        eul = qeuler(data.view(-1, 4))

    ori_shp[-1] = ori_shp[-1] / 4 * 3
    return eul.view(ori_shp)

def get_train_quatern_to_euler_loss(output, target):
    # data is (time, batch, dim)
    angle_distance = torch.remainder(tensor_quaternion_to_euler(output) - tensor_quaternion_to_euler(target) + np.pi, 2 * np.pi) - np.pi
    return torch.mean(torch.sqrt(torch.sum(torch.pow(angle_distance, 2), -1)) )

def get_train_expmap_to_quaternion_loss(output, target):
    # data is (time, batch, dim)
    out = tensor_expmap_to_quaternion(output)
    trg = tensor_expmap_to_quaternion(target)
    # Todo, it seems that abs better than pow2/sqrt for quaternion
    # torch.mean(torch.sqrt(torch.sum(torch.pow(distance, 2), -1)) )
    if 0:
        distance = out - trg
    else:
        out = out.view(out.size(0), out.size(1), out.size(2)/4, 4)
        trg = trg.view(trg.size(0), trg.size(1), trg.size(2)/4, 4)
        # distance = 1 - torch.pow(torch.sum(out*trg, dim=-1), 2)

        if 1:
            dis1 = torch.norm(torch.abs(out - trg), p=1, dim=-1)
            dis2 = torch.norm(torch.abs(out + trg), p=1, dim=-1)
            distance = torch.where(dis1 > dis2, dis2, dis1)
        else:
            eps = 1e-5
            distance = torch.sum(out*trg, dim=-1).clamp(-1.0 + eps, 1.0 - eps)
            theta = torch.acos(distance)
            distance = torch.where(2*theta > math.pi, math.pi - theta, theta)

    return torch.mean(torch.abs(distance))

def get_train_expmap_to_euler_loss(output, target):
    # data is (time, batch, dim)
    angle_distance = torch.remainder(tensor_expmap_to_euler(output) - tensor_expmap_to_euler(target) + np.pi, 2 * np.pi) - np.pi
    return torch.mean(torch.abs(angle_distance))
    # return torch.mean(torch.sqrt(torch.sum(torch.pow(angle_distance, 2), -1)) )

def get_train_expmap_to_rotmat_loss(output, target):
    # data is (time, batch, dim)
    rm1 = expmap2rotmat(output.view(-1, 3))
    rm2 = expmap2rotmat(target.view(-1, 3))
    # for ablation study
    # return nn.SmoothL1Loss()(rm1, rm2)

    # cannot add the second term or not, skew-symmetric_matrix
    dm = torch.bmm(rm1, rm2.permute(0, 2, 1))
    if 0:
        # torch.bmm(rm2, rm1.permute(0, 2, 1))
        ds = dm - torch.eye(3).unsqueeze(dim=0).to(output.device)
        dd = dm[:,0,0]*(dm[:,1,1]*dm[:,2,2]-dm[:,1,2]*dm[:,2,1]) - dm[:,0,1]*(dm[:,1,0]*dm[:,2,2]-dm[:,2,0]*dm[:,1,2]) \
            + dm[:,0,2]*(dm[:,1,0]*dm[:,2,1] - dm[:,1,1]*dm[:,2,0])
        # print(torch.mean(torch.abs(ds)), torch.mean(torch.abs(dd - 1)) )
        # Todo, the second term is 0 always
        return torch.mean(torch.abs(ds)) # + torch.mean(torch.abs(dd - 1))
    # return F.mse_loss(dm, torch.eye(3).unsqueeze(dim=0).to(output.device) )
    else:
        if 0:
            return torch.mean(torch.abs(rm1 - rm2))
        else:
            if 1:
                # implement geometry loss on 20210914, Adversarial geometry-aware human motion prediction
                dm1 = (dm - dm.permute(0, 2, 1))/2
                a = torch.stack([dm1[:,2,1], dm1[:,0,2], dm1[:,1,0]], dim=1).float()
                return torch.mean(torch.norm(a, dim=1))
                an = torch.norm(a, dim=1).float()
                b = torch.asin(an)/(an + 1e-4)
                c = torch.where(an < 1e-4, torch.ones(b.shape, device=b.device), b)

                # print(torch.mean(torch.norm(a * c.unsqueeze(dim=-1), dim=1)) )
                # if torch.any(torch.isnan(torch.mean(torch.norm(a * c.unsqueeze(dim=-1), dim=1)) ) ):
                #     print(output)
                #     import pdb
                #     pdb.set_trace()
                return torch.mean(torch.norm(a * c.unsqueeze(dim=-1), dim=1))
            else:
                dm1 = dm - dm.permute(0, 2, 1)
                theta = torch.acos((dm[:,0,0] + dm[:,1,1] + dm[:,2,2] - 1)/2)
                theta = theta.unsqueeze(dim=-1).unsqueeze(dim=-1)
                logm = theta/(torch.sin(theta)*2 + 1e-6)*dm1
                return torch.mean(torch.abs(logm))

def get_train_expmap_to_quaternion_kinematic_loss(output, target, db_name, w_kn=0.001):
    # data is (time, batch, dim)
    out = tensor_expmap_to_quaternion(output)
    trg = tensor_expmap_to_quaternion(target)
    distance = out - trg
    loss1 = torch.mean(torch.abs(distance))

    from kinematic_fwd import kinematic_forward
    rot_out = out.view(out.shape[0], out.shape[1], out.shape[2]/4, 4)
    rot_trg = trg.view(trg.shape[0], trg.shape[1], trg.shape[2]/4, 4)
    rot_out = rot_out.transpose(0, 1).contiguous()
    rot_trg = rot_trg.transpose(0, 1).contiguous()

    root_positions = torch.zeros(rot_out.shape[0], rot_out.shape[1], 3).to(output.device)
    pos_out = kinematic_forward(rot_out, root_positions, dataset=db_name)
    pos_trg = kinematic_forward(rot_trg, root_positions, dataset=db_name)
    loss2 = torch.mean(torch.abs(pos_out - pos_trg))
    return loss1 + loss2*w_kn

def get_expmap_to_euler_error_time(output, target):
    angle_distance = torch.remainder(tensor_expmap_to_euler(output) - tensor_expmap_to_euler(target) + np.pi, 2 * np.pi) - np.pi
    # along batch size
    # return torch.mean(torch.pow(angle_distance, 2), 1)
    return torch.mean(torch.abs(angle_distance), 1)

def rand_rotate_expmap(source, target):
    # (batch, time, dim)
    joint_dim = source.shape[2]
    batch = source.shape[0]
    src_len = source.shape[1]
    tgt_len = target.shape[1]
    ep = torch.rand(batch, 3).to(source.device) - 0.5
    R = expmap2rotmat(ep)
    R = R.double()
    source_tf = torch.bmm(source.view(batch, -1, 3), R)
    source_tf = source_tf.view(batch, src_len, joint_dim)
    target_tf = torch.bmm(target.view(batch, -1, 3), R)
    target_tf = target_tf.view(batch, tgt_len, joint_dim)
    return source_tf, target_tf


if __name__ == '__main__':
    e = torch.rand(1, 5, 3)
    # q = expmap_to_quaternion(e)
    # e2 = quaternion_to_expmap(q)
    # u = qeuler(q)
    #
    # R = expmap2rotmat(e.view(-1, 3))
    # print(R.shape)
    # print(torch.det(R[0]), torch.det(R[1]), torch.det(R[2]))
    #
    # data = expmap2rotmat_new(e.view(-1, 3))

    print(get_train_expmap_to_rotmat_loss(e, torch.rand(1, 5, 3)) )