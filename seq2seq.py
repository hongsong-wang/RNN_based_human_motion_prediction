import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.optim as optim

from enc_dec import Encoder_Decoder
import torch_utils

class EncodeDecodeModel(object):
    def __init__(self, input_size, hidden_size, num_layer, rnn_unit, out_dropout, std_mask, learning_rate, step_size, gamma,
                 residual=False, cuda=False, veloc=False, pos_sum=False, loss_type=0, veloc_only=False, quat_rnn=False,
                 pos_embed=False, pos_embed_dim=96, accel=False, trj_embed=False, trj_embed_dim=32,
                 db_name='h36m', weight_kn=0):
        super(EncodeDecodeModel, self).__init__()
        self.epoch = 0
        self.quat_rnn = quat_rnn
        if quat_rnn:
            from enc_dec_quat import Encoder_Decoder_Quaternion
            self.model = Encoder_Decoder_Quaternion(input_size, hidden_size, rnn_unit, residual=residual, cuda=cuda)
        else:
            if veloc_only:
                from enc_dec_vel import Encoder_Decoder_Velocity
                self.model = Encoder_Decoder_Velocity(input_size, hidden_size, num_layer, rnn_unit, residual=residual, out_dropout=out_dropout,
                                             std_mask=std_mask, pos_sum=pos_sum, pos_embed=pos_embed, pos_embed_dim=pos_embed_dim, cuda=cuda,
                                                      accel=accel, trj_embed=trj_embed, trj_embed_dim=trj_embed_dim)
            else:
                self.model = Encoder_Decoder(input_size, hidden_size, num_layer, rnn_unit, residual=residual, out_dropout=out_dropout,
                                         std_mask=std_mask, veloc=veloc, pos_sum=pos_sum, pos_embed=pos_embed, pos_embed_dim=pos_embed_dim,
                                             cuda=cuda, trj_embed=trj_embed, trj_embed_dim=trj_embed_dim)
        self.loss_type = loss_type
        self.db_name = db_name
        self.weight_kn = weight_kn
        # self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()
        if cuda:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, step_size=step_size, gamma=gamma)

    def scheduler_step(self):
        self.epoch = self.epoch + 1
        self.model_scheduler.step()

    def get_loss(self, outputs, target):
        if self.loss_type == 0:
            total_loss = self.loss(outputs, target[1:])
            return total_loss
        if self.loss_type == 1:
            total_loss = torch_utils.get_train_expmap_to_quaternion_loss(outputs, target[1:])
            return total_loss
        # combine euler loss and quat loss give no improvement
        if self.loss_type == 2:
            total_loss = torch_utils.get_train_expmap_to_euler_loss(outputs, target[1:])
            return total_loss
        if self.loss_type == 3:
            total_loss = torch_utils.get_train_expmap_to_rotmat_loss(outputs, target[1:] )
            return total_loss
        if self.loss_type == 5:
            total_loss = torch_utils.get_train_expmap_to_quaternion_kinematic_loss(outputs, target[1:], self.db_name, w_kn=self.weight_kn)
            return total_loss

    def train(self, input, target, trj_seq):
        if self.quat_rnn:
            input = torch_utils.tensor_expmap_to_quaternion(input, channel_first=True)
            target = torch_utils.tensor_expmap_to_quaternion(target, channel_first=True)

        self.model.train()
        outputs_enc, outputs = self.model(input, target, trj_seq)
        total_loss = self.get_loss(outputs, target)

        gradient_clip = 0.1
        nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

        self.model_optimizer.zero_grad()
        total_loss.backward()
        self.model_optimizer.step()

        return total_loss.item()

    def eval(self, input, target, trj_seq_tst):
        if self.quat_rnn:
            input = torch_utils.tensor_expmap_to_quaternion(input, channel_first=True)
            target = torch_utils.tensor_expmap_to_quaternion(target, channel_first=True)

        self.model.eval()
        outputs_enc, outputs = self.model(input, target, trj_seq_tst)
        return outputs

if __name__ == '__main__':
    pass
