import torch
import torch.nn as nn
import torch.nn.functional as F
import quaternion_rnn
from quaternion_linear import QuaternionLinearAutograd
import torch_utils

class Encoder_Decoder_Quaternion(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_unit, residual=False, cuda=False):
        super(Encoder_Decoder_Quaternion, self).__init__()
        self.input_size = input_size
        self.residual = residual
        self.linear = QuaternionLinearAutograd(hidden_size, input_size)

        if rnn_unit == 'lstm':
            self.rnn = quaternion_rnn.QLSTM(input_size, hidden_size)
        else:
            self.rnn = quaternion_rnn.QRNN(input_size, hidden_size)

    def forward_seq(self, input, hidden=None):
        pred_pre = input[:, :, 0:self.input_size].clone()
        output, hidden_state = self.rnn(input, hidden=hidden)
        pred = self.linear(output)

        pred_norm = F.normalize(torch_utils.tensor_split_quaternion(pred), p=2, dim=-1)
        pred_norm = pred_norm.transpose(-1, -2).contiguous().view(pred.shape)

        if self.residual:
            # check norm of quaternion dimension is 1
            # print(torch.norm(torch_utils.tensor_split_quaternion(pred_pre), p=2, dim=-1))
            pred = torch_utils.qmul(torch_utils.tensor_split_quaternion(pred_norm), \
                                    torch_utils.tensor_split_quaternion(pred_pre) )
            # notice: multiples of two unit quaterion is also a unit quaterion, do not need to norm again
            pred_norm = pred.transpose(-1, -2).contiguous().view(pred_pre.shape)

        return pred_norm, hidden_state

    def forward(self, input, target):
        input_en = input
        outputs_enc, hidden_state_en = self.forward_seq(input_en)

        outputs_dec = torch.zeros(target.size(0) - 1, target.size(1), target.size(2)).to(input.device)
        for i in range(len(target) - 1):
            inp_cur = target[i][None] if i == 0 else pred
            pred, hidden_state = self.forward_seq(inp_cur, hidden=(hidden_state_en if i == 0 else hidden_state))
            outputs_dec[i:i + 1] = pred
        return outputs_enc, outputs_dec