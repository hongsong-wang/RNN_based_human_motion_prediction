# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from quaternion_linear import QuaternionLinearAutograd

class QRNN(nn.Module):
    def __init__(self, feat_size, hidden_size):
        super(QRNN, self).__init__()

        # Reading options:
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.num_classes = feat_size

        # List initialization (Not used, but could be if multiple layers)
        self.wx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.uh = QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim)

    def forward(self, x, hidden=None):
        output = torch.zeros(x.shape[0], x.shape[1], self.hidden_dim ).to(x.device)
        if hidden is None:
            h_init = Variable(torch.zeros(x.shape[1], self.hidden_dim).to(x.device))
        else:
            h_init = hidden

        # Compute W * X in parallel
        wx_out = self.wx(x)
        h = h_init
        # Navigate trough timesteps
        for k in range(x.shape[0]):
            at = wx_out[k] + self.uh(h)
            h = at
            output[k] = h

        return output, h


class QLSTM(nn.Module):
    def __init__(self, feat_size, hidden_size):
        super(QLSTM, self).__init__()

        # Reading options:
        self.act = nn.Tanh()
        self.act_gate = nn.Sigmoid()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size

        # +1 because feat_size = the number on the sequence, and the output one hot will also have
        # a blank dimension so FEAT_SIZE + 1 BLANK
        self.num_classes = feat_size + 1

        # Gates initialization
        self.wfx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)  # Forget
        self.ufh = QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim, bias=False)  # Forget

        self.wix = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)  # Input
        self.uih = QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim, bias=False)  # Input

        self.wox = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)  # Output
        self.uoh = QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim, bias=False)  # Output

        self.wcx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)  # Cell
        self.uch = QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim, bias=False)  # Cell

    def forward(self, x, hidden=None):
        output = torch.zeros(x.shape[0], x.shape[1], self.hidden_dim).to(x.device)
        if hidden is None:
            h_init = Variable(torch.zeros(x.shape[1], self.hidden_dim).to(x.device))
        else:
            h_init = hidden

        # Feed-forward affine transformation (done in parallel)
        wfx_out = self.wfx(x)
        wix_out = self.wix(x)
        wox_out = self.wox(x)
        wcx_out = self.wcx(x)

        # Processing time steps
        c = h_init
        h = h_init

        for k in range(x.shape[0]):
            ft = self.act_gate(wfx_out[k] + self.ufh(h))
            it = self.act_gate(wix_out[k] + self.uih(h))
            ot = self.act_gate(wox_out[k] + self.uoh(h))

            at = wcx_out[k] + self.uch(h)
            c = it * self.act(at) + ft * c
            h = ot * self.act(c)
            output[k] = h

        return output, h

if __name__ == '__main__':
    feat_size = 20
    hidden_size = 1024
    CUDA = False
    model = QLSTM(feat_size, hidden_size, CUDA)

    inp = torch.randn(100, 2, 20)
    out = model(inp)
    print(out.shape)