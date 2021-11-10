import math
import torch
import torch.nn as nn

def position_embedding(d_model, max_len=75+25*0): # +25*4
    if d_model <= 0:
        pe = torch.eye(max_len).float()
        pe.require_grad = False
        return pe

    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class RNN_Frame_Classify(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer, rnn_unit, dropout=0):
        super(RNN_Frame_Classify, self).__init__()
        if rnn_unit == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layer)
        if rnn_unit == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layer)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, target):
        output, hidden_state = self.rnn(input)
        output, _ = self.rnn(target, hidden_state)
        output = self.linear(self.dropout(output))
        return output

class Encoder_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, rnn_unit, residual=False, out_dropout=0, std_mask=False,
                 veloc=False, pos_sum=False, pos_embed=False, pos_embed_dim=96, cuda=False, trj_embed=False, trj_embed_dim=32):
        super(Encoder_Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.residual = residual
        self.dropout_out = nn.Dropout(p=out_dropout)
        self.linear = nn.Linear(hidden_size, input_size)
        self.std_mask = std_mask
        self.veloc = veloc
        self.pos_sum = pos_sum
        rnn_input = 2*input_size if (veloc or pos_sum) else input_size
        rnn_input = 3 * input_size if (veloc and pos_sum) else rnn_input

        self.pos_embed = pos_embed
        if pos_embed:
            self.position_embeding = position_embedding(d_model=pos_embed_dim)
            rnn_input = rnn_input + self.position_embeding.size(1)
            if cuda:
                self.position_embeding = self.position_embeding.cuda()

        self.trj_embed = trj_embed
        self.trj_embed_dim = trj_embed_dim
        if trj_embed:
            # trajectory at a time step is 3 dimensional coodinates
            self.trj_fc = nn.Linear(3, trj_embed_dim)
            rnn_input = rnn_input + trj_embed_dim

        if rnn_unit == 'gru':
            self.rnn = nn.GRU(rnn_input, hidden_size, num_layers=num_layer)
        if rnn_unit == 'lstm':
            self.rnn = nn.LSTM(rnn_input, hidden_size, num_layers=num_layer)

    def forward_seq(self, input, hidden=None):
        pred_pre = input[:,:,0:self.input_size].clone()
        if hidden is None:
            output, hidden_state = self.rnn(input)
        else:
            output, hidden_state = self.rnn(input, hidden)
        pred = self.linear(self.dropout_out(output))

        if self.residual:
            pred = pred + pred_pre

        return pred, hidden_state


    def forward(self, input, target, trj_seq):
        # check std of previous time
        mask_pred = torch.std(input, dim=0, keepdim=True) > 1e-4
        # Encoder
        input_en = input
        if self.veloc and (not self.pos_sum):
            input_vl = torch.zeros(input.size()).to(input.device)
            input_vl[1:] = input[1:] - input[0:-1]
            input_en = torch.cat((input, input_vl), dim=-1)
        if (not self.veloc) and self.pos_sum:
            input_sum = torch.cumsum(input, dim=0) / torch.arange(1., input.size(0) + 1.0).unsqueeze(-1).unsqueeze(
                -1).to(input.device)
            input_en = torch.cat((input, input_sum), dim=-1)
        if self.veloc and self.pos_sum:
            input_vl = torch.zeros(input.size()).to(input.device)
            input_vl[1:] = input[1:] - input[0:-1]
            input_sum = torch.cumsum(input, dim=0) / torch.arange(1., input.size(0) + 1.0).unsqueeze(-1).unsqueeze(
                -1).to(input.device)
            input_en = torch.cat((input, input_vl, input_sum), dim=-1)

        if self.pos_embed:
            pos_emb = self.position_embeding[0:input_en.size(0)].unsqueeze(1).repeat(1, input_en.size(1), 1)
            input_en = torch.cat((input_en, pos_emb), dim=-1)

        if self.trj_embed:
            trj_hid = self.trj_fc(trj_seq)
            input_en = torch.cat((input_en, trj_hid[0:input_en.size(0)]), dim=-1)

        outputs_enc, hidden_state_en = self.forward_seq(input_en)

        # Decoder
        # the last frame of given poses is placed in the target
        pos_sum = torch.sum(input, dim=0, keepdim=True) + target[0][None]
        count = input.size(0) + 1
        outputs_dec = torch.zeros(target.size(0) - 1, target.size(1), target.size(2)).to(input.device)

        for i in range(len(target) - 1):
            inp_cur = target[i][None] if i == 0 else pred
            if self.veloc and (not self.pos_sum):
                inp_cur_vl = (target[i][None] - input[-1][None]) if i == 0 else (pred - outputs_dec[i - 1:i])
                inp_cur = torch.cat((inp_cur, inp_cur_vl), dim=-1)
            if (not self.veloc) and self.pos_sum:
                inp_cur_sum = pos_sum / count
                inp_cur = torch.cat((inp_cur, inp_cur_sum), dim=-1)
            if self.veloc and self.pos_sum:
                inp_cur_vl = (target[i][None] - input[-1][None]) if i == 0 else (pred - outputs_dec[i - 1:i])
                inp_cur_sum = pos_sum / count
                inp_cur = torch.cat((inp_cur, inp_cur_vl, inp_cur_sum), dim=-1)

            if self.pos_embed:
                pos_emb = self.position_embeding[count - 1:count].unsqueeze(1).repeat(1, inp_cur.size(1), 1)
                inp_cur = torch.cat((inp_cur, pos_emb), dim=-1)

            if self.trj_embed:
                inp_cur = torch.cat((inp_cur, trj_hid[count-1:count]), dim=-1)

            pred, hidden_state = self.forward_seq(inp_cur, hidden=(hidden_state_en if i == 0 else hidden_state))

            if self.std_mask:
                pred = mask_pred.float() * pred

            outputs_dec[i:i + 1] = pred
            pos_sum = pos_sum + pred
            count += 1
        return outputs_enc, outputs_dec

if __name__ == '__main__':
    feat = position_embedding(max_len=20, d_model=512)
    print(feat.shape)
    print(feat)