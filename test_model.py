from __future__ import print_function
import numpy as np
import h5py
import torch
from enc_dec import Encoder_Decoder
from enc_dec_vel import Encoder_Decoder_Velocity
import torch_utils

def test_model(model_file, dataset, input_size, get_batch):
    # default parameter, do not change
    hid_size = 1024 # 2048
    num_layer = 1
    rnn_unit = 'gru' # ''gru'
    source_seq_len = 50
    target_seq_len = 25 # 25*4

    if 1:
        model = Encoder_Decoder(input_size, hid_size, num_layer, rnn_unit, residual=False, veloc=False,
                                std_mask=True, pos_embed=False, pos_embed_dim=96, cuda=False)
    else:
        model = Encoder_Decoder_Velocity(input_size, hid_size, num_layer, rnn_unit, residual=True,
                                         std_mask=True, accel=False)

    model.load_state_dict(torch.load(model_file) )
    model.eval()
    test_set = dataset.load_data(dataset.get_test_subject_ids())

    result = {}
    # groundtruth, prediction
    # hf = h5py.File('data/test_data_cmu.h5', 'r')
    hf = h5py.File('data/bef_2019/test_data.h5', 'r')
    print("{0: <18} |".format("milliseconds"), end="")
    for ms in [80, 160, 320, 400, 560, 1000]:
        print(" {0:5d} |".format(ms), end="")
    print()
    for action, action_idx in dataset.get_test_actions():
        # Evaluate the model on the test batches
        source_tst, target_tst = get_batch(test_set, action, source_seq_len, target_seq_len, input_size+3)
        # hf.create_dataset(action + 'source', data=source_tst)
        # hf.create_dataset(action + 'target', data=target_tst)
        source_tst = hf.get(action + 'source')
        target_tst = hf.get(action + 'target')
        gt_map = np.concatenate((source_tst, target_tst), axis=1)

        source_tst = torch.tensor(source_tst)
        target_tst = torch.tensor(target_tst)
        source_tst = source_tst[:, :, 3:]
        target_tst = target_tst[:, :, 3:]
        source_tst = source_tst.permute(1, 0, 2).float()
        target_tst = target_tst.permute(1, 0, 2).float()

        _, pred_target = model(source_tst, target_tst, trj_seq=0)

        pred_map = torch.cat((source_tst, target_tst[0:1], pred_target), dim=0).permute(1, 0, 2).detach().numpy()
        pred_map = np.concatenate((gt_map[:,:,0:3], pred_map), axis=-1)
        result[action] = (gt_map, pred_map)

        # Convert from exponential map to Euler angles
        target_tst = torch_utils.tensor_expmap_to_euler(target_tst)
        pred_target = torch_utils.tensor_expmap_to_euler(pred_target)

        # global rotation the first 3 entries are also not considered in the error
        error = torch.pow(target_tst[1:, :, 3:] - pred_target[:, :, 3:], 2)
        error = torch.sqrt(torch.sum(error, dim=-1))
        error = torch.mean(error, dim=1)
        mean_mean_errors = error.cpu().detach().numpy()
        # if action == "walking":
        #     print(mean_mean_errors)

        # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
        print("{0: <18} |".format(action), end="")
        for ms in [1, 3, 7, 9, 13, 24]:
            if mean_mean_errors.shape[0] >= ms + 1:
                print(" {0:.3f} |".format(mean_mean_errors[ms]), end="")
            else:
                print("   n/a |", end="")
        print()  # start new line
    hf.close()
    return result


if __name__ == '__main__':
    # model_file = 'data/bef_2019/motpred_h36m_epoch_15105_err_1.08650.pt'  # motpred_h36m_epoch_19969_err_1.12090.pt
    model_file = 'data/bef_2019/motpred_rnn_baseline_h36m_epoch_27457_err_2.02304.pt'
    model_file = 'data/bef_2019/motpred_rnn_baseline_h36m_epoch_5009_err_1.38891.pt'
    model_file = 'data/model/rot_det_20_v2v_acel_res_gru_h36m_epoch_13185_err_1.14296.pt'
    model_file = 'data/model/res_gru_h36m_epoch_25809_err_1.45784.pt'

    data_dir = '/home/hust/data/Human_3.6M/h3.6m/dataset'

    from human_36m import human_36m_dataset
    from batch_sample import get_batch_srnn
    dataset = human_36m_dataset(data_dir)
    input_size = 3 * 32

    test_model(model_file, dataset, input_size, get_batch_srnn)
