from __future__ import print_function
import sys
import argparse
import torch
from human_36m import human_36m_dataset
from cmu_mocap import cmu_mocap_dataset
from batch_sample import get_batch_srnn, get_batch_srnn_cmu
import torch_utils

def average_k(source_tst, target_tst, k=2):
    pred_target = torch.zeros(target_tst[1:].size())
    if k<=1:
        pred_target[:] = target_tst[0:1]
        return pred_target

    last_k1 = source_tst[(-k+1):]
    last_k = torch.cat((last_k1, target_tst[0:1]), dim=0)
    for j in range(len(target_tst) - 1):
        avg = torch.mean(last_k, dim=0, keepdim=True)
        last_k = torch.cat((last_k[1:], avg), dim=0)
        pred_target[j] = avg
    return pred_target

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cmu', help='h36m or cmu')
    parser.add_argument('--data_dir', default='/home/hust/data/Human_3.6M/h3.6m/dataset')
    parser.add_argument('--data_dir_cmu', default='/home/hust/data/Human_3.6M/cmu_mocap/')
    parser.add_argument('--source_seq_len', type=int, default=50, help='length of encode sequence')
    parser.add_argument('--target_seq_len', type=int, default=25, help='length of output decode sequence')
    parser.add_argument('--input_size', type=int, default=96, help='input size at each timestep')
    parser.add_argument('--num_joint_cmu', type=int, default=38, help='number of joints for cmu dataset')
    parser.add_argument('--num_joint', type=int, default=32, help='input size at each timestep')
    args = parser.parse_args()

    if args.dataset == 'h36m':
        dataset = human_36m_dataset
    if args.dataset == 'cmu':
        dataset = cmu_mocap_dataset
        args.data_dir = args.data_dir_cmu

    dataset = dataset(args.data_dir)
    test_set = dataset.load_data(dataset.get_test_subject_ids())

    print("{0: <16} |".format("milliseconds"), end="")
    for ms in [80, 160, 320, 400, 560, 1000]:
        print(" {0:5d} |".format(ms), end="")
    print()
    for action, _ in dataset.get_test_actions():
        # Evaluate the model on the test batches
        if args.dataset == 'h36m':
            source_tst, target_tst = get_batch_srnn(test_set, action, args.source_seq_len, args.target_seq_len, 3 * args.num_joint + 3)
        else:
            source_tst, target_tst = get_batch_srnn_cmu(test_set, action, args.source_seq_len, args.target_seq_len, 3 * args.num_joint_cmu + 3)

        # Discard the first joint, which represents a corrupted translation
        source_tst = source_tst[:, :, 3:]
        target_tst = target_tst[:, :, 3:]
        source_tst = torch.tensor(source_tst)
        target_tst = torch.tensor(target_tst)
        source_tst = source_tst.permute(1, 0, 2).float()
        target_tst = target_tst.permute(1, 0, 2).float()

        pred_target = average_k(source_tst, target_tst)

        # Convert from exponential map to Euler angles
        target_tst = torch_utils.tensor_expmap_to_euler(target_tst)
        pred_target = torch_utils.tensor_expmap_to_euler(pred_target)

        # global rotation the first 3 entries are also not considered in the error
        error = torch.pow(target_tst[1:, :, 3:] - pred_target[:, :, 3:], 2)
        error = torch.sqrt(torch.sum(error, dim=-1))
        error = torch.mean(error, dim=1)
        mean_mean_errors = error.cpu().detach().numpy()

        # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
        print("{0: <18} |".format(action), end="")
        for ms in [1, 3, 7, 9, 13, 24]:
            if mean_mean_errors.shape[0] >= ms + 1:
                print(" {0:.3f} |".format(mean_mean_errors[ms]), end="")
            else:
                print("   n/a |", end="")
        print()  # start new line

if __name__ == '__main__':
    main()
