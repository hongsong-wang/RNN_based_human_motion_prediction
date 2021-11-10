from __future__ import print_function
import sys
import os
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from human_36m import human_36m_dataset
from batch_sample import generate_train_data, get_batch_srnn
import torch_utils
from enc_dec import RNN_Frame_Classify
from utils import Logger, AverageMeter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', default=True, help='use cpu only')
    parser.add_argument('--data_dir', default='/home/hust/data/Human_3.6M/h3.6m/dataset')
    parser.add_argument('--save_model', default='/temp/wanghongsong/data/savetemp/h36m_classify_epoch_%d_acc_%0.5f.pt')
    parser.add_argument('--log_dir', default='log/h36m')
    parser.add_argument('--log_file', default='log_train.txt')

    parser.add_argument('--source_seq_len', type=int, default=50, help='length of encode sequence')
    parser.add_argument('--target_seq_len', type=int, default=25, help='length of output decode sequence')
    parser.add_argument('--num_joint', type=int, default=32, help='input size at each timestep')
    parser.add_argument('--num_class', type=int, default=15, help='number of output class')
    parser.add_argument('--hid_size', type=int, default=128, help='hidden size of RNN')
    parser.add_argument('--rnn_unit', default='lstm', help='gru or lstm')
    parser.add_argument('--num_layer', type=int, default=2, help='number of rnn layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio for output prediction')
    parser.add_argument('--data_aug', action='store_true', default=True, help='whether perform data augmentation')

    parser.add_argument('--batch_size', type=int, default=128, help='make sure batch size large than 8 due to testing, only 180 sequences')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate reduce ratio')
    parser.add_argument('--num_max_epoch', type=int, default=30000, help='number of epoch for all training samples')
    parser.add_argument('--step_size', type=int, default=10000, help='step epoch to reduce learning rate')

    args = parser.parse_args()
    sys.stdout = Logger(os.path.join(args.log_dir, args.log_file))
    print(args)

    criterion = nn.CrossEntropyLoss()
    model = RNN_Frame_Classify(3*args.num_joint, args.hid_size, args.num_class, args.num_layer, args.rnn_unit, dropout=args.dropout)
    if not args.cpu:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    dataset = human_36m_dataset(args.data_dir)
    train_set = dataset.load_data(dataset.get_train_subject_ids())
    test_set = dataset.load_data(dataset.get_test_subject_ids())

    train_gen = generate_train_data(train_set, args.source_seq_len, args.target_seq_len)

    for epoch_i in range(int(args.num_max_epoch)):
        scheduler.step()
        model.train()
        losses = AverageMeter()
        for source, target, action in DataLoader(train_gen, batch_size=args.batch_size, shuffle=True):
            if not args.cpu:
                source = source.cuda()
                target = target.cuda()
                action = action.cuda()
            if args.data_aug:
                source, target = torch_utils.rand_rotate_expmap(source, target)

            # convert seq to (time, batch, dim)
            source = source.permute(1, 0, 2).float()
            target = target.permute(1, 0, 2).float()

            pred = model.forward(source, target)
            pred = pred.permute(1, 2, 0)
            action_time = action.unsqueeze(1).repeat(1, target.size(0)).long()
            loss = criterion(pred, action_time)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), source.size(1) )

        print('train epoch %d, loss = %0.5f, lr = %0.5f' % (epoch_i + 1, losses.val, optimizer.param_groups[-1]['lr']))

        # === Validation with srnn's seeds ===
        if epoch_i % 8 ==0 and epoch_i > 0:
            total_count = 0
            acc_count = 0
            for action, action_idx in dataset.get_test_actions():
                # Evaluate the model on the test batches
                source_tst, target_tst = get_batch_srnn(test_set, action, args.source_seq_len, args.target_seq_len, 3*args.num_joint)

                source_tst = torch.tensor(source_tst).to(source.device)
                target_tst = torch.tensor(target_tst).to(source.device)

                source_tst = source_tst.permute(1, 0, 2).float()
                target_tst = target_tst.permute(1, 0, 2).float()

                pred_target = model.forward(source_tst, target_tst)
                pred_target = torch.argmax(pred_target, dim=-1)
                acc_count += torch.sum(pred_target == action_idx).item()
                total_count += pred_target.size(0)*pred_target.size(1)
            acc = acc_count*1.0/total_count
            print('epoch %d, test frame-wise accuracy = %0.5f' % (epoch_i + 1, acc) )

        if epoch_i % 1000==0 and epoch_i > 0:
            torch.save(model.state_dict(), args.save_model%(epoch_i + 1, acc) )

if __name__ == '__main__':
    main()