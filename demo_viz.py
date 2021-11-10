import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  #<-- Note the capitalization!
from test_model import test_model
from human_36m import human_36m_dataset
from cmu_mocap import cmu_mocap_dataset
from batch_sample import get_batch_srnn, get_batch_srnn_cmu

h36m_db = True

if h36m_db:
    from viz_h36m import _some_variables
    import viz_h36m as viz
    from kinematics import fkl, revert_coordinate_space
else:
    from viz_cmu import _some_variables
    import viz_cmu as viz
    from kinematics_cmu import fkl, revert_coordinate_space

def main():
    model_file = 'data/bef_2019/motpred_h36m_epoch_15105_err_1.08650.pt'
    # model_file = 'data/bef_2019/motpred4s_lstm_2048_h36m_epoch_58177_err_2.25741.pt'
    # model_file = 'data/bef_2019/motpred_rnn_baseline_h36m_epoch_27457_err_2.02304.pt'
    # model_file = 'data/bef_2019/motpred4s_baseline_lstm_2048_h36m_epoch_16929_err_2.54094.pt'
    model_file = 'data/model/rot_det_20_v2v_acel_res_gru_h36m_epoch_13185_err_1.14296.pt'
    model_file = 'data/model/rot_det_30_v2v_res_gru_h36m_epoch_16865_err_1.14445.pt'
    model_file = 'data/model/res_gru_h36m_epoch_25809_err_1.45784.pt'
    model_file = 'data/model/gru_h36m_epoch_12817_err_1.54736.pt'

    if h36m_db:
        data_dir = '/home/hust/data/Human_3.6M/h3.6m/dataset'
        dataset = human_36m_dataset(data_dir)
        input_size = 3 * 32
        get_batch = get_batch_srnn
        result = test_model(model_file, dataset, input_size, get_batch)
        action_set = ["walking", "eating", "smoking", "discussion", "directions",
         "greeting", "phoning", "posing", "purchases", "sitting",
         "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"]
    else:
        data_dir = '/home/hust/data/Human_3.6M/cmu_mocap/'
        dataset = cmu_mocap_dataset(data_dir)
        input_size = 3 * 38
        get_batch = get_batch_srnn_cmu
        result = test_model(model_file, dataset, input_size, get_batch)
        action_set = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking", "washwindow"]

    index = 6
    for action in action_set:
        expmap_gt, expmap_pred = result[action]
        expmap_gt = expmap_gt[index]
        expmap_pred = expmap_pred[index]

        nframes_gt, nframes_pred = expmap_gt.shape[0], expmap_pred.shape[0]

        # Put them together and revert the coordinate space
        expmap_gt = revert_coordinate_space( expmap_gt, np.eye(3), np.zeros(3) )
        expmap_pred = revert_coordinate_space(expmap_pred, np.eye(3), np.zeros(3))

        # Load all the data
        parent, offset, rotInd, expmapInd = _some_variables()
        # Compute 3d points for each frame
        if h36m_db:
            xyz_gt, xyz_pred = np.zeros((nframes_gt, 96)), np.zeros((nframes_pred, 96))
        else:
            xyz_gt, xyz_pred = np.zeros((nframes_gt, 114)), np.zeros((nframes_pred, 114))
        for i in range( nframes_gt ):
            xyz_gt[i,:] = fkl( expmap_gt[i,:], parent, offset, rotInd, expmapInd )
        for i in range( nframes_pred ):
            xyz_pred[i,:] = fkl( expmap_pred[i,:], parent, offset, rotInd, expmapInd )

        # calculate velocity
        vlc_gt, vlc_pred = np.zeros(xyz_gt.shape), np.zeros(xyz_pred.shape)
        vlc_gt[1:] = xyz_gt[1:] - xyz_gt[0:-1]
        vlc_pred[1:] = xyz_pred[1:] - xyz_pred[0:-1]

        # === Plot and animate ===
        fig = plt.figure()
        # Todo, set axis on, use default elev and azmin when make video
        ax = Axes3D(fig, elev=30, azim=10)
        ax.set_axis_off()
        ob = viz.Ax3DPose(ax)

        # Todo, change save folder for long term prediction
        save_fold = 'data/201904/short/%s_%d' % (action, index)
        if not os.path.exists(save_fold):
            os.makedirs(save_fold)

        # Plot the conditioning ground truth
        for i in range(nframes_gt):
            ob.update( xyz_gt[i,:])
            # ax.quiver()
            plt.show(block=False)
            fig.canvas.draw()
            # if (i+1) in [52, 54, 58, 60, 64, 75,  70, 66,   10, 20, 30, 40, 50]:
            if (i + 1) in range(40, 51 + 25, 1): # range(10, 51+4*25, 4)
                # fig.savefig('%s/gt_frm_%03d.jpg'%(save_fold, i+1), dpi=1000, bbox_inches='tight', pad_inches=0)
                pass
            plt.pause(0.01)

        # Plot the prediction
        for i in range(nframes_pred):
            # ob.update( xyz_pred[i,:], lcolor="#9b59b6", rcolor="#2ecc71" )
            # ob.update(xyz_pred[i, :], lcolor='b', rcolor='g')
            ob.update(xyz_pred[i, :], lcolor='c', rcolor='y')
            plt.show(block=False)
            # if (i + 1) in [52, 54, 58, 60, 64, 75, 70, 66]:
            if (i + 1) in range(50, 51 + 25, 1): # range(50, 51 + 4*25, 4)
                fig.savefig('%s/red_frm_%03d.jpg' % (save_fold, i + 1), dpi=1000, bbox_inches='tight', pad_inches=0)
                pass
            fig.canvas.draw()
            plt.pause(0.01)

if __name__ == '__main__':
    main()