import pickle
import glob
from utils import *
from metrics import *
from model import social_stgcnn
from torch.utils.data import DataLoader
import copy


def show_res(KSTEPS=1):
    global loader_test, model
    model.eval()
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, obs_traj_rel, non_linear_ped, V_obs, A_obs = batch

        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        # print(V_pred.shape)
        # torch.Size([1, 5, 6, 2])
        # torch.Size([6, 2, 5])
        V_pred = V_pred.permute(0, 2, 3, 1)
        # torch.Size([1, 6, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()

        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred = V_pred[:, :num_of_objs, :]

        ### Rel to abs
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len

        # Now sample 20 samples
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        # print(V_x.shape)
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(), V_x[0, :, :].copy())
        V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(), V_x[-1, :, :].copy())
        # 历史轨迹和未来轨迹也是相对预测起始点的

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['pred'] = []
        raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

    return raw_data_dict


if __name__ == '__main__':
    # paths = ['./checkpoint/*social-stgcnn*']
    paths = ['./checkpoint/tag/']
    KSTEPS = 1

    print("*" * 50)

    for feta in range(len(paths)):
        path = paths[feta]
        print(path)
        exps = glob.glob(path)
        print('Model being tested are:', exps)

        for exp_path in exps:
            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = exp_path + '/val_best.pth'
            args_path = exp_path + '/args.pkl'
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            stats = exp_path + '/constant_metrics.pkl'
            with open(stats, 'rb') as f:
                cm = pickle.load(f)
            print("Stats:", cm)

            # Data prep
            obs_seq_len = args.obs_seq_len
            data_set = '../datasets/' + args.dataset + '/'

            dset_test = DatasetForResult(data_set + 'data/test/', obs_len=obs_seq_len, skip=1, norm_lap_matr=True)
            loader_test = DataLoader(dset_test, batch_size=1, shuffle=False, num_workers=1)

            # Defining the model
            model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                                  output_feat=args.output_size, seq_len=args.obs_seq_len,
                                  kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()
            model.load_state_dict(torch.load(model_path))

            print("Testing ....")
            raw_data_dic_ = show_res()
            print("Finish.")