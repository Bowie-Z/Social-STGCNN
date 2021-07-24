import pickle
import glob
from utils import *
from metrics import *
from model import social_stgcnn
from torch.utils.data import DataLoader
import copy


def show_res():
    global loader_test, model
    model.eval()
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, obs_traj_rel, non_linear_agent, V_obs, A_obs = batch

        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        # torch.Size([1, 5, 6, 2])
        V_pred = V_pred.permute(0, 2, 3, 1)
        # torch.Size([1, 6, 2, 5])>>seq,node,feat

        V_pred = V_pred.squeeze()
        V_pred = V_pred[:, :, 0:2]
        num_of_objs = obs_traj_rel.shape[1]
        V_pred = V_pred[:, :num_of_objs, :]

        ### Rel to abs
        ##obs_traj.shape = torch.Size([1, nodes, 5, 6]) Batch, num_of_agents, frame_id|agent_id|type|x|y, Seq Len
        id_and_type = obs_traj[:, :, 0:3, :].permute(0, 3, 1, 2).data.cpu().numpy().squeeze()
        # print(id_and_type.shape)
        V_x = seq_to_nodes(obs_traj[:, :, 3:5, :].data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(), V_x[0, :, :].copy())
        V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(), V_x[-1, :, :].copy())
        # print(V_pred_rel_to_abs.shape)
        show = np.concatenate([id_and_type, V_pred_rel_to_abs], axis=2)

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['pred'] = []
        raw_data_dict[step]['pred'].append(copy.deepcopy(show))

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

            dset_test = DatasetForResult(data_set + 'data/test/', obs_len=obs_seq_len, skip=6, norm_lap_matr=True)
            loader_test = DataLoader(dset_test, batch_size=1, shuffle=False, num_workers=1)

            # Defining the model
            model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                                  output_feat=args.output_size, seq_len=args.obs_seq_len,
                                  kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()
            model.load_state_dict(torch.load(model_path))

            print("Testing ....")
            raw_data_dic_ = show_res()

            count = 0
            for key in raw_data_dic_:
                temp = raw_data_dic_[key]['pred']
                for i in range(len(temp)):
                    for j in range(temp[i].shape[0]):
                        np.savetxt(str(count) + '.txt', temp[i][j, :, :], fmt="%4f", delimiter=' ')
                        count += 1

            print("Finish.")
