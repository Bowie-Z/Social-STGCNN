import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from math import sqrt, ceil

import torch
import numpy as np

from torch.utils.data import Dataset
import networkx as nx
from tqdm import tqdm


def anorm(p1, p2):
    NORM = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloader for the Trajectory datasets"""

    def __init__(self, data_dir, obs_len=6, pred_len=6, skip=1, threshold=0.002,
            min_ped=1, delim='\t', norm_lap_matr=True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()  # get all numbers of frames
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])  # get all data of a certain frame
            num_sequences = int(ceil((len(frames) - self.seq_len + 1) / skip))  # number of utilized sequences in all data

            for idx in range(0, num_sequences * self.skip + 1, skip):  # 处理一个历史+未来序列中所有行人的轨迹
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                # curr_seq_data是从idx到idx+seq_len的所有帧中所有行人的轨迹
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # 当前序列中所有行人的编号
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                # 寻找具有最多行人的帧，其行人数目即图的节点数
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):  # 处理单个行人的轨迹
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]  # 单个行人的轨迹
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)  # 保留四位小数
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    # pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    pad_end = pad_front + len(curr_ped_seq)
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 3:5])  # coordinates here
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)  # seq_list维度是行人数目：坐标维度：序列长度
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        # Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))  # show a process bar
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_, a_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]
        ]
        return out


class DatasetForResult(Dataset):
    """Dataloader for the Trajectory datasets when output results"""

    def __init__(self, data_dir, obs_len=6, pred_len=6, skip=1, threshold=0.002,
                 min_ped=1, delim='\t', norm_lap_matr=True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(DatasetForResult, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.skip = skip
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        obs_seq_list = []
        obs_seq_list_rel = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()  # get all numbers of frames
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])  # get all data of a certain frame
            num_sequences = int(ceil((len(frames) - self.seq_len + 1) / skip))  # number of utilized sequences in all data

            for idx in range(0, num_sequences * self.skip + 1, skip):  # 处理一个历史序列中所有行人的轨迹
                curr_obs_seq_data = np.concatenate(frame_data[idx:idx + self.obs_len], axis=0)
                # curr_obs_seq_data是从idx到idx+obs_len的所有帧中所有行人的轨迹
                peds_in_curr_seq = np.unique(curr_obs_seq_data[:, 1])  # 当前序列中所有行人的编号
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                # 寻找具有最多行人的帧，其行人数目即图的节点数
                curr_obs_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.obs_len))
                curr_obs_seq = np.zeros((len(peds_in_curr_seq), 2, self.obs_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.obs_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):  # 处理单个行人的轨迹
                    curr_ped_obs_seq = curr_obs_seq_data[curr_obs_seq_data[:, 1] == ped_id, :]  # 单个行人的轨迹
                    curr_ped_obs_seq = np.around(curr_ped_obs_seq, decimals=4)  # 保留四位小数
                    pad_front = frames.index(curr_ped_obs_seq[0, 0]) - idx
                    # pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    pad_end = pad_front + len(curr_ped_obs_seq)
                    if pad_end - pad_front != self.obs_len:
                        continue
                    curr_ped_obs_seq = np.transpose(curr_ped_obs_seq[:, 3:5])  # get coordinates here
                    # Make coordinates relative
                    rel_curr_ped_obs_seq = np.zeros(curr_ped_obs_seq.shape)
                    rel_curr_ped_obs_seq[:, 1:] = curr_ped_obs_seq[:, 1:] - curr_ped_obs_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_obs_seq[_idx, :, pad_front:pad_end] = curr_ped_obs_seq
                    curr_obs_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_obs_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_obs_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    obs_seq_list.append(curr_obs_seq[:num_peds_considered])
                    obs_seq_list_rel.append(curr_obs_seq_rel[:num_peds_considered])

        self.num_seq = len(obs_seq_list)
        obs_seq_list = np.concatenate(obs_seq_list, axis=0)
        obs_seq_list_rel = np.concatenate(obs_seq_list_rel, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(obs_seq_list[:, :, :self.obs_len]).type(torch.float)  # seq_list维度是行人数目：坐标维度：序列长度
        self.obs_traj_rel = torch.from_numpy(obs_seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        # Convert to Graphs
        self.v_obs = []
        self.A_obs = []

        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))  # show a process bar
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]
            v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.v_obs[index], self.A_obs[index],
        ]
        return out
