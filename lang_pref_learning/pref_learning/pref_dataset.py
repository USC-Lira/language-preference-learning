from torch.utils.data import Dataset


class LangPrefDataset(Dataset):
    def __init__(self, trajs, feature_values):
        self.trajs = trajs
        self.feature_values = feature_values

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        traj = self.trajs[idx]
        feature_value = self.feature_values[idx]
        return traj, feature_value, idx
    

class CompPrefDataset(Dataset):
    def __init__(self, trajs, feature_values):
        self.trajs = trajs
        self.feature_values = feature_values
        self.traj_pairs = [(i, j) for i in range(len(trajs)) for j in range(i+1, len(trajs))]
        self.feature_pairs = [(i, j) for i in range(len(feature_values)) for j in range(i+1, len(feature_values))]
        assert len(self.traj_pairs) == len(self.feature_pairs)
    
    def __len__(self):
        return len(self.traj_pairs)
    
    def __getitem__(self, idx):
        traj_a_idx, traj_b_idx = self.traj_pairs[idx]
        traj_a, traj_b = self.trajs[traj_a_idx], self.trajs[traj_b_idx]
        feature_a, feature_b = self.feature_values[traj_a_idx], self.feature_values[traj_b_idx]
        return traj_a, traj_b, feature_a, feature_b, traj_a_idx, traj_b_idx


class EvalDataset(Dataset):
    def __init__(self, trajs):
        self.trajs = trajs
        self.traj_pairs = [(i, j) for i in range(len(trajs)) for j in range(i+1, len(trajs))]

    def __len__(self):
        return len(self.traj_pairs)

    def __getitem__(self, idx):
        traj_a_idx, traj_b_idx = self.traj_pairs[idx]
        traj_a, traj_b = self.trajs[traj_a_idx], self.trajs[traj_b_idx]
        return traj_a, traj_b, traj_a_idx, traj_b_idx
