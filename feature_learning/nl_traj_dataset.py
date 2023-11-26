import torch
from torch.utils.data import Dataset
import json
import numpy as np


# nlcomp_file is a json file with the list of comparisons in NL.
# traj_a_file is a .npy or .npz file with the first trajectory and has a shape of (n_trajectories, n_timesteps, STATE_DIM+ACTION_DIM)
# traj_b_file is a .npy or .npz file with the second trajectory and has a shape of (n_trajectories, n_timesteps, STATE_DIM+ACTION_DIM)
# If id_mapped is true, traj_*_file and nlcomp_file are instead .npy files with INDICES corresponding to traj_file and unique_nlcomp_file.
class NLTrajComparisonDataset(Dataset):
    def __init__(self, nlcomp_file, traj_a_file, traj_b_file,
                 preprocessed_nlcomps=False, id_mapped=False, unique_nlcomp_file=None, traj_file=None):
        if id_mapped:
            assert unique_nlcomp_file is not None
            assert traj_file is not None
            self.id_mapped = True

            # These are actually just indexes. (So we don't need to mmap them.)
            self.nlcomps = np.load(nlcomp_file)
            self.traj_as = np.load(traj_a_file)
            self.traj_bs = np.load(traj_b_file)

            # These are the actual embeddings.
            self.unique_nlcomps = np.load(unique_nlcomp_file)
            self.trajs = np.load(traj_file)
        else:
            self.id_mapped = False
            if preprocessed_nlcomps:
                self.nlcomps = np.load(nlcomp_file)
            else:
                with open(nlcomp_file, 'rb') as f:
                    self.nlcomps = json.load(f)
            # self.traj_as = np.load(traj_a_file)
            # NOTE: `mmap_mode` memory-maps the file, enabling us to read directly from disk
            # rather than loading the entire (huge) array into memory.
            self.traj_as = np.load(traj_a_file, mmap_mode='r')
            self.traj_bs = np.load(traj_b_file, mmap_mode='r')

    def __len__(self):
        return len(self.nlcomps)

    def __getitem__(self, idx):
        if self.id_mapped:
            traj_a = np.copy(self.trajs[self.traj_as[idx], :, :])
            traj_b = np.copy(self.trajs[self.traj_bs[idx], :, :])
            nlcomp = np.copy(self.unique_nlcomps[self.nlcomps[idx]])
        else:
            traj_a = np.copy(self.traj_as[idx, :, :])
            traj_b = np.copy(self.traj_bs[idx, :, :])
            nlcomp = np.copy(self.nlcomps[idx])
        return traj_a, traj_b, nlcomp
