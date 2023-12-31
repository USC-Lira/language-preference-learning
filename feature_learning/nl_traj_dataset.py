import torch
from torch.utils.data import Dataset
import json
import numpy as np


# nlcomp_file is a json file with the list of comparisons in NL.
# traj_a_file is a .npy or .npz file with the first trajectory and has a shape of (n_trajectories, n_timesteps, STATE_DIM+ACTION_DIM)
# traj_b_file is a .npy or .npz file with the second trajectory and has a shape of (n_trajectories, n_timesteps, STATE_DIM+ACTION_DIM)
# If id_mapped is true, traj_*_file and nlcomp_file are instead .npy files with INDICES corresponding to traj_file and unique_nlcomp_file.
class NLTrajComparisonDataset(Dataset):
    def __init__(self, nlcomp_file, traj_a_file, traj_b_file, seq_len=64, tokenizer=None,
                 preprocessed_nlcomps=False, unique_nlcomp_file=None, traj_file=None):

            assert unique_nlcomp_file is not None
            assert traj_file is not None

            # These are actually just indexes. (So we don't need to mmap them.)
            self.nlcomps = np.load(nlcomp_file)
            self.traj_as = np.load(traj_a_file)
            self.traj_bs = np.load(traj_b_file)

            self.max_len = seq_len
            self.preprocessed_nlcomps = preprocessed_nlcomps

            self.unique_nlcomps = None
            self.unique_nlcomps_tokens = None
            self.unique_nlcomps_attention_masks = None

            # These are the sentences and trajectories corresponding to the indexes.
            if preprocessed_nlcomps:
                self.unique_nlcomps = np.load(unique_nlcomp_file)
            else:
                assert tokenizer is not None, "Must provide tokenizer if not using preprocessed_nlcomps."
                self.tokenizer = tokenizer
                with open(unique_nlcomp_file, 'rb') as f:
                    nlcomps= json.load(f)
                self.unique_nlcomps_tokens = []
                self.unique_nlcomps_attention_masks = []
                for nlcomp in nlcomps:
                    tokens = self.tokenizer.tokenize(self.tokenizer.cls_token + " " + nlcomp + " " + self.tokenizer.sep_token)
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                    # Pad sequences to the common length
                    padding_length = self.max_len - len(token_ids)

                    # Create attention mask
                    attention_mask = [1] * len(token_ids) + [0] * padding_length
                    token_ids += [self.tokenizer.pad_token_id] * padding_length

                    self.unique_nlcomps_tokens.append(token_ids)
                    self.unique_nlcomps_attention_masks.append(attention_mask)

                self.unique_nlcomps_tokens = np.array(self.unique_nlcomps_tokens)
                self.unique_nlcomps_attention_masks = np.array(self.unique_nlcomps_attention_masks)

            self.trajs = np.load(traj_file)

    def __len__(self):
        return len(self.nlcomps)

    def __getitem__(self, idx):
        traj_a = self.trajs[self.traj_as[idx], :, :]
        traj_b = self.trajs[self.traj_bs[idx], :, :]

        data = {
            'traj_a': torch.tensor(traj_a, dtype=torch.float32),
            'traj_b': torch.tensor(traj_b, dtype=torch.float32)
        }
        if self.preprocessed_nlcomps:
            data['nlcomp'] = torch.tensor(self.unique_nlcomps[self.nlcomps[idx]])
        else:
            data['nlcomp_tokens'] = torch.tensor(self.unique_nlcomps_tokens[self.nlcomps[idx]],  dtype=torch.long)
            data['attention_mask'] = torch.tensor(self.unique_nlcomps_attention_masks[self.nlcomps[idx]], dtype=torch.long)

        return data
