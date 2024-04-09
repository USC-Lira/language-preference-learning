import torch
from torch.utils.data import Dataset
import json
import numpy as np

from einops import rearrange


# nlcomp_file is a json file with the list of comparisons in NL.
# traj_a_file is a .npy or .npz file with the first trajectory and has a shape of (n_trajectories, n_timesteps, STATE_DIM+ACTION_DIM)
# traj_b_file is a .npy or .npz file with the second trajectory and has a shape of (n_trajectories, n_timesteps, STATE_DIM+ACTION_DIM)
# If id_mapped is true, traj_*_file and nlcomp_file are instead .npy files with INDICES corresponding to traj_file and unique_nlcomp_file.
class NLTrajComparisonDataset(Dataset):
    def __init__(
        self,
        nlcomp_file,
        traj_a_file,
        traj_b_file,
        seq_len=64,
        tokenizer=None,
        preprocessed_nlcomps=False,
        unique_nlcomp_file=None,
        traj_file=None,
        use_img_obs=False,
        img_obs_file=None,
        action_file=None,
        use_visual_features=False,
        resample=False,
        resample_factor=1.0,
        device="cpu",
    ):

        assert unique_nlcomp_file is not None
        assert traj_file is not None

        # These are actually just indexes. (So we don't need to mmap them.)
        self.trajs = np.load(traj_file)
        self.nlcomps = np.load(nlcomp_file)
        self.traj_as = np.load(traj_a_file)
        self.traj_bs = np.load(traj_b_file)
        if use_img_obs:
            self.img_observations = np.load(img_obs_file)
            self.actions = np.load(action_file)

            # self.img_observations = self.img_observations[:, :seq_len]
            # self.actions = self.actions[:, :seq_len]

            # if resample:
            #     resample_frames = int(1 / resample_factor)
            #     self.img_observations = self.img_observations[:, ::resample_frames]
            #     self.actions = self.actions[:, ::resample_frames]

            # assert self.img_observations.shape[1] == int(
            #     seq_len * resample_factor
            # ), f"Image shape: {self.img_observations.shape}, expected: {seq_len * resample_factor}"

            # assert self.actions.shape[1] == int(seq_len * resample_factor)

            if not use_visual_features:
                self.img_observations = rearrange(self.img_observations, "b t h w c -> b t c h w")

        self.max_len = seq_len
        self.preprocessed_nlcomps = preprocessed_nlcomps
        self.use_img_obs = use_img_obs

        self.unique_nlcomps = None
        self.unique_nlcomps_tokens = None
        self.unique_nlcomps_attention_masks = None

        # These are the sentences and trajectories corresponding to the indexes.
        if preprocessed_nlcomps:
            self.unique_nlcomps = np.load(unique_nlcomp_file)
        else:
            assert (
                tokenizer is not None
            ), "Must provide tokenizer if not using preprocessed_nlcomps."
            self.tokenizer = tokenizer
            with open(unique_nlcomp_file, "rb") as f:
                nlcomps = json.load(f)
            self.unique_nlcomps_tokens = []
            self.unique_nlcomps_attention_masks = []
            for nlcomp in nlcomps:
                tokens = self.tokenizer.tokenize(
                    self.tokenizer.cls_token + " " + nlcomp + " " + self.tokenizer.sep_token
                )
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

        # transform to torch tensors and move to device
        self.trajs = torch.tensor(self.trajs, dtype=torch.float32).to(device)
        self.unique_nlcomps_tokens = torch.tensor(self.unique_nlcomps_tokens, dtype=torch.long).to(
            device
        )
        self.unique_nlcomps_attention_masks = torch.tensor(
            self.unique_nlcomps_attention_masks, dtype=torch.long
        ).to(device)

        if self.use_img_obs:
            self.img_observations = torch.tensor(self.img_observations, dtype=torch.float32).to(
                device
            )
            self.actions = torch.tensor(self.actions, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.nlcomps)

    def __getitem__(self, idx):
        traj_a = self.trajs[self.traj_as[idx], :, :]
        traj_b = self.trajs[self.traj_bs[idx], :, :]

        data = {
            "traj_a": traj_a,
            "traj_b": traj_b,
        }

        if self.use_img_obs:
            data["traj_a_img_obs"] = self.img_observations[self.traj_as[idx]]
            data["traj_b_img_obs"] = self.img_observations[self.traj_bs[idx]]
            data["actions_a"] = self.actions[self.traj_as[idx]]
            data["actions_b"] = self.actions[self.traj_bs[idx]]

        if self.preprocessed_nlcomps:
            data["nlcomp"] = torch.tensor(self.unique_nlcomps[self.nlcomps[idx]])
        else:
            data["nlcomp_tokens"] = self.unique_nlcomps_tokens[self.nlcomps[idx]]
            data["attention_mask"] = self.unique_nlcomps_attention_masks[self.nlcomps[idx]]

        return data
