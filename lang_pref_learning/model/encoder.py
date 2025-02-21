import numpy as np
import torch
import torch.nn as nn

from lang_pref_learning.model.cnn import CNNEncoder
from lang_pref_learning.model.visual_mlp import VisualMLP


class NLTrajAutoencoder(nn.Module):
    def __init__(
        self,
        STATE_OBS_DIM,
        ACTION_DIM,
        PROPRIO_STATE_DIM,
        OBJECT_STATE_DIM,
        encoder_hidden_dim=128,
        feature_dim=256,
        decoder_hidden_dim=128,
        lang_embed_dim=768,
        lang_encoder=None,
        traj_encoder="mlp",
        use_stack_img_obs=False,
        n_frames=3,
        use_visual_features=False,
        visual_feature_dim=256,
        seq_len=500,
    ):
        super().__init__()
        # TODO: can later make encoders and decoders transformers
        self.traj_encoder_cls = traj_encoder
        self.use_visual_features = use_visual_features
        if traj_encoder == "mlp":
            self.traj_encoder = nn.Sequential(
                nn.Linear(in_features=STATE_OBS_DIM + ACTION_DIM, out_features=encoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=encoder_hidden_dim, out_features=feature_dim),
            )
        elif traj_encoder == "cnn":
            if use_visual_features:
                self.traj_encoder = VisualMLP(
                    visual_feature_dim=visual_feature_dim,
                    state_dim=PROPRIO_STATE_DIM + ACTION_DIM, 
                    hidden_dim=encoder_hidden_dim, 
                    out_dim=feature_dim,
                )
            else:
                self.traj_encoder = CNNEncoder(
                    in_channels=3 if not use_stack_img_obs else 3 * n_frames,
                    state_dim=PROPRIO_STATE_DIM + ACTION_DIM,
                    hidden_dim=encoder_hidden_dim,
                    output_dim=feature_dim,
                )
        else:
            raise ValueError(f"Trajectory encoder {traj_encoder} not found")

        self.traj_decoder = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=decoder_hidden_dim, out_features=STATE_OBS_DIM + ACTION_DIM),
        )

        assert lang_encoder is not None
        self.lang_encoder = lang_encoder

        # state_dim = proprio_state_dim + object_state_dim
        self.state_dim = STATE_OBS_DIM
        self.action_dim = ACTION_DIM
        self.proprio_state_dim = PROPRIO_STATE_DIM
        self.object_state_dim = OBJECT_STATE_DIM

    # Input is a tuple with (trajectory_a, trajectory_b, language)
    # traj_a has shape (n_trajs, n_timesteps, state+action)
    def forward(self, inputs, train=False):
        # NOTE: traj_a is the reference, traj_b is the updated
        traj_a = inputs["traj_a"]
        traj_b = inputs["traj_b"]

        # Encode trajectories
        if self.traj_encoder_cls == "cnn":
            inputs_a = {
                "states": inputs["traj_a"][:, :, self.object_state_dim: self.state_dim + self.action_dim],
                "img_obs": inputs["img_obs_a"],
            }
            inputs_b = {
                "states": inputs["traj_b"][:, :, self.object_state_dim: self.state_dim + self.action_dim],
                "img_obs": inputs["img_obs_b"],
            }
            encoded_traj_a = self.traj_encoder(inputs_a, train=train)
            encoded_traj_b = self.traj_encoder(inputs_b, train=train)
        else:
            encoded_traj_a = self.traj_encoder(traj_a)
            encoded_traj_b = self.traj_encoder(traj_b)

        if self.traj_encoder_cls == "lstm":
            encoded_traj_a = encoded_traj_a
            encoded_traj_b = encoded_traj_b

        elif self.traj_encoder_cls == "mlp" or self.traj_encoder_cls == "cnn":
            # Take the mean over timesteps
            encoded_traj_a = torch.mean(encoded_traj_a, dim=-2)
            encoded_traj_b = torch.mean(encoded_traj_b, dim=-2)

        else:
            raise ValueError(f"Trajectory encoder {self.traj_encoder} not found")

        # Encode the language
        lang_tokens = inputs["nlcomp_tokens"]
        lang_attention_mask = inputs["attention_mask"]
        bert_outputs = self.lang_encoder(lang_tokens, attention_mask=lang_attention_mask)
        bert_embeddings = bert_outputs.last_hidden_state
        encoded_lang = torch.mean(bert_embeddings, dim=1, keepdim=False)

        # NOTE: traj_a is the reference, traj_b is the updated
        # if not (self.traj_encoder_cls == "cnn" or self.traj_encoder_cls == "visual-transformer"):
        #     decoded_traj_a = self.traj_decoder(encoded_traj_a)
        #     decoded_traj_b = self.traj_decoder(encoded_traj_b)
        # else:
        decoded_traj_a, decoded_traj_b = None, None

        output = (
            encoded_traj_a,
            encoded_traj_b,
            encoded_lang,
            decoded_traj_a,
            decoded_traj_b,
        )
        return output
