import numpy as np
import torch
import torch.nn as nn

from lang_pref_learning.model.transformer import TransformerEncoder
from lang_pref_learning.model.lstm import LSTMEncoder
from lang_pref_learning.model.cnn import CNNEncoder
from lang_pref_learning.model.visual_mlp import VisualMLP

STATE_DIM = 65
ACTION_DIM = 4  # NOTE: we use OSC_POSITION as our controller


class NLTrajEncoder(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim=128,
        remove_lang_encoder_hidden=False,
        preprocessed_nlcomps=False,
        **kwargs,
    ):
        super().__init__()

        # Use a transformer to encode the NL comparison.
        self.embedding = nn.Embedding(STATE_DIM + ACTION_DIM, encoder_hidden_dim)
        self.position_encoding = nn.Embedding(500, encoder_hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=encoder_hidden_dim, nhead=4),
            num_layers=4,
        )


class NLTrajAutoencoder(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim=128,
        feature_dim=256,
        decoder_hidden_dim=128,
        bert_output_dim=768,
        lang_encoder=None,
        preprocessed_nlcomps=False,
        use_bert_encoder=False,
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
        if traj_encoder == "mlp":
            self.traj_encoder = nn.Sequential(
                nn.Linear(in_features=STATE_DIM + ACTION_DIM, out_features=encoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=encoder_hidden_dim, out_features=feature_dim),
            )
        elif traj_encoder == "cnn":
            if use_visual_features:
                self.traj_encoder = VisualMLP(
                    in_feature_channel=visual_feature_dim // n_frames if use_stack_img_obs else visual_feature_dim,
                    action_dim=ACTION_DIM,
                    hidden_dim=encoder_hidden_dim,
                    out_dim=feature_dim,
                )
            else:
                self.traj_encoder = CNNEncoder(
                    in_channels=3 if not use_stack_img_obs else 3 * n_frames,
                    action_dim=ACTION_DIM,
                    hidden_dim=encoder_hidden_dim,
                    output_dim=feature_dim,
                )
        elif traj_encoder == "lstm":
            self.traj_encoder = LSTMEncoder(
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                hidden_dim=encoder_hidden_dim,
                output_dim=feature_dim,
            )
        else:
            raise ValueError(f"Trajectory encoder {traj_encoder} not found")

        self.traj_decoder = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=decoder_hidden_dim, out_features=STATE_DIM + ACTION_DIM),
        )

        self.preprocessed_nlcomps = preprocessed_nlcomps
        # Note: the first language encoder layer is BERT.
        self.use_bert_encoder = use_bert_encoder
        if use_bert_encoder:
            assert lang_encoder is not None
            self.lang_encoder = lang_encoder
        else:
            self.lang_encoder = nn.Sequential(
                nn.Linear(in_features=bert_output_dim, out_features=encoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=encoder_hidden_dim, out_features=feature_dim),
            )

    # Input is a tuple with (trajectory_a, trajectory_b, language)
    # traj_a has shape (n_trajs, n_timesteps, state+action)
    def forward(self, inputs):
        # NOTE: traj_a is the reference, traj_b is the updated
        if self.traj_encoder_cls == "cnn":
            traj_a, actions_a = inputs["traj_a_img_obs"], inputs["actions_a"]
            traj_b, actions_b = inputs["traj_b_img_obs"], inputs["actions_b"]
        else:
            traj_a = inputs["traj_a"]
            traj_b = inputs["traj_b"]

        # Encode trajectories
        if self.traj_encoder_cls == "cnn":
            actions_a = inputs["actions_a"]
            actions_b = inputs["actions_b"]
            encoded_traj_a = self.traj_encoder(traj_a, actions_a)
            encoded_traj_b = self.traj_encoder(traj_b, actions_b)
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
        if self.use_bert_encoder:
            lang_tokens = inputs["nlcomp_tokens"]
            lang_attention_mask = inputs["attention_mask"]
            bert_outputs = self.lang_encoder(lang_tokens, attention_mask=lang_attention_mask)
            bert_embeddings = bert_outputs.last_hidden_state
            encoded_lang = torch.mean(bert_embeddings, dim=1, keepdim=False)
        else:
            lang_embeds = inputs["nlcomp"]
            encoded_lang = self.lang_encoder(lang_embeds)

        # NOTE: traj_a is the reference, traj_b is the updated
        if not self.traj_encoder_cls == "cnn":
            decoded_traj_a = self.traj_decoder(encoded_traj_a)
            decoded_traj_b = self.traj_decoder(encoded_traj_b)
        else:
            decoded_traj_a, decoded_traj_b = None, None

        output = (
            encoded_traj_a,
            encoded_traj_b,
            encoded_lang,
            decoded_traj_a,
            decoded_traj_b,
        )
        return output
