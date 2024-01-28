import numpy as np
import torch
import torch.nn as nn

from feature_learning.transformer import TransformerEncoder
from feature_learning.lstm import LSTMEncoder

STATE_DIM = 65
ACTION_DIM = 4  # NOTE: we use OSC_POSITION as our controller


class NLTrajEncoder(nn.Module):
    def __init__(self, encoder_hidden_dim=128, remove_lang_encoder_hidden=False, preprocessed_nlcomps=False, **kwargs):
        super().__init__()

        # Use a transformer to encode the NL comparison.
        self.embedding = nn.Embedding(STATE_DIM + ACTION_DIM, encoder_hidden_dim)
        self.position_encoding = nn.Embedding(500, encoder_hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=encoder_hidden_dim, nhead=4),
            num_layers=4
        )


class NLTrajAutoencoder(nn.Module):
    def __init__(self, encoder_hidden_dim=128, feature_dim=256, decoder_hidden_dim=128,
                 bert_output_dim=768, lang_encoder=None, preprocessed_nlcomps=False,
                 use_bert_encoder=False, traj_encoder='mlp', use_cnn_in_transformer=False,
                 use_casual_attention=False, use_cls_token=False):
        super().__init__()
        # TODO: can later make encoders and decoders transformers
        self.traj_encoder_cls = traj_encoder
        self.use_cls_token = use_cls_token
        if traj_encoder == 'transformer':
            self.traj_encoder = TransformerEncoder(
                input_size=STATE_DIM + ACTION_DIM, d_model=encoder_hidden_dim, nhead=4, d_hid=encoder_hidden_dim,
                nlayers=2, d_ff=feature_dim, dropout=0.1, use_cnn_in_transformer=use_cnn_in_transformer,
                use_casual_attention=use_casual_attention, use_cls_token=use_cls_token
            )
        elif traj_encoder == 'mlp':
            self.traj_encoder = nn.Sequential(
                nn.Linear(in_features=STATE_DIM + ACTION_DIM, out_features=encoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=encoder_hidden_dim, out_features=feature_dim),
            )
        elif traj_encoder == 'lstm':
            self.traj_encoder = LSTMEncoder(
                state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=encoder_hidden_dim, output_dim=feature_dim
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
        traj_a = inputs['traj_a']
        traj_b = inputs['traj_b']

        # Encode trajectories
        encoded_traj_a = self.traj_encoder(traj_a)
        encoded_traj_b = self.traj_encoder(traj_b)

        if self.traj_encoder_cls == 'transformer':
            if self.use_cls_token:
                # Use the CLS token as the embedding
                encoded_traj_a = encoded_traj_a[:, -1, :]
                encoded_traj_b = encoded_traj_b[:, -1, :]
            else:
                encoded_traj_a = torch.mean(encoded_traj_a, dim=-2)
                encoded_traj_b = torch.mean(encoded_traj_b, dim=-2)
        elif self.traj_encoder_cls == 'lstm':
            encoded_traj_a = encoded_traj_a
            encoded_traj_b = encoded_traj_b
        elif self.traj_encoder_cls == 'mlp':
            # Take the mean over timesteps
            encoded_traj_a = torch.mean(encoded_traj_a, dim=-2)
            encoded_traj_b = torch.mean(encoded_traj_b, dim=-2)
        else:
            raise ValueError(f"Trajectory encoder {self.traj_encoder} not found")

        # Encode the language
        if self.use_bert_encoder:
            lang_tokens = inputs['nlcomp_tokens']
            lang_attention_mask = inputs['attention_mask']
            bert_outputs = self.lang_encoder(lang_tokens, attention_mask=lang_attention_mask)
            bert_embeddings = bert_outputs.last_hidden_state
            encoded_lang = torch.mean(bert_embeddings, dim=1, keepdim=False)
        else:
            lang_embeds = inputs['nlcomp']
            encoded_lang = self.lang_encoder(lang_embeds)

        # NOTE: traj_a is the reference, traj_b is the updated
        decoded_traj_a = self.traj_decoder(encoded_traj_a)
        decoded_traj_b = self.traj_decoder(encoded_traj_b)

        output = (encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b)
        return output
