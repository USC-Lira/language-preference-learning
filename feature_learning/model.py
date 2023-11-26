import numpy as np
import torch
import torch.nn as nn

STATE_DIM = 65
ACTION_DIM = 4  # NOTE: we use OSC_POSITION as our controller
BERT_OUTPUT_DIM = 768


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=500):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, d_model)


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
    def __init__(self, encoder_hidden_dim=128, decoder_hidden_dim=128, remove_lang_encoder_hidden=False,
                 preprocessed_nlcomps=False, **kwargs):
        super().__init__()
        # TODO: can later make encoders and decoders transformers
        self.traj_encoder_hidden_layer = nn.Linear(
            in_features=STATE_DIM + ACTION_DIM, out_features=encoder_hidden_dim
        )
        self.traj_encoder_output_layer = nn.Linear(
            in_features=encoder_hidden_dim, out_features=16
        )
        self.traj_decoder_hidden_layer = nn.Linear(
            in_features=16, out_features=decoder_hidden_dim
        )
        self.traj_decoder_output_layer = nn.Linear(
            in_features=decoder_hidden_dim, out_features=STATE_DIM + ACTION_DIM
        )

        self.preprocessed_nlcomps = preprocessed_nlcomps
        # Note: the first language encoder layer is BERT.
        self.remove_lang_encoder_hidden = remove_lang_encoder_hidden
        if self.remove_lang_encoder_hidden:
            self.lang_encoder_layer = nn.Linear(
                in_features=BERT_OUTPUT_DIM, out_features=16
            )
        else:
            self.lang_encoder_hidden_layer = nn.Linear(
                in_features=BERT_OUTPUT_DIM, out_features=encoder_hidden_dim
            )
            self.lang_encoder_output_layer = nn.Linear(
                in_features=encoder_hidden_dim, out_features=16
            )
        self.lang_decoder_output_layer = None  # TODO: implement language decoder later.

    # Input is a tuple with (trajectory_a, trajectory_b, language)
    # traj_a has shape (n_trajs, n_timesteps, state+action)
    def forward(self, inputs):
        traj_a = inputs[0]
        traj_b = inputs[1]
        lang = inputs[2]

        # Encode trajectories
        encoded_traj_a = self.traj_encoder_output_layer(torch.relu(self.traj_encoder_hidden_layer(traj_a)))
        encoded_traj_b = self.traj_encoder_output_layer(torch.relu(self.traj_encoder_hidden_layer(traj_b)))
        # Take the mean over timesteps
        encoded_traj_a = torch.mean(encoded_traj_a, dim=-2)
        encoded_traj_b = torch.mean(encoded_traj_b, dim=-2)

        bert_output_embeddings = lang

        # Encode the language
        if self.remove_lang_encoder_hidden:
            encoded_lang = torch.relu(self.lang_encoder_layer(bert_output_embeddings))
        else:
            encoded_lang = self.lang_encoder_output_layer(
                torch.relu(self.lang_encoder_hidden_layer(bert_output_embeddings)))

        # NOTE: traj_a is the reference, traj_b is the updated

        decoded_traj_a = self.traj_decoder_output_layer(torch.relu(self.traj_decoder_hidden_layer(encoded_traj_a)))
        decoded_traj_b = self.traj_decoder_output_layer(torch.relu(self.traj_decoder_hidden_layer(encoded_traj_b)))

        output = (encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b)
        return output
