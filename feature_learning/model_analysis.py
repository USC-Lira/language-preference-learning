import numpy as np
import os
import torch
import argparse
import json

from transformers import AutoModel, AutoTokenizer
from feature_learning.utils import BERT_MODEL_NAME, BERT_OUTPUT_DIM
from feature_learning.model import NLTrajAutoencoder
from data.utils import gt_reward, speed, height, distance_to_cube, distance_to_bottle


def get_nearest_embed_distance(embed, embeds):
    """
        Get the nearest embedding to embed from embeds
        Input:
            embed: the embedding to compare to
            embeds: the list of embeddings to compare against
        Output:
            the index of the nearest embedding in embeds
    """
    return np.argmin(np.linalg.norm(embeds - embed, axis=1))


def get_nearest_embed_cosine(embed, lang_embed, embeds):
    """
        Get the nearest embedding to embed from embeds
        Input:
            embed: the embedding to compare to
            embeds: the list of embeddings to compare against
        Output:
            the index of the nearest embedding in embeds
    """
    return np.argmax(np.dot(embeds - embed, lang_embed) / (np.linalg.norm(embeds - embed + 1e-5, axis=1) * np.linalg.norm(lang_embed)))


def main(model_dir, use_bert_encoder, bert_model, encoder_hidden_dim, decoder_hidden_dim, preprocessed_nlcomps,
         old_model=False):
    # Load the val trajectories and language comparisons first
    trajs = np.load('data/dataset/val/trajs.npy')
    nlcomps = json.load(open(f'data/dataset/val/unique_nlcomps_{bert_model}.json', 'rb'))
    nlcomps_bert_embeds = np.load(f'data/dataset/val/unique_nlcomps_{bert_model}.npy')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    if use_bert_encoder:
        lang_encoder = AutoModel.from_pretrained(BERT_MODEL_NAME[bert_model])
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME[bert_model])
        feature_dim = BERT_OUTPUT_DIM[bert_model]
    else:
        lang_encoder = None
        tokenizer = None
        feature_dim = 16
    model = NLTrajAutoencoder(encoder_hidden_dim=encoder_hidden_dim, feature_dim=feature_dim,
                              decoder_hidden_dim=decoder_hidden_dim, lang_encoder=lang_encoder,
                              preprocessed_nlcomps=preprocessed_nlcomps, bert_output_dim=BERT_OUTPUT_DIM[bert_model],
                              use_bert_encoder=use_bert_encoder)

    state_dict = torch.load(os.path.join(model_dir, 'best_model_state_dict.pth'))

    # Compatibility with old model
    if old_model:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('_hidden_layer', '.0')
            new_k = new_k.replace('_output_layer', '.2')
            new_state_dict[new_k] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Get all the embeddings for the trajectories
    encoded_trajs = model.traj_encoder(torch.from_numpy(trajs).float().to(device)).detach().cpu().numpy()
    traj_embeds = np.mean(encoded_trajs, axis=-2, keepdims=False)

    # Get the nearest trajectory embedding for each language comparison
    traj = trajs[0]
    traj_embed = traj_embeds[0]

    value_func = {
        "gt_reward": gt_reward,
        "speed": speed,
        "height": height,
        "distance_to_bottle": distance_to_bottle,
        "distance_to_cube": distance_to_cube
    }

    lang_embeds = []
    if use_bert_encoder:
        for i, nlcomp in enumerate(nlcomps):
            # First tokenize the NL comparison and get the embedding
            tokens = tokenizer.tokenize(tokenizer.cls_token + " " + nlcomp + " " + tokenizer.sep_token)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            # Pad sequences to the common length
            padding_length = 64 - len(token_ids)
            # Create attention mask
            attention_mask = [1] * len(token_ids) + [0] * padding_length
            token_ids += [tokenizer.pad_token_id] * padding_length
            token_ids = torch.from_numpy(np.asarray(token_ids)).unsqueeze(0).to(device)
            attention_mask = torch.from_numpy(np.asarray(attention_mask)).unsqueeze(0).to(device)
            lang_embed = model.lang_encoder(token_ids, attention_mask=attention_mask).detach().cpu().numpy()
            lang_embeds.append(lang_embed)
    else:
        for i, nlcomp_bert_embed in enumerate(nlcomps_bert_embeds):
            lang_embed = model.lang_encoder(
                torch.from_numpy(nlcomp_bert_embed).float().to(device)).detach().cpu().numpy()
            lang_embeds.append(lang_embed)

    # Get the nearest trajectory embedding given the language embedding
    for lang_embed in lang_embeds:
        nearest_traj_idx = get_nearest_embed_distance(traj_embed + lang_embed, traj_embeds)
        nearest_traj = trajs[nearest_traj_idx]
        traj1_feature_values = [[value_func[feature_name](traj[t]) for t in range(500)]
                                for feature_name in value_func.keys()]
        traj2_feature_values = [[value_func[feature_name](nearest_traj[t]) for t in range(500)]
                                for feature_name in value_func.keys()]

        print(f"GT reward: {traj1_feature_values[0], traj2_feature_values[0]},"
              f"Speed: {traj1_feature_values[1], traj2_feature_values[1]},",
              f"Height: {traj1_feature_values[1], traj2_feature_values[2]},"
              f"Distance to cube: {traj1_feature_values[3], traj2_feature_values[3]},"
              f"Distance to bottle: {traj1_feature_values[4], traj2_feature_values[4]},")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model-dir', type=str, default='exp/linear_bert-mini')
    parser.add_argument('--use-bert-encoder', action='store_true')
    parser.add_argument('--bert-model', type=str, default='bert-base-uncased')
    parser.add_argument('--encoder-hidden-dim', type=int, default=128)
    parser.add_argument('--decoder-hidden-dim', type=int, default=128)
    parser.add_argument('--preprocessed-nlcomps', action='store_true')
    parser.add_argument('--old-model', action='store_true')
    args = parser.parse_args()
    main(args.model_dir, args.use_bert_encoder, args.bert_model, args.encoder_hidden_dim, args.decoder_hidden_dim,
         args.preprocessed_nlcomps, args.old_model)
