import numpy as np
import os
import torch
import argparse
import json

from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from lang_pref_learning.feature_learning.utils import LANG_MODEL_NAME, LANG_OUTPUT_DIM
from lang_pref_learning.model.encoder import NLTrajAutoencoder
from lang_pref_learning.model_analysis.utils import get_traj_lang_embeds

from data.utils import gt_reward, speed, height, distance_to_cube, distance_to_bottle
from data.utils import RS_STATE_OBS_DIM, RS_ACTION_DIM, RS_PROPRIO_STATE_DIM, RS_OBJECT_STATE_DIM
from data.utils import WidowX_STATE_OBS_DIM, WidowX_ACTION_DIM, WidowX_PROPRIO_STATE_DIM, WidowX_OBJECT_STATE_DIM



def get_nearest_embed_distance(embed, lang_embed, embeds, index=None):
    """
        Get the nearest embedding to embed from embeds
        Input:
            embed: the embedding to compare to
            embeds: the list of embeddings to compare against
        Output:
            the index of the nearest embedding in embeds
    """
    new_embed = embed + lang_embed
    norm = np.linalg.norm(embeds - new_embed, axis=1)
    if index:
        norm = np.delete(norm, index)
    return np.argmin(norm)


def get_nearest_embed_cosine(embed, lang_embed, embeds, index=None):
    """
        Get the nearest embedding to embed from embeds
        Input:
            embed: the embedding to compare to
            embeds: the list of embeddings to compare against
        Output:
            the index of the nearest embedding in embeds
    """
    cos_sim = np.dot(embeds, embed + lang_embed) / (
            np.linalg.norm(embeds, axis=1) * np.linalg.norm(embed + lang_embed))
    if index:
        cos_sim = np.delete(cos_sim, index)
    return np.argmax(cos_sim)


def get_nearest_embed_project(embed, lang_embed, embeds, index=None):
    """
        Get the nearest embedding based on the projection
        Input:
            embed: the embedding to compare to
            embeds: the list of embeddings to compare against
        Output:
            the index of the nearest embedding in embeds
    """
    new_embed = embed + lang_embed
    embeds_norm = np.linalg.norm(embeds, axis=1)
    proj = np.dot(embeds, new_embed) / embeds_norm
    if index:
        proj = np.delete(proj, index)
    return np.argmax(proj)
    


def get_feature_class(nlcomp, classified_nlcomps):
    if nlcomp in classified_nlcomps['gt_reward']:
        value_func = gt_reward
        feature_name = "gt_reward"
    elif nlcomp in classified_nlcomps['speed']:
        value_func = speed
        feature_name = "speed"
    elif nlcomp in classified_nlcomps['height']:
        value_func = height
        feature_name = "height"
    elif nlcomp in classified_nlcomps['distance_to_cube']:
        value_func = distance_to_cube
        feature_name = "distance_to_cube"
    elif nlcomp in classified_nlcomps['distance_to_bottle']:
        value_func = distance_to_bottle
        feature_name = "distance_to_bottle"
    else:
        raise ValueError(f"NL comparison {nlcomp} not found in classified NL comparisons")
    return value_func, feature_name


def main(model_dir, data_dir, use_bert_encoder, bert_model, encoder_hidden_dim, decoder_hidden_dim, preprocessed_nlcomps,
         old_model=False, traj_encoder='mlp', debug=False):
    # Load the val trajectories and language comparisons first
    trajs = np.load(f'{data_dir}/test/trajs.npy')
    nlcomps = json.load(open(f'{data_dir}/test/unique_nlcomps.json', 'rb'))
    nlcomps_bert_embeds = np.load(f'{data_dir}/test/unique_nlcomps_{bert_model}.npy')
    classified_nlcomps = json.load(open(f'data/classified_nlcomps.json', 'rb'))
    greater_nlcomps = json.load(open(f'data/greater_nlcomps.json', 'rb'))
    less_nlcomps = json.load(open(f'data/less_nlcomps.json', 'rb'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    if args.use_bert_encoder:
        if 't5' in args.lang_model:
            lang_encoder = T5EncoderModel.from_pretrained(args.lang_model)
        else:
            lang_encoder = AutoModel.from_pretrained(LANG_MODEL_NAME[args.lang_model])

        tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME[args.lang_model])
        feature_dim = LANG_OUTPUT_DIM[args.lang_model]
    else:
        lang_encoder = None
        tokenizer = None
        feature_dim = 128

    if args.env == "robosuite":
        STATE_OBS_DIM = RS_STATE_OBS_DIM
        ACTION_DIM = RS_ACTION_DIM
        PROPRIO_STATE_DIM = RS_PROPRIO_STATE_DIM
        OBJECT_STATE_DIM = RS_OBJECT_STATE_DIM
    elif args.env == "widowx":
        STATE_OBS_DIM = WidowX_STATE_OBS_DIM
        ACTION_DIM = WidowX_ACTION_DIM
        PROPRIO_STATE_DIM = WidowX_PROPRIO_STATE_DIM
        OBJECT_STATE_DIM = WidowX_OBJECT_STATE_DIM
    elif args.env == "metaworld":
        # TODO: fill in the dimensions for metaworld
        pass
    else:
        raise ValueError("Invalid environment")

    model = NLTrajAutoencoder(STATE_OBS_DIM, ACTION_DIM, PROPRIO_STATE_DIM, OBJECT_STATE_DIM,
                              encoder_hidden_dim=encoder_hidden_dim, feature_dim=feature_dim,
                              decoder_hidden_dim=decoder_hidden_dim, lang_encoder=lang_encoder,
                              preprocessed_nlcomps=preprocessed_nlcomps, bert_output_dim=LANG_OUTPUT_DIM[bert_model],
                              use_bert_encoder=use_bert_encoder, traj_encoder=traj_encoder)

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

    # Get embeddings for the trajectories and language comparisons
    traj_embeds, lang_embeds = get_traj_lang_embeds(trajs, nlcomps, model, device, use_bert_encoder, tokenizer,
                                                    nlcomps_bert_embeds)

    # Get the nearest trajectory embedding given the language embedding
    total = 0
    correct = 0
    improvements = []
    for traj_idx in range(len(trajs)):
        traj = trajs[traj_idx]
        traj_embed = traj_embeds[traj_idx]
        for i, lang_embed in enumerate(lang_embeds):
            nlcomp = nlcomps[i]
            # Get the feature class and values for the trajectories
            value_func, feature_name = get_feature_class(nlcomp, classified_nlcomps)
            nearest_traj_idx = get_nearest_embed_cosine(traj_embed, lang_embed, traj_embeds)
            nearest_traj = trajs[nearest_traj_idx]
            traj1_feature_values = [value_func(traj[t]) for t in range(500)]
            traj2_feature_values = [value_func(nearest_traj[t]) for t in range(500)]

            # If the language comparison is greater, then the feature value of traj2 should be greater than traj1
            # vice versa for less
            if nlcomp in greater_nlcomps[feature_name]:
                correct += np.mean(traj1_feature_values) <= np.mean(traj2_feature_values)
                greater = True
                improvement = (np.mean(traj2_feature_values) - np.mean(traj1_feature_values)) / np.mean(
                    traj1_feature_values)
            elif nlcomp in less_nlcomps[feature_name]:
                correct += np.mean(traj1_feature_values) >= np.mean(traj2_feature_values)
                greater = False
                improvement = (np.mean(traj1_feature_values) - np.mean(traj2_feature_values)) / np.mean(
                    traj1_feature_values)
            else:
                print(nlcomp)
                raise ValueError(f"{nlcomp} not found in greater or less NL comparisons")
            total += 1
            if debug:
                print(
                    f"{nlcomp}, {greater}\n{feature_name}, traj1: {np.mean(traj1_feature_values)}, traj2: {np.mean(traj2_feature_values)}, {correct}")

            # how much the trajectory gets improved on the feature
            improvements.append(improvement)

    print(f"Correct: {correct}/{total} ({correct / total}), Avg. Improvement: {np.mean(improvements)}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model-dir', type=str, default='exp/linear_bert-mini')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--use-bert-encoder', action='store_true')
    parser.add_argument('--bert-model', type=str, default='bert-base-uncased')
    parser.add_argument('--encoder-hidden-dim', type=int, default=128)
    parser.add_argument('--decoder-hidden-dim', type=int, default=128)
    parser.add_argument('--preprocessed-nlcomps', action='store_true')
    parser.add_argument('--old-model', action='store_true')
    parser.add_argument('--traj_encoder', type=str, default='mlp')
    args = parser.parse_args()
    main(args.model_dir, args.data_dir, args.use_bert_encoder, args.bert_model,
         args.encoder_hidden_dim, args.decoder_hidden_dim,
         args.preprocessed_nlcomps, args.old_model, traj_encoder=args.traj_encoder)
