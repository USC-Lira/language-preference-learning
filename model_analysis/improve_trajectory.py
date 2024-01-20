import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import torch
import os
from transformers import AutoModel, AutoTokenizer

from feature_learning.utils import BERT_MODEL_NAME, BERT_OUTPUT_DIM
from data.utils import gt_reward, speed, height, distance_to_cube, distance_to_bottle
from model_analysis.find_nearest_traj import get_nearest_embed_cosine
from feature_learning.model import NLTrajAutoencoder
from model_analysis.utils import get_traj_lang_embeds


def initialize_reward(num_features):
    # Randomly initialize weight for each feature
    # All weights sum to 1
    reward_func = np.random.rand(num_features)
    reward_func /= np.sum(reward_func)
    # # Randomly assign + or - to each weight， + means greater, - means less
    # reward_func = np.random.choice([-1, 1], size=num_features) * reward_func
    return reward_func


def get_feature_value(traj):
    # Get feature values for each timestep
    feature_values = [
        [gt_reward(state), speed(state), height(state), distance_to_cube(state), distance_to_bottle(state)]
        for state in traj]
    return np.mean(feature_values, axis=0)


def get_best_lang(optimal_traj_embed, curr_traj_embed, lang_embeds, softmax=False):
    if softmax:
        cos_sim = np.dot(lang_embeds, optimal_traj_embed - curr_traj_embed) / (
                np.linalg.norm(lang_embeds, axis=1) * np.linalg.norm(optimal_traj_embed - curr_traj_embed))
        cos_sim = torch.from_numpy(cos_sim)
        probs = torch.softmax(cos_sim, dim=0)
        # add some noise to the probabilities
        probs = probs + torch.randn_like(probs) * 1e-5
        # sample a language comparison with the probabilities
        idx = torch.multinomial(probs, 1).item()
    else:
        dot_product = np.dot(lang_embeds, optimal_traj_embed - curr_traj_embed)
        idx = np.argmax(dot_product)
    return lang_embeds[idx], idx


def improve_trajectory(args):
    trajs = np.load('data/dataset/val/trajs.npy')
    nlcomps = json.load(open(f'data/dataset/val/unique_nlcomps_{args.bert_model}.json', 'rb'))
    nlcomps_bert_embeds = np.load(f'data/dataset/val/unique_nlcomps_{args.bert_model}.npy')

    # Load the model
    if args.use_bert_encoder:
        lang_encoder = AutoModel.from_pretrained(BERT_MODEL_NAME[args.bert_model])
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME[args.bert_model])
        feature_dim = BERT_OUTPUT_DIM[args.bert_model]
    else:
        lang_encoder = None
        tokenizer = None
        feature_dim = 16

    model = NLTrajAutoencoder(encoder_hidden_dim=args.encoder_hidden_dim, feature_dim=feature_dim,
                              decoder_hidden_dim=args.decoder_hidden_dim, lang_encoder=lang_encoder,
                              preprocessed_nlcomps=args.preprocessed_nlcomps,
                              bert_output_dim=BERT_OUTPUT_DIM[args.bert_model],
                              use_bert_encoder=args.use_bert_encoder)

    state_dict = torch.load(os.path.join(args.model_dir, 'best_model_state_dict.pth'))

    # Compatibility with old model
    if args.old_model:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('_hidden_layer', '.0')
            new_k = new_k.replace('_output_layer', '.2')
            new_state_dict[new_k] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Get embeddings for the trajectories and language comparisons
    traj_embeds, lang_embeds = get_traj_lang_embeds(trajs, nlcomps, model, device, args.use_bert_encoder, tokenizer,
                                                    nlcomps_bert_embeds)

    # Randomly initialize a reward function
    reward_func = initialize_reward(5)

    # Find the optimal trajectory given the reward function
    feature_values = np.array([get_feature_value(traj) for traj in trajs])
    # Normalize feature values
    feature_values = (feature_values - np.min(feature_values, axis=0)) / (
                np.max(feature_values, axis=0) - np.min(feature_values, axis=0))

    # Randomly select some features and set their values to 1 - original value
    less_idx = np.random.choice(5, size=2, replace=False)
    for i in less_idx:
        feature_values[:, i] = 1 - feature_values[:, i]

    reward_values = np.dot(feature_values, reward_func)
    optimal_traj_idx = np.argmax(reward_values)
    optimal_traj_value = reward_values[optimal_traj_idx]
    curr_traj_idx = np.argmin(reward_values)
    curr_traj_value = reward_values[curr_traj_idx]
    if args.debug:
        print(f'Optimal trajectory: {optimal_traj_idx}, optimal value: {optimal_traj_value}\n')
        print(f'Initial trajectory: {curr_traj_idx}, initial value: {curr_traj_value}\n')

    optimal_reached = False
    traj_values = [curr_traj_value]
    for i in range(args.iterations):
        if curr_traj_value == optimal_traj_value:
            optimal_reached = True
            traj_values.extend([optimal_traj_value for _ in range(args.iterations - i)])
            break

        # Find the nearest language embedding to the optimal trajectory embedding
        lang_embed, lang_idx = get_best_lang(traj_embeds[optimal_traj_idx], traj_embeds[curr_traj_idx], lang_embeds)
        nlcomp = nlcomps[lang_idx]
        next_traj_idx = get_nearest_embed_cosine(traj_embeds[curr_traj_idx], lang_embed, traj_embeds)
        next_traj_value = reward_values[next_traj_idx]
        traj_values.append(next_traj_value)

        curr_traj_idx = next_traj_idx
        curr_traj_value = next_traj_value

        if args.debug:
            print(f'Iteration {i}')
            print(f'Current trajectory: {curr_traj_idx}, current value: {curr_traj_value}')
            print(f'Language comparison: {nlcomp}\n')

    return optimal_reached, optimal_traj_value, traj_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model-dir', type=str, default='exp/linear_bert-mini')
    parser.add_argument('--use-bert-encoder', action='store_true')
    parser.add_argument('--bert-model', type=str, default='bert-base')
    parser.add_argument('--encoder-hidden-dim', type=int, default=128)
    parser.add_argument('--decoder-hidden-dim', type=int, default=128)
    parser.add_argument('--preprocessed-nlcomps', action='store_true')
    parser.add_argument('--old-model', action='store_true')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    optimal_traj_values = []
    all_traj_values = []
    for _ in range(100):
        optimal_reached, optimal_traj_value, traj_values = improve_trajectory(args)
        optimal_traj_values.append(optimal_traj_value)
        all_traj_values.append(traj_values)

    all_traj_values = np.array(all_traj_values)
    np.save('model_analysis/all_traj_values_argmax.npy', all_traj_values)
    optimal_traj_values = np.array(optimal_traj_values)
    np.save('model_analysis/optimal_traj_values_argmax.npy', optimal_traj_values)
    plt.plot(np.mean(all_traj_values, axis=0), label='Current Trajectory')
    # Draw optimal trajecotry values as a dashed horizontal line
    plt.plot([0, args.iterations], [np.mean(optimal_traj_values), np.mean(optimal_traj_values)], 'k--', label='Optimal Trajectory')
    plt.xlabel('Iteration')
    plt.ylabel('Avg. Reward')
    plt.title('Average Reward vs. Iteration (Val Set)')
    plt.legend()

    plt.savefig('model_analysis/improve_traj.png')
    plt.show()
