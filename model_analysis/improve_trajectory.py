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
from model_analysis.utils import get_traj_lang_embeds, get_lang_embed


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


def get_lang_feedback(optimal_traj_feature_value, curr_traj_feature_value, reward_function, less_feature_idx,
                      greater_nlcomps, less_nlcomps, softmax=False):
    # Based on the reward function, determine which feature aspect to improve
    feature_diff = optimal_traj_feature_value - curr_traj_feature_value
    reward_diff = np.multiply(feature_diff, reward_function)
    if softmax:
        feature_probs = torch.softmax(torch.from_numpy(reward_diff), dim=0)
        feature_aspect = torch.multinomial(feature_probs, 1).item()
    else:
        feature_aspect = np.argmax(np.abs(reward_diff))

    feature_names = ["gt_reward", "speed", "height", "distance_to_cube", "distance_to_bottle"]
    if feature_aspect in less_feature_idx:
        # If the feature aspect is less than the optimal trajectory, then the language comparison should be greater
        nlcomp = np.random.choice(less_nlcomps[feature_names[feature_aspect]])
    else:
        # If the feature aspect is greater than the optimal trajectory, then the language comparison should be less
        nlcomp = np.random.choice(greater_nlcomps[feature_names[feature_aspect]])
    return nlcomp


def improve_trajectory(reward_func, feature_values, less_idx, greater_nlcomps, less_nlcomps, traj_embeds, lang_embeds,
                       model, device, tokenizer, lang_encoder, args, use_softmax=False):
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
        # lang_embed, lang_idx = get_best_lang(traj_embeds[optimal_traj_idx], traj_embeds[curr_traj_idx], lang_embeds)
        # nlcomp = nlcomps[lang_idx]
        nlcomp = get_lang_feedback(feature_values[optimal_traj_idx], feature_values[curr_traj_idx], reward_func,
                                   less_idx, greater_nlcomps, less_nlcomps, softmax=use_softmax)
        lang_embed = get_lang_embed(nlcomp, model, device, tokenizer, use_bert_encoder=args.use_bert_encoder,
                                    bert_model=lang_encoder)
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


def main(args):
    trajs = np.load(os.path.join(args.data_dir, 'test/trajs.npy'))
    nlcomps = json.load(open(os.path.join(args.data_dir, 'test/unique_nlcomps.json'), 'rb'))
    nlcomps_bert_embeds = np.load(os.path.join(args.data_dir, f'test/unique_nlcomps_{args.bert_model}.npy'))
    greater_nlcomps = json.load(open(os.path.join(args.data_dir, '../greater_nlcomps.json'), 'rb'))
    less_nlcomps = json.load(open(os.path.join(args.data_dir, '../less_nlcomps.json'), 'rb'))

    # Load the model
    if args.use_bert_encoder:
        lang_encoder = AutoModel.from_pretrained(BERT_MODEL_NAME[args.bert_model])
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME[args.bert_model])
        feature_dim = BERT_OUTPUT_DIM[args.bert_model]
    else:
        lang_encoder = AutoModel.from_pretrained(BERT_MODEL_NAME[args.bert_model])
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME[args.bert_model])
        feature_dim = 128

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

    # Get BERT embeddings for the language comparisons in greater_nlcomps and less_nlcomps
    greater_nlcomps_bert_embeds = {}
    less_nlcomps_bert_embeds = {}
    # for feature_name in greater_nlcomps:
    #     greater_nlcomps_bert_embeds[feature_name] = []
    #     for nlcomp in greater_nlcomps[feature_name]:
    #         inputs = tokenizer(nlcomp, return_tensors="pt")
    #         bert_output = lang_encoder(**inputs)
    #         embedding = bert_output.last_hidden_state
    #
    #         # Average across the sequence to get a sentence-level embedding
    #         embedding = torch.mean(embedding, dim=1, keepdim=False)
    #         greater_nlcomps_bert_embeds[feature_name].append(embedding.detach().cpu().numpy())
    #
    # for feature_name in less_nlcomps:
    #     less_nlcomps_bert_embeds[feature_name] = []
    #     for nlcomp in less_nlcomps[feature_name]:
    #         inputs = tokenizer(nlcomp, return_tensors="pt")
    #         bert_output = lang_encoder(**inputs)
    #         embedding = bert_output.last_hidden_state
    #
    #         # Average across the sequence to get a sentence-level embedding
    #         embedding = torch.mean(embedding, dim=1, keepdim=False)
    #         less_nlcomps_bert_embeds[feature_name].append(embedding.detach().cpu().numpy())

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

    optimal_reached, optimal_traj_value_argmax, traj_values_argmax = improve_trajectory(reward_func, feature_values, less_idx,
                                                                           greater_nlcomps, less_nlcomps, traj_embeds,
                                                                           lang_embeds, model, device, tokenizer,
                                                                           lang_encoder, args, use_softmax=False)
    optimal_reached, optimal_traj_value_softmax, traj_values_softmax = improve_trajectory(reward_func, feature_values, less_idx,
                                                                            greater_nlcomps, less_nlcomps, traj_embeds,
                                                                            lang_embeds, model, device, tokenizer,
                                                                            lang_encoder, args, use_softmax=True)

    return optimal_traj_value_softmax, traj_values_softmax, optimal_traj_value_argmax, traj_values_argmax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model-dir', type=str, default='exp/linear_bert-mini')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--use-bert-encoder', action='store_true')
    parser.add_argument('--bert-model', type=str, default='bert-base')
    parser.add_argument('--encoder-hidden-dim', type=int, default=128)
    parser.add_argument('--decoder-hidden-dim', type=int, default=128)
    parser.add_argument('--preprocessed-nlcomps', action='store_true')
    parser.add_argument('--old-model', action='store_true')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    exp_name = args.model_dir.split('/')[-1]
    optimal_traj_values_softmax = []
    all_traj_values_softmax = []
    optimal_traj_values_argmax = []
    all_traj_values_argmax = []
    for i in range(100):
        if i % 10 == 0:
            print(f'Attempt {i}')
        optimal_traj_value_softmax, traj_values_softmax, optimal_traj_value_argmax, traj_values_argmax = main(args)
        optimal_traj_values_softmax.append(optimal_traj_value_softmax)
        all_traj_values_softmax.append(traj_values_softmax)
        optimal_traj_values_argmax.append(optimal_traj_value_argmax)
        all_traj_values_argmax.append(traj_values_argmax)

    all_traj_values_argmax = np.array(all_traj_values_argmax)
    all_traj_values_softmax = np.array(all_traj_values_softmax)
    optimal_traj_values_argmax = np.array(optimal_traj_values_argmax)
    optimal_traj_values_softmax = np.array(optimal_traj_values_softmax)

    save_dir = os.path.join('model_analysis', exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(f'{save_dir}/all_traj_values_softmax.npy', all_traj_values_softmax)
    np.save(f'{save_dir}/optimal_traj_values_softmax.npy', optimal_traj_values_softmax)
    np.save(f'{save_dir}/all_traj_values_argmax.npy', all_traj_values_argmax)
    np.save(f'{save_dir}/optimal_traj_values_argmax.npy', optimal_traj_values_argmax)
    # plt.plot(np.mean(all_traj_values, axis=0), label='Improved Trajectory')
    # # Draw optimal trajecotry values as a dashed horizontal line
    # plt.plot([0, args.iterations], [np.mean(optimal_traj_values), np.mean(optimal_traj_values)], 'k--', label='Optimal Trajectory')
    # plt.xlabel('Iteration')
    # plt.ylabel('Avg. Reward')
    # plt.title('Average Reward vs. Iteration (Test Set)')
    # plt.legend()
    #
    # plt.savefig(f'model_analysis/{exp_name}/improve_traj_softmax.png')
    # plt.show()
