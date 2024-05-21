"""
Show trajectory on real robot and improve it based on the feedback from the user.
Load trajectories -> Start from the worst one -> Get the input from user -> Improve the trajectory
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import torch
import os
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from lang_pref_learning.feature_learning.utils import LANG_MODEL_NAME, LANG_OUTPUT_DIM
from lang_pref_learning.model_analysis.find_nearest_traj import get_nearest_embed_cosine, get_nearest_embed_distance, get_nearest_embed_project
from lang_pref_learning.model.encoder import NLTrajAutoencoder
from lang_pref_learning.model_analysis.utils import get_traj_embeds, get_lang_embed
from lang_pref_learning.real_robot_exp.utils import replay_trajectory_robot, replay_trajectory_video, remove_special_characters

from data.utils import speed_wx, distance_to_pan_wx, distance_to_spoon_wx
from data.utils import WidowX_STATE_OBS_DIM, WidowX_ACTION_DIM, WidowX_PROPRIO_STATE_DIM, WidowX_OBJECT_STATE_DIM


def get_feature_value(traj, traj_mean=False):
    # Get feature values for each timestep
    if traj_mean:
        features_values = np.array([speed_wx(traj), distance_to_pan_wx(traj), distance_to_spoon_wx(traj)])
        return features_values

    else:
        feature_values = [
            [speed_wx(state), distance_to_pan_wx(state), distance_to_spoon_wx(state)]
            for state in traj
            ]
        return np.mean(feature_values, axis=0)


def improve_trajectory_human(feature_values, traj_embeds, traj_images, model, device, tokenizer, lang_encoder, args):
    # Normalize reward values to be between 0 and 1
    dummy_reward_func = np.array([-1, -1, -1])
    reward_values = np.dot(feature_values, dummy_reward_func)
    reward_values = (reward_values - np.min(reward_values)) / (np.max(reward_values) - np.min(reward_values))

    curr_traj_idx = np.argmin(reward_values)
    curr_traj_value = reward_values[curr_traj_idx]
    if args.debug:
        print(f'Initial trajectory: {curr_traj_idx}, initial value: {curr_traj_value}\n')

    optimal_reached = False
    traj_values = [curr_traj_value]
    for i in range(args.iterations):

        # Show current trajecotry to the user
        curr_traj_images = traj_images[curr_traj_idx]
        # replay_trajectory_video(curr_traj_images, frame_rate=10)
        # replay_trajectory_robot(robot, trajs[curr_traj_idx])


        # TODO: Get the feedback from the users
        nlcomp = input(f'\nPlease enter the language feedback: ')
        # remove whitespace and from the input
        nlcomp = nlcomp.strip()

        lang_embed = get_lang_embed(nlcomp, model, device, tokenizer, use_bert_encoder=args.use_bert_encoder,
                                    lang_model=lang_encoder)
        next_traj_idx = get_nearest_embed_cosine(traj_embeds[curr_traj_idx], lang_embed, traj_embeds, curr_traj_idx)
        # next_traj_idx = get_nearest_embed_project(traj_embeds[curr_traj_idx], lang_embed, traj_embeds, curr_traj_idx)
        next_traj_value = reward_values[next_traj_idx]

        if next_traj_value > curr_traj_value:
            curr_traj_idx = next_traj_idx
            curr_traj_value = next_traj_value
        
        traj_values.append(curr_traj_value)

        if args.debug:
            print(f'========= Iteration {i} =========')
            print(f'Current trajectory: {curr_traj_idx}, current value: {curr_traj_value}')
            print(f'Language comparison: {nlcomp}\n')

    return optimal_reached, traj_values


def plot_results(optimal_traj_values, all_traj_values, postfix):
    plt.figure()
    plt.plot(np.mean(all_traj_values, axis=0), label='Improved Trajectory')
    # Draw optimal trajecotry values as a dashed horizontal line
    plt.plot([0, args.iterations], [np.mean(optimal_traj_values), np.mean(optimal_traj_values)], 'k--', label='Optimal Trajectory')
    plt.xlabel('Iteration')
    plt.ylabel('Avg. Reward')
    plt.title('Average Reward vs. Iteration (Test Set)')
    plt.legend()
    
    plt.savefig(f'model_analysis/{exp_name}/improve_traj_{postfix}.png')
    # plt.show()


def main(args):
    trajs = np.load(os.path.join(args.data_dir, 'test/trajs.npy'))
    if args.use_image_obs:
        traj_img_obs = np.load(os.path.join(args.data_dir, 'test/traj_img_obs.npy'))
        actions = np.load(os.path.join(args.data_dir, 'test/actions.npy'))

    # Load the model
    if args.use_bert_encoder:
        if 't5' in args.lang_model:
            lang_encoder = T5EncoderModel.from_pretrained(args.lang_model)
        else:
            lang_encoder = AutoModel.from_pretrained(LANG_MODEL_NAME[args.lang_model])

        tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME[args.lang_model])
        feature_dim = LANG_OUTPUT_DIM[args.lang_model]
    else:
        lang_encoder = AutoModel.from_pretrained(LANG_MODEL_NAME[args.lang_model])
        tokenizer = AutoTokenizer.from_pretrained(LANG_OUTPUT_DIM[args.lang_model])
        feature_dim = 128

    if args.env == "widowx":
        STATE_OBS_DIM = WidowX_STATE_OBS_DIM
        ACTION_DIM = WidowX_ACTION_DIM
        PROPRIO_STATE_DIM = WidowX_PROPRIO_STATE_DIM
        OBJECT_STATE_DIM = WidowX_OBJECT_STATE_DIM
    else:
        raise ValueError("Invalid environment")

    model = NLTrajAutoencoder(STATE_OBS_DIM, ACTION_DIM, PROPRIO_STATE_DIM, OBJECT_STATE_DIM,
                              encoder_hidden_dim=args.encoder_hidden_dim, feature_dim=feature_dim,
                              decoder_hidden_dim=args.decoder_hidden_dim, lang_encoder=lang_encoder,
                              preprocessed_nlcomps=args.preprocessed_nlcomps,
                              lang_embed_dim=LANG_OUTPUT_DIM[args.lang_model],
                              use_bert_encoder=args.use_bert_encoder, 
                              traj_encoder=args.traj_encoder
                              )

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
    traj_embeds = get_traj_embeds(trajs, model, device, args.use_bert_encoder, tokenizer,
                                                use_img_obs=args.use_image_obs,
                                                img_obs=traj_img_obs,
                                                actions=actions)

    # Find the optimal trajectory given the reward function
    feature_values = np.array([get_feature_value(traj) for traj in trajs])
    # Normalize feature values
    feature_values = (feature_values - np.min(feature_values, axis=0)) / (
            np.max(feature_values, axis=0) - np.min(feature_values, axis=0))

    # Randomly select some features and set their values to 1 - original value
    less_idx = np.random.choice(5, size=2, replace=False)
    for i in less_idx:
        feature_values[:, i] = 1 - feature_values[:, i]

    improve_trajectory_human(feature_values, traj_embeds, traj_img_obs, 
                             model, device, tokenizer, lang_encoder, args)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model-dir', type=str, default='exp/linear_bert-mini')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--use-bert-encoder', action='store_true')
    parser.add_argument('--lang-model', type=str, default='bert-base')
    parser.add_argument('--encoder-hidden-dim', type=int, default=128)
    parser.add_argument('--decoder-hidden-dim', type=int, default=128)
    parser.add_argument('--preprocessed-nlcomps', action='store_true')
    parser.add_argument('--old-model', action='store_true')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use-image-obs', action='store_true')
    parser.add_argument('--traj-encoder', type=str, default='cnn',
                        choices=['mlp', 'cnn'], help='which trajectory encoder to use')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-trails', type=int, default=10)
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    exp_name = args.model_dir.split('/')[-1]
    optimal_traj_values_softmax = []
    all_traj_values_softmax = []
    optimal_traj_values_argmax = []
    all_traj_values_argmax = []
    for i in range(args.num_trails):
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


    plot_results(optimal_traj_values_softmax, all_traj_values_softmax, 'softmax')
    plot_results(optimal_traj_values_argmax, all_traj_values_argmax, 'argmax')