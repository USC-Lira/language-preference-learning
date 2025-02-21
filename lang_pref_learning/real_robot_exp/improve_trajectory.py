"""
Show trajectory on real robot and improve it based on the feedback from the user.
Load trajectories -> Start from the worst one -> Get the input from user -> Improve the trajectory
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
import torch
import pickle
import os
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from lang_pref_learning.feature_learning.utils import HF_LANG_MODEL_NAME, LANG_OUTPUT_DIM
from lang_pref_learning.model_analysis.find_nearest_traj import get_nearest_embed#, get_nearest_embed_project
from lang_pref_learning.model.encoder import NLTrajAutoencoder
from lang_pref_learning.real_robot_exp.utils import get_lang_embed, get_traj_embeds_wx
from lang_pref_learning.real_robot_exp.utils import replay_traj_widowx, replay_trajectory_video, remove_special_characters

from data.utils import speed_wx, distance_to_pan_wx, distance_to_spoon_wx
from data.utils import WidowX_STATE_OBS_DIM, WidowX_ACTION_DIM, WidowX_PROPRIO_STATE_DIM, WidowX_OBJECT_STATE_DIM

try:
    import time
    import rospy
    import pickle as pkl

    from multicam_server.topic_utils import IMTopic
    from widowx_envs.widowx_env import WidowXEnv

    env_params = {
    'camera_topics': [IMTopic('/blue/image_raw')
                      ],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    'move_to_rand_start_freq': -1,
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'adaptive_wait': True,
    'action_clipping': None
}
    
except ImportError as e:
    print(e)





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


def improve_trajectory_human(feature_values, traj_embeds, traj_images, traj_policy_outs, model, device, tokenizer, lang_encoder, args):
    # Normalize reward values to be between 0 and 1
    dummy_reward_func = np.array([-1, -1, -1])
    reward_values = np.dot(feature_values, dummy_reward_func)
    reward_values = (reward_values - np.min(reward_values)) / (np.max(reward_values) - np.min(reward_values))

    # get the 5 trajectories with the lowest reward values
    init_traj_idxs = np.argsort(reward_values)[:5]
    curr_traj_idx = np.random.choice(init_traj_idxs)
    curr_traj_value = reward_values[curr_traj_idx]
    if args.debug:
        print(f'Initial trajectory: {curr_traj_idx}, initial value: {curr_traj_value}\n')

    optimal_reached = False
    traj_values = [curr_traj_value]

    if args.real_robot:
        widowx_env = WidowXEnv(env_params)
        widowx_env.start()
        widowx_env.move_to_neutral()

    for i in range(args.iterations):
        if i == 0:
            print(f'Replay Initial Trajecotry.....')
        else:
            print(f'Replay Current Trajecotry.....')

        # Show current trajecotry to the user
        if args.real_robot:
            curr_traj_policy_out = traj_policy_outs[curr_traj_idx]
            replay_traj_widowx(widowx_env, curr_traj_policy_out)
        else:
            curr_traj_images = traj_images[curr_traj_idx]
            replay_trajectory_video(curr_traj_images)
        
        satisfied = input(f'\nAre you satisfied with the current trajectory? (y/n): ')
        satisfied = satisfied.strip().lower()
        if satisfied == 'y':
            optimal_reached = True
            print(f'\nOptimal trajectory reached at iteration {i}!\n')
            break
        

        # TODO: Get the feedback from the users
        nlcomp = input(f'\nPlease enter the language feedback: ')
        # remove whitespace and from the input
        nlcomp = remove_special_characters(nlcomp)
        print(nlcomp)

        lang_embed = get_lang_embed(nlcomp, model, device, tokenizer)
        # next_traj_idx = get_nearest_embed_cosine(traj_embeds[curr_traj_idx], lang_embed, traj_embeds, curr_traj_idx)
        next_traj_idx = get_nearest_embed(traj_embeds[curr_traj_idx], lang_embed, traj_embeds, curr_traj_idx)
        curr_traj_idx = next_traj_idx
        next_traj_value = reward_values[next_traj_idx]

        # if next_traj_value > curr_traj_value:
        #     curr_traj_idx = next_traj_idx
        #     curr_traj_value = next_traj_value
        
        traj_values.append(curr_traj_value)

        if args.debug:
            print(f'========= Iteration {i} =========')
            print(f'Next trajectory: {next_traj_idx}, next value: {next_traj_value}')
            print(f'Language comparison: {nlcomp}\n')
            
    if args.real_robot:
        widowx_env._controller.open_gripper(True)
        widowx_env.move_to_neutral()

    return optimal_reached, traj_values


def load_trajectories(traj_root_dir):
    traj_root_dir = os.path.join(args.data_dir, 'trajectory')
    trajs = []
    traj_images = []
    traj_policy_outs = []

    # Read all trajectory directories by numerical order
    traj_dirs = os.listdir(traj_root_dir)

    def extract_number(filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        return None
    
    sorted_traj_dirs = sorted(traj_dirs, key=extract_number)

    for traj_dir in sorted_traj_dirs:
        traj = np.load(os.path.join(traj_root_dir, traj_dir, 'trajs.npy'))
        traj_img_obs = np.load(os.path.join(traj_root_dir, traj_dir, 'traj_img_obs.npy'))
        trajs.append(traj)
        traj_images.append(traj_img_obs)
        with open(os.path.join(traj_root_dir, traj_dir, 'policy_out.pkl'), 'rb') as f:
            policy_out = pickle.load(f)
            traj_policy_outs.append(policy_out)

    return trajs, traj_images, traj_policy_outs



def main(args):
    # trajs = np.load(os.path.join(args.data_dir, 'trajs.npy'))
    # if args.use_image_obs:
    #     traj_img_obs = np.load(os.path.join(args.data_dir, 'test/traj_img_obs.npy'))
    trajs, traj_img_obs, traj_policy_outs = load_trajectories(args.data_dir)

    # Load the model
    if args.use_lang_encoder:
        if 't5' in args.lang_model_name:
            lang_encoder = T5EncoderModel.from_pretrained(args.lang_model_name)
        else:
            lang_encoder = AutoModel.from_pretrained(HF_LANG_MODEL_NAME[args.lang_model_name])

        tokenizer = AutoTokenizer.from_pretrained(HF_LANG_MODEL_NAME[args.lang_model_name])
        feature_dim = LANG_OUTPUT_DIM[args.lang_model_name]
    else:
        lang_encoder = AutoModel.from_pretrained(HF_LANG_MODEL_NAME[args.lang_model_name])
        tokenizer = AutoTokenizer.from_pretrained(HF_LANG_MODEL_NAME[args.lang_model_name])
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
                              lang_embed_dim=LANG_OUTPUT_DIM[args.lang_model_name],
                              traj_encoder=args.traj_encoder)

    state_dict = torch.load(os.path.join(args.model_dir, 'best_model_state_dict.pth'))

    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Get trajectory embeddings
    if not os.path.exists(f'{args.data_dir}/traj_embeds.npy'):
        traj_embeds = get_traj_embeds_wx(trajs, model, device,
                                      use_img_obs=args.use_image_obs, 
                                      img_obs=traj_img_obs)
        np.save(f'{args.data_dir}/traj_embeds.npy', traj_embeds)
    else:
        traj_embeds = np.load(f'{args.data_dir}/traj_embeds.npy')
    

    # Find the optimal trajectory given the reward function
    feature_values = np.array([get_feature_value(traj) for traj in trajs])
    # Normalize feature values
    feature_values = (feature_values - np.min(feature_values, axis=0)) / (
            np.max(feature_values, axis=0) - np.min(feature_values, axis=0))

    improve_trajectory_human(feature_values, traj_embeds, traj_img_obs, traj_policy_outs,
                             model, device, tokenizer, model.lang_encoder, args)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--env', type=str, default='widowx')
    parser.add_argument('--model-dir', type=str, default='exp/linear_bert-mini')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--use-lang-encoder', action='store_true')
    parser.add_argument('--lang-model-name', type=str, default='bert-base')
    parser.add_argument('--encoder-hidden-dim', type=int, default=128)
    parser.add_argument('--decoder-hidden-dim', type=int, default=128)
    parser.add_argument('--old-model', action='store_true')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use-image-obs', action='store_true')
    parser.add_argument('--traj-encoder', type=str, default='cnn',
                        choices=['mlp', 'cnn'], help='which trajectory encoder to use')
    # parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-trails', type=int, default=10)
    parser.add_argument('--real-robot', action='store_true')
    args = parser.parse_args()

    # randomly set a seed
    np.random.seed(np.random.randint(0, 1000000))

    main(args)