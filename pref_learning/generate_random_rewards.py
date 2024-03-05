import numpy as np
import argparse
from torch.utils.data import DataLoader


from model_analysis.improve_trajectory import initialize_reward
from pref_learning.pref_based_learning import load_data, get_feature_value, evaluate
from pref_learning.pref_dataset import EvalDataset

true_reward_dim = 5

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='data', help='')
parser.add_argument('--bert-model', type=str, default='bert-tiny', help='which BERT model to use')
args = parser.parse_args()

# Load the data
train_data_dict = load_data(args)
train_trajs = train_data_dict['trajs']
train_nlcomps, train_nlcomps_embed = train_data_dict['nlcomps'], train_data_dict['nlcomp_embeds']
train_greater_nlcomps, train_less_nlcomps = train_data_dict['greater_nlcomps'], train_data_dict['less_nlcomps']
train_classified_nlcomps = train_data_dict['classified_nlcomps']
train_feature_values = np.array([get_feature_value(traj) for traj in train_trajs])

test_data_dict = load_data(args, test=True)
test_trajs = test_data_dict['trajs']
test_nlcomps, test_nlcomps_embed = test_data_dict['nlcomps'], test_data_dict['nlcomp_embeds']
test_feature_values = np.array([get_feature_value(traj) for traj in test_trajs])
test_dataset = EvalDataset(test_trajs)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)

all_features = np.concatenate([train_feature_values, test_feature_values], axis=0)
feature_value_means = np.mean(all_features, axis=0)
feature_value_stds = np.std(all_features, axis=0)

# Normalize the feature values
train_feature_values = (train_feature_values - feature_value_means) / feature_value_stds
test_feature_values = (test_feature_values - feature_value_means) / feature_value_stds

true_rewards = []
for _ in range(4):
    cross_entropy = 1.0
    true_reward = None
    while cross_entropy > 0.4:
        true_reward = initialize_reward(true_reward_dim)
        print(true_reward)
        true_traj_rewards = test_feature_values @ true_reward
        test_entropy = evaluate(test_data, true_traj_rewards, true_reward, test_nlcomps_embed, test=True)
        cross_entropy = test_entropy
        print(cross_entropy)
    true_rewards.append(true_reward)

np.save('true_rewards.npy', true_rewards)
print(true_rewards)