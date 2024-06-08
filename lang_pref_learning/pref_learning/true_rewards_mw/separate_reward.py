import numpy as np

all_rewards = np.load("true_rewards.npy")

# Get the best trajectory for each true reward
train_trajs = np.load("../../../data/metaworld/train/trajs.npy")
val_trajs = np.load("../../../data/metaworld/val/trajs.npy")
test_trajs = np.load("../../../data/metaworld/test/trajs.npy")
all_trajs = np.concatenate([train_trajs, val_trajs, test_trajs], axis=0)

train_trajs_features = np.load("../../../data/metaworld/train/feature_vals.npy")
val_trajs_features = np.load("../../../data/metaworld/val/feature_vals.npy")
test_trajs_features = np.load("../../../data/metaworld/test/feature_vals.npy")
all_trajs_features_ori = np.concatenate([train_trajs_features, val_trajs_features, test_trajs_features], axis=0)
all_trajs_features = np.mean(all_trajs_features_ori, axis=-1)
all_trajs_features = all_trajs_features[:, :3]

for i in range(3, 4):
    true_reward = all_rewards[i]
    np.save(f"{2}/true_rewards.npy", true_reward)

    true_reward_values = np.dot(all_trajs_features, true_reward)
    best_traj_idx = np.argmax(true_reward_values)

    np.save(f"{2}/traj.npy", all_trajs[best_traj_idx])
    np.save(f"{2}/traj_vals.npy", all_trajs_features_ori[best_traj_idx])

