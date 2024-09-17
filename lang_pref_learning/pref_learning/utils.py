import os
import json
import numpy as np
import matplotlib.pyplot as plt

rs_feature_aspects = ['gt_reward', 'speed', 'height', 'distance_to_bottle', 'distance_to_cube']

mw_feature_aspects = ['height', 'velocity', 'distance']


def load_data(args, split='train', DEBUG=False):
    # Load the test trajectories and language comparisons
    trajs = np.load(f"{args.data_dir}/{split}/trajs.npy")
    nlcomps = json.load(open(f"{args.data_dir}/{split}/unique_nlcomps.json", "rb"))

    nlcomp_embeds = None
    traj_img_obs = None
    actions = None

    if not args.use_lang_encoder:
        nlcomp_embeds = np.load(f"{args.data_dir}/{split}/unique_nlcomps_{args.lang_model_name}.npy")

    if args.use_img_obs:
        traj_img_obs = np.load(f"{args.data_dir}/{split}/traj_img_obs.npy")
        actions = np.load(f"{args.data_dir}/{split}/actions.npy")
        trajs = trajs[:, ::10, :]
        traj_img_obs = traj_img_obs[:, ::10, :]
        actions = actions[:, ::10, :]

    if DEBUG:
        print("len of trajs: " + str(len(trajs)))

    # need to run categorize.py first to get these files
    greater_nlcomps = json.load(open(f"{args.data_dir}/train/greater_nlcomps.json", "rb"))
    less_nlcomps = json.load(open(f"{args.data_dir}/train/less_nlcomps.json", "rb"))
    classified_nlcomps = json.load(open(f"{args.data_dir}/train/classified_nlcomps.json", "rb"))
    if DEBUG:
        print("greater nlcomps size: " + str(len(greater_nlcomps)))
        print("less nlcomps size: " + str(len(less_nlcomps)))

    data = {
        "trajs": trajs,
        "nlcomps": nlcomps,
        "nlcomp_embeds": nlcomp_embeds,
        "greater_nlcomps": greater_nlcomps,
        "less_nlcomps": less_nlcomps,
        "classified_nlcomps": classified_nlcomps,
        'traj_img_obs': traj_img_obs,
        'actions': actions
    }

    return data


def save_results(args, results, postfix="noisy"):
    save_dir = f"{args.true_reward_dir}/pref_learning"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    eval_cross_entropies = results["cross_entropy"]
    learned_reward_norms = results["learned_reward_norms"]
    optimal_learned_rewards = results["optimal_learned_rewards"]
    optimal_true_rewards = results["optimal_true_rewards"]

    result_dict = {
        "eval_cross_entropies": eval_cross_entropies,
        "learned_reward_norms": learned_reward_norms,
        "optimal_learned_rewards": optimal_learned_rewards,
        "optimal_true_rewards": optimal_true_rewards,
    }

    np.savez(f"{save_dir}/{postfix}.npz", **result_dict)


def plot_results(args, results, test_ce):
    """ Plot the results """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(results["cross_entropy"], label="Noisy")
    ax1.plot([0, len(results["cross_entropy"])], [test_ce, test_ce], 'k--', label='Ground Truth')
    ax1.set_xlabel("Number of Queries")
    ax1.set_ylabel("Cross-Entropy")
    ax1.set_title("Feedback, True Dist: Softmax")
    ax1.legend()

    ax2.plot(results["optimal_learned_rewards"], label="Noisy, Learned Reward")
    ax2.plot(results["optimal_true_rewards"], label="True Reward", c="r")
    ax2.set_xlabel("Number of Queries")
    ax2.set_ylabel("Reward Value")
    ax2.set_title("True Reward of Optimal Trajectory")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{args.true_reward_dir}/pref_learning/{args.method}_pref.png")