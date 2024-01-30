from matplotlib import rcParams
import numpy as np

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Palatino Linotype']
rcParams['font.size'] = 40

import matplotlib.pyplot as plt

all_traj_rewards_softmax = np.load('../model_analysis/finetune-bert-tiny-newdata-aug-lr0.0001/all_traj_values_softmax.npy')
optimal_traj_rewards_softmax = np.load('../model_analysis/finetune-bert-tiny-newdata-aug-lr0.0001/optimal_traj_values_softmax.npy')

all_traj_rewards_argmax = np.load('../model_analysis/finetune-bert-tiny-newdata-aug-lr0.0001/all_traj_values_argmax.npy')
optimal_traj_rewards_argmax = np.load('../model_analysis/finetune-bert-tiny-newdata-aug-lr0.0001/optimal_traj_values_argmax.npy')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 8))

axs = [ax1, ax2]
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=28)  # Adjust font size of ticks

ax1.plot(np.mean(all_traj_rewards_argmax, axis=0), color='#E48B10', linewidth=4, label='Improved Trajectory')
ax1.fill_between(np.arange(0, len(all_traj_rewards_argmax[0]), 1),
                    np.mean(all_traj_rewards_argmax, axis=0) - 0.5 * np.std(all_traj_rewards_argmax, axis=0),
                    np.mean(all_traj_rewards_argmax, axis=0) + 0.5 * np.std(all_traj_rewards_argmax, axis=0),
                    alpha=0.2, color='#E48B10')
ax1.plot([0, len(all_traj_rewards_argmax[0])-1],
         [np.mean(optimal_traj_rewards_argmax), np.mean(optimal_traj_rewards_argmax)],
         'k--', linewidth=4, label='Optimal Trajectory')
ax1.set_xticks(np.arange(0, len(all_traj_rewards_argmax[0]), 5))
ax1.set_ylim([0.2, 0.8])

ax2.plot(np.mean(all_traj_rewards_softmax, axis=0), color='#E48B10', linewidth=4, label='Improved Trajectory')
ax2.fill_between(np.arange(0, len(all_traj_rewards_argmax[0]), 1),
                    np.mean(all_traj_rewards_softmax, axis=0) - 0.5 * np.std(all_traj_rewards_softmax, axis=0),
                    np.mean(all_traj_rewards_softmax, axis=0) + 0.5 * np.std(all_traj_rewards_softmax, axis=0),
                    alpha=0.2, color='#E48B10')
ax2.plot([0, len(all_traj_rewards_softmax[0])-1],
         [np.mean(optimal_traj_rewards_softmax), np.mean(optimal_traj_rewards_softmax)],
         'k--', linewidth=4, label='Optimal Trajectory')
ax2.set_xticks(np.arange(0, len(all_traj_rewards_argmax[0]), 5))
ax2.set_ylim([0.2, 0.8])

ax1.set_xlabel('Iterations')
ax2.set_xlabel('Iterations')
ax1.set_ylabel('Reward')
ax2.set_ylabel('Reward')
ax2.legend(fontsize=28, frameon=False)

handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncols=2, prop={'size': 20})
plt.tight_layout(pad=2.0)
plt.subplots_adjust(wspace=0.25, hspace=0.2)
plt.savefig('finetune_improve_traj.pdf', dpi=300)
plt.show()