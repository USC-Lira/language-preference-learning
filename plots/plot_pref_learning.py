# from matplotlib import rcParams
#
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Palatino Linotype']
# rcParams['font.size'] = 40

import numpy as np
import matplotlib.pyplot as plt


def combine_results(results, total_results):
    for key, value in results.items():
        total_results[key] = np.concatenate((total_results[key], value), axis=0)
    return total_results


def load_results(base_data_dir, postfix):
    total_noisy_results = {
        'all_optimal_true_rewards': [],
        'all_optimal_learned_rewards': [],
        'all_eval_cross_entropies': []
    }
    total_noiseless_results = {
        'all_optimal_true_rewards': [],
        'all_optimal_learned_rewards': [],
        'all_eval_cross_entropies': []
    }

    def get_data(results, total_results):
        optimal_true_rewards = results['optimal_true_rewards']
        optimal_learned_rewards = results['optimal_learned_rewards']
        eval_cross_entropies = results['eval_cross_entropies']

        total_results['all_optimal_true_rewards'].append(optimal_true_rewards.reshape(1, -1))
        total_results['all_optimal_learned_rewards'].append(optimal_learned_rewards.reshape(1, -1))
        total_results['all_eval_cross_entropies'].append(eval_cross_entropies.reshape(1, -1))
        return total_results

    for i in range(3):
        data_dir = f'{base_data_dir}/{i}/pref_learning'
        noisy_results = np.load(f'{data_dir}/noisy_{postfix}.npz')
        noiseless_results = np.load(f'{data_dir}/noiseless_{postfix}.npz')

        total_noisy_results = get_data(noisy_results, total_noisy_results)
        total_noiseless_results = get_data(noiseless_results, total_noiseless_results)

    for key, value in total_noisy_results.items():
        total_noisy_results[key] = np.concatenate(value, axis=0)

    for key, value in total_noiseless_results.items():
        total_noiseless_results[key] = np.concatenate(value, axis=0)

    return total_noisy_results, total_noiseless_results


def plot(base_data_dir):
    noisy_results_baseline, noiseless_results_baseline = load_results(base_data_dir, 'baseline')
    noisy_results_other_feedback, noiseless_results_other_feedback = load_results(base_data_dir,
                                                                                  'other_feedback_10_temp_1.0')
    noisy_results_other_feedback_10_temp_cos, noiseless_results_other_feedback_10_temp_cos = load_results(base_data_dir,
                                                                                                          'other_feedback_10_temp_cos')

    # Plot cross-entropies
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[1].plot(np.mean(noisy_results_baseline['all_eval_cross_entropies'], axis=0), label='Baseline')
    ax[1].fill_between(np.arange(0, len(noisy_results_baseline['all_eval_cross_entropies'][0]), 1),
                       np.mean(noisy_results_baseline['all_eval_cross_entropies'], axis=0) - 0.5 * np.std(
                           noisy_results_baseline['all_eval_cross_entropies'], axis=0),
                       np.mean(noisy_results_baseline['all_eval_cross_entropies'], axis=0) + 0.5 * np.std(
                           noisy_results_baseline['all_eval_cross_entropies'], axis=0),
                       alpha=0.2)
    ax[1].plot(np.mean(noisy_results_other_feedback['all_eval_cross_entropies'], axis=0),
               label='Other Feedback, Constant Temp')
    ax[1].fill_between(np.arange(0, len(noisy_results_other_feedback['all_eval_cross_entropies'][0]), 1),
                       np.mean(noisy_results_other_feedback['all_eval_cross_entropies'], axis=0) - 0.5 * np.std(
                           noisy_results_other_feedback['all_eval_cross_entropies'], axis=0),
                       np.mean(noisy_results_other_feedback['all_eval_cross_entropies'], axis=0) + 0.5 * np.std(
                           noisy_results_other_feedback['all_eval_cross_entropies'], axis=0),
                       alpha=0.2)
    ax[1].plot(np.mean(noisy_results_other_feedback_10_temp_cos['all_eval_cross_entropies'], axis=0),
               label='Other Feedback, Temp Cosine')
    ax[1].fill_between(np.arange(0, len(noisy_results_other_feedback_10_temp_cos['all_eval_cross_entropies'][0]), 1),
                       np.mean(noisy_results_other_feedback_10_temp_cos['all_eval_cross_entropies'],
                               axis=0) - 0.5 * np.std(
                           noisy_results_other_feedback_10_temp_cos['all_eval_cross_entropies'], axis=0),
                       np.mean(noisy_results_other_feedback_10_temp_cos['all_eval_cross_entropies'],
                               axis=0) + 0.5 * np.std(
                           noisy_results_other_feedback_10_temp_cos['all_eval_cross_entropies'], axis=0),
                       alpha=0.2)
    ax[1].set_xlabel('Number of Queries')
    ax[1].set_ylabel('Cross-Entropy')
    ax[1].legend()
    ax[1].set_title('Noisy Feedback')

    # Do the same for noiseless feedback
    ax[0].plot(np.mean(noiseless_results_baseline['all_eval_cross_entropies'], axis=0), label='Baseline')
    ax[0].fill_between(np.arange(0, len(noiseless_results_baseline['all_eval_cross_entropies'][0]), 1),
                       np.mean(noiseless_results_baseline['all_eval_cross_entropies'], axis=0) - 0.5 * np.std(
                           noiseless_results_baseline['all_eval_cross_entropies'], axis=0),
                       np.mean(noiseless_results_baseline['all_eval_cross_entropies'], axis=0) + 0.5 * np.std(
                           noiseless_results_baseline['all_eval_cross_entropies'], axis=0),
                       alpha=0.2)
    ax[0].plot(np.mean(noiseless_results_other_feedback['all_eval_cross_entropies'], axis=0),
               label='Other Feedback, Constant Temp')
    ax[0].fill_between(np.arange(0, len(noiseless_results_other_feedback['all_eval_cross_entropies'][0]), 1),
                       np.mean(noiseless_results_other_feedback['all_eval_cross_entropies'], axis=0) - 0.5 * np.std(
                           noiseless_results_other_feedback['all_eval_cross_entropies'], axis=0),
                       np.mean(noiseless_results_other_feedback['all_eval_cross_entropies'], axis=0) + 0.5 * np.std(
                           noiseless_results_other_feedback['all_eval_cross_entropies'], axis=0),
                       alpha=0.2)
    ax[0].plot(np.mean(noiseless_results_other_feedback_10_temp_cos['all_eval_cross_entropies'], axis=0),
               label='Other Feedback, Temp Cosine')
    ax[0].fill_between(
        np.arange(0, len(noiseless_results_other_feedback_10_temp_cos['all_eval_cross_entropies'][0]), 1),
        np.mean(noiseless_results_other_feedback_10_temp_cos['all_eval_cross_entropies'], axis=0) - 0.5 * np.std(
            noiseless_results_other_feedback_10_temp_cos['all_eval_cross_entropies'], axis=0),
        np.mean(noiseless_results_other_feedback_10_temp_cos['all_eval_cross_entropies'], axis=0) + 0.5 * np.std(
            noiseless_results_other_feedback_10_temp_cos['all_eval_cross_entropies'], axis=0),
        alpha=0.2)
    ax[0].set_xlabel('Number of Queries')
    ax[0].set_ylabel('Cross-Entropy')
    ax[0].legend()
    ax[0].set_title('Noiseless Feedback')

    # set y-axis limit
    ax[1].set_ylim([0.51, 0.72])
    ax[0].set_ylim([0.51, 0.72])

    plt.savefig(f'pref_learning_comparison.png')
    plt.show()

    # First plot noiseless results

    # # Plot cross-entropies, learned reward norms, and optimal rewards in one figure
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # ax1.plot(np.mean(total_noisy_results['all_eval_cross_entropies'], axis=0), label='Noisy Feedback')
    # ax1.fill_between(np.arange(0, len(total_noisy_results['all_eval_cross_entropies'][0]), 1),
    #                  np.mean(total_noisy_results['all_eval_cross_entropies'], axis=0) - 0.5 * np.std(
    #                      total_noisy_results['all_eval_cross_entropies'], axis=0),
    #                  np.mean(total_noisy_results['all_eval_cross_entropies'], axis=0) + 0.5 * np.std(
    #                      total_noisy_results['all_eval_cross_entropies'], axis=0),
    #                  alpha=0.2)
    # ax1.plot(np.mean(total_noiseless_results['all_eval_cross_entropies'], axis=0), label='Noiseless Feedback')
    # ax1.fill_between(np.arange(0, len(total_noiseless_results['all_eval_cross_entropies'][0]), 1),
    #                  np.mean(total_noiseless_results['all_eval_cross_entropies'], axis=0) - 0.5 * np.std(
    #                      total_noiseless_results['all_eval_cross_entropies'], axis=0),
    #                  np.mean(total_noiseless_results['all_eval_cross_entropies'], axis=0) + 0.5 * np.std(
    #                      total_noiseless_results['all_eval_cross_entropies'], axis=0),
    #                  alpha=0.2)
    # ax1.set_xlabel('Number of Queries')
    # ax1.set_ylabel('Cross-Entropy')
    # ax1.legend()
    # ax1.set_title('Baseline')
    #
    # ax2.plot([0, len(total_noisy_results['all_optimal_true_rewards'][0]) - 1],
    #          [np.mean(total_noisy_results['all_optimal_true_rewards']),
    #           np.mean(total_noisy_results['all_optimal_true_rewards'])],
    #          'k--', linewidth=2, label='Optimal Trajectory')
    # ax2.plot(np.mean(total_noisy_results['all_optimal_learned_rewards'], axis=0),
    #          label='Learned Reward, Noisy Feedback')
    # ax2.fill_between(np.arange(0, len(total_noisy_results['all_optimal_learned_rewards'][0]), 1),
    #                  np.mean(total_noisy_results['all_optimal_learned_rewards'], axis=0) - 0.5 * np.std(
    #                      total_noisy_results['all_optimal_learned_rewards'], axis=0),
    #                  np.mean(total_noisy_results['all_optimal_learned_rewards'], axis=0) + 0.5 * np.std(
    #                      total_noisy_results['all_optimal_learned_rewards'], axis=0),
    #                  alpha=0.2)
    # ax2.plot(np.mean(total_noiseless_results['all_optimal_learned_rewards'], axis=0),
    #          label='Learned Reward, Noiseless Feedback')
    # ax2.fill_between(np.arange(0, len(total_noiseless_results['all_optimal_learned_rewards'][0]), 1),
    #                  np.mean(total_noiseless_results['all_optimal_learned_rewards'], axis=0) - 0.5 * np.std(
    #                      total_noiseless_results['all_optimal_learned_rewards'], axis=0),
    #                  np.mean(total_noiseless_results['all_optimal_learned_rewards'], axis=0) + 0.5 * np.std(
    #                      total_noiseless_results['all_optimal_learned_rewards'], axis=0),
    #                  alpha=0.2)
    #
    # ax2.set_xlabel('Number of Queries')
    # ax2.set_ylabel('Reward Value')
    # ax2.set_title(f'True Reward of Optimal Trajectory')
    # ax2.legend()
    #
    # plt.tight_layout()
    # plt.savefig(f'pref_learning_other_feedback_10_temp_cos.png')


if __name__ == '__main__':
    base_data_dir = '../pref_learning/true_rewards'
    plot(base_data_dir)
