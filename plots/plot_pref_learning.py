from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Palatino Linotype']
rcParams['font.size'] = 40
rcParams["legend.loc"] = 'lower left'

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
    # noisy_results_other_feedback_10_temp_cos, noiseless_results_other_feedback_10_temp_cos = load_results(base_data_dir,
    #                                                                                                       'other_feedback_10_temp_cos_lc_1.0')
    # noisy_results_other_feedback_10_temp_cos_lc, noiseless_results_other_feedback_10_temp_cos_lc = load_results(
    #     base_data_dir,
    #     'other_feedback_10_temp_cos_lc_1.2')
    # noisy_results_lc_1_5, noiseless_results_lc_1_5 = load_results(base_data_dir, 'other_feedback_10_temp_cos_lc_1.5')
    # noisy_results_lc_0_8, noiseless_results_lc_0_8 = load_results(base_data_dir, 'other_feedback_10_temp_cos_lc_0.8')

    all_noisy_results = [noisy_results_baseline, noisy_results_other_feedback]
    all_noiseless_results = [noiseless_results_baseline, noiseless_results_other_feedback]
    labels = ["Baseline", "Constant Temperature"]
    # all_noisy_results = [noisy_results_other_feedback_10_temp_cos, noisy_results_other_feedback_10_temp_cos_lc,
    #                      noisy_results_lc_1_5, noisy_results_lc_0_8]
    # all_noiseless_results = [noiseless_results_other_feedback_10_temp_cos, noiseless_results_other_feedback_10_temp_cos_lc,
    #                          noiseless_results_lc_1_5, noiseless_results_lc_0_8]
    # labels = ["Other Feedback, Temp Cosine, alpha=1.0", "Other Feedback, Temp Cosine, alpha=1.2", "Other Feedback, Temp Cosine, alpha=1.5", "Other Feedback, Temp Cosine, alpha=0.8"]

    def plot_curve_and_std(ax, mean, std, label, color='#E48B10'):
        ax.plot(
            mean,
            color=color,
            linewidth=4,
            label=label)
        ax.fill_between(np.arange(0, len(mean), 1),
                        mean - 0.5 * std,
                        mean + 0.5 * std,
                        alpha=0.2,
                        color=color)

    # Plot cross-entropies
    fig, ax = plt.subplots(1, 2, figsize=(20, 9))

    for ax_idx in range(2):
        ax[ax_idx].spines["top"].set_visible(False)
        ax[ax_idx].spines["right"].set_visible(False)
        ax[ax_idx].tick_params(axis="both", which="major", labelsize=25)

    colors = ['#E48B10', '#298c8c']
    for noisy_results, noiseless_results, label, color in zip(all_noisy_results, all_noiseless_results, labels, colors):
        plot_curve_and_std(ax[0], np.mean(noiseless_results['all_eval_cross_entropies'], axis=0),
                           np.std(noiseless_results['all_eval_cross_entropies'], axis=0), label,
                           color=color)
        plot_curve_and_std(ax[1], np.mean(noisy_results['all_eval_cross_entropies'], axis=0),
                           np.std(noisy_results['all_eval_cross_entropies'], axis=0), label,
                           color=color)

    ax[0].set_xlabel('Number of Queries')
    ax[0].set_ylabel('Cross-Entropy')
    # ax[0].legend(fontsize=25, frameon=False)
    ax[0].set_title('Noiseless Feedback')

    ax[1].set_xlabel('Number of Queries')
    ax[1].set_ylabel('Cross-Entropy')
    ax[1].legend(fontsize=25, frameon=False)
    ax[1].set_title('Noisy Feedback')

    # set y-axis limit
    ax[0].set_ylim([0.51, 0.72])
    ax[1].set_ylim([0.51, 0.72])

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(wspace=0.25, hspace=0.2)
    plt.savefig(f'pref_learning_comparison.pdf', dpi=250)
    plt.show()


if __name__ == '__main__':
    base_data_dir = '../pref_learning/true_rewards'
    plot(base_data_dir)
