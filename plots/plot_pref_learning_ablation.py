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


def load_results(base_data_dir, method, postfix):
    total_results = {
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
        single_results = np.load(f'{data_dir}/{method}_noisy_{postfix}.npz')

        total_results = get_data(single_results, total_results)

    for key, value in total_results.items():
        total_results[key] = np.concatenate(value, axis=0)

    return total_results


def plot(rs_data_dir, mw_data_dir):
    # noisy_results_baseline, noiseless_results_baseline = load_results(rs_data_dir, 'lr_0.004_other_feedback_10_temp_1.0_lc_1.0')
    # noisy_results_other_feedback, noiseless_results_other_feedback = load_results(lang_data_dir, 'lang',
    #                                                                               'lr_0.005_other_feedback_20_temp_1.0_lc_1.0')
    
    # rs_comp_results = load_results(rs_data_dir, 'comp', 'lr_0.004_other_feedback_10_temp_1.0_lc_1.0')
    # rs_lang_results = load_results(rs_data_dir, 'lang', 'lr_0.005_other_feedback_20_temp_1.0_lc_1.0')

    # mw_comp_results = load_results(mw_data_dir, 'comp', 'lr_0.0005_other_feedback_20_temp_1.0_lc_1.0')
    # mw_lang_results = load_results(mw_data_dir, 'lang', 'lr_0.006_other_feedback_20_temp_1.0_lc_1.0')

    # Load results
    rs_lang_pref_only = load_results(rs_data_dir, 'lang', 'lr_0.01_lang_pref')
    rs_lang_pref_other_feedback = load_results(rs_data_dir, 'lang', 'lr_0.01_other_feedback_20_temp_1.0_lc_1.0_lang_pref')
    rs_other_feedback_only = load_results(rs_data_dir, 'lang', 'lr_0.01_other_feedback_20_temp_1.0_lc_1.0_no_lang_pref')

    mw_lang_pref_only = load_results(mw_data_dir, 'lang', 'lr_0.006_lang_pref')
    mw_lang_pref_other_feedback = load_results(mw_data_dir, 'lang', 'lr_0.006_other_feedback_20_temp_1.0_lc_1.0_lang_pref')
    mw_other_feedback_only = load_results(mw_data_dir, 'lang', 'lr_0.006_other_feedback_20_temp_1.0_lc_1.0_no_lang_pref')

    all_rs_results = [rs_lang_pref_other_feedback, rs_lang_pref_only, rs_other_feedback_only]
    all_mw_results = [mw_lang_pref_other_feedback, mw_lang_pref_only, mw_other_feedback_only]
    labels = ["Traj + Lang", "Traj Only", "Lang Only"]


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

    colors = ['#E48B10', '#9BBB59', '#9B870C']
    for rs_results, mw_results, label, color in zip(all_rs_results, all_mw_results, labels, colors):
        plot_curve_and_std(ax[0], np.mean(rs_results['all_eval_cross_entropies'], axis=0),
                           np.std(rs_results['all_eval_cross_entropies'], axis=0), label,
                           color=color)
        plot_curve_and_std(ax[1], np.mean(mw_results['all_eval_cross_entropies'], axis=0),
                           np.std(mw_results['all_eval_cross_entropies'], axis=0), label,
                           color=color)

    ax[1].set_xlabel('Number of Queries')
    ax[1].set_ylabel('Cross-Entropy')
    # ax[0].legend(fontsize=25, frameon=False)
    ax[1].set_title('Meta-World')

    ax[0].set_xlabel('Number of Queries')
    ax[0].set_ylabel('Cross-Entropy')
    ax[0].set_title('Robosuite')

    # set y-axis limit
    ax[0].set_ylim([0.51, 0.72])
    ax[1].set_ylim([0.38, 0.72])

    ax[0].legend(fontsize=25, frameon=False)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(wspace=0.25, hspace=0.2)
    plt.savefig(f'pref_learning_ablation.pdf', dpi=250)
    plt.show()


if __name__ == '__main__':
    rs_data_dir = '../lang_pref_learning/pref_learning/true_rewards_rs'
    mw_data_dir = '../lang_pref_learning/pref_learning/true_rewards_mw'
    plot(rs_data_dir, mw_data_dir)
