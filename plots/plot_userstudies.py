import matplotlib.patches as patches
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Palatino Linotype"]
rcParams["font.size"] = 15

import matplotlib.pyplot as plt

# Define the start and end colors
start_color = "#ca8742"
end_color = "#517a80"

# Create a colormap from the two colors
cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color])

lang_color = "#F79646"
pairwise_color = "#4BACC6"
improve_color = "#B3A2C7"

# current directory of this file
# dir = os.path.dirname(os.path.realpath(__file__))
dir = "C:/Users/Rosies/Desktop/Things/LiraLab/CoRL_2025/figures"
os.chdir(dir)


def satisfaction():
    global cmap

    lvls = [1, 2, 3, 4, 5]
    satisfaction = [5, 5, 3, 4, 5, 5, 3, 5, 4]    

    # Sample 5 colors from the colormap
    # num_colors = 3
    # colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    # cmap2 = plt.get_cmap('cividis')
    # colors = cmap2(np.linspace(0, 1, len(lvls)))


    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # count
    count = [satisfaction.count(i) for i in lvls]
    ax.bar(lvls, count, color=improve_color, alpha=0.8, label="Count")

    ax.set_title("Improve Trajectory:\nSatisfaction with Final Trajectory", fontsize=18)
    ax.set_xticks(lvls)
    ax.set_ylim([0, 5.5])
    ax.set_xlabel("Satisfaction Level\n[1 Completely unsatisfied ~ 5 Completely satisfied]")
    ax.set_xticklabels(lvls)
    ax.set_ylabel("Count")
    plt.tight_layout(pad=2.0)

    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()

    plt.savefig("satisfaction.pdf")

def speed_improve():
    global cmap

    improve_traj_speed = [5, 4, 4, 5, 5, 5, 3, 4, 3]
    # lang_speed = [2, 4, 4, 2, 4, 1, 4, 5, 2, 4]
    # pairwise_speed = [2, 3, 1, 3, 2, 4, 1, 3, 3, 2]

    # Sample 5 colors from the colormap
    # num_colors = 3
    # colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    fig.set_figwidth(fig.get_figwidth() / 1.4)

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # avg
    avg_improve_traj_speed = np.mean(improve_traj_speed)
    # avg_lang_speed = np.mean(lang_speed)
    # avg_pairwise_speed = np.mean(pairwise_speed)

    # error
    std_improve_traj_speed = np.std(improve_traj_speed)
    # std_lang_speed = np.std(lang_speed)
    # std_pairwise_speed = np.std(pairwise_speed)

    # positions = [0.4, 0.7, 0.85]

    # plot
    ax.bar(0, avg_improve_traj_speed, color=improve_color, alpha=0.8, label="Average", width=0.2)
    # ax.bar(positions[1], avg_lang_speed, color=colors[1], alpha=0.8, label="Average", width=0.1)
    # ax.bar(positions[2], avg_pairwise_speed, color=colors[2], alpha=0.8, label="Average", width=0.1)
    ax.errorbar(0, avg_improve_traj_speed, yerr=std_improve_traj_speed, color='black', label="Standard Deviation", capsize=3, fmt='o', markersize=0)

    # plt.xticks(positions, experiments)
    ax.set_title("Improve Trajectory:\nSpeed to Adapt to Feedback", fontsize=15)
    ax.set_ylim([1, 5.5])
    ax.set_xlim([-0.3, 0.3])
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Speed\n[1 Very slow ~ 5 Very fast]")
    plt.tight_layout(pad=2.0)

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()

    plt.savefig("speed_improve.pdf")

def speed_pref():
    global cmap

    # improve_traj_speed = [5, 4, 4, 5, 5, 5, 3, 4, 3]
    lang_speed = [2, 4, 4, 2, 4, 1, 4, 5, 2, 4]
    pairwise_speed = [2, 3, 1, 3, 2, 4, 1, 3, 3, 2]

    # Sample 5 colors from the colormap
    # num_colors = 3
    # colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    fig.set_figwidth(fig.get_figwidth() / 1.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # y axis 1-5
    lvls = [1, 2, 3, 4, 5]
    # x axis
    experiments = ["Language", "Pairwise"]

    # avg
    # avg_improve_traj_speed = np.mean(improve_traj_speed)
    avg_lang_speed = np.mean(lang_speed)
    avg_pairwise_speed = np.mean(pairwise_speed)

    # error
    # std_improve_traj_speed = np.std(improve_traj_speed)
    std_lang_speed = np.std(lang_speed)
    std_pairwise_speed = np.std(pairwise_speed)

    positions = [0.4, 0.7, 0.85]

    # plot
    # ax.bar(positions[0], avg_improve_traj_speed, color=colors[0], alpha=0.8, label="Average", width=0.2)
    ax.bar(positions[1], avg_lang_speed, color=lang_color, alpha=0.8, label="Average", width=0.1)
    ax.bar(positions[2], avg_pairwise_speed, color=pairwise_color, alpha=0.8, label="Average", width=0.1)
    ax.errorbar(positions[1:], [avg_lang_speed, avg_pairwise_speed], yerr=[std_lang_speed, std_pairwise_speed], color='black', label="Standard Deviation", capsize=3, fmt='o', markersize=0)

    plt.xticks(positions[1:], experiments)
    ax.set_title("Preference Learning:\nSpeed to Adapt to Feedback", fontsize=15)
    ax.set_ylim([1, 5.5])
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Speed\n[1 Very slow ~ 5 Very fast]")
    plt.tight_layout(pad=1.0)

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()

    plt.savefig("speed_pref.pdf")


def experience_improve():
    global cmap

    improve_traj_exp = [5, 5, 4, 5, 5, 4, 4, 5, 4]
    lang_exp = [4, 5, 4, 2, 5, 4, 3, 4, 2, 3]
    pairwise_exp = [4, 5, 1, 3, 4, 1, 3, 2, 3, 3]

    # Sample 5 colors from the colormap
    # num_colors = 3
    # colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    fig, ax = plt.subplots()

    fig.set_figwidth(fig.get_figwidth() / 1.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", which="major", labelsize=14)  # Adjust font size of ticks

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    # avg
    avg_improve_traj_exp = np.mean(improve_traj_exp)

    # error
    std_improve_traj_exp = np.std(improve_traj_exp)

    # plot
    ax.bar(0, avg_improve_traj_exp, color=improve_color, alpha=0.8, label="Average", width=0.2)
    ax.errorbar(0, avg_improve_traj_exp, yerr=std_improve_traj_exp, color='black', label="Standard Deviation", capsize=3, fmt='o', markersize=0)

    ax.set_title("Improve Trajectory:\nExperience with Feedback", fontsize=16)
    ax.set_ylim([1, 5.5])
    ax.set_xlim([-0.2, 0.2])
    ax.set_xlabel("Improve Trajectory")
    ax.set_ylabel("Experience\n[1 Negatively ~ 5 Positively]")
    plt.tight_layout(pad=2.0)

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()

    plt.savefig("experience_improve.pdf")

def experience_pref():
    global cmap

    improve_traj_exp = [5, 5, 4, 5, 5, 4, 4, 5, 4]
    lang_exp = [4, 5, 4, 2, 5, 4, 3, 4, 2, 3]
    pairwise_exp = [4, 5, 1, 3, 4, 1, 3, 2, 3, 3]

    # Sample 5 colors from the colormap
    # num_colors = 3
    # colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    fig.set_figwidth(fig.get_figwidth() / 1.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # y axis 1-5
    lvls = [1, 2, 3, 4, 5]
    # x axis
    experiments = ["Language", "Pairwise"]

    # avg
    avg_improve_traj_exp = np.mean(improve_traj_exp)
    avg_lang_exp = np.mean(lang_exp)
    avg_pairwise_exp = np.mean(pairwise_exp)

    # error
    std_improve_traj_exp = np.std(improve_traj_exp)
    std_lang_exp = np.std(lang_exp)
    std_pairwise_exp = np.std(pairwise_exp)

    positions = [0.4, 0.7, 0.85]

    # plot
    # ax.bar(positions[0], [avg_improve_traj_exp], color=colors[0], alpha=0.8, label="Average", width=0.2)
    ax.bar(positions[1], avg_lang_exp, color=lang_color, alpha=0.8, label="Average", width=0.1)
    ax.bar(positions[2], avg_pairwise_exp, color=pairwise_color, alpha=0.8, label="Average", width=0.1)
    ax.errorbar(positions[1:], [avg_lang_exp, avg_pairwise_exp], yerr=[std_lang_exp, std_pairwise_exp], color='black', label="Standard Deviation", capsize=3, fmt='o', markersize=0)

    plt.xticks(positions[1:], experiments)
    ax.set_title("Preference Learning:\nExperience with Feedback", fontsize=15)
    ax.set_ylim([1, 5.5])
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Experience\n[1 Negatively ~ 5 Positively]")
    plt.tight_layout(pad=1.0)

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()

    plt.savefig("experience_pref.pdf")

def iterations():
    iters = [9, 3, 5, 3, 4, 4, 9, 3, 10]
    lvls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.style.use('tableau-colorblind10')

    # Sample 10 colors from the colormap
    # num_colors = 10
    # colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    # cmap2 = plt.get_cmap('cividis')
    # colors = cmap2(np.linspace(0, 1, len(lvls)))

    fig, ax = plt.subplots()

    fig.set_figwidth(fig.get_figwidth() / 1.5)
    fig.set_figheight(fig.get_figheight() / 1.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # y axis 1-10
    lvls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


    # count
    count = [iters.count(i) for i in lvls]

    ax.bar(lvls, count, color=improve_color, alpha=0.8, label="Count")

    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)

    ax.set_title("Improve Trajectory:\nIterations to Satisfaction", fontsize=23)
    ax.set_xlabel("Number of Iterations", fontsize=21)
    ax.set_xticks(lvls)
    ax.set_xticklabels(lvls, fontsize=18)
    ax.set_xlim([0, 11])

    ax.set_ylabel("Count", fontsize=21)
    ax.set_yticks([0, 1, 2, 3, 4])
    # plt.tight_layout(pad=2.0)
    plt.tight_layout(rect=[-0.05, 0.02, 1.05, 1.03])

    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()

    plt.savefig("iterations.pdf")

def adaptation():
    lang_adapt = [4, 4, 4, 4, 5, 2, 4, 5, 2, 5]
    pairwise_adapt = [1, 4, 3, 4, 3, 3, 4, 1]

    # # Sample 5 colors from the colormap
    # num_colors = 3
    # colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    # colors = colors[1:]

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    fig.set_figwidth(fig.get_figwidth() / 1.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # y axis 1-5
    lvls = [1, 2, 3, 4, 5]
    # x axis
    experiments = ["Language", "Pairwise"]

    # avg
    avg_lang_adapt = np.mean(lang_adapt)
    avg_pairwise_adapt = np.mean(pairwise_adapt)

    # error
    std_lang_adapt = np.std(lang_adapt)
    std_pairwise_adapt = np.std(pairwise_adapt)

    # bars
    positions = [0.4, 0.7]

    # plot
    ax.bar(positions, [avg_lang_adapt, avg_pairwise_adapt], color=[lang_color, pairwise_color], alpha=0.8, label="Average", width=0.2)

    ax.errorbar(positions, [avg_lang_adapt, avg_pairwise_adapt], yerr=[std_lang_adapt, std_pairwise_adapt], color='black', label="Standard Deviation", capsize=3, fmt='o', markersize=0)

    plt.xticks(positions, experiments)
    ax.set_title("Preference Learning:\nAdaptation to Feedback", fontsize=15)
    ax.set_ylim([1, 5.5])
    ax.set_xlabel("Method")
    ax.set_ylabel("Adaptation\n[1 Not at all ~ 5 Constantly]")
    plt.tight_layout(pad=1.0)

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()

    plt.savefig("adaptation.pdf")

def trust():
    lang_trust = [3, 5, 4, 3, 5, 1, 3, 4, 2, 4]
    pairwise_trust = [3, 4, 1, 4, 3, 4, 3, 3, 4, 3]

    # Sample 5 colors from the colormap
    # num_colors = 3
    # colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    # colors = colors[1:]

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    fig.set_figwidth(fig.get_figwidth() / 1.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=13)  # Adjust font size of ticks

    # y axis 1-5
    lvls = [1, 2, 3, 4, 5]
    # x axis
    experiments = ["Language", "Pairwise"]

    # avg
    avg_lang_trust = np.mean(lang_trust)
    avg_pairwise_trust = np.mean(pairwise_trust)

    # error
    std_lang_trust = np.std(lang_trust)
    std_pairwise_trust = np.std(pairwise_trust)

    # bars
    positions = [0.4, 0.7]

    # plot
    ax.bar(positions, [avg_lang_trust, avg_pairwise_trust], color=[lang_color, pairwise_color], alpha=0.8, label="Average", width=0.2)

    ax.errorbar(positions, [avg_lang_trust, avg_pairwise_trust], yerr=[std_lang_trust, std_pairwise_trust], color='black', label="Standard Deviation", capsize=3, fmt='o', markersize=0)

    plt.xticks(positions, experiments)
    ax.set_title("Preference Learning:\nImpact on Trust", fontsize=15)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(ymin=1)
    ax.set_xlabel("Method")
    ax.set_ylabel("Adaptation\n[1 Negatively ~ 5 Positively]", fontsize=13)
    plt.tight_layout(pad=1.0)

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()

    plt.savefig("trust.pdf")

def aspect():
    lang_aspects = [
                "Time efficiency", "Convenience", 
"Adaptability", "Time efficiency", "Convenience", 
"Adaptability", "Time efficiency", "Convenience", 
"Adaptability", 
"Adaptability", "Time efficiency", "Convenience", 
                "Time efficiency", 
"Adaptability", "Time efficiency", "Convenience", 
"Adaptability", "Time efficiency", "Convenience", 
"Adaptability", "Time efficiency", 
"Adaptability", "Time efficiency"]
    
    lang_aspects_with_neg = [
"Not Adaptability", "Time efficiency",      "Convenience", 
"Adaptability",     "Time efficiency",      "Convenience", 
"Adaptability",     "Time efficiency",      "Convenience", 
"Adaptability",     "Not Time efficiency",  "Not Convenience", 
"Adaptability",     "Time efficiency",      "Convenience", 
"Not Adaptability", "Time efficiency",      "Not Convenience", 
"Adaptability",     "Time efficiency",      "Convenience", 
"Adaptability",     "Time efficiency",      "Convenience", 
"Adaptability",     "Time efficiency",      "Not Convenience", 
"Adaptability",     "Time efficiency",      "Not Convenience"]
    
    pairwise_aspects = [
                                    "Convenience",
"Adaptability",                     "Convenience",
"None of them",
"Adaptability",                     "Convenience",
                                    "Convenience",
"Adaptability",                     "Convenience",
                "Time efficiency",  "Convenience",
"None of them",
                                    "Convenience",
                                    "Convenience",
    ]

    pairwise_aspects_with_neg = [
"Not Adaptability", "Not Time efficiency",  "Convenience",
"Adaptability",     "Not Time efficiency",  "Convenience",
"Not Adaptability", "Not Time efficiency",  "Not Convenience",
"Adaptability",     "Not Time efficiency",  "Convenience",
"Not Adaptability", "Not Time efficiency",  "Convenience",
"Adaptability",     "Not Time efficiency",  "Convenience",
"Not Adaptability", "Time efficiency",      "Convenience",
"Not Adaptability", "Not Time efficiency",  "Not Convenience",
"Not Adaptability", "Not Time efficiency",  "Convenience",
"Not Adaptability", "Not Time efficiency",  "Convenience",
    ]

    # Sample 5 colors from the colormap
    # num_colors = 3
    # colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # y axis 1-5
    lvls = [1, 2, 3, 4, 5]
    # x axis
    aspects = ["Time efficiency", "Convenience", "Adaptability"]
    neg_aspects = ["Not Time efficiency", "Not Convenience", "Not Adaptability"]
    experiments = ["Language", "Pairwise"]

    # count
    count_lang = [lang_aspects.count(i) for i in aspects]
    count_lang_neg = [lang_aspects_with_neg.count(i) for i in neg_aspects]

    count_pairwise = [pairwise_aspects.count(i) for i in aspects]
    count_pairwise_neg = [pairwise_aspects_with_neg.count(i) for i in neg_aspects]

    positions = [0.2, 0.7, 1.3]

    # plot
    ax.bar(positions[0] - 0.11, count_lang[0], color=lang_color, alpha=0.8, label="Time efficiency", width=0.2)
    ax.bar(positions[1] - 0.11, count_lang[1], color=lang_color, alpha=0.8, label="Convenience", width=0.2)
    ax.bar(positions[2] - 0.11, count_lang[2], color=lang_color, alpha=0.8, label="Adaptability", width=0.2)

    ax.bar(positions[0] + 0.11, count_pairwise[0], color=pairwise_color, alpha=0.8, label="Time efficiency", width=0.2)
    ax.bar(positions[1] + 0.11, count_pairwise[1], color=pairwise_color, alpha=0.8, label="Convenience", width=0.2)
    ax.bar(positions[2] + 0.11, count_pairwise[2], color=pairwise_color, alpha=0.8, label="Adaptability", width=0.2)

    plt.xticks(positions, aspects)
    ax.set_title("Preference Learning:\nSatisfactory Aspects", fontsize=14)
    ax.set_ylim([0, 10])
    ax.set_xlabel("Aspects")
    ax.set_ylabel("Count")
    plt.tight_layout(pad=2.0)

    # legend
    ax.legend(["Language", "Pairwise"], loc="upper right")
    # mod colors
    ax.get_legend().legend_handles[0].set_color(lang_color)
    ax.get_legend().legend_handles[1].set_color(pairwise_color)

    plt.savefig("aspects_count.pdf")

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(2)
    plt.close()


    # ======================

    # plt.style.use('tableau-colorblind10')

    # fig, ax = plt.subplots()

    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # # y axis 1-5
    # lvls = [-1, 0, 1]
    
    # # percent
    # perc_lang = [(lang - neg) / 10 for lang, neg in zip(count_lang, count_lang_neg)]
    # perc_pairwise = [(pair - neg) / 10 for pair, neg in zip(count_pairwise, count_pairwise_neg)]

    # positions = [0.2, 0.7, 1.3]

    # # plot
    # ax.bar(positions[0] - 0.11, perc_lang[0], color=colors[1], alpha=0.8, label="Time efficiency", width=0.2)
    # ax.bar(positions[1] - 0.11, perc_lang[1], color=colors[1], alpha=0.8, label="Convenience", width=0.2)
    # ax.bar(positions[2] - 0.11, perc_lang[2], color=colors[1], alpha=0.8, label="Adaptability", width=0.2)

    # ax.bar(positions[0] + 0.11, perc_pairwise[0], color=colors[2], alpha=0.8, label="Time efficiency", width=0.2)
    # ax.bar(positions[1] + 0.11, perc_pairwise[1], color=colors[2], alpha=0.8, label="Convenience", width=0.2)
    # ax.bar(positions[2] + 0.11, perc_pairwise[2], color=colors[2], alpha=0.8, label="Adaptability", width=0.2)

    # plt.xticks(positions, aspects)
    # ax.set_title("Preference Learning:\nAspect Satisfaction", fontsize=18)
    # ax.set_ylim([-1, 1])
    # ax.set_xlabel("Aspects")
    # ax.set_ylabel("Satisfaction\n[-1 Negative ~ 1 Positive]")
    # plt.tight_layout(pad=2.0)

    # # legend
    # ax.legend(["Language", "Pairwise"], loc="lower right")
    # # mod colors
    # ax.get_legend().legend_handles[0].set_color(colors[1])
    # ax.get_legend().legend_handles[1].set_color(colors[2])

    # plt.savefig("aspects_perc.pdf")

    # # plt.show(block=True)
    # # plt.show(block=False)
    # # plt.pause(2)
    # # plt.close()

def all_pref():
    global cmap

    plt.close()

    improve_traj_exp = [5, 5, 4, 5, 5, 4, 4, 5, 4]
    lang_exp = [4, 5, 4, 2, 5, 4, 3, 4, 2, 3]
    pairwise_exp = [4, 5, 1, 3, 4, 1, 3, 2, 3, 3]

    lang_speed = [2, 4, 4, 2, 4, 1, 4, 5, 2, 4]
    pairwise_speed = [2, 3, 1, 3, 2, 4, 1, 3, 3, 2]

    lang_adapt = [4, 4, 4, 4, 5, 2, 4, 5, 2, 5]
    pairwise_adapt = [1, 4, 3, 4, 3, 3, 4, 1]

    lang_trust = [3, 5, 4, 3, 5, 1, 3, 4, 2, 4]
    pairwise_trust = [3, 4, 1, 4, 3, 4, 3, 3, 4, 3]

    # Sample 5 colors from the colormap
    # num_colors = 3
    # colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    fig.set_figwidth(fig.get_figwidth() / 1.2)
    fig.set_figheight(fig.get_figheight() / 1.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # y axis 1-5
    lvls = [1, 2, 3, 4, 5]
    # x axis
    experiments = ["Experience", "Speed", "Adaptation", "Trust"]

    # avg
    avg_lang_exp = np.mean(lang_exp)
    avg_pairwise_exp = np.mean(pairwise_exp)

    avg_lang_speed = np.mean(lang_speed)
    avg_pairwise_speed = np.mean(pairwise_speed)

    avg_lang_adapt = np.mean(lang_adapt)
    avg_pairwise_adapt = np.mean(pairwise_adapt)

    avg_lang_trust = np.mean(lang_trust)
    avg_pairwise_trust = np.mean(pairwise_trust)

    print("avg_lang_exp:", avg_lang_exp)
    print("avg_pairwise_exp:", avg_pairwise_exp)
    print("avg_lang_speed:", avg_lang_speed)
    print("avg_pairwise_speed:", avg_pairwise_speed)
    print("avg_lang_adapt:", avg_lang_adapt)
    print("avg_pairwise_adapt:", avg_pairwise_adapt)
    print("avg_lang_trust:", avg_lang_trust)
    print("avg_pairwise_trust:", avg_pairwise_trust)

    # error
    std_lang_exp = np.std(lang_exp)
    std_pairwise_exp = np.std(pairwise_exp)

    std_lang_speed = np.std(lang_speed)
    std_pairwise_speed = np.std(pairwise_speed)

    std_lang_adapt = np.std(lang_adapt)
    std_pairwise_adapt = np.std(pairwise_adapt)

    std_lang_trust = np.std(lang_trust)
    std_pairwise_trust = np.std(pairwise_trust)

    width = 0.4
    positions = [0, 1, 2, 3]
    positions_lang = [i - (width/2) for i in positions]
    positions_pair = [i + (width/2) for i in positions]

    # widtht
    # ax.bar(positions[0], [avg_improve_traj_exp], color=colors[0], alpha=0.8, label="Average", width=0.2)
    temp_langs = [avg_lang_exp, avg_lang_speed, avg_lang_adapt, avg_lang_trust]
    temp_pairs = [avg_pairwise_exp, avg_pairwise_speed, avg_pairwise_adapt, avg_pairwise_trust]
    ax.bar(positions_lang, temp_langs, color=lang_color, alpha=0.8, width=width, label="Language")
    ax.bar(positions_pair, temp_pairs, color=pairwise_color, alpha=0.8, width=width, label="Comparison")
    # ax.bar(positions[1]-(width/2), avg_lang_speed, color=lang_color, alpha=0.8, width=width, label="Language")
    # ax.bar(positions[1]+(width/2), avg_pairwise_speed, color=pairwise_color, alpha=0.8, width=width, label="Comparison")
    # ax.bar(positions[2]-(width/2), avg_lang_adapt, color=lang_color, alpha=0.8, width=width, label="Language")
    # ax.bar(positions[2]+(width/2), avg_pairwise_adapt, color=pairwise_color, alpha=0.8, width=width, label="Comparison")
    # ax.bar(positions[3]-(width/2), avg_lang_trust, color=lang_color, alpha=0.8, width=width, label="Language")
    # ax.bar(positions[3]+(width/2), avg_pairwise_trust, color=pairwise_color, alpha=0.8, width=width, label="Comparison")

    ax.errorbar(positions[0]-(width/2), avg_lang_exp, yerr=std_lang_exp, color='black', capsize=3, fmt='o', markersize=0)
    ax.errorbar(positions[0]+(width/2), avg_pairwise_exp, yerr=std_pairwise_exp, color='black', capsize=3, fmt='o', markersize=0)
    ax.errorbar(positions[1]-(width/2), avg_lang_speed, yerr=std_lang_speed, color='black', capsize=3, fmt='o', markersize=0)
    ax.errorbar(positions[1]+(width/2), avg_pairwise_speed, yerr=std_pairwise_speed, color='black', capsize=3, fmt='o', markersize=0)
    ax.errorbar(positions[2]-(width/2), avg_lang_adapt, yerr=std_lang_adapt, color='black', capsize=3, fmt='o', markersize=0)
    ax.errorbar(positions[2]+(width/2), avg_pairwise_adapt, yerr=std_pairwise_adapt, color='black', capsize=3, fmt='o', markersize=0)
    ax.errorbar(positions[3]-(width/2), avg_lang_trust, yerr=std_lang_trust, color='black', capsize=3, fmt='o', markersize=0)
    ax.errorbar(positions[3]+(width/2), avg_pairwise_trust, yerr=std_pairwise_trust, color='black', capsize=3, fmt='o', markersize=0)

    # ax.errorbar(positions, [avg_lang_exp, avg_pairwise_exp, avg_lang_speed, avg_pairwise_speed, avg_lang_adapt, avg_pairwise_adapt, avg_lang_trust, avg_pairwise_trust], yerr=[std_lang_exp, std_pairwise_exp, std_lang_speed, std_pairwise_speed, std_lang_adapt, std_pairwise_adapt, std_lang_trust, std_pairwise_trust], color='black', capsize=3, fmt='o', markersize=0)

    plt.xticks(positions, experiments, fontsize=20)
    plt.yticks(fontsize=20)

    ax.set_title("Preference Learning:\nAverage Score for Attributes", fontsize=21)
    ax.set_xlabel("Attributes", fontsize=22)
    ax.set_ylim([1, 5.5])
    ax.set_ylabel("Average Score", fontsize=22)

    plt.tight_layout(rect=[-0.04, -0.05, 1.05, 1.03])
    # plt.tight_layout(pad=1.0)

    # ax.legend(frameon=False)
    # plt.legend(loc='upper left', fontsize=19)
    # ax.get_legend().legend_handles[0].set_color(lang_color)
    # ax.get_legend().legend_handles[1].set_color(pairwise_color)

    # r = patches.Patch(facecolor=lang_color, label='Language')
    # b = patches.Patch(facecolor=pairwise_color, label='Comparison')

    # plt.legend(loc='lower right', fontsize=13)
    # plt.legend().get_frame().set_alpha(0)
    legend = plt.legend(bbox_to_anchor=(0.2, 1.05), loc='upper left', fontsize=12)
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')
    plt.savefig("all_pref.pdf")
    plt.savefig("all_pref.png")

 
    # plt.show(block=True)
    # # plt.show(block=False)
    # plt.pause(2)
    # plt.close()

def all_improve():
    global cmap

    plt.close()

    improve_traj_exp = [5, 5, 4, 5, 5, 4, 4, 5, 4]
    exp_avg = np.mean(improve_traj_exp)
    exp_std = np.std(improve_traj_exp)
    improve_traj_speed = [5, 4, 4, 5, 5, 5, 3, 4, 3]
    speed_avg = np.mean(improve_traj_speed)
    speed_std = np.std(improve_traj_speed)
    satisfaction = [5, 5, 3, 4, 5, 5, 3, 5, 4]    
    sat_avg = np.mean(satisfaction)
    sat_std = np.std(satisfaction)


    # Sample 5 colors from the colormap
    # num_colors = 3
    # colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    fig.set_figwidth(fig.get_figwidth() / 1.7)
    fig.set_figheight(fig.get_figheight() / 1.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # y axis 1-5
    lvls = [1, 2, 3, 4, 5]
    # x axis
    experiments = ["Experience", "Speed", "Satisfaction"]

    positions = [0, 1, 2]

    # widtht
    # ax.bar(positions[0], [avg_improve_traj_exp], color=colors[0], alpha=0.8, label="Average", width=0.2)
    width = 0.6
    ax.bar(positions[0], exp_avg, color=improve_color, alpha=0.8, width=width)
    ax.bar(positions[1], speed_avg, color=improve_color, alpha=0.8, width=width)
    ax.bar(positions[2], sat_avg, color=improve_color, alpha=0.8, width=width)

    ax.errorbar(positions[0], exp_avg, yerr=exp_std, color='black', capsize=3, fmt='o', markersize=0)
    ax.errorbar(positions[1], speed_avg, yerr=speed_std, color='black', capsize=3, fmt='o', markersize=0)
    ax.errorbar(positions[2], sat_avg, yerr=sat_std, color='black', capsize=3, fmt='o', markersize=0)

    # ax.errorbar(positions, [avg_lang_exp, avg_pairwise_exp, avg_lang_speed, avg_pairwise_speed, avg_lang_adapt, avg_pairwise_adapt, avg_lang_trust, avg_pairwise_trust], yerr=[std_lang_exp, std_pairwise_exp, std_lang_speed, std_pairwise_speed, std_lang_adapt, std_pairwise_adapt, std_lang_trust, std_pairwise_trust], color='black', capsize=3, fmt='o', markersize=0)

    plt.xticks(positions, experiments, fontsize=17)
    plt.yticks(fontsize=18)
    ax.set_title("Improve Trajectory:\nAverage Attribute Scores", fontsize=21)
    ax.set_ylim([1, 6])
    ax.set_xlabel("Attributes", fontsize=20)
    ax.set_ylabel("Average Score", fontsize=20)

    plt.tight_layout(rect=[-0.05,0,1.05,1])

    plt.savefig("all_improve.pdf")
 
    # plt.show(block=True)
    # # plt.show(block=False)
    # plt.pause(2)
    # plt.close()


if __name__ == "__main__":
    iterations()
    aspect()

    # experience_improve()
    # speed_improve()
    # satisfaction()
    all_improve()
    # experience_pref()
    # speed_pref()
    # adaptation()
    # trust()
    all_pref()