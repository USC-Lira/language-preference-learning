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
    cmap2 = plt.get_cmap('cividis')
    colors = cmap2(np.linspace(0, 1, len(lvls)))


    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # count
    count = [satisfaction.count(i) for i in lvls]
    ax.bar(lvls, count, color=colors, alpha=0.8, label="Count")

    ax.set_title("Improve Trajectory:\nSatisfaction with Final Trajectory", fontsize=18)
    ax.set_xticks(lvls)
    ax.set_ylim([0, 5.5])
    ax.set_xlabel("Satisfaction Level")
    ax.set_xticklabels(lvls)
    ax.set_ylabel("Count")
    plt.tight_layout(pad=2.0)

    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()

    plt.savefig("satisfaction.png")

def speed_adapt():
    global cmap

    improve_traj_speed = [5, 4, 4, 5, 5, 5, 3, 4, 3]
    lang_speed = [2, 4, 4, 2, 4, 1, 4, 5, 2, 4]
    pairwise_speed = [2, 3, 1, 3, 2, 4, 1, 3, 3, 2]

    # Sample 5 colors from the colormap
    num_colors = 3
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # y axis 1-5
    lvls = [1, 2, 3, 4, 5]
    # x axis
    experiments = ["Improve Trajectory", "Language", "Pairwise"]

    # avg
    avg_improve_traj_speed = np.mean(improve_traj_speed)
    avg_lang_speed = np.mean(lang_speed)
    avg_pairwise_speed = np.mean(pairwise_speed)

    # error
    std_improve_traj_speed = np.std(improve_traj_speed)
    std_lang_speed = np.std(lang_speed)
    std_pairwise_speed = np.std(pairwise_speed)

    positions = [0.4, 0.7, 0.85]

    # plot
    ax.bar(positions[0], avg_improve_traj_speed, color=colors[0], alpha=0.8, label="Average", width=0.2)
    ax.bar(positions[1], avg_lang_speed, color=colors[1], alpha=0.8, label="Average", width=0.1)
    ax.bar(positions[2], avg_pairwise_speed, color=colors[2], alpha=0.8, label="Average", width=0.1)
    ax.errorbar(positions, [avg_improve_traj_speed, avg_lang_speed, avg_pairwise_speed], yerr=[std_improve_traj_speed, std_lang_speed, std_pairwise_speed], color='black', label="Standard Deviation", capsize=3, fmt='o', markersize=0)

    plt.xticks(positions, experiments)
    ax.set_title("Speed to Adapt to Feedback", fontsize=18)
    ax.set_ylim([0, 5.5])
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Speed [1 Very slow ~ 5 Very fast]")
    plt.tight_layout(pad=2.0)

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()

    plt.savefig("speed_adapt.png")


def experience():
    global cmap

    improve_traj_exp = [5, 5, 4, 5, 5, 4, 4, 5, 4]
    lang_exp = [4, 5, 4, 2, 5, 4, 3, 4, 2, 3]
    pairwise_exp = [4, 5, 1, 3, 4, 1, 3, 2, 3, 3]

    # Sample 5 colors from the colormap
    num_colors = 3
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # y axis 1-5
    lvls = [1, 2, 3, 4, 5]
    # x axis
    experiments = ["Improve Trajectory", "Language", "Pairwise"]

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
    ax.bar(positions[0], [avg_improve_traj_exp], color=colors[0], alpha=0.8, label="Average", width=0.2)
    ax.bar(positions[1], avg_lang_exp, color=colors[1], alpha=0.8, label="Average", width=0.1)
    ax.bar(positions[2], avg_pairwise_exp, color=colors[2], alpha=0.8, label="Average", width=0.1)
    ax.errorbar(positions, [avg_improve_traj_exp, avg_lang_exp, avg_pairwise_exp], yerr=[std_improve_traj_exp, std_lang_exp, std_pairwise_exp], color='black', label="Standard Deviation", capsize=3, fmt='o', markersize=0)

    plt.xticks(positions, experiments)
    ax.set_title("Experience with Feedback", fontsize=18)
    ax.set_ylim([0, 5.5])
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Experience\n[1 Negatively ~ 5 Positively]")
    plt.tight_layout(pad=2.0)

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()

    plt.savefig("experience.png")

def iterations():
    iters = [9, 3, 5, 3, 4, 4, 9, 3, 10]
    lvls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.style.use('tableau-colorblind10')

    # Sample 10 colors from the colormap
    # num_colors = 10
    # colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    cmap2 = plt.get_cmap('cividis')
    colors = cmap2(np.linspace(0, 1, len(lvls)))

    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # y axis 1-10
    lvls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


    # count
    count = [iters.count(i) for i in lvls]

    ax.bar(lvls, count, color=colors, alpha=0.8, label="Count")

    ax.set_title("Improve Trajectory:\nIterations to Satisfaction", fontsize=18)
    ax.set_xticks(lvls)
    ax.set_ylim([0, 5.5])
    ax.set_xlabel("Number of Iterations")
    ax.set_xticklabels(lvls)
    ax.set_ylabel("Count")
    plt.tight_layout(pad=2.0)

    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()

    plt.savefig("iterations.png")

def adaptation():
    lang_adapt = [4, 4, 4, 4, 5, 2, 4, 5, 2, 5]
    pairwise_adapt = [1, 4, 3, 4, 3, 3, 4, 1]

    # Sample 5 colors from the colormap
    num_colors = 3
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    colors = colors[1:]

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

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
    ax.bar(positions, [avg_lang_adapt, avg_pairwise_adapt], color=colors, alpha=0.8, label="Average", width=0.2)

    ax.errorbar(positions, [avg_lang_adapt, avg_pairwise_adapt], yerr=[std_lang_adapt, std_pairwise_adapt], color='black', label="Standard Deviation", capsize=3, fmt='o', markersize=0)

    plt.xticks(positions, experiments)
    ax.set_title("Preference Learning:\nAdaptation to Feedback", fontsize=18)
    ax.set_ylim([0, 5.5])
    ax.set_xlabel("Method")
    ax.set_ylabel("Adaptation\n[1 Not at all ~ 5 Constantly]")
    plt.tight_layout(pad=2.0)

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()

    plt.savefig("adaptation.png")

def trust():
    lang_trust = [3, 5, 4, 3, 5, 1, 3, 4, 2, 4]
    pairwise_trust = [3, 4, 1, 4, 3, 4, 3, 3, 4, 3]

    # Sample 5 colors from the colormap
    num_colors = 3
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    colors = colors[1:]

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

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
    ax.bar(positions, [avg_lang_trust, avg_pairwise_trust], color=colors, alpha=0.8, label="Average", width=0.2)

    ax.errorbar(positions, [avg_lang_trust, avg_pairwise_trust], yerr=[std_lang_trust, std_pairwise_trust], color='black', label="Standard Deviation", capsize=3, fmt='o', markersize=0)

    plt.xticks(positions, experiments)
    ax.set_title("Preference Learning:\nImpact on Trust in Capabilities", fontsize=18)
    ax.set_ylim([0, 5.5])
    ax.set_xlabel("Method")
    ax.set_ylabel("Adaptation\n[1 Negatively ~ 5 Positively]")
    plt.tight_layout(pad=2.0)

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()

    plt.savefig("trust.png")

def aspect():
    lang_aspects = ["Time efficiency", "Convenience", 
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
"Not Adaptability", "Time efficiency", "Convenience", 
"Adaptability", "Time efficiency", "Convenience", 
"Adaptability", "Time efficiency", "Convenience", 
"Adaptability", "Not Time efficiency", "Not Convenience", 
"Adaptability", "Time efficiency", "Convenience", 
"Not Adaptability", "Time efficiency", "Not Convenience", 
"Adaptability", "Time efficiency", "Convenience", 
"Adaptability", "Time efficiency", "Convenience", 
"Adaptability", "Time efficiency", "Not Convenience", 
"Adaptability", "Time efficiency", "Not Convenience"]
    
    pairwise_aspects = [
"Convenience",
"Adaptability", "Convenience",
"None of them",
"Adaptability", "Convenience",
"Convenience",
"Adaptability", "Convenience",
"Time efficiency", "Convenience",
"None of them",
"Convenience",
"Convenience",
    ]

    pairwise_aspects_with_neg = [
"Not Adaptability", "Not Time efficiency", "Convenience",
"Adaptability", "Not Time efficiency", "Convenience",
"Not Adaptability", "Not Time efficiency", "Not Convenience",
"Adaptability", "Not Time efficiency", "Convenience",
"Not Adaptability", "Not Time efficiency", "Convenience",
"Adaptability", "Not Time efficiency", "Convenience",
"Not Adaptability", "Time efficiency", "Convenience",
"Not Adaptability", "Not Time efficiency", "Not Convenience",
"Not Adaptability", "Not Time efficiency","Convenience",
"Not Adaptability", "Not Time efficiency", "Convenience",
    ]

    # Sample 5 colors from the colormap
    num_colors = 3
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

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
    ax.bar(positions[0] - 0.11, count_lang[0], color=colors[1], alpha=0.8, label="Time efficiency", width=0.2)
    ax.bar(positions[1] - 0.11, count_lang[1], color=colors[1], alpha=0.8, label="Convenience", width=0.2)
    ax.bar(positions[2] - 0.11, count_lang[2], color=colors[1], alpha=0.8, label="Adaptability", width=0.2)

    ax.bar(positions[0] + 0.11, count_pairwise[0], color=colors[2], alpha=0.8, label="Time efficiency", width=0.2)
    ax.bar(positions[1] + 0.11, count_pairwise[1], color=colors[2], alpha=0.8, label="Convenience", width=0.2)
    ax.bar(positions[2] + 0.11, count_pairwise[2], color=colors[2], alpha=0.8, label="Adaptability", width=0.2)

    plt.xticks(positions, aspects)
    ax.set_title("Preference Learning:\nSatisfactory Aspects", fontsize=18)
    ax.set_ylim([0, 10])
    ax.set_xlabel("Aspects")
    ax.set_ylabel("Count")
    plt.tight_layout(pad=2.0)

    # legend
    ax.legend(["Language", "Pairwise"], loc="upper right")
    # mod colors
    ax.get_legend().legend_handles[0].set_color(colors[1])
    ax.get_legend().legend_handles[1].set_color(colors[2])

    plt.savefig("aspects_count.png")

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(2)
    plt.close()


    # ======================

    plt.style.use('tableau-colorblind10')

    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # y axis 1-5
    lvls = [-1, 0, 1]
    
    # percent
    perc_lang = [(lang - neg) / 10 for lang, neg in zip(count_lang, count_lang_neg)]
    perc_pairwise = [(pair - neg) / 10 for pair, neg in zip(count_pairwise, count_pairwise_neg)]

    positions = [0.2, 0.7, 1.3]

    # plot
    ax.bar(positions[0] - 0.11, perc_lang[0], color=colors[1], alpha=0.8, label="Time efficiency", width=0.2)
    ax.bar(positions[1] - 0.11, perc_lang[1], color=colors[1], alpha=0.8, label="Convenience", width=0.2)
    ax.bar(positions[2] - 0.11, perc_lang[2], color=colors[1], alpha=0.8, label="Adaptability", width=0.2)

    ax.bar(positions[0] + 0.11, perc_pairwise[0], color=colors[2], alpha=0.8, label="Time efficiency", width=0.2)
    ax.bar(positions[1] + 0.11, perc_pairwise[1], color=colors[2], alpha=0.8, label="Convenience", width=0.2)
    ax.bar(positions[2] + 0.11, perc_pairwise[2], color=colors[2], alpha=0.8, label="Adaptability", width=0.2)

    plt.xticks(positions, aspects)
    ax.set_title("Preference Learning:\nAspect Satisfaction", fontsize=18)
    ax.set_ylim([-1, 1])
    ax.set_xlabel("Aspects")
    ax.set_ylabel("Satisfaction\n[-1 Negative ~ 1 Positive]")
    plt.tight_layout(pad=2.0)

    # legend
    ax.legend(["Language", "Pairwise"], loc="upper right")
    # mod colors
    ax.get_legend().legend_handles[0].set_color(colors[1])
    ax.get_legend().legend_handles[1].set_color(colors[2])

    plt.savefig("aspects_perc.png")

    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()



if __name__ == "__main__":
    satisfaction()
    speed_adapt()
    experience()
    iterations()
    adaptation()
    trust()
    aspect()