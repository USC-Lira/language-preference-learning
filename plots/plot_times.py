import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Palatino Linotype"]
rcParams["font.size"] = 15

def plot_lang():

    # xiang_lang  = [40, 38, 51, 33, 33, 44, 26, 35, 38, 32, 32, 39, 37, 41, 26, 35, 25, 41, 30, 32]
    # xiang_pair  = [64, 61, 73, 63, 76, 81, 68, 59, 50, 51, 59, 45, 48, 63, 55, 58, 59, 48, 61, 48]

    # bas_lang_1  = [70, 175, 127, 74, 104, 98, 65, 85, 50, 83, 53, 84, 60, 95, 137]
    # bas_lang_2  = [54, 70, 35, 60, 120, 56, 91, 21, 136, 123, 56, 104, 64, 39, 61]
    # bas_pair    = [51, 51, 37, 60, 71, 66, 57, 57, 67, 54, 73, 39, 62, 57, 73, 60, 55, 73, 44, 72]

    # shreya_lang = [41, 59, 47, 31, 64, 40, 40, 59, 48, 60, 48, 30, 53, 40, 37, 40, 44, 33, 41, 35]
    # shreya_pair = [73, 75, 54, 54, 52, 53, 57, 56, 54, 46, 69, 57, 64, 59, 58, 62, 46, 45, 62, 46]

    # eisuke_lang = [87, 50, 36, 27, 28, 35, 35, 44, 32, 39, 40, 42, 39, 42, 27, 30, 26, 30, 31, 31]
    # eisuke_pair = [65, 48, 76, 57, 56, 63, 49, 54, 78, 56, 82, 50, 64, 47, 59, 51, 44, 64, 40, 54]

    # cait_lang   = [119, 92, 53, 101, 47, 56, 40, 87, 107, 31, 45, 55, 45, 62, 37, 44, 50, 37, 44, 53]
    # cait_pair1  = [42, 68, 67]
    # cait_pair2  = [58, 45, 70, 66, 56, 72]
    # cait_pair3  = [52, 53, 55, 57, 64, 59, 62]

    xiang_lang  = [40, 38, 51, 33, 33, 44, 26, 35, 38, 32, 32, 39, 37, 41, 26, 35, 25, 41, 30, 32]
    bas_lang_1  = [70, 175, 127, 74, 104, 98, 65, 85, 50, 83, 53, 84, 60, 95, 137, 54, 70, 35, 60, 120, 56, 91, 21, 136, 123, 56, 104, 64, 39, 61]
    shreya_lang = [41, 59, 47, 31, 64, 40, 40, 59, 48, 60, 48, 30, 53, 40, 37, 40, 44, 33, 41, 35]
    eisuke_lang = [87, 50, 36, 27, 28, 35, 35, 44, 32, 39, 40, 42, 39, 42, 27, 30, 26, 30, 31, 31]
    cait_lang   = [119, 92, 53, 101, 47, 56, 40, 87, 107, 31, 45, 55, 45, 62, 37, 44, 50, 37, 44, 53]
    langs = [xiang_lang] + [bas_lang_1] + [shreya_lang] + [eisuke_lang] + [cait_lang]
    xiang_lang_avg = np.mean(xiang_lang)
    bas_lang_avg = np.mean(bas_lang_1)
    shreya_lang_avg = np.mean(shreya_lang)
    eisuke_lang_avg = np.mean(eisuke_lang)
    cait_lang_avg = np.mean(cait_lang)

    xiang_lang_std = np.std(xiang_lang)
    bas_lang_std = np.std(bas_lang_1)
    shreya_lang_std = np.std(shreya_lang)
    eisuke_lang_std = np.std(eisuke_lang)
    cait_lang_std = np.std(cait_lang)
    all_lang_std = np.std(xiang_lang + bas_lang_1 + shreya_lang + eisuke_lang + cait_lang)

    all_counts = []
    all_counts_T = []

    for l in langs:
        counts = np.array([0] * 18)
        for time in l:
            counts[time // 10] += 1
        print(counts)
        all_counts = all_counts + [counts.tolist()]

    print("==============")
    # all_counts[person][bin] = count for that person in that bin
    all_counts = np.array(all_counts)
    print(all_counts) 
    # all_counts_T[bin][person] = count for that person in that bin
    all_counts_T = np.array(all_counts).T
    print(all_counts_T)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    width = 2

    # plot
    fig, ax = plt.subplots()

    bins = np.arange(0, 190, 10)
    print(bins)

    for i in range(len(bins) - 1):
        sum = 0
        for j in range(len(all_counts_T[i])):
            print("bin: ", i, " person: ", j, " sum: ", sum, " count: ", all_counts_T[i][j])
            sum += all_counts_T[i][j]
            ax.bar(bins[i]+(width*j), all_counts_T[i][j], color=colors[j], alpha=0.8, label=colors[j], width=width, align='edge')
        print("----------")

    # lang avgs
    plt.axvline(x=xiang_lang_avg, color=colors[0], linestyle='--', linewidth=1)
    plt.axvline(x=bas_lang_avg, color=colors[1], linestyle='--', linewidth=1)
    plt.axvline(x=shreya_lang_avg, color=colors[2], linestyle='--', linewidth=1)
    plt.axvline(x=eisuke_lang_avg, color=colors[3], linestyle='--', linewidth=1)
    plt.axvline(x=cait_lang_avg, color=colors[4], linestyle='--', linewidth=1)

    all_lang_avg = [xiang_lang_avg, bas_lang_avg, shreya_lang_avg, eisuke_lang_avg, cait_lang_avg]
    all_lang_avg = np.mean(all_lang_avg)
    plt.axvline(x=all_lang_avg, color='black', linestyle='-', linewidth=1.5)

    # stds
    ax.axvspan(all_lang_avg - all_lang_std, all_lang_avg + all_lang_std, color='black', alpha=0.1)
    # ax.axvspan(xiang_lang_avg - xiang_lang_std, xiang_lang_avg + xiang_lang_std, color=colors[0], alpha=0.1)
    # ax.axvspan(bas_lang_avg - bas_lang_std, bas_lang_avg + bas_lang_std, color=colors[1], alpha=0.1)
    # ax.axvspan(shreya_lang_avg - shreya_lang_std, shreya_lang_avg + shreya_lang_std, color=colors[2], alpha=0.1)
    # ax.axvspan(eisuke_lang_avg - eisuke_lang_std, eisuke_lang_avg + eisuke_lang_std, color=colors[3], alpha=0.1)
    # ax.axvspan(cait_lang_avg - cait_lang_std, cait_lang_avg + cait_lang_std, color=colors[4], alpha=0.1)

    ax.autoscale(tight=True)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    ax.set_xticks(bins)
    ax.set_xticklabels(bins)

    plt.title("Language Preference Learning:\nTime Per Query", fontsize=14)

    ax.set_ylim(0, 15)

    plt.savefig("C:/Users/Rosies/Desktop/Things/LiraLab/CoRL_2025/figures/time_lang.png")


    plt.show(block=False)
    plt.pause(3)
    plt.close()

def plot_pair():
    xiang_pair  = [64, 61, 73, 63, 76, 81, 68, 59, 50, 51, 59, 45, 48, 63, 55, 58, 59, 48, 61, 48]
    bas_pair    = [51, 51, 37, 60, 71, 66, 57, 57, 67, 54, 73, 39, 62, 57, 73, 60, 55, 73, 44, 72]
    shreya_pair = [73, 75, 54, 54, 52, 53, 57, 56, 54, 46, 69, 57, 64, 59, 58, 62, 46, 45, 62, 46]
    eisuke_pair = [65, 48, 76, 57, 56, 63, 49, 54, 78, 56, 82, 50, 64, 47, 59, 51, 44, 64, 40, 54]
    cait_pair1  = [42, 68, 67, 58, 45, 70, 66, 56, 72, 52, 53, 55, 57, 64, 59, 62]
    pairs = [xiang_pair] + [bas_pair] + [shreya_pair] + [eisuke_pair] + [cait_pair1]
    xiang_pair_avg = np.mean(xiang_pair)
    bas_pair_avg = np.mean(bas_pair)
    shreya_pair_avg = np.mean(shreya_pair)
    eisuke_pair_avg = np.mean(eisuke_pair)
    cait_pair_avg = np.mean(cait_pair1)

    xiang_pair_std = np.std(xiang_pair)
    bas_pair_std = np.std(bas_pair)
    shreya_pair_std = np.std(shreya_pair)
    eisuke_pair_std = np.std(eisuke_pair)
    cait_pair_std = np.std(cait_pair1)
    all_pair_std = np.std(xiang_pair + bas_pair + shreya_pair + eisuke_pair + cait_pair1)

    all_counts = []
    all_counts_T = []

    for l in pairs:
        counts = np.array([0] * 18)
        for time in l:
            counts[time // 10] += 1
        print(counts)
        all_counts = all_counts + [counts.tolist()]

    print("==============")
    # all_counts[person][bin] = count for that person in that bin
    all_counts = np.array(all_counts)
    print(all_counts) 
    # all_counts_T[bin][person] = count for that person in that bin
    all_counts_T = np.array(all_counts).T
    print(all_counts_T)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    width = 2

    # plot
    fig, ax = plt.subplots()

    bins = np.arange(0, 190, 10)
    print(bins)

    for i in range(len(bins) - 1):
        sum = 0
        for j in range(len(all_counts_T[i])):
            print("bin: ", i, " person: ", j, " sum: ", sum, " count: ", all_counts_T[i][j])
            sum += all_counts_T[i][j]
            ax.bar(bins[i]+(width*j), all_counts_T[i][j], color=colors[j], alpha=0.8, label=colors[j], width=width, align='edge')
        print("----------")

    # lang avgs
    plt.axvline(x=xiang_pair_avg, color=colors[0], linestyle='--', linewidth=1)
    plt.axvline(x=bas_pair_avg, color=colors[1], linestyle='--', linewidth=1)
    plt.axvline(x=shreya_pair_avg, color=colors[2], linestyle='--', linewidth=1)
    plt.axvline(x=eisuke_pair_avg, color=colors[3], linestyle='--', linewidth=1)
    plt.axvline(x=cait_pair_avg, color=colors[4], linestyle='--', linewidth=1)

    all_lang_avg = [xiang_pair_avg, bas_pair_avg, shreya_pair_avg, eisuke_pair_avg, cait_pair_avg]
    all_lang_avg = np.mean(all_lang_avg)
    plt.axvline(x=all_lang_avg, color='black', linestyle='-', linewidth=1.5)

    # stds
    ax.axvspan(all_lang_avg - all_pair_std, all_lang_avg + all_pair_std, color='black', alpha=0.1)
    # ax.axvspan(xiang_lang_avg - xiang_lang_std, xiang_lang_avg + xiang_lang_std, color=colors[0], alpha=0.1)
    # ax.axvspan(bas_lang_avg - bas_lang_std, bas_lang_avg + bas_lang_std, color=colors[1], alpha=0.1)
    # ax.axvspan(shreya_lang_avg - shreya_lang_std, shreya_lang_avg + shreya_lang_std, color=colors[2], alpha=0.1)
    # ax.axvspan(eisuke_lang_avg - eisuke_lang_std, eisuke_lang_avg + eisuke_lang_std, color=colors[3], alpha=0.1)
    # ax.axvspan(cait_lang_avg - cait_lang_std, cait_lang_avg + cait_lang_std, color=colors[4], alpha=0.1)

    ax.autoscale(tight=True)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    ax.set_xticks(bins)
    ax.set_xticklabels(bins)

    plt.title("Pairwise Preference Learning:\nTime Per Query", fontsize=14)

    ax.set_ylim(0, 15)

    plt.savefig("C:/Users/Rosies/Desktop/Things/LiraLab/CoRL_2025/figures/time_pair.png")


    plt.show(block=False)
    plt.pause(3)
    plt.close()


def plot_avg():
    xiang_lang  = [40, 38, 51, 33, 33, 44, 26, 35, 38, 32, 32, 39, 37, 41, 26, 35, 25, 41, 30, 32]
    bas_lang_1  = [70, 175, 127, 74, 104, 98, 65, 85, 50, 83, 53, 84, 60, 95, 137, 54, 70, 35, 60, 120, 56, 91, 21, 136, 123, 56, 104, 64, 39, 61]
    shreya_lang = [41, 59, 47, 31, 64, 40, 40, 59, 48, 60, 48, 30, 53, 40, 37, 40, 44, 33, 41, 35]
    eisuke_lang = [87, 50, 36, 27, 28, 35, 35, 44, 32, 39, 40, 42, 39, 42, 27, 30, 26, 30, 31, 31]
    cait_lang   = [119, 92, 53, 101, 47, 56, 40, 87, 107, 31, 45, 55, 45, 62, 37, 44, 50, 37, 44, 53]
    langs = [xiang_lang] + [bas_lang_1] + [shreya_lang] + [eisuke_lang] + [cait_lang]
    xiang_lang_avg = np.mean(xiang_lang)
    bas_lang_avg = np.mean(bas_lang_1)
    shreya_lang_avg = np.mean(shreya_lang)
    eisuke_lang_avg = np.mean(eisuke_lang)
    cait_lang_avg = np.mean(cait_lang)
    all_lang_avg = [xiang_lang_avg, bas_lang_avg, shreya_lang_avg, eisuke_lang_avg, cait_lang_avg]
    all_lang_std = np.std(xiang_lang + bas_lang_1 + shreya_lang + eisuke_lang + cait_lang)
    all_lang_avg = np.mean(all_lang_avg)

    xiang_pair  = [64, 61, 73, 63, 76, 81, 68, 59, 50, 51, 59, 45, 48, 63, 55, 58, 59, 48, 61, 48]
    bas_pair    = [51, 51, 37, 60, 71, 66, 57, 57, 67, 54, 73, 39, 62, 57, 73, 60, 55, 73, 44, 72]
    shreya_pair = [73, 75, 54, 54, 52, 53, 57, 56, 54, 46, 69, 57, 64, 59, 58, 62, 46, 45, 62, 46]
    eisuke_pair = [65, 48, 76, 57, 56, 63, 49, 54, 78, 56, 82, 50, 64, 47, 59, 51, 44, 64, 40, 54]
    cait_pair1  = [42, 68, 67, 58, 45, 70, 66, 56, 72, 52, 53, 55, 57, 64, 59, 62]
    pairs = [xiang_pair] + [bas_pair] + [shreya_pair] + [eisuke_pair] + [cait_pair1]
    xiang_pair_avg = np.mean(xiang_pair)
    bas_pair_avg = np.mean(bas_pair)
    shreya_pair_avg = np.mean(shreya_pair)
    eisuke_pair_avg = np.mean(eisuke_pair)
    cait_pair_avg = np.mean(cait_pair1)
    all_pair_avg = [xiang_pair_avg, bas_pair_avg, shreya_pair_avg, eisuke_pair_avg, cait_pair_avg]
    all_pair_std = np.std(xiang_pair + bas_pair + shreya_pair + eisuke_pair + cait_pair1)
    all_pair_avg = np.mean(all_pair_avg)

    lang_color = "#F79646"
    pairwise_color = "#4BACC6"
    improve_color = "#B3A2C7"

    fig, ax = plt.subplots()

    fig.set_figwidth(fig.get_figwidth() / 1.8)
    fig.set_figheight(fig.get_figheight() / 1.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)  # Adjust font size of ticks

    # y axis 1-5
    lvls = [1, 2, 3, 4, 5]
    # x axis
    experiments = ["Language", "Pairwise"]

    positions = [0.4, 0.7, 0.85]

    # plot
    # ax.bar(positions[0], avg_improve_traj_speed, color=colors[0], alpha=0.8, label="Average", width=0.2)
    ax.bar(positions[1], all_lang_avg, color=lang_color, alpha=0.8, label="Average", width=0.1)
    ax.bar(positions[2], all_pair_avg, color=pairwise_color, alpha=0.8, label="Average", width=0.1)
    ax.errorbar(positions[1:], [all_lang_avg, all_pair_avg], yerr=[all_lang_std, all_pair_std], color='black', label="Standard Deviation", capsize=3, fmt='o', markersize=0)

    plt.xticks(positions[1:], experiments, fontsize=17)
    plt.yticks(fontsize=16)

    ax.set_title("Preference Learning:\nAverage Time per Query", fontsize=18)
    ax.set_ylabel("Time per Query (s)", fontsize=20)
    ax.set_ylim([1, 85])
    ax.set_xlabel("Method", fontsize=20)

    # plt.tight_layout(pad=1.0)
    plt.tight_layout(rect=[-0.03,-0.075,0.9,1])
    # plt.show(block=True)
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()

    plt.savefig("C:/Users/Rosies/Desktop/Things/LiraLab/CoRL_2025/figures/avgtime_both.png")

if __name__ == "__main__":
    # plot_lang()
    # plot_pair()
    plot_avg()