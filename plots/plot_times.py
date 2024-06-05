import numpy as np
import matplotlib.pyplot as plt

# xiang_lang  = [40, 38, 51, 33, 33, 44, 26, 35, 38, 32, 32, 39, 37, 41, 26, 35, 25, 41, 30, 32]
# xiang_pair  = [64, 61, 73, 63, 76, 81, 68, 59, 50, 51, 59, 45, 48, 63, 55, 58, 59, 48, 61, 48]

# bas_lang_1  = [70, 175, 127, 74, 104, 98, 65, 85, 50, 83, 53, 84, 60, 95, 137]
# bas_lang_2  = [54, 70, 35, 60, 12, 56, 91, 21, 13, 12, 56, 10, 64, 39, 61]
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
bas_lang_1  = [70, 175, 127, 74, 104, 98, 65, 85, 50, 83, 53, 84, 60, 95, 137, 54, 70, 35, 60, 12, 56, 91, 21, 13, 12, 56, 10, 64, 39, 61]
shreya_lang = [41, 59, 47, 31, 64, 40, 40, 59, 48, 60, 48, 30, 53, 40, 37, 40, 44, 33, 41, 35]
eisuke_lang = [87, 50, 36, 27, 28, 35, 35, 44, 32, 39, 40, 42, 39, 42, 27, 30, 26, 30, 31, 31]
cait_lang   = [119, 92, 53, 101, 47, 56, 40, 87, 107, 31, 45, 55, 45, 62, 37, 44, 50, 37, 44, 53]
langs = [xiang_lang] + [bas_lang_1] + [shreya_lang] + [eisuke_lang] + [cait_lang]

xiang_pair  = [64, 61, 73, 63, 76, 81, 68, 59, 50, 51, 59, 45, 48, 63, 55, 58, 59, 48, 61, 48]
bas_pair    = [51, 51, 37, 60, 71, 66, 57, 57, 67, 54, 73, 39, 62, 57, 73, 60, 55, 73, 44, 72]
shreya_pair = [73, 75, 54, 54, 52, 53, 57, 56, 54, 46, 69, 57, 64, 59, 58, 62, 46, 45, 62, 46]
eisuke_pair = [65, 48, 76, 57, 56, 63, 49, 54, 78, 56, 82, 50, 64, 47, 59, 51, 44, 64, 40, 54]
cait_pair1  = [42, 68, 67, 58, 45, 70, 66, 56, 72, 52, 53, 55, 57, 64, 59, 62]
pairs = [xiang_pair] + [bas_pair] + [shreya_pair] + [eisuke_pair] + [cait_pair1]

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

ax.autoscale(tight=True)

plt.xlabel('Time', fontsize=14)
plt.ylabel('Count', fontsize=14)

ax.set_xticks(bins)
ax.set_xticklabels(bins)

ax.set_ylim(0, 15)
plt.show()