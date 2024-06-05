import numpy as np
import os

# read in np file 
file_path = "C:/Users/Rosies/Desktop/Things/LiraLab/CoRL_2025/optim_ratings/"
# get all files in the directory
files = os.listdir(file_path)

lang_color = "#F79646"
pairwise_color = "#4BACC6"

l = 0
p = 0
langs = []
prefs = []
# for file in files:
#     if file.endswith('.npy') and file.startswith('lang_comp'):
#         print(file)
#         lang_comp = np.load(file_path + file)
#         print(l, ": ", lang_comp)
#         l += 1
#         if (len(lang_comp) == 4):
#             langs.append(lang_comp)
#     if file.endswith('.npy') and file.startswith('pref_comp'):
#         print(file)
#         pref_comp = np.load(file_path + file)
#         print(p, ": ", pref_comp)
#         p += 1
#         if (len(pref_comp) == 4):
#             prefs.append(pref_comp)

langs = [[5, 2, 5, 5], [1, 1, 3, 3], [4, 1, 5, 5], [3, 1, 4, 4], [3, 3, 5, 5]]
prefs = [[1, 3, 1, 5], [2, 3, 2, 3], [1, 3, 1, 4], [2, 1, 2, 5], [1, 1, 3, 4]]



# plot
import matplotlib.pyplot as plt
langs = np.array(langs)
prefs = np.array(prefs)

# flip shape
langs = langs.T
prefs = prefs.T

print(langs)

# average
langs_avg = np.mean(langs, axis=1)
prefs_avg = np.mean(prefs, axis=1)

print(langs_avg)
print(prefs_avg)

# plot
fig, ax = plt.subplots()
plt.plot(langs_avg, label='Language', color=lang_color)
plt.plot(prefs_avg, label='Preference', color=pairwise_color)

plt.legend(loc='upper left', fontsize=14)
ax.get_legend().legend_handles[0].set_color(lang_color)
ax.get_legend().legend_handles[1].set_color(pairwise_color)

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Average Rating\n[1 Very Bad ~ 5 Very Good]', fontsize=14)

ax.set_xticks(range(4))
ax.set_xticklabels([5, 10, 15, 20])

# error
langs_std = np.std(langs, axis=1)
prefs_std = np.std(prefs, axis=1)

# connect and shade in error
plt.fill_between(range(4), langs_avg - langs_std, langs_avg + langs_std, alpha=0.1, color=lang_color)
plt.fill_between(range(4), prefs_avg - prefs_std, prefs_avg + prefs_std, alpha=0.1, color=pairwise_color)



plt.savefig('C:/Users/Rosies/Desktop/Things/LiraLab/CoRL_2025/figures/avg_ratings.png')

plt.show(block=False)
plt.pause(3)
plt.close()

    # pref_comp_20240526_112531.npy
    # 0 :  [1]

# lang_comp_20240526_144455.npy
# 0 :  [5 2 5 5]
# pref_comp_20240526_150643.npy
# 1 :  [1 3 1 5]

# lang_comp_20240526_162602.npy
# 1 :  [1 1 3 3]
# pref_comp_20240526_164936.npy
# 2 :  [2 3 2 3]

# ======== 5/27/2024 14:12:42 pref first

# lang_comp_20240527_112913.npy
# 2 :  [4 1 5 5]
# pref_comp_20240527_115112.npy
# 3 :  [1 3 1 4]

    # pref_comp_20240527_144903.npy
    # 4 :  [1 1 4 5]

# lang_comp_20240528_113120.npy
# 3 :  [3 1 4 4]
# pref_comp_20240528_111344.npy
# 5 :  [2 1 2 5]

# pref_comp_20240528_160423.npy
# 6 :  [1 1 3 4]
# lang_comp_20240528_162044.npy
# 4 :  [3 3 5 5]

    # lang_comp_20240529_184420.npy
    # 5 :  [4 2 5 5]

    # pref_comp_20240529_191034.npy
    # 7 :  [4 4]
    # lang_comp_20240529_193031.npy
    # 6 :  [1 1 4 3]

