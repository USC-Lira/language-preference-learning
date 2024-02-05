import json
from data.utils import greater_speed_adjs, greater_height_adjs
from data.utils import greater_gtreward_adjs, less_gtreward_adjs
from data.utils import greater_distance_adjs, less_distance_adjs
from data.utils import less_speed_adjs, less_height_adjs
from data.utils import height_nouns, speed_nouns, distance_nouns

with open('/home/mdjun/language-preference-learning/data/GPT_augmented_comps.json', 'rb') as f:
# with open('GPT_augmented_comps.json', 'rb') as f:
    a = json.load(f)

classified = {
    "gt_reward": [],
    "speed": [],
    "height": [],
    "distance_to_cube": [],
    "distance_to_bottle": [],
}

greater = {
    "gt_reward": [],
    "speed": [],
    "height": [],
    "distance_to_cube": [],
    "distance_to_bottle": [],
}
less = {
    "gt_reward": [],
    "speed": [],
    "height": [],
    "distance_to_cube": [],
    "distance_to_bottle": [],
}

DEBUG = False

greater_adjs = greater_speed_adjs + greater_height_adjs + greater_gtreward_adjs + greater_distance_adjs
less_adjs = less_speed_adjs + less_height_adjs + less_gtreward_adjs + less_distance_adjs

for i in range(len(a)):
    flag = False
    flag2 = False
    original_sentence = a[i][0]
    original_sentence = original_sentence.lower()
    
    if "lift the cube" in original_sentence or "lifting the cube" in original_sentence or "cube-lifting" in original_sentence or "raise the cube" in original_sentence:
        classified["gt_reward"].extend(a[i])
        flag = True
        for adj in greater_adjs:
            if adj in original_sentence and not flag2:
                greater["gt_reward"].extend(a[i])
                flag2 = True
                if DEBUG: print("increase gtreward: " + a[i][0])
        for adj in less_adjs:
            if adj in original_sentence and not flag2:
                less["gt_reward"].extend(a[i])
                flag2 = True
                if DEBUG: print("decrease gtreward: " + a[i][0])
    elif "bottle" in original_sentence:
        classified["distance_to_bottle"].extend(a[i])
        flag = True
        for adj in greater_adjs:
            if adj in original_sentence and not flag2:
                greater["distance_to_bottle"].extend(a[i])
                flag2 = True
                if DEBUG: print("increase dist to bottle: " + a[i][0])
        for adj in less_adjs:
            if adj in original_sentence and not flag2:
                less["distance_to_bottle"].extend(a[i])
                flag2 = True
                if DEBUG: print("decrease dist to bottle: " + a[i][0])
    elif "cube" in original_sentence:
        classified["distance_to_cube"].extend(a[i])
        flag = True
        for adj in greater_adjs:
            if adj in original_sentence and not flag2:
                greater["distance_to_cube"].extend(a[i])
                flag2 = True
                if DEBUG: print("increase dist to cube: " + a[i][0])
        for adj in less_adjs:
            if adj in original_sentence and not flag2:
                less["distance_to_cube"].extend(a[i])
                flag2 = True
                if DEBUG: print("decrease dist to cube: " + a[i][0])
    else:
        for adj in greater_speed_adjs + less_speed_adjs + speed_nouns:
            if adj in original_sentence and not flag:
                classified["speed"].extend(a[i])
                flag = True
                for adj_2 in greater_adjs:
                    if adj_2 in original_sentence and not flag2:
                        greater["speed"].extend(a[i])
                        flag2 = True
                        if DEBUG: print("increase speed: " + a[i][0])
                for adj_2 in less_adjs:
                    if adj_2 in original_sentence and not flag2:
                        less["speed"].extend(a[i])
                        flag2 = True
                        if DEBUG: print("decrease speed: " + a[i][0])
        for adj in greater_height_adjs + less_height_adjs + height_nouns:
            if adj in original_sentence and not flag:
                classified["height"].extend(a[i])
                flag = True
                for adj_2 in greater_adjs:
                    if adj_2 in original_sentence and not flag2:
                        greater["height"].extend(a[i])
                        flag2 = True
                        if DEBUG: print("increase height: " + a[i][0])
                for adj_2 in less_adjs:
                    if adj_2 in original_sentence and not flag2:
                        less["height"].extend(a[i])
                        flag2 = True
                        if DEBUG: print("decrease height: " + a[i][0])
    if not flag:
        print("unknown feature: " + original_sentence)
    if not flag2:
        print("missing adj: " + original_sentence)
    
for feature in classified:
    print("classified " + feature + " len: " + str(len(classified[feature])))
for feature in greater:
    print("greater " + feature + " len: " + str(len(greater[feature])))
for feature in less:
    print("lesser " + feature + " len: " + str(len(less[feature])))

with open('classified_nlcomps.json', 'w') as f:
    json.dump(classified, f, indent=4)

with open('greater_nlcomps.json', 'w') as f:
    json.dump(greater, f, indent=4)

with open('less_nlcomps.json', 'w') as f:
    json.dump(less, f, indent=4)
