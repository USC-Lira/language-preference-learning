import json
from data.utils import greater_speed_adjs, greater_height_adjs
from data.utils import greater_gtreward_adjs, less_gtreward_adjs
from data.utils import greater_distance_adjs, less_distance_adjs
from data.utils import less_speed_adjs, less_height_adjs
from data.utils import height_nouns, speed_nouns, distance_nouns

with open('/home/mdjun/language-preference-learning/data/GPT_augmented_dataset.json', 'rb') as f:
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

greater_adjs = greater_speed_adjs + greater_height_adjs + greater_gtreward_adjs + greater_distance_adjs
less_adjs = less_speed_adjs + less_height_adjs + less_gtreward_adjs + less_distance_adjs

for i in range(len(a)):
    flag = False
    original_sentence = a[i][0]
    if "lift the cube" in original_sentence.lower():
        classified["gt_reward"].extend(a[i])
        flag = True
        for adj in greater_adjs:
            if adj in original_sentence:
                greater["gt_reward"].extend(a[i])
        for adj in less_adjs:
            if adj in original_sentence:
                less["gt_reward"].extend(a[i])
    elif "bottle" in original_sentence:
        classified["distance_to_bottle"].extend(a[i])
        flag = True
        for adj in greater_adjs:
            if adj in original_sentence:
                greater["distance_to_bottle"].extend(a[i])
        for adj in less_adjs:
            if adj in original_sentence:
                less["distance_to_bottle"].extend(a[i])
    elif "cube" in original_sentence:
        classified["distance_to_cube"].extend(a[i])
        flag = True
        for adj in greater_adjs:
            if adj in original_sentence:
                greater["distance_to_cube"].extend(a[i])
        for adj in less_adjs:
            if adj in original_sentence:
                less["distance_to_cube"].extend(a[i])
    else:
        for adj in greater_speed_adjs + less_speed_adjs + speed_nouns:
            if adj in original_sentence:
                classified["speed"].extend(a[i])
                flag = True
                for adj_2 in greater_adjs:
                    if adj_2 in original_sentence:
                        greater["speed"].extend(a[i])
                for adj_2 in less_adjs:
                    if adj_2 in original_sentence:
                        less["speed"].extend(a[i])
        for adj in greater_height_adjs + less_height_adjs + height_nouns:
            if adj in original_sentence:
                classified["height"].extend(a[i])
                flag = True
                for adj_2 in greater_adjs:
                    if adj_2 in original_sentence:
                        greater["height"].extend(a[i])
                for adj_2 in less_adjs:
                    if adj_2 in original_sentence:
                        less["height"].extend(a[i])
    if not flag:
        print(original_sentence)

with open('classified_nlcomps.json', 'w') as f:
    json.dump(classified, f, indent=4)

with open('greater_nlcomps.json', 'w') as f:
    json.dump(greater, f, indent=4)

with open('less_nlcomps.json', 'w') as f:
    json.dump(less, f, indent=4)
