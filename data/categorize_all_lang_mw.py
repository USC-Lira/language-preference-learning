import os
import json
from collections import defaultdict

data_dir = 'metaworld'

all_unique_nlcomps = []
all_greater_nlcomps = defaultdict(list)
all_less_nlcomps = defaultdict(list)

with open(os.path.join(data_dir, 'ver2_gpt_augmented_dataset_metaworld.json'), 'r') as f:
    augmented_lang = json.load(f)

for key in augmented_lang:
    nlcomps = augmented_lang[key]
    all_unique_nlcomps.extend(nlcomps)
    if 'greater' in key:
        feature = key.split('_')[1]
        all_greater_nlcomps[feature].extend(nlcomps)
    elif 'lesser' in key:
        feature = key.split('_')[1]
        all_less_nlcomps[feature].extend(nlcomps)
    else:
        print(key)

import ipdb; ipdb.set_trace()
with open(os.path.join(data_dir, 'all_unique_nlcomps.json'), 'w') as f:
    json.dump(all_unique_nlcomps, f)

with open(os.path.join(data_dir, 'all_greater_nlcomps.json'), 'w') as f:
    json.dump(all_greater_nlcomps, f)

with open(os.path.join(data_dir, 'all_less_nlcomps.json'), 'w') as f:
    json.dump(all_less_nlcomps, f)