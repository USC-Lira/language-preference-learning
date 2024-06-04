import os
import json
import numpy as np


def replace_nlcomps(data_dir, new_nlcomps, old_nlcomps, replace_ori_nlcomps):
    """
    Replace the natural language comparisons in the given data directory

    Args:
        - data_dir (str): Directory containing the data files
        - new_nlcomps (List[str]): New natural language comparisons to replace with
        - old_nlcomps (List[str]): Old natural language comparisons to replace
    """
    # load the old unique language comparisons
    with open(os.path.join(data_dir, 'unique_nlcomps.json'), 'r') as f:
        old_unique_nlcomps = json.load(f)
    
    # find the language comparisons to replace
    new_unique_nlcomps = old_unique_nlcomps.copy()
    for i, nlcomps in enumerate(old_nlcomps):
        ori_nlcomp = nlcomps[0]
        if ori_nlcomp in replace_ori_nlcomps:
            for j, nlcomp in enumerate(nlcomps[1:]):
                if nlcomp in old_unique_nlcomps:
                    new_unique_nlcomps[old_unique_nlcomps.index(nlcomp)] = new_nlcomps[i][j+1]
    
    
    # save the new unique language comparisons
    with open(os.path.join(data_dir, 'new_unique_nlcomps.json'), 'w') as f:
        json.dump(new_unique_nlcomps, f)


if __name__ == '__main__':
    data_root_dir = 'data/data_img_obs_res_224_more'

    with open(os.path.join('data/GPT_augmented_comps.json'), 'r') as f:
        new_nlcomps = json.load(f)
    
    with open(os.path.join('data/GPT_augmented_comps_old.json'), 'r') as f:
        old_nlcomps = json.load(f)

    replace_ori_nlcomps = ['Lift the cube worse.', 'Lift the cube not as well.']

    replace_nlcomps(os.path.join(data_root_dir, 'train'), new_nlcomps, old_nlcomps, replace_ori_nlcomps)
    replace_nlcomps(os.path.join(data_root_dir, 'val'), new_nlcomps, old_nlcomps, replace_ori_nlcomps)
    replace_nlcomps(os.path.join(data_root_dir, 'test'), new_nlcomps, old_nlcomps, replace_ori_nlcomps)

