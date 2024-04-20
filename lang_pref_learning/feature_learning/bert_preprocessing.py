import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import argparse
import os

from lang_pref_learning.feature_learning.utils import LANG_MODEL_NAME

os.environ["OMP_NUM_THREADS"] = "4"


def preprocess_strings(nlcomp_dir, LANG_MODEL, nlcomp_list=None, id_mapping=False, save=False):
    if nlcomp_list is None:
        assert nlcomp_dir != ''
        # nlcomp_file is a json file with the list of comparisons in NL.
        nlcomp_file = os.path.join(os.getcwd(), nlcomp_dir, 'nlcomps.json')

        with open(nlcomp_file, 'rb') as f:
            nlcomps = json.load(f)
    else:
        nlcomps = nlcomp_list

    # Get unique nlcomps (sorted by alphabetical order)
    unique_nlcomps = list(sorted(set(nlcomps)))
    id_map = dict()
    for i, unique_nlcomp in enumerate(unique_nlcomps):
        id_map[unique_nlcomp] = i

    # Save the index mapping and unique nlcomps
    nlcomp_indexes = []
    for nlcomp in nlcomps:
        nlcomp_indexes.append(id_map[nlcomp])
    if save:
        np.save(os.path.join(nlcomp_dir, 'nlcomp_indexes.npy'), np.asarray(nlcomp_indexes, dtype=np.int32))

    unbatched_input = unique_nlcomps

    # Get BERT embeddings
    tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME[LANG_MODEL])
    model = AutoModel.from_pretrained(LANG_MODEL_NAME[LANG_MODEL])
    lang_embeddings = []
    for sentence in unbatched_input:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state

        # Average across the sequence to get a sentence-level embedding
        embedding = torch.mean(embedding, dim=1, keepdim=False)
        lang_embeddings.append(embedding.detach().numpy())

    if id_mapping:
        outfile = os.path.join(nlcomp_dir, 'unique_nlcomps_{}.npy'.format(LANG_MODEL))
    else:
        outfile = os.path.join(nlcomp_dir, 'nlcomps.npy')

    lang_embeddings = np.concatenate(lang_embeddings, axis=0)
    print(lang_embeddings.shape)
    if save:
        np.save(outfile, lang_embeddings)
    return lang_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data-dir', type=str, default='', help='')
    parser.add_argument('--batch-size', type=int, default=5000, help='')
    parser.add_argument('--id-mapping', action="store_true", help='')
    parser.add_argument('--lang-model', default='bert-base', help='Which BERT model to use')

    args = parser.parse_args()
    preprocess_strings(args.data_dir, args.LANG_MODEL, id_mapping=args.id_mapping, save=True)
