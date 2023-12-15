import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import argparse
import os
from feature_learning.utils import BERT_MODEL_NAME

os.environ["OMP_NUM_THREADS"] = "4"


def preprocess_strings(nlcomp_dir, bert_model, nlcomp_list=None, id_mapping=False, save=False):
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
        np.save(os.path.join(nlcomp_dir, 'nlcomp_indexes_{}.npy'.format(bert_model)), np.asarray(nlcomp_indexes))
        json.dump(unique_nlcomps, open(os.path.join(nlcomp_dir, 'unique_nlcomps_{}.json'.format(bert_model)), 'w'))

    unbatched_input = unique_nlcomps

    # Get BERT embeddings
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME[bert_model])
    model = AutoModel.from_pretrained(BERT_MODEL_NAME[bert_model])
    bert_output_embeddings = []
    for sentence in unbatched_input:
        inputs = tokenizer(sentence, return_tensors="pt")
        bert_output = model(**inputs)
        embedding = bert_output.last_hidden_state

        # Average across the sequence to get a sentence-level embedding
        embedding = torch.mean(embedding, dim=1, keepdim=False)
        bert_output_embeddings.append(embedding.detach().numpy())

    if id_mapping:
        outfile = os.path.join(nlcomp_dir, 'unique_nlcomps_{}.npy'.format(bert_model))
    else:
        outfile = os.path.join(nlcomp_dir, 'nlcomps.npy')

    bert_output_embeddings = np.concatenate(bert_output_embeddings, axis=0)
    print(bert_output_embeddings.shape)
    if save:
        np.save(outfile, bert_output_embeddings)
    return bert_output_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data-dir', type=str, default='', help='')
    parser.add_argument('--batch-size', type=int, default=5000, help='')
    parser.add_argument('--id-mapping', action="store_true", help='')
    parser.add_argument('--bert-model', default='bert-base', help='Which BERT model to use')

    args = parser.parse_args()
    preprocess_strings(args.data_dir, args.bert_model, id_mapping=args.id_mapping, save=True)
