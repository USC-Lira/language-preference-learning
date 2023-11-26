import json
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import argparse
import os


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def preprocess_strings(nlcomp_dir, batch_size, nlcomp_list=None, id_mapping=False, save=False):
    if nlcomp_list is None:
        assert nlcomp_dir != ''
        # nlcomp_file is a json file with the list of comparisons in NL.
        nlcomp_file = os.path.join(os.getcwd(), nlcomp_dir, 'nlcomps.json')

        with open(nlcomp_file, 'rb') as f:
            nlcomps = json.load(f)
    else:
        nlcomps = nlcomp_list

    if id_mapping:
        unique_nlcomps = list(set(nlcomps))
        id_map = dict()
        for i, unique_nlcomp in enumerate(unique_nlcomps):
            id_map[unique_nlcomp] = i

        nlcomp_indexes = []
        for nlcomp in nlcomps:
            nlcomp_indexes.append(id_map[nlcomp])
        if save:
            np.save(os.path.join(nlcomp_dir, 'nlcomp_indexes.npy'), np.asarray(nlcomp_indexes))

        unbatched_input = unique_nlcomps
    else:
        unbatched_input = nlcomps

    bert_output_embeddings = []
    for sentence in unbatched_input:
        inputs = tokenizer(sentence, return_tensors="pt")
        bert_output = model(**inputs)

        embedding = bert_output.last_hidden_state
        embedding = torch.mean(embedding, dim=1, keepdim=False)
        # print("bert_output_embedding:", bert_output_embedding.shape)
        bert_output_embeddings.append(embedding.detach().numpy())

    if id_mapping:
        outfile = os.path.join(nlcomp_dir, 'unique_nlcomps.npy')
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

    args = parser.parse_args()
    preprocess_strings(args.nlcomp_dir, args.batch_size, id_mapping=args.id_mapping, save=True)

