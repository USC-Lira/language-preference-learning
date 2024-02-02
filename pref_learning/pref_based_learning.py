# preference based learning

# Importing the libraries
import json
import os
import torch
import torch.nn as nn
import numpy as np
import argparse

from transformers import AutoModel, AutoTokenizer

from feature_learning.model import NLTrajAutoencoder
from feature_learning.nl_traj_dataset import NLTrajComparisonDataset
from feature_learning.learn_features import load_data
from feature_learning.utils import timeStamped, BERT_MODEL_NAME, BERT_OUTPUT_DIM, create_logger, AverageMeter
from model_analysis.utils import get_traj_lang_embeds
from model_analysis.improve_trajectory import initialize_reward, get_feature_value


# learned and true reward func
class RewardFunc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RewardFunc, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, traj_cur, traj_opt, lang_feedback):
        # neg dot product
        dot_product = torch.dot(lang_feedback, (traj_opt - traj_cur))
        return (-dot_product)


def run(args):
    # data
    # Load the val trajectories and language comparisons first
    trajs = np.load(f'{args.data_dir}/val/trajs.npy')
    nlcomps = json.load(open(f'{args.data_dir}/val/unique_nlcomps.json', 'rb'))
    nlcomps_bert_embeds = np.load(f'{args.data_dir}/val/unique_nlcomps_{bert_model}.npy')
    classified_nlcomps = json.load(open(f'data/classified_nlcomps.json', 'rb'))
    greater_nlcomps = json.load(open(f'data/greater_nlcomps.json', 'rb'))
    less_nlcomps = json.load(open(f'data/less_nlcomps.json', 'rb'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

    # Current learned language encoder
    # Load the model
    if args.use_bert_encoder:
        lang_encoder = AutoModel.from_pretrained(BERT_MODEL_NAME[args.bert_model])
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME[args.bert_model])
        feature_dim = BERT_OUTPUT_DIM[args.bert_model]
    else:
        lang_encoder = None
        tokenizer = None
        feature_dim = 128

    model = NLTrajAutoencoder(
                    encoder_hidden_dim=args.encoder_hidden_dim, 
                    feature_dim=feature_dim,
                    decoder_hidden_dim=args.decoder_hidden_dim, 
                    lang_encoder=lang_encoder,
                    preprocessed_nlcomps=args.preprocessed_nlcomps,
                    bert_output_dim=BERT_OUTPUT_DIM[args.bert_model],
                    use_bert_encoder=args.use_bert_encoder, 
                    traj_encoder=args.traj_encoder,
                    use_cnn_in_transformer=args.use_cnn_in_transformer,
                    use_casual_attention=args.use_casual_attention
    )
    state_dict = torch.load(os.path.join(args.model_dir, 'best_model_state_dict.pth'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

        
    # random init both reward functions (learned, true)
    learned_reward = RewardFunc(128, 128)
    true_reward = RewardFunc(128, 128)
    learned_reward = initialize_rewards(5)
    true_reward = initialize_rewards(5)
    # nn.init.normal_(learned_reward.linear.weight, mean=0, std=0.01)
    

     # loss func
    criteria = Loss()
    optimizer = torch.optim.Adam(learned_reward.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    traj_embeds, lang_embeds = get_traj_lang_embeds(trajs, nlcomps, model, device, args.use_bert_encoder, tokenizer, nlcomps_bert_embeds)

    # Find the optimal trajectory given the reward function
    feature_values = np.array([get_feature_value(traj) for traj in trajs])
    # Normalize feature values
    feature_values = (feature_values - np.min(feature_values, axis=0)) / (
                np.max(feature_values, axis=0) - np.min(feature_values, axis=0))

    np.random.seed(args.seed)

    for i in range(args.num_iterations):
        # random select traj
        rand = np.random.randint(0, len(trajs))
        traj_cur = trajs[rand]

        # use current learned encoder for traj to get features
        # TODO: how to get features?
        feature_values_cur = feature_values[rand]

        # Use current learned reward func to get language feedback
            # TODO: get_lang_feedback given the feature_values_cur

        # Based on language feedback, use learned lang encoder to get the feature in that feedback (select from set)
            # TODO: ??


        # Select optimal traj based on current reward func
            # TODO: argmax...?


        # Compute dot product of lang(traj_opt - traj_cur)
            # Minimize the negative dot product (loss)!
        loss = criteria(traj_cur, traj_opt, lang_feedback)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: Cosine sim of (current learned reward, true reward) (erdem help)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data_dir', type=str, default='data', help='')
    parser.add_argument('--model_dir', type=str, default='models', help='')
    parser.add_argument('--encoder_hidden_dim')
    parser.add_argument('--decoder_hidden_dim')
    parser.add_argument('--preprocessed_nlcomps', action='store_true', help="")
    parser.add_argument('--bert_model', type=str, default='bert-base', help='which BERT model to use')
    parser.add_argument('--use-bert-encoder', action="store_true", help='whether to use BERT in the language encoder')
    parser.add_argument('--traj-encoder', default='mlp', choices=['mlp', 'transformer', 'lstm'],
                        help='which trajectory encoder to use')
    parser.add_argument('--use-cnn-in-transformer', action="store_true", help='whether to use CNN in the transformer')
    parser.add_argument('--use-casual-attention', action="store_true",
                        help='whether to use casual attention in the transformer')
    parser.add_argument('--weight-decay', type=float, default=0, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--num_iterations', type=int, default=10, help='')

    args=parser.parse_args()
    run(args)