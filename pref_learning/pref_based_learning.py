# preference based learning

# Importing the libraries
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

from transformers import AutoModel, AutoTokenizer

from feature_learning.model import NLTrajAutoencoder
from pref_learning.pref_dataset import PrefDataset, EvaluationDataset
from pref_learning.utils import feature_aspects
from feature_learning.utils import timeStamped, BERT_MODEL_NAME, BERT_OUTPUT_DIM, create_logger, AverageMeter
from model_analysis.utils import get_traj_lang_embeds
from model_analysis.improve_trajectory import initialize_reward, get_feature_value, get_lang_feedback

DEBUG = True


# learned and true reward func (linear for now)
class RewardFunc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RewardFunc, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)


def get_lang_feedback_aspect(feature_value, reward, noisy=False):
    """
    Get language feedback based on feature value and true reward function
    """
    potential = torch.tensor(reward * (1 - feature_value.numpy()))
    probs = torch.softmax(potential, dim=1)
    if noisy:
        # add some noise to the probabilities
        probs = probs + torch.randn_like(probs) * 1e-5
        # sample a language comparison with the probabilities
        idx = torch.multinomial(probs, 1).item()
    else:
        idx = torch.argmax(probs).item()
    return idx


def calculate_cross_entropy(p_probs, q_probs):
    # Ensure probabilities are within valid range
    p_probs = torch.clamp(p_probs, 1e-15, 1 - 1e-15)
    q_probs = torch.clamp(q_probs, 1e-15, 1 - 1e-15)

    # Calculate cross-entropy
    cross_entropy = -(p_probs * torch.log(q_probs) + (1 - p_probs) * torch.log(1 - q_probs))

    # Sum over probabilities
    cross_entropy = torch.sum(cross_entropy)

    return cross_entropy.item()


def load_data(args, test=False):
    # Load the test trajectories and language comparisons
    if not test:
        trajs = np.load(f'{args.data_dir}/val/trajs.npy')
        nlcomps = json.load(open(f'{args.data_dir}/val/unique_nlcomps.json', 'rb'))
    else:
        trajs = np.load(f'{args.data_dir}/test/trajs.npy')
        nlcomps = json.load(open(f'{args.data_dir}/test/unique_nlcomps.json', 'rb'))

    if DEBUG:
        print("len of trajs: " + str(len(trajs)))

    if args.use_all_datasets:
        if DEBUG:
            print("using all datasets, so appending them....")
        # train+val+test all datasets
        # append train and val datasets
        trajs = np.append(trajs, np.load(f'{args.data_dir}/train/trajs.npy'), axis=0)
        nlcomps = json.load(open(f'{args.data_dir}/train/unique_nlcomps.json', 'rb')) + nlcomps

        if DEBUG:
            print("length of trajs after using all datasets: " + str(len(trajs)))

    # need to run categorize.py first
    greater_nlcomps = json.load(open(f'{args.data_dir}/val/greater_nlcomps.json', 'rb'))
    less_nlcomps = json.load(open(f'{args.data_dir}/val/less_nlcomps.json', 'rb'))
    if DEBUG:
        print("greater nlcomps size: " + str(len(greater_nlcomps)))
        print("less nlcomps size: " + str(len(less_nlcomps)))

    return trajs, nlcomps, greater_nlcomps, less_nlcomps


def reconstruct_traj(traj_embed, model, nlcomp_embeds):
    new_trajs = traj_embed + nlcomp_embeds
    recon_trajs = model.traj_decoder(new_trajs)
    return recon_trajs


def _pref_learning(train_dataloader, test_dataloader, model, greater_nlcomps, less_nlcomps, learned_reward,
                   true_reward, traj_embeds, lang_embeds, test_traj_embeds, test_traj_true_rewards,
                   less_idx, optimizer, args):
    eval_cross_entropies = []
    logsigmoid = nn.LogSigmoid()
    for train_data in train_dataloader:
        curr_traj, curr_feature_value, idx = train_data
        curr_traj_embed = traj_embeds[idx]
        # Use true reward func to get language feedback (select from set)
        # First find the feature aspect to give feedback on and positive / negative
        feature_aspect_idx = get_lang_feedback_aspect(curr_feature_value, true_reward, args.use_softmax)
        if feature_aspect_idx in less_idx:
            # randomly choose from less_nlcomps
            nlcomps = less_nlcomps[feature_aspects[feature_aspect_idx]]
        else:
            # randomly choose from greater_nlcomps
            nlcomps = greater_nlcomps[feature_aspects[feature_aspect_idx]]

        # Get the feature of the language comparison
        # traj_feature = traj_embeds[idx]
        nlcomp_features = torch.tensor([lang_embeds[nlcomps.index(nlcomp)] for nlcomp in nlcomps])
        nlcomp_features_norm = torch.norm(nlcomp_features, dim=1, keepdim=True)

        # recon_trajs = reconstruct_traj(curr_traj_embed, model, nlcomp_features)
        # true_recon_features = [get_feature_value(recon_traj) for recon_traj in recon_trajs]

        for i in range(args.num_iterations):
            # Compute dot product of lang(traj_opt - traj_cur)
            # Minimize the negative dot product (loss)!
            # loss = torch.mean(torch.exp(learned_reward(nlcomp_features)) / (torch.exp(learned_reward(nlcomp_features)) + 1), dim=0)
            loss = -logsigmoid(learned_reward(nlcomp_features)).mean()
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print('=====================')
        # print(loss)
        # Do the evaluation
        cross_entropy = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds)
        eval_cross_entropies.append(cross_entropy)
    return eval_cross_entropies


def evaluate(test_dataloader, true_traj_rewards, learned_reward, traj_embeds):
    total_cross_entropy = AverageMeter('cross-entropy')
    for i, test_data in enumerate(test_dataloader):
        traj_a, traj_b, idx_a, idx_b = test_data

        # get the embeddings of the two trajectories
        traj_a_embed = traj_embeds[idx_a]
        traj_b_embed = traj_embeds[idx_b]

        # get bernoulli distributions for the two trajectories
        true_rewards = torch.tensor([true_traj_rewards[idx_a], true_traj_rewards[idx_b]])
        # make true probs 0 and 1
        true_probs = torch.tensor([torch.argmax(true_rewards) == 0, torch.argmax(true_rewards) == 1]).float()
        # make true probs with softmax
        # true_probs = torch.softmax(true_rewards, dim=0)

        traj_a_embed = torch.tensor(traj_a_embed)
        traj_b_embed = torch.tensor(traj_b_embed)
        learned_rewards = torch.tensor([learned_reward(traj_a_embed), learned_reward(traj_b_embed)])
        learned_probs = torch.softmax(learned_rewards, dim=0)

        # if i < 5:
        #     print("true_probs: " + str(true_probs))
        #     print("learned_probs: " + str(learned_probs))

        # calculate cross-entropy between learned and true distributions
        cross_entropy = calculate_cross_entropy(true_probs, learned_probs)
        total_cross_entropy.update(cross_entropy, 1)
    return total_cross_entropy.avg


def run(args):
    np.random.seed(args.seed)

    # Load train data
    trajs, nlcomps, greater_nlcomps, less_nlcomps = load_data(args)
    feature_values = np.array([get_feature_value(traj) for traj in trajs])
    # Normalize feature values
    feature_values = (feature_values - np.min(feature_values, axis=0)) / (
            np.max(feature_values, axis=0) - np.min(feature_values, axis=0))

    less_idx = np.random.choice(5, size=2, replace=False)
    for i in less_idx:
        feature_values[:, i] = 1 - feature_values[:, i]

    train_dataset = PrefDataset(trajs, feature_values)
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NLTrajAutoencoder(
        encoder_hidden_dim=args.encoder_hidden_dim,
        feature_dim=feature_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        lang_encoder=lang_encoder,
        preprocessed_nlcomps=args.preprocessed_nlcomps,
        bert_output_dim=BERT_OUTPUT_DIM[args.bert_model],
        use_bert_encoder=args.use_bert_encoder,
        traj_encoder=args.traj_encoder,
    )
    state_dict = torch.load(os.path.join(args.model_dir, 'best_model_state_dict.pth'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    traj_embeds, lang_embeds = get_traj_lang_embeds(trajs, nlcomps, model, device, args.use_bert_encoder, tokenizer)

    # random init both reward functions (learned, true)
    learned_reward = RewardFunc(128, 1)
    true_reward = initialize_reward(5)
    nn.init.normal_(learned_reward.linear.weight, mean=0.5, std=0.01)
    optimizer = torch.optim.SGD(learned_reward.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Load test data
    trajs_test, nlcomps_test, greater_nlcomps_test, less_nlcomps_test = load_data(args, test=True)
    test_dataset = EvaluationDataset(trajs_test)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_traj_embeds, test_lang_embeds = get_traj_lang_embeds(trajs_test, nlcomps_test, model, device,
                                                              args.use_bert_encoder, tokenizer)
    test_feature_values = np.array([get_feature_value(traj) for traj in trajs_test])

    # Normalize feature values
    test_feature_values = (test_feature_values - np.min(test_feature_values, axis=0)) / (
            np.max(test_feature_values, axis=0) - np.min(test_feature_values, axis=0))

    for i in less_idx:
        test_feature_values[:, i] = 1 - test_feature_values[:, i]

    test_traj_true_rewards = np.dot(test_feature_values, true_reward)

    eval_cross_entropies = _pref_learning(train_data, test_data, nlcomps, greater_nlcomps, less_nlcomps, learned_reward,
                                          true_reward, traj_embeds, lang_embeds, test_traj_embeds,
                                          test_traj_true_rewards,
                                          less_idx, optimizer, args)
    print(eval_cross_entropies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data-dir', type=str, default='data', help='')
    parser.add_argument('--model-dir', type=str, default='models', help='')
    parser.add_argument('--num-batches', type=int, default=2, help='')
    parser.add_argument('--encoder-hidden-dim', type=int, default=128)
    parser.add_argument('--decoder-hidden-dim', type=int, default=128)
    parser.add_argument('--preprocessed-nlcomps', action='store_true', help="")
    parser.add_argument('--bert-model', type=str, default='bert-base', help='which BERT model to use')
    parser.add_argument('--use-bert-encoder', action="store_true", help='whether to use BERT in the language encoder')
    parser.add_argument('--traj-encoder', default='mlp', choices=['mlp', 'transformer', 'lstm'],
                        help='which trajectory encoder to use')
    parser.add_argument('--weight-decay', type=float, default=0, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--num-iterations', type=int, default=10, help='')
    parser.add_argument('--use-all-datasets', action="store_true", help='whether to use all datasets or just test set')
    parser.add_argument('--use-softmax', action="store_true", help='whether to use softmax or argmax for feedback')

    args = parser.parse_args()
    run(args)
