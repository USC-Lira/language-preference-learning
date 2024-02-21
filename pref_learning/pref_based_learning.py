# preference based learning

# Importing the libraries
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer

from feature_learning.model import NLTrajAutoencoder
from pref_learning.pref_dataset import PrefDataset, EvalDataset
from pref_learning.utils import feature_aspects
from feature_learning.utils import BERT_MODEL_NAME, BERT_OUTPUT_DIM, AverageMeter
from model_analysis.utils import get_traj_lang_embeds
from model_analysis.improve_trajectory import initialize_reward, get_feature_value, get_lang_feedback


# learned and true reward func (linear for now)
def init_weights_with_norm_one(m):
    if isinstance(m, nn.Linear):  # Check if the module is a linear layer
        weight_shape = m.weight.size()
        # Initialize weights with a standard method
        weights = torch.normal(mean=0, std=0.01, size=weight_shape)
        # Normalize weights to have a norm of 1
        # weights /= weights.norm(2)  # Adjust this if you need a different norm
        m.weight.data = weights
        # You can also initialize biases here if needed
        if m.bias is not None:
            m.bias.data.fill_(0)


class RewardFunc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RewardFunc, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.apply(init_weights_with_norm_one)

    def forward(self, x):
        return self.linear(x)


def get_lang_feedback_aspect(curr_feature, reward, optimal_traj_feature, noisy=False, temperature=1.0):
    """
    Get language feedback based on feature value and true reward function
    """
    potential_pos = torch.tensor((optimal_traj_feature - curr_feature.numpy()))
    potential_neg = torch.tensor((curr_feature.numpy() - optimal_traj_feature))
    potential = torch.cat([potential_pos, potential_neg], dim=1)
    potential = potential / temperature
    probs = torch.softmax(potential, dim=1)
    if noisy:
        # sample a language comparison with the probabilities
        idx = torch.multinomial(probs, 1).item()
        feature_idx = idx % 5
        pos = idx // 5
        # pos_prob = reward[idx]
        # if np.random.rand() < pos_prob:
        #     pos = True
        # else:
        #     pos = False
    else:
        idx = torch.argmax(probs).item()
        feature_idx = idx % 5
        pos = idx // 5
    # probs = torch.softmax(torch.from_numpy(reward), dim=0)
    # feature_idx = torch.multinomial(probs, 1).item()
    # if optimal_traj_feature[feature_idx] > curr_feature[0, feature_idx]:
    #     pos = True
    # else:
    #     pos = False
    return feature_idx, pos


def load_data(args, test=False, DEBUG=False):
    # Load the test trajectories and language comparisons
    if not test:
        trajs = np.load(f'{args.data_dir}/train/trajs.npy')
        nlcomps = json.load(open(f'{args.data_dir}/train/unique_nlcomps.json', 'rb'))
        nlcomp_embeds = np.load(f'{args.data_dir}/train/unique_nlcomps_{args.bert_model}.npy')
    else:
        trajs = np.load(f'{args.data_dir}/val/trajs.npy')
        nlcomps = json.load(open(f'{args.data_dir}/val/unique_nlcomps.json', 'rb'))
        nlcomp_embeds = np.load(f'{args.data_dir}/val/unique_nlcomps_{args.bert_model}.npy')

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
    greater_nlcomps = json.load(open(f'{args.data_dir}/train/greater_nlcomps.json', 'rb'))
    less_nlcomps = json.load(open(f'{args.data_dir}/train/less_nlcomps.json', 'rb'))
    if DEBUG:
        print("greater nlcomps size: " + str(len(greater_nlcomps)))
        print("less nlcomps size: " + str(len(less_nlcomps)))

    return trajs, nlcomps, nlcomp_embeds, greater_nlcomps, less_nlcomps


def reconstruct_traj(traj_embeds, model, nlcomp_embeds):
    new_trajs = torch.from_numpy(traj_embeds) + nlcomp_embeds
    recon_trajs = model.traj_decoder(new_trajs.to('cuda'))
    return recon_trajs.detach().cpu().numpy()


def _pref_learning(train_dataloader, test_dataloader, model, nlcomps, greater_nlcomps, less_nlcomps, learned_reward,
                   true_reward, traj_embeds, lang_embeds, test_traj_embeds, test_traj_true_rewards,
                   less_idx, optimizer, args, optimal_traj_feature, feature_mins, feature_maxs, DEBUG=False):
    # Transform numpy array to torch tensor (improve this with a function)
    traj_embeds = torch.from_numpy(traj_embeds)

    eval_cross_entropies = []
    learned_reward_norms = []
    all_lang_feedback = []
    sampled_traj_embeds = []
    sampled_traj_true_rewards = []
    optimal_learned_rewards, optimal_true_rewards = [], []
    logsigmoid = nn.LogSigmoid()
    init_ce = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds)
    print("Initial Cross Entropy:", init_ce)
    eval_cross_entropies.append(init_ce)

    learned_rewards = torch.tensor([learned_reward(torch.from_numpy(traj_embed)) for traj_embed in test_traj_embeds])
    optimal_learned_reward = test_traj_true_rewards[torch.argmax(learned_rewards)]
    optimal_true_reward = test_traj_true_rewards[torch.argmax(torch.tensor(test_traj_true_rewards))]
    optimal_learned_rewards.append(optimal_learned_reward)
    optimal_true_rewards.append(optimal_true_reward)

    for train_data in train_dataloader:
        curr_traj, curr_feature_value, idx = train_data
        curr_traj_embed = traj_embeds[idx]
        sampled_traj_embeds.append(curr_traj_embed.view(1, -1))
        curr_traj_reward = torch.sum(torch.from_numpy(true_reward) * curr_feature_value.numpy())
        sampled_traj_true_rewards.append(curr_traj_reward.view(1, -1))

        # Use true reward func to get language feedback (select from set)
        # First find the feature aspect to give feedback on and positive / negative
        feature_aspect_idx, pos = get_lang_feedback_aspect(curr_feature_value, true_reward, optimal_traj_feature,
                                                           args.use_softmax, temperature=0.5)

        # if pos:
        #     nlcomp = np.random.choice(greater_nlcomps[feature_aspects[feature_aspect_idx]])
        # else:
        #     nlcomp = np.random.choice(less_nlcomps[feature_aspects[feature_aspect_idx]])

        if feature_aspect_idx in less_idx:
            # randomly choose from less_nlcomps
            if pos:
                nlcomp = np.random.choice(less_nlcomps[feature_aspects[feature_aspect_idx]])
            else:
                nlcomp = np.random.choice(greater_nlcomps[feature_aspects[feature_aspect_idx]])
        else:
            # randomly choose from greater_nlcomps
            if pos:
                nlcomp = np.random.choice(greater_nlcomps[feature_aspects[feature_aspect_idx]])
            else:
                nlcomp = np.random.choice(less_nlcomps[feature_aspects[feature_aspect_idx]])

        all_lang_feedback.append(nlcomp)

        # Get the feature of the language comparison
        nlcomp_features = torch.concat([torch.from_numpy(lang_embeds[nlcomps.index(nlcomp)]).view(1, -1)
                                        for nlcomp in all_lang_feedback], dim=0)

        if DEBUG:
            nlcomp_feature_norm = torch.norm(nlcomp_features, dim=1, keepdim=True)
            print("nlcomp_feature_norm: ", nlcomp_feature_norm)

        # recon_trajs = reconstruct_traj(np.array(), model, nlcomp_features)
        # recon_features = np.array([get_feature_value(recon_traj, traj_mean=True) for recon_traj in recon_trajs])
        # recon_features = (recon_features - feature_mins) / (feature_maxs - feature_mins)

        for i in range(args.num_iterations):
            # Compute dot product of lang(traj_opt - traj_cur)
            # Minimize the negative dot product (loss)!
            loss = -logsigmoid(learned_reward(nlcomp_features)).mean()
            loss += F.mse_loss(torch.concat(sampled_traj_true_rewards).float(),
                               learned_reward(torch.concat(sampled_traj_embeds)).float())
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Loss: {loss.item()}, Norm of learned reward: {torch.norm(learned_reward.linear.weight)}')
        learned_reward_norms.append(torch.norm(learned_reward.linear.weight).item())

        # Evaluation
        cross_entropy = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds, test=False)
        eval_cross_entropies.append(cross_entropy)

        # Fine the optimal trajectory with learned reward
        learned_rewards = torch.tensor(
            [learned_reward(torch.from_numpy(traj_embed)) for traj_embed in test_traj_embeds])
        optimal_learned_reward = test_traj_true_rewards[torch.argmax(learned_rewards)]
        optimal_true_reward = test_traj_true_rewards[torch.argmax(torch.tensor(test_traj_true_rewards))]
        optimal_learned_rewards.append(optimal_learned_reward)
        optimal_true_rewards.append(optimal_true_reward)

    return_dict = {
        'cross_entropy': eval_cross_entropies,
        'learned_reward_norms': learned_reward_norms,
        'optimal_learned_rewards': optimal_learned_rewards,
        'optimal_true_rewards': optimal_true_rewards
    }
    return return_dict


def evaluate(test_dataloader, true_traj_rewards, learned_reward, traj_embeds, test=False):
    total_cross_entropy = AverageMeter('cross-entropy')
    bce_loss = nn.BCELoss()
    for i, test_data in enumerate(test_dataloader):
        traj_a, traj_b, idx_a, idx_b = test_data

        # get the embeddings of the two trajectories
        traj_a_embed = traj_embeds[idx_a]
        traj_b_embed = traj_embeds[idx_b]

        # get bernoulli distributions for the two trajectories
        true_rewards = torch.tensor([true_traj_rewards[idx_a], true_traj_rewards[idx_b]])
        # make true probs 0 and 1
        # true_probs = torch.tensor([torch.argmax(true_rewards) == 0, torch.argmax(true_rewards) == 1]).float()
        # make true probs with softmax
        true_probs = torch.softmax(true_rewards, dim=0).float()

        traj_a_embed = torch.tensor(traj_a_embed)
        traj_b_embed = torch.tensor(traj_b_embed)
        learned_rewards = torch.tensor([learned_reward(traj_a_embed), learned_reward(traj_b_embed)])
        learned_probs = torch.softmax(learned_rewards, dim=0)
        print('True probs:', true_probs)
        if test:
            learned_probs = true_probs

        # calculate cross-entropy between learned and true distributions
        cross_entropy = bce_loss(learned_probs, true_probs)
        # kl_div_loss = kl_div(learned_probs.log(), true_probs)
        total_cross_entropy.update(cross_entropy, 1)
    return total_cross_entropy.avg


def run(args):
    np.random.seed(args.seed)

    # true_reward = initialize_reward(5)
    # For now fixed true reward
    true_reward = np.array([0.2, 0.1, 0.3, 0.35, 0.05])

    # Load train data
    train_trajs, train_nlcomps, train_nlcomps_embed, train_greater_nlcomps, train_less_nlcomps = load_data(args)
    train_feature_values = np.array([get_feature_value(traj) for traj in train_trajs])

    # Load test data
    test_trajs, test_nlcomps, test_nlcomps_embed, test_greater_nlcomps, test_less_nlcomps = load_data(args, test=True)
    test_feature_values = np.array([get_feature_value(traj) for traj in test_trajs])

    all_features = np.concatenate([train_feature_values, test_feature_values], axis=0)
    feature_value_mins = np.min(all_features, axis=0)
    feature_value_maxs = np.max(all_features, axis=0)

    # Normalize the feature values
    train_feature_values = (train_feature_values - feature_value_mins) / (feature_value_maxs - feature_value_mins)
    test_feature_values = (test_feature_values - feature_value_mins) / (feature_value_maxs - feature_value_mins)

    # Flip the feature values for distance to cube and distance to bottle
    less_idx = [3, 4]
    for i in less_idx:
        train_feature_values[:, i] = 1 - train_feature_values[:, i]
        test_feature_values[:, i] = 1 - test_feature_values[:, i]

    train_traj_true_rewards = np.dot(train_feature_values, true_reward)
    test_traj_true_rewards = np.dot(test_feature_values, true_reward)

    # Initialize the dataset and dataloader
    train_dataset = PrefDataset(train_trajs, train_feature_values)
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = EvalDataset(test_trajs)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)

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

    # Compatibility with old models
    state_dict = torch.load(os.path.join(args.model_dir, 'best_model_state_dict.pth'))
    if args.old_model:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('_hidden_layer', '.0')
            new_k = new_k.replace('_output_layer', '.2')
            new_state_dict[new_k] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    train_traj_embeds, train_lang_embeds = get_traj_lang_embeds(train_trajs, train_nlcomps, model, device,
                                                                args.use_bert_encoder, tokenizer,
                                                                nlcomps_bert_embeds=train_nlcomps_embed)
    test_traj_embeds, test_lang_embeds = get_traj_lang_embeds(test_trajs, test_nlcomps, model, device,
                                                              args.use_bert_encoder, tokenizer,
                                                              nlcomps_bert_embeds=test_nlcomps_embed)

    # Random init learned reward
    learned_reward = RewardFunc(feature_dim, 1)
    optimizer = torch.optim.SGD(learned_reward.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Load optimal trajectory given the true reward
    optimal_traj = np.load(f'{args.optimal_traj_dir}/traj.npy').reshape(500, 69)
    optimal_traj_feature = get_feature_value(optimal_traj)
    # Normalize the feature value
    optimal_traj_feature = (optimal_traj_feature - feature_value_mins) / (feature_value_maxs - feature_value_mins)

    results = _pref_learning(train_data, test_data, model, train_nlcomps, train_greater_nlcomps, train_less_nlcomps,
                             learned_reward, true_reward, train_traj_embeds, train_lang_embeds, test_traj_embeds,
                             test_traj_true_rewards,
                             less_idx, optimizer, args, optimal_traj_feature,
                             feature_value_mins, feature_value_maxs)

    eval_cross_entropies = results['cross_entropy']
    learned_reward_norms = results['learned_reward_norms']

    # plot the cross-entropies
    plt.figure()
    plt.plot(eval_cross_entropies)
    plt.xlabel('Number of Queries')
    plt.xticks(np.arange(0, len(eval_cross_entropies), 20))
    plt.ylabel('Cross-Entropy')
    plt.title(f'Noisy Feedback, True Dist: Softmax, {args.num_iterations} Iterations')
    # plt.savefig(f'cross_entropy_noisy_softmax_{args.num_iterations}_itrs.png')
    plt.show()

    plt.figure()
    plt.plot(learned_reward_norms)
    plt.xlabel('Number of Queries')
    plt.xticks(np.arange(0, len(learned_reward_norms), 20))
    plt.ylabel('Norm of Learned Reward')
    plt.title(f'Noisy Feedback, True Dist: Softmax, {args.num_iterations} Iterations')
    # plt.savefig(f'learned_reward_norm_noisy_softmax_{args.num_iterations}_itrs.png')
    plt.show()

    optimal_learned_rewards = results['optimal_learned_rewards']
    optimal_true_rewards = results['optimal_true_rewards']
    plt.figure()
    plt.plot(optimal_learned_rewards, label='Learned Reward')
    plt.plot(optimal_true_rewards, label='True Reward', c='r')
    plt.xlabel('Number of Queries')
    plt.xticks(np.arange(0, len(optimal_learned_rewards), 20))
    plt.ylabel('Reward')
    plt.title(f'The Reward of Optimal Trajectory')
    plt.legend()
    # plt.savefig(f'optimal_rewards_noisy_softmax_{args.num_iterations}_itrs.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data-dir', type=str, default='data', help='')
    parser.add_argument('--model-dir', type=str, default='models', help='')
    parser.add_argument('--optimal-traj-dir', type=str, default='optim_traj',
                        help='the directory of saved optimal trajs')
    parser.add_argument('--old-model', action="store_true", help='whether to use old model')
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
