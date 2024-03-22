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

    Args:
        curr_feature: the feature value of the current trajectory
        reward: the true reward function
        optimal_traj_feature: the feature value of the optimal trajectory
        noisy: whether to add noise to the feedback
        temperature: the temperature for the softmax

    Returns:
        feature_idx: the index of the feature to give feedback on
        pos: whether the feedback is positive or negative
    """
    # potential_pos = torch.tensor(reward * (optimal_traj_feature - curr_feature.numpy()))
    # potential_neg = torch.tensor(reward * (curr_feature.numpy() - optimal_traj_feature))
    # potential = torch.cat([potential_pos, potential_neg], dim=1)
    potential = torch.tensor(reward * (optimal_traj_feature - curr_feature.numpy()))
    potential = potential / temperature
    probs = torch.softmax(potential, dim=1)
    if noisy:
        # sample a language comparison with the probabilities
        feature_idx = torch.multinomial(probs, 1).item()
    else:
        feature_idx = torch.argmax(probs).item()
    if optimal_traj_feature[feature_idx] - curr_feature[0, feature_idx] > 0:
        pos = True
    else:
        pos = False

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

    # need to run categorize.py first
    greater_nlcomps = json.load(open(f'{args.data_dir}/train/greater_nlcomps.json', 'rb'))
    less_nlcomps = json.load(open(f'{args.data_dir}/train/less_nlcomps.json', 'rb'))
    classified_nlcomps = json.load(open(f'{args.data_dir}/train/classified_nlcomps.json', 'rb'))
    if DEBUG:
        print("greater nlcomps size: " + str(len(greater_nlcomps)))
        print("less nlcomps size: " + str(len(less_nlcomps)))

    data = {
        'trajs': trajs,
        'nlcomps': nlcomps,
        'nlcomp_embeds': nlcomp_embeds,
        'greater_nlcomps': greater_nlcomps,
        'less_nlcomps': less_nlcomps,
        'classified_nlcomps': classified_nlcomps
    }

    return data


def reconstruct_traj(traj_embeds, model, nlcomp_embeds):
    new_trajs = torch.from_numpy(traj_embeds) + nlcomp_embeds
    recon_trajs = model.traj_decoder(new_trajs.to('cuda'))
    return recon_trajs.detach().cpu().numpy()


def get_optimal_traj(learned_reward, traj_embeds, traj_true_rewards):
    # Fine the optimal trajectory with learned reward
    learned_rewards = torch.tensor(
        [learned_reward(torch.from_numpy(traj_embed)) for traj_embed in traj_embeds])
    optimal_learned_reward = traj_true_rewards[torch.argmax(learned_rewards)]
    optimal_true_reward = traj_true_rewards[torch.argmax(torch.tensor(traj_true_rewards))]

    return optimal_learned_reward, optimal_true_reward


def _pref_learning(args, train_dataloader, test_dataloader, model,
                   nlcomps, greater_nlcomps, less_nlcomps, classified_nlcomps,
                   learned_reward, true_reward, traj_embeds, lang_embeds, test_traj_embeds, test_traj_true_rewards,
                   optimal_traj_feature, optimizer, lr_scheduler=None, DEBUG=False):
    # Transform numpy array to torch tensor (improve this with a function)
    traj_embeds = torch.from_numpy(traj_embeds)

    eval_cross_entropies = []
    learned_reward_norms = []
    all_lang_feedback = []
    all_other_language_feedback_feats = []
    sampled_traj_embeds = []
    sampled_traj_true_rewards = []
    optimal_learned_rewards, optimal_true_rewards = [], []
    logsigmoid = nn.LogSigmoid()
    init_ce = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds)
    print("Initial Cross Entropy:", init_ce)
    eval_cross_entropies.append(init_ce)

    optimal_learned_reward, optimal_true_reward = get_optimal_traj(learned_reward, test_traj_embeds, test_traj_true_rewards)
    optimal_learned_rewards.append(optimal_learned_reward)
    optimal_true_rewards.append(optimal_true_reward)

    for it, train_data in enumerate(train_dataloader):
        curr_traj, curr_feature_value, idx = train_data
        curr_traj_embed = traj_embeds[idx]
        sampled_traj_embeds.append(curr_traj_embed.view(1, -1))
        curr_traj_reward = torch.sum(torch.from_numpy(true_reward) * curr_feature_value.numpy())
        sampled_traj_true_rewards.append(curr_traj_reward.view(1, -1))

        # Use true reward func to get language feedback (select from set)
        # First find the feature aspect to give feedback on and positive / negative
        feature_aspect_idx, pos = get_lang_feedback_aspect(curr_feature_value, true_reward, optimal_traj_feature,
                                                           args.use_softmax, temperature=1.0)

        if pos:
            nlcomp = np.random.choice(greater_nlcomps[feature_aspects[feature_aspect_idx]])
        else:
            nlcomp = np.random.choice(less_nlcomps[feature_aspects[feature_aspect_idx]])

        all_lang_feedback.append(nlcomp)
        # Get the feature of the language comparison
        nlcomp_features = torch.concat([torch.from_numpy(lang_embeds[nlcomps.index(nlcomp)]).view(1, -1)
                                        for nlcomp in all_lang_feedback], dim=0)

        # Randomly sample feedback for other features in the training set
        if args.use_other_feedback:
            other_nlcomps = []
            for i in range(len(feature_aspects)):
                if i != feature_aspect_idx:
                    other_nlcomps.extend(classified_nlcomps[feature_aspects[i]])

            sampled_nlcomps = np.random.choice(other_nlcomps, args.num_other_feedback, replace=False)
            all_other_language_feedback_feats.append([lang_embeds[nlcomps.index(nlcomp)] for nlcomp in sampled_nlcomps])
        
            # Get the feature of the other language comparisons
            all_other_language_feedback_feats_np = np.array(all_other_language_feedback_feats)
            other_nlcomp_features = torch.concat([torch.from_numpy(feature).unsqueeze(0)
                                                 for feature in all_other_language_feedback_feats_np], dim=0)

        if DEBUG:
            nlcomp_feature_norm = torch.norm(nlcomp_features, dim=1, keepdim=True)
            print("nlcomp_feature_norm: ", nlcomp_feature_norm)

        for i in range(args.num_iterations):
            # Compute dot product of lang(traj_opt - traj_cur)
            # Minimize the negative dot product (loss)!
            loss = -logsigmoid(learned_reward(nlcomp_features)).mean()

            # Compute preference loss of selected language feedback over other language feedback
            # now nlcomp_features is of shape N x feature_dim, need to change it to N x n x feature_dim
            # where n is the number of other language feedback
            if args.use_other_feedback:
                nlcomp_features_expand = nlcomp_features.unsqueeze(1).expand(-1, other_nlcomp_features.shape[1], -1)
                if args.use_constant_temp:
                    loss_lang_pref = -logsigmoid((learned_reward(nlcomp_features_expand) - learned_reward(other_nlcomp_features)) / args.lang_temp).mean()
                else:
                    # Compute the cosine similarity between the language feedback and the other language feedback
                    cos_sim = F.cosine_similarity(nlcomp_features_expand, other_nlcomp_features, dim=2)
                    # Transform cosine similarity to temperature
                    # temp_cos_sim = (1 - cos_sim) / 2
                    temp_cos_sim = 1 / (1 + torch.exp(-5 * cos_sim))
                    temp_cos_sim = temp_cos_sim.unsqueeze(2)
                    # Compute the preference loss
                    loss_lang_pref = -logsigmoid((learned_reward(nlcomp_features_expand) - learned_reward(other_nlcomp_features)) / temp_cos_sim).mean()

                loss += args.lang_loss_coeff * loss_lang_pref

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

        print(f'Loss: {loss.item():.4f}, Norm of learned reward: {torch.norm(learned_reward.linear.weight):.4f}')
        learned_reward_norms.append(torch.norm(learned_reward.linear.weight).item())

        # norm_scale_factor = np.linalg.norm(true_reward) / torch.norm(learned_reward.linear.weight).item()
        # Evaluation
        cross_entropy = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds)

        eval_cross_entropies.append(cross_entropy)
        optimal_learned_reward, optimal_true_reward = get_optimal_traj(learned_reward, test_traj_embeds,
                                                                       test_traj_true_rewards)
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
    """
    Evaluate the cross-entropy between the learned and true distributions

    Input:
        - test_dataloader: DataLoader for the test data
        - true_traj_rewards: true rewards of the test trajectories
        - learned_reward: the learned reward function
        - traj_embeds: the embeddings of the test trajectories
        - feature_scale_factor: the scale factor for the learned reward
        - test: whether to use the true reward for evaluation

    Output:
        - total_cross_entropy: the average cross-entropy between the learned and true distributions
    """
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

        if test:
            learned_probs = true_probs

        else:
            traj_a_embed = torch.tensor(traj_a_embed)
            traj_b_embed = torch.tensor(traj_b_embed)
            learned_rewards = torch.tensor([learned_reward(traj_a_embed), learned_reward(traj_b_embed)])
            learned_probs = torch.softmax(learned_rewards, dim=0)

        # calculate cross-entropy between learned and true distributions
        cross_entropy = bce_loss(learned_probs, true_probs)
        # kl_div_loss = kl_div(learned_probs.log(), true_probs)
        total_cross_entropy.update(cross_entropy, 1)
    return total_cross_entropy.avg


def save_results(args, results, postfix='noisy'):
    save_dir = f'{args.true_reward_dir}/pref_learning'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    eval_cross_entropies = results['cross_entropy']
    learned_reward_norms = results['learned_reward_norms']
    optimal_learned_rewards = results['optimal_learned_rewards']
    optimal_true_rewards = results['optimal_true_rewards']

    result_dict = {
        'eval_cross_entropies': eval_cross_entropies,
        'learned_reward_norms': learned_reward_norms,
        'optimal_learned_rewards': optimal_learned_rewards,
        'optimal_true_rewards': optimal_true_rewards
    }

    np.savez(f'{save_dir}/{postfix}.npz', **result_dict)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load the weight of true reward
    true_reward = np.load(f'{args.true_reward_dir}/true_rewards.npy')

    # Load train data
    train_data_dict = load_data(args)
    train_trajs = train_data_dict['trajs']
    train_nlcomps, train_nlcomps_embed = train_data_dict['nlcomps'], train_data_dict['nlcomp_embeds']
    train_greater_nlcomps, train_less_nlcomps = train_data_dict['greater_nlcomps'], train_data_dict['less_nlcomps']
    train_classified_nlcomps = train_data_dict['classified_nlcomps']

    train_feature_values = np.array([get_feature_value(traj) for traj in train_trajs])

    # Load test data
    test_data_dict = load_data(args, test=True)
    test_trajs = test_data_dict['trajs']
    test_nlcomps, test_nlcomps_embed = test_data_dict['nlcomps'], test_data_dict['nlcomp_embeds']
    # test_greater_nlcomps, test_less_nlcomps = test_data_dict['greater_nlcomps'], test_data_dict['less_nlcomps']
    # test_classified_nlcomps = test_data_dict['classified_nlcomps']

    test_feature_values = np.array([get_feature_value(traj) for traj in test_trajs])

    all_features = np.concatenate([train_feature_values, test_feature_values], axis=0)
    # feature_value_mins = np.min(all_features, axis=0)
    # feature_value_maxs = np.max(all_features, axis=0)
    feature_value_means = np.mean(all_features, axis=0)
    feature_value_stds = np.std(all_features, axis=0)

    # Normalize the feature values
    train_feature_values = (train_feature_values - feature_value_means) / feature_value_stds
    test_feature_values = (test_feature_values - feature_value_means) / feature_value_stds

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

    print("Mean Norm of Traj Embeds:", np.linalg.norm(train_traj_embeds, axis=1).mean())
    print("Mean Norm of Lang Embeds:", np.linalg.norm(train_lang_embeds, axis=1).mean())
    print('Norm of feature values:', np.linalg.norm(train_feature_values, axis=1).mean())

    # feature_scale_factor = np.linalg.norm(train_feature_values, axis=1).mean() / np.linalg.norm(train_traj_embeds,
    #                                                                                             axis=1).mean()

    # Random init learned reward
    learned_reward = RewardFunc(feature_dim, 1)
    optimizer = torch.optim.SGD(learned_reward.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

    learned_reward_noiseless = RewardFunc(feature_dim, 1)
    # Copy the weights from the noisy reward
    learned_reward_noiseless.linear.weight.data = learned_reward.linear.weight.data.clone()
    optimizer_noiseless = torch.optim.SGD(learned_reward_noiseless.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
    # lr_scheduler_noiseless = torch.optim.lr_scheduler.StepLR(optimizer_noiseless, step_size=20, gamma=0.95)

    # Test the entropy of test data
    test_entropy = evaluate(test_data, test_traj_true_rewards, learned_reward, test_traj_embeds, test=True)

    # Load optimal trajectory given the true reward
    optimal_traj = np.load(f'{args.true_reward_dir}/traj.npy').reshape(500, 69)
    optimal_traj_feature = get_feature_value(optimal_traj)
    # Normalize the feature value
    optimal_traj_feature = (optimal_traj_feature - feature_value_means) / feature_value_stds

    noisy_results = _pref_learning(args, train_data, test_data, model, train_nlcomps, train_greater_nlcomps,
                                   train_less_nlcomps, train_classified_nlcomps,
                                   learned_reward, true_reward, train_traj_embeds, train_lang_embeds, test_traj_embeds,
                                   test_traj_true_rewards, optimal_traj_feature, optimizer)

    args.use_softmax = False
    noiseless_results = _pref_learning(args, train_data, test_data, model, train_nlcomps, train_greater_nlcomps,
                                       train_less_nlcomps, train_classified_nlcomps,
                                       learned_reward_noiseless, true_reward, train_traj_embeds, train_lang_embeds,
                                       test_traj_embeds, test_traj_true_rewards, optimal_traj_feature,
                                       optimizer_noiseless)

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(noisy_results['cross_entropy'], label='Noisy')
    ax1.plot(noiseless_results['cross_entropy'], label='Noiseless')
    ax1.set_xlabel('Number of Queries')
    ax1.set_ylabel('Cross-Entropy')
    ax1.set_title('Feedback, True Dist: Softmax')
    ax1.legend()

    ax2.plot(noisy_results['optimal_learned_rewards'], label='Noisy, Learned Reward')
    ax2.plot(noiseless_results['optimal_learned_rewards'], label='Noiseless, Learned Reward')
    ax2.plot(noisy_results['optimal_true_rewards'], label='True Reward', c='r')
    ax2.set_xlabel('Number of Queries')
    ax2.set_ylabel('Reward Value')
    ax2.set_title('True Reward of Optimal Trajectory')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{args.true_reward_dir}/pref_learning/lang_pref.png')
    plt.show()

    postfix_noisy = 'noisy'
    postfix_noiseless = 'noiseless'
    if args.use_other_feedback:
        postfix_noisy += '_other_feedback_' + str(args.num_other_feedback)
        postfix_noiseless += '_other_feedback_' + str(args.num_other_feedback)
        if args.use_constant_temp:
            postfix_noisy += f'_temp_{args.lang_temp}' + f"_lc_{args.lang_loss_coeff}"
            postfix_noiseless += f'_temp_{args.lang_temp}' + f"_lc_{args.lang_loss_coeff}"
        else:
            postfix_noisy += '_temp_cos' + f"_lc_{args.lang_loss_coeff}"
            postfix_noiseless += '_temp_cos' + f"_lc_{args.lang_loss_coeff}"

    save_results(args, noisy_results, postfix=postfix_noisy)
    save_results(args, noiseless_results, postfix=postfix_noiseless)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data-dir', type=str, default='data', help='')
    parser.add_argument('--model-dir', type=str, default='models', help='')
    parser.add_argument('--true-reward-dir', type=str, default='true_rewards/0',
                        help='the directory of trajectories and true rewards')
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
    parser.add_argument('--num-iterations', type=int, default=5, help='')
    parser.add_argument('--use-all-datasets', action="store_true", help='whether to use all datasets or just test set')
    parser.add_argument('--use-softmax', action="store_true", help='whether to use softmax or argmax for feedback')
    parser.add_argument('--use-other-feedback', action="store_true", help='whether to use other feedback')
    parser.add_argument('--num-other-feedback', default=1, type=int, help='number of other feedback to use')
    parser.add_argument('--coeff-other-feedback', default=1.0, type=float, help='coefficient for loss of other feedback')
    parser.add_argument('--use-constant-temp', action="store_true", help='whether to use constant temperature')
    parser.add_argument('--lang-temp', default=1.0, type=float, help='temperature for compare with other language feedback')
    parser.add_argument('--lang-loss-coeff', default=1.0, type=float, help='coefficient for language preference loss')

    args = parser.parse_args()
    run(args)
