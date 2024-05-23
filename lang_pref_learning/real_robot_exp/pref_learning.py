"""
Preference learning for real robot experiments
"""

# Importing the libraries
import json
import re
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from lang_pref_learning.model.encoder import NLTrajAutoencoder
from lang_pref_learning.pref_learning.pref_dataset import LangPrefDataset, CompPrefDataset, EvalDataset
from lang_pref_learning.pref_learning.utils import feature_aspects
from lang_pref_learning.feature_learning.utils import LANG_MODEL_NAME, LANG_OUTPUT_DIM, AverageMeter
from lang_pref_learning.real_robot_exp.utils import get_traj_embeds_wx
from lang_pref_learning.real_robot_exp.improve_trajectory import get_feature_value
from lang_pref_learning.real_robot_exp.utils import replay_trajectory_video, remove_special_characters

from data.utils import WidowX_STATE_OBS_DIM, WidowX_ACTION_DIM, WidowX_PROPRIO_STATE_DIM, WidowX_OBJECT_STATE_DIM


# learned and true reward func (linear for now)
def init_weights_with_norm_one(m):
    if isinstance(m, nn.Linear):  # Check if the module is a linear layer
        weight_shape = m.weight.size()
        # Initialize weights with a standard method
        weights = torch.normal(mean=0, std=0.001, size=weight_shape)
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


def load_data(args, DEBUG=False):
    # # Load the trajectories and language comparisons
    # nlcomps = json.load(open(f"{args.data_dir}/unique_nlcomps.json", "rb"))

    # nlcomp_embeds = None
    
    traj_root_dir = os.path.join(args.data_dir, 'trajectory')
    trajs = []
    traj_images = []

    # Read all trajectory directories by numerical order
    traj_dirs = os.listdir(traj_root_dir)

    def extract_number(filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        return None
    
    sorted_traj_dirs = sorted(traj_dirs, key=extract_number)

    for traj_dir in sorted_traj_dirs:
        traj = np.load(os.path.join(traj_root_dir, traj_dir, 'trajs.npy'))
        traj_img_obs = np.load(os.path.join(traj_root_dir, traj_dir, 'traj_img_obs.npy'))
        trajs.append(traj)
        traj_images.append(traj_img_obs)

    # need to run categorize.py first to get these files
    # greater_nlcomps = json.load(open(f"{args.data_dir}/greater_nlcomps.json", "rb"))
    # less_nlcomps = json.load(open(f"{args.data_dir}/less_nlcomps.json", "rb"))
    # classified_nlcomps = json.load(open(f"{args.data_dir}/classified_nlcomps.json", "rb"))
    # if DEBUG:
    #     print("greater nlcomps size: " + str(len(greater_nlcomps)))
    #     print("less nlcomps size: " + str(len(less_nlcomps)))

    data = {
        "trajs": trajs,
        # "nlcomps": nlcomps,
        # "nlcomp_embeds": nlcomp_embeds,
        # "greater_nlcomps": greater_nlcomps,
        # "less_nlcomps": less_nlcomps,
        # "classified_nlcomps": classified_nlcomps,
        'traj_img_obs': traj_images,
    }

    return data


def get_optimal_traj(learned_reward, traj_embeds):
    # Fine the optimal trajectory with learned reward
    learned_rewards = learned_reward(traj_embeds)
    optim_traj_idx = torch.argmax(learned_rewards)

    return optim_traj_idx


def comp_pref_learning(
    args,
    dataloader,
    learned_reward,
    traj_embeds,
    traj_img_obs,
    optimizer,
    DEBUG=False,
):
    # TODO: write a new train dataloader
    # Transform numpy array to torch tensor (improve this with a function)
    traj_embeds = torch.from_numpy(traj_embeds)
    sampled_traj_a_embeds, sampled_traj_b_embeds = [], []
    sampled_labels = []
    optim_traj_scores = []

    CEloss = nn.CrossEntropyLoss()

    for it, comp_data in enumerate(dataloader):
        if it >= 20:
            break
        traj_a, traj_b, feature_a, feature_b, idx_a, idx_b = comp_data
        traj_a_embed, traj_b_embed = traj_embeds[idx_a], traj_embeds[idx_b]
        sampled_traj_a_embeds.append(traj_a_embed.view(1, -1))
        sampled_traj_b_embeds.append(traj_b_embed.view(1, -1))

        traj_a_images, traj_b_images = traj_img_obs[idx_a], traj_img_obs[idx_b]

        # Replay two trajectories to the user
        replay_trajectory_video(traj_a_images, 
                                title='Trajectory A',
                                frame_rate=10)
        
        replay_trajectory_video(traj_b_images, 
                                title='Trajectory B',
                                frame_rate=10)
        
        print(f"\nIteration {it + 1}")
        comp_feedback = input("Please choose your preferred one (a or b): ")

        receive_input = False
        while not receive_input:
            if comp_feedback == "a":
                sampled_labels.append(0)
                receive_input = True
            elif comp_feedback == "b":
                sampled_labels.append(1)
                receive_input = True
            else:
                # let the user choose again
                comp_feedback = input("Invalid input. Please choose again (a or b): ")

        for _ in range(args.num_iterations):
            pred_rewards_a = learned_reward(torch.concat(sampled_traj_a_embeds, dim=0))
            pred_rewards_b = learned_reward(torch.concat(sampled_traj_b_embeds, dim=0))
            pred_rewards = torch.cat([pred_rewards_a, pred_rewards_b], dim=-1)

            labels = torch.tensor(sampled_labels).long()

            assert pred_rewards.shape[0] == labels.shape[0]

            loss = CEloss(pred_rewards, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (it + 1) % 5 == 0:
            # Show users the best trajectory so far and let them rate it
            optim_traj_idx = get_optimal_traj(learned_reward, traj_embeds)
            traj_images = traj_img_obs[optim_traj_idx]
            replay_trajectory_video(traj_images, title='Optimal Trajectory', frame_rate=10)
            score = input("Please rate the trajectory (0-5, 5 is the best): ")
            optim_traj_scores.append(int(score))


def lang_pref_learning(
    args,
    dataloader,
    model,
    nlcomps,
    greater_nlcomps,
    less_nlcomps,
    classified_nlcomps,
    learned_reward,
    true_reward,
    traj_embeds,
    traj_img_obs,
    lang_embeds,
    optimal_traj_feature,
    optimizer,
    device,
    tokenizer,
    lang_encoder,
    lr_scheduler=None,
    DEBUG=False,
):
    # Transform numpy array to torch tensor (improve this with a function)
    traj_embeds = torch.from_numpy(traj_embeds)

    eval_cross_entropies = []
    learned_reward_norms = []
    all_lang_feedback = []
    all_other_language_feedback_feats = []
    sampled_traj_embeds = []
    optim_traj_scores = []
    logsigmoid = nn.LogSigmoid()


    for it, train_lang_data in enumerate(dataloader):
        if it > 20:
            break
        curr_traj, curr_feature_value, idx = train_lang_data
        curr_traj_embed = traj_embeds[idx]
        sampled_traj_embeds.append(curr_traj_embed.view(1, -1))

        # get the language feedback from the user
        curr_traj_images = traj_img_obs[idx]
        replay_trajectory_video(curr_traj_images, title='Current Trajectory', frame_rate=10)

        print(f"\nIteration {it + 1}")
        nlcomp = input("Please provide the language feedback: ")
        nlcomp = remove_special_characters(nlcomp)
        lang_embed = get_lang_embed(nlcomp, model, device, tokenizer, lang_model=lang_encoder)
        all_lang_feedback.append(nlcomp)

        # if pos:
        #     nlcomp = np.random.choice(greater_nlcomps[feature_aspects[feature_aspect_idx]])
        # else:
        #     nlcomp = np.random.choice(less_nlcomps[feature_aspects[feature_aspect_idx]])

        all_lang_feedback.append(nlcomp)
        # Get the feature of the language comparison
        nlcomp_features = torch.concat(
            [torch.from_numpy(lang_embeds[nlcomps.index(nlcomp)]).view(1, -1) for nlcomp in all_lang_feedback],
            dim=0,
        )

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
            other_nlcomp_features = torch.concat(
                [torch.from_numpy(feature).unsqueeze(0) for feature in all_other_language_feedback_feats_np],
                dim=0,
            )

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
                # Compute the cosine similarity between the language feedback and the other language feedback
                cos_sim = F.cosine_similarity(nlcomp_features_expand, other_nlcomp_features, dim=2)
                if args.use_constant_temp:
                    loss_lang_pref = -logsigmoid(
                        (learned_reward(nlcomp_features_expand) - learned_reward(other_nlcomp_features))
                        / args.lang_temp
                    )
                    if args.adaptive_weights:
                        weights = (1 - cos_sim) / 2
                        weights = weights / torch.sum(weights, dim=1, keepdim=True)
                        weights = weights.unsqueeze(2)
                        loss_lang_pref = torch.sum(weights * loss_lang_pref, dim=1).mean()
                    else:
                        loss_lang_pref = loss_lang_pref.mean()
                else:
                    # Transform cosine similarity to temperature
                    # temp_cos_sim = (1 - cos_sim) / 2
                    temp_cos_sim = 1 / (1 + torch.exp(-5 * cos_sim))
                    temp_cos_sim = temp_cos_sim.unsqueeze(2)
                    # Compute the preference loss
                    loss_lang_pref = -logsigmoid(
                        (learned_reward(nlcomp_features_expand) - learned_reward(other_nlcomp_features)) / temp_cos_sim
                    ).mean()

                loss += args.lang_loss_coeff * loss_lang_pref

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
                
        if (it + 1) % 5 == 0:
            # Show users the best trajectory so far and let them rate it
            optim_traj_idx = get_optimal_traj(learned_reward, traj_embeds)
            traj_images = traj_img_obs[optim_traj_idx]
            replay_trajectory_video(traj_images, title='Optimal Trajectory', frame_rate=10)
            score = input("Please rate the trajectory (0-5, 5 is the best): ")
            optim_traj_scores.append(int(score))
    
    return optim_traj_scores


def evaluate(test_dataloader, true_traj_rewards, learned_reward, traj_embeds, test=False, device="cuda"):
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
    total_cross_entropy = AverageMeter("cross-entropy")
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


def save_results(args, results, postfix="noisy"):
    save_dir = f"{args.true_reward_dir}/pref_learning"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    eval_cross_entropies = results["cross_entropy"]
    learned_reward_norms = results["learned_reward_norms"]
    optimal_learned_rewards = results["optimal_learned_rewards"]
    optimal_true_rewards = results["optimal_true_rewards"]

    result_dict = {
        "eval_cross_entropies": eval_cross_entropies,
        "learned_reward_norms": learned_reward_norms,
        "optimal_learned_rewards": optimal_learned_rewards,
        "optimal_true_rewards": optimal_true_rewards,
    }

    np.savez(f"{save_dir}/{postfix}.npz", **result_dict)


def run(args):
    # Load data
    
    data_dict = load_data(args)
    trajs = data_dict["trajs"]
    traj_img_obs = data_dict["traj_img_obs"]
    # nlcomps, nlcomps_embed = (
    #     data_dict["nlcomps"],
    #     data_dict["nlcomp_embeds"],
    # )
    # greater_nlcomps, less_nlcomps = (
    #     data_dict["greater_nlcomps"],
    #     data_dict["less_nlcomps"],
    # )
    # classified_nlcomps = data_dict["classified_nlcomps"]

    feature_values = np.array([get_feature_value(traj) for traj in trajs])

    # feature_value_mins = np.min(all_features, axis=0)
    # feature_value_maxs = np.max(all_features, axis=0)
    feature_value_means = np.mean(feature_values, axis=0)
    feature_value_stds = np.std(feature_values, axis=0)

    # Normalize the feature values
    feature_values = (feature_values - feature_value_means) / feature_value_stds

    # Initialize the dataset and dataloader
    lang_dataset = LangPrefDataset(trajs, feature_values)
    lang_data = DataLoader(lang_dataset, batch_size=1, shuffle=True)
    
    comp_dataset = CompPrefDataset(trajs, feature_values)
    comp_data = DataLoader(comp_dataset, batch_size=1, shuffle=True)

    # Current learned language encoder
    # Load the model
    if args.use_bert_encoder:
        if 't5' in args.lang_model:
            lang_encoder = T5EncoderModel.from_pretrained(args.lang_model)
        else:
            lang_encoder = AutoModel.from_pretrained(LANG_MODEL_NAME[args.lang_model])

        tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME[args.lang_model])
        feature_dim = LANG_OUTPUT_DIM[args.lang_model]
    else:
        lang_encoder = None
        tokenizer = None
        feature_dim = 128

    if args.env == "widowx":
        STATE_OBS_DIM = WidowX_STATE_OBS_DIM
        ACTION_DIM = WidowX_ACTION_DIM
        PROPRIO_STATE_DIM = WidowX_PROPRIO_STATE_DIM
        OBJECT_STATE_DIM = WidowX_OBJECT_STATE_DIM
    else:
        raise ValueError("Invalid environment")

    model = NLTrajAutoencoder(
        STATE_OBS_DIM, ACTION_DIM, PROPRIO_STATE_DIM, OBJECT_STATE_DIM,
        encoder_hidden_dim=args.encoder_hidden_dim,
        feature_dim=feature_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        lang_encoder=lang_encoder,
        preprocessed_nlcomps=args.preprocessed_nlcomps,
        lang_embed_dim=LANG_OUTPUT_DIM[args.lang_model],
        use_bert_encoder=args.use_bert_encoder,
        traj_encoder=args.traj_encoder,
    )

    # Compatibility with old models
    state_dict = torch.load(os.path.join(args.model_dir, "best_model_state_dict.pth"))
    if args.old_model:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("_hidden_layer", ".0")
            new_k = new_k.replace("_output_layer", ".2")
            new_state_dict[new_k] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Check if the embeddings are already computed
    # If not, compute the embeddings
    if not os.path.exists(f"{args.data_dir}/traj_embeds.npy"):
        traj_embeds = get_traj_embeds_wx(trajs, model, device, use_img_obs=True, img_obs=traj_img_obs)

        # Save the embeddings
        np.save(f"{args.data_dir}/traj_embeds.npy", traj_embeds)
    
    else:
        traj_embeds = np.load(f"{args.data_dir}/traj_embeds.npy")

    print("Mean Norm of Traj Embeds:", np.linalg.norm(traj_embeds, axis=1).mean())
    print("Norm of feature values:", np.linalg.norm(feature_values, axis=1).mean())

    # Random init learned reward
    learned_reward = RewardFunc(feature_dim, 1)
    optimizer = torch.optim.SGD(learned_reward.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

    learned_reward_noiseless = RewardFunc(feature_dim, 1)
    # Copy the weights from the noisy reward
    learned_reward_noiseless.linear.weight.data = learned_reward.linear.weight.data.clone()
    optimizer_noiseless = torch.optim.SGD(
        learned_reward_noiseless.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.method == "comp":
        results = comp_pref_learning(
            args,
            comp_data,
            learned_reward,
            traj_embeds,
            traj_img_obs,
            optimizer,
        )

    elif args.method == "lang":
        pass
    
    else:
        raise ValueError("Invalid method")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--env", type=str, default="widowx", help="")
    parser.add_argument("--data-dir", type=str, default="data", help="")
    parser.add_argument("--model-dir", type=str, default="models", help="")
    parser.add_argument("--old-model", action="store_true", help="")

    parser.add_argument("--encoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--preprocessed-nlcomps", action="store_true", help="")
    parser.add_argument(
        "--lang-model", type=str, default="t5-small", 
        choices=["bert-base", "bert-mini", "bert-tiny", "t5-small", "t5-base"],
        help="which language model to use"
    )
    parser.add_argument(
        "--use-bert-encoder",
        action="store_true",
        help="whether to use BERT in the language encoder",
    )
    parser.add_argument("--use-img-obs", action="store_true", help="whether to use image observations")
    parser.add_argument(
        "--traj-encoder",
        default="mlp",
        choices=["mlp", "transformer", "lstm", "cnn"],
        help="which trajectory encoder to use",
    )
    parser.add_argument("--weight-decay", type=float, default=0, help="")
    parser.add_argument("--lr", type=float, default=1e-3, help="")
    parser.add_argument("--num-iterations", type=int, default=1, help="")
    parser.add_argument(
        "--use-all-datasets",
        action="store_true",
        help="whether to use all datasets or just test set",
    )
    parser.add_argument(
        "--use-softmax",
        action="store_true",
        help="whether to use softmax or argmax for feedback",
    )
    parser.add_argument(
        "--use-other-feedback",
        action="store_true",
        help="whether to use other feedback",
    )
    parser.add_argument(
        "--num-other-feedback",
        default=1,
        type=int,
        help="number of other feedback to use",
    )
    parser.add_argument(
        "--coeff-other-feedback",
        default=1.0,
        type=float,
        help="coefficient for loss of other feedback",
    )
    parser.add_argument(
        "--use-constant-temp",
        action="store_true",
        help="whether to use constant temperature",
    )
    parser.add_argument(
        "--lang-temp",
        default=1.0,
        type=float,
        help="temperature for compare with other language feedback",
    )
    parser.add_argument(
        "--lang-loss-coeff",
        default=1.0,
        type=float,
        help="coefficient for language preference loss",
    )
    parser.add_argument(
        "--method",
        default="lang",
        type=str,
        choices=["lang", "comp"],
    )

    args = parser.parse_args()
    run(args)
