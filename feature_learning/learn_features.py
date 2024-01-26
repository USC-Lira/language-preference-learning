import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from feature_learning.nl_traj_dataset import NLTrajComparisonDataset
from feature_learning.model import NLTrajAutoencoder
from feature_learning.utils import timeStamped, BERT_MODEL_NAME, BERT_OUTPUT_DIM, create_logger, AverageMeter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"


def train(logger, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # Load model to the specified device, either gpu or cpu
    logger.info("Initializing model and loading to device...")
    if args.use_bert_encoder:
        lang_encoder = AutoModel.from_pretrained(BERT_MODEL_NAME[args.bert_model])
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME[args.bert_model])
        feature_dim = BERT_OUTPUT_DIM[args.bert_model]
    else:
        lang_encoder = None
        tokenizer = None
        feature_dim = 128

    model = NLTrajAutoencoder(encoder_hidden_dim=args.encoder_hidden_dim, feature_dim=feature_dim,
                              decoder_hidden_dim=args.decoder_hidden_dim, lang_encoder=lang_encoder,
                              preprocessed_nlcomps=args.preprocessed_nlcomps, bert_output_dim=BERT_OUTPUT_DIM[args.bert_model],
                              use_bert_encoder=args.use_bert_encoder, traj_encoder=args.traj_encoder,
                              use_cnn_in_transformer=args.use_cnn_in_transformer,
                              use_casual_attention=args.use_casual_attention)

    if args.use_bert_encoder:
        if not args.finetune_bert:
            # Freeze BERT in the first training stage
            for param in model.lang_encoder.parameters():
                param.requires_grad = False
        else:
            # Load the model.
            model_path = os.path.join(exp_dir, "best_model_state_dict.pth")
            logger.info(f"Loaded model from: {model_path}")
            model.load_state_dict(torch.load(model_path))

    for name, param in model.named_parameters():
        logger.debug(f"{name}: {param.requires_grad}")

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)

    # mean-squared error loss
    logsigmoid = nn.LogSigmoid()

    logger.info("Loading dataset...")

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # Some file-handling logic first.
    # If we don't use bert in the language encoder, then we need to load the preprocessed nlcomps.
    if not args.use_bert_encoder:
        train_nlcomp_index_file = os.path.join(args.data_dir, "train/nlcomp_indexes_{}.npy".format(args.bert_model))
        train_unique_nlcomp_file = os.path.join(args.data_dir, "train/unique_nlcomps_{}.npy".format(args.bert_model))
        val_nlcomp_index_file = os.path.join(args.data_dir, "val/nlcomp_indexes_{}.npy".format(args.bert_model))
        val_unique_nlcomp_file = os.path.join(args.data_dir, "val/unique_nlcomps_{}.npy".format(args.bert_model))
    else:
        train_nlcomp_index_file = os.path.join(args.data_dir, "train/nlcomp_indexes_{}.npy".format(args.bert_model))
        train_unique_nlcomp_file = os.path.join(args.data_dir, "train/unique_nlcomps_{}.json".format(args.bert_model))
        val_nlcomp_index_file = os.path.join(args.data_dir, "val/nlcomp_indexes_{}.npy".format(args.bert_model))
        val_unique_nlcomp_file = os.path.join(args.data_dir, "val/unique_nlcomps_{}.json".format(args.bert_model))

    train_traj_a_index_file = os.path.join(args.data_dir, "train/traj_a_indexes.npy")
    train_traj_b_index_file = os.path.join(args.data_dir, "train/traj_b_indexes.npy")
    train_traj_file = os.path.join(args.data_dir, "train/trajs.npy")
    val_traj_a_index_file = os.path.join(args.data_dir, "val/traj_a_indexes.npy")
    val_traj_b_index_file = os.path.join(args.data_dir, "val/traj_b_indexes.npy")
    val_traj_file = os.path.join(args.data_dir, "val/trajs.npy")

    train_dataset = NLTrajComparisonDataset(train_nlcomp_index_file, train_traj_a_index_file,
                                            train_traj_b_index_file, tokenizer=tokenizer,
                                            preprocessed_nlcomps=args.preprocessed_nlcomps,
                                            unique_nlcomp_file=train_unique_nlcomp_file, traj_file=train_traj_file)
    val_dataset = NLTrajComparisonDataset(val_nlcomp_index_file, val_traj_a_index_file, val_traj_b_index_file,
                                          tokenizer=tokenizer,
                                          preprocessed_nlcomps=args.preprocessed_nlcomps,
                                          unique_nlcomp_file=val_unique_nlcomp_file, traj_file=val_traj_file)

    # NOTE: this creates a dataset that doesn't have trajectories separated across datasets. DEPRECATED.
    # generator = torch.Generator().manual_seed(seed)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[0.9, 0.1], generator=generator)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    if args.initial_loss_check:
        logger.info("Initial loss sanity check...")
        temp_loss = 0
        num_correct = 0
        for val_data in val_loader:
            with torch.no_grad():
                val_data = {key: value.to(device) for key, value in val_data.items()}
                pred = model(val_data)

                encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = pred
                reconstruction_loss = F.mse_loss(decoded_traj_a, torch.mean(val_data['traj_a'], dim=-2)) + \
                                      F.mse_loss(decoded_traj_b, torch.mean(val_data['traj_b'], dim=-2))

                dot_prod = torch.einsum('ij,ij->i', encoded_traj_b - encoded_traj_a, encoded_lang)
                log_likelihood = torch.mean(logsigmoid(dot_prod))
                log_likelihood_loss = -1 * log_likelihood

                temp_loss += (reconstruction_loss + log_likelihood_loss).item()
                num_correct += np.sum(dot_prod.detach().cpu().numpy() > 0)

        temp_loss /= len(val_loader)
        accuracy = num_correct / len(val_dataset)
        logger.info("initial val loss: {:.4f}, accuracy: {:.4f}".format(temp_loss, accuracy))

    logger.info("Beginning training...")
    train_losses = []
    val_losses = []
    val_reconstruction_losses = []
    val_cosine_similarities = []
    val_log_likelihoods = []
    accuracies = []
    best_val_acc = 0
    for epoch in range(args.epochs):
        ep_loss = AverageMeter('loss')
        ep_log_likelihood_loss = AverageMeter('log_likelihood_loss')
        ep_reconstruction_loss = AverageMeter('reconstruction_loss')
        ep_cosine_sim = AverageMeter('cosine_similarity')
        ep_norm_loss = AverageMeter('norm_loss')

        with tqdm(total=len(train_loader), unit='batch') as pbar:
            pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")
            for train_data in train_loader:
                # load it to the active device
                train_data = {key: value.to(device) for key, value in train_data.items()}

                # reset the gradients back to zero
                optimizer.zero_grad()

                # compute reconstructions
                output = model(train_data)
                encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = output

                # compute training reconstruction loss
                # MSELoss already takes the mean over the batch.
                reconstruction_loss = F.mse_loss(decoded_traj_a, torch.mean(train_data['traj_a'], dim=-2))
                reconstruction_loss += F.mse_loss(decoded_traj_b, torch.mean(train_data['traj_b'], dim=-2))

                # F.cosine_similarity only reduces along the feature dimension, so we take the mean over the batch later.
                cos_sim = F.cosine_similarity(encoded_traj_b - encoded_traj_a, encoded_lang)
                cos_sim = torch.mean(cos_sim)  # Take the mean over the batch.
                distance_loss = 1 - cos_sim  # Then convert the value to a loss.

                dot_prod = torch.einsum('ij,ij->i', encoded_traj_b - encoded_traj_a, encoded_lang)
                log_likelihood = logsigmoid(dot_prod)
                log_likelihood = torch.mean(log_likelihood)  # Take the mean over the batch.
                log_likelihood_loss = -1 * log_likelihood  # Then convert the value to a loss.

                # Norm loss, to make sure the encoded vectors are unit vectors
                norm_loss = F.mse_loss(torch.norm(encoded_traj_a, dim=-1),
                                       torch.ones(encoded_traj_a.shape[0]).to(device))
                norm_loss += F.mse_loss(torch.norm(encoded_traj_b, dim=-1),
                                        torch.ones(encoded_traj_b.shape[0]).to(device))
                norm_loss += F.mse_loss(torch.norm(encoded_lang, dim=-1), torch.ones(encoded_lang.shape[0]).to(device))

                # norm loss
                # By now, train_loss is a scalar.
                # train_loss = reconstruction_loss + distance_loss
                train_loss = reconstruction_loss + log_likelihood_loss

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                ep_loss.update(train_loss.item(), train_data['traj_a'].shape[0])
                ep_reconstruction_loss.update(reconstruction_loss.item(), train_data['traj_a'].shape[0])
                ep_log_likelihood_loss.update(log_likelihood_loss.item(), train_data['traj_a'].shape[0])
                ep_cosine_sim.update(cos_sim.item(), train_data['traj_a'].shape[0])

                tqdm_postfix = {"log_likelihood_loss": ep_log_likelihood_loss.avg,
                                "reconstruction_loss": ep_reconstruction_loss.avg,
                                "cosine_similarity": ep_cosine_sim.avg,
                                }

                pbar.set_postfix(tqdm_postfix)

                pbar.update()

        # Evaluation
        val_loss = 0
        val_reconstruction_loss = 0
        val_norm_loss = 0
        val_cosine_similarity = 0
        val_log_likelihood = 0
        num_correct = 0
        for val_data in val_loader:
            with torch.no_grad():
                val_data = {key: value.to(device) for key, value in val_data.items()}
                pred = model(val_data)

                encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = pred

                reconstruction_loss = F.mse_loss(decoded_traj_a, torch.mean(val_data['traj_a'], dim=-2))
                reconstruction_loss += F.mse_loss(decoded_traj_b, torch.mean(val_data['traj_b'], dim=-2))
                val_reconstruction_loss += reconstruction_loss.item()  # record

                norm_loss = F.mse_loss(torch.norm(encoded_traj_a, dim=-1),
                                       torch.ones(encoded_traj_a.shape[0]).to(device))
                norm_loss += F.mse_loss(torch.norm(encoded_traj_b, dim=-1),
                                        torch.ones(encoded_traj_b.shape[0]).to(device))
                norm_loss += F.mse_loss(torch.norm(encoded_lang, dim=-1), torch.ones(encoded_lang.shape[0]).to(device))
                val_norm_loss += norm_loss.item()

                cos_sim = torch.mean(F.cosine_similarity(encoded_traj_b - encoded_traj_a, encoded_lang))
                distance_loss = 1 - cos_sim
                val_cosine_similarity += cos_sim.item()

                dot_prod = torch.einsum('ij,ij->i', encoded_traj_b - encoded_traj_a, encoded_lang)
                log_likelihood = torch.mean(logsigmoid(dot_prod))
                log_likelihood_loss = -1 * log_likelihood
                val_log_likelihood += log_likelihood.item()

                val_loss += (reconstruction_loss + log_likelihood_loss).item()
                num_correct += np.sum(dot_prod.detach().cpu().numpy() > 0)

        val_loss /= len(val_loader)
        val_reconstruction_loss /= len(val_loader)
        val_cosine_similarity /= len(val_loader)
        val_log_likelihood /= len(val_loader)
        val_acc = num_correct / len(val_dataset)
        val_norm_loss /= len(val_loader)

        # display the epoch training loss
        logger.info(
            "epoch : {}/{}, [train] loss = {:.4f}, [val] loss = {:.4f}, [val] reconstruction_loss = {:.4f}, "
            "[val] norm loss = {:.4f}, [val] cos_similarity = {:.4f}, [val] log_likelihood = {:.4f}, [val] accuracy = {:.6f}".format(
                epoch + 1, args.epochs, ep_loss.avg, val_loss, val_reconstruction_loss, val_norm_loss, val_cosine_similarity,
                val_log_likelihood,
                val_acc))
        # Don't forget to save the model (as we go)!
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_state_dict_{}.pth'.format(epoch)))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_state_dict.pth'))
        train_losses.append(ep_loss.avg)
        val_losses.append(val_loss)
        val_reconstruction_losses.append(val_reconstruction_loss)
        val_cosine_similarities.append(val_cosine_similarity)
        val_log_likelihoods.append(val_log_likelihood)
        accuracies.append(val_acc)
        np.save(os.path.join(args.save_dir, 'train_losses.npy'), np.asarray(train_losses))
        np.save(os.path.join(args.save_dir, 'val_losses.npy'), np.asarray(val_losses))
        np.save(os.path.join(args.save_dir, 'val_reconstruction_losses.npy'), np.asarray(val_reconstruction_losses))
        np.save(os.path.join(args.save_dir, 'val_cosine_similarities.npy'), np.asarray(val_cosine_similarities))
        np.save(os.path.join(args.save_dir, 'val_log_likelihoods.npy'), np.asarray(val_log_likelihoods))
        np.save(os.path.join(args.save_dir, 'accuracies.npy'), np.asarray(accuracies))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--exp-name', type=str, default='feature_learning', help='The name of experiment')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--data-dir', type=str, default='data/', help='')
    parser.add_argument('--epochs', type=int, default=5, help='')
    parser.add_argument('--batch-size', type=int, default=1024, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--weight-decay', type=float, default=0, help='')
    parser.add_argument('--encoder-hidden-dim', type=int, default=128, help='')
    parser.add_argument('--decoder-hidden-dim', type=int, default=128, help='')
    parser.add_argument('--preprocessed-nlcomps', action="store_true", help='')
    parser.add_argument('--initial-loss-check', action="store_true", help='whether to check initial loss')
    parser.add_argument('--finetune-bert', action="store_true", help='whether to finetune BERT')
    parser.add_argument('--save-dir', type=str, default='feature_learning/', help='where to save the model')
    parser.add_argument('--bert-model', type=str, default='bert-base', help='which BERT model to use')
    parser.add_argument('--use-bert-encoder', action="store_true", help='whether to use BERT in the language encoder')
    parser.add_argument('--traj-encoder', default='mlp', choices=['mlp', 'transformer', 'lstm'],
                        help='which trajectory encoder to use')
    parser.add_argument('--n-heads', type=int, default=4, help='number of heads in the multi-head attention')
    parser.add_argument('--n-layers', type=int, default=3, help='number of layers in the trajectory transformer')
    parser.add_argument('--use-cnn-in-transformer', action="store_true", help='whether to use CNN in the transformer')
    parser.add_argument('--use-casual-attention', action="store_true", help='whether to use casual attention in the transformer')

    args = parser.parse_args()

    # Create exp directory and logger
    exp_dir = os.path.join('exp', timeStamped(args.exp_name))
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)

    if not args.use_bert_encoder:
        # Linear model: one-stage training
        trained_model = train(logger, args)

    else:
        # BERT as the language encoder: two-stage training
        # Stage 1: train the trajectory encoder with BERT frozen
        logger.info('\n------------------ Freeze BERT ------------------')
        train(logger, args)

        # Stage 2: co-finetune BERT and the trajectory encoder
        logger.info('\n------------------ Co-finetune BERT ------------------')
        args.finetune_bert = True
        train(logger, args)
