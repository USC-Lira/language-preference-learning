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
from feature_learning.utils import timeStamped, BERT_MODEL_NAME, BERT_OUTPUT_DIM, create_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(logger, seed, data_dir, save_dir, epochs, batch_size, learning_rate=1e-3, weight_decay=0, encoder_hidden_dim=128,
          decoder_hidden_dim=128, preprocessed_nlcomps=False, id_mapped=False, initial_loss_check=False,
          finetune_bert=False, bert_model='bert-base'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # load it to the specified device, either gpu or cpu
    lang_encoder = AutoModel.from_pretrained(BERT_MODEL_NAME[bert_model])
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME[bert_model])
    logger.info("Initializing model and loading to device...")
    model = NLTrajAutoencoder(encoder_hidden_dim=encoder_hidden_dim, feature_dim=BERT_OUTPUT_DIM[bert_model],
                              decoder_hidden_dim=decoder_hidden_dim, lang_encoder=lang_encoder,
                              preprocessed_nlcomps=preprocessed_nlcomps)

    if not finetune_bert:
        # Freeze BERT in the first training stage
        for param in model.lang_encoder.parameters():
            param.requires_grad = False
    else:
        # Load the model to the specified device.
        model_path = os.path.join(exp_dir, "best_model_state_dict.pth")
        logger.info(f"Loaded model from: {model_path}")
        model.load_state_dict(torch.load(model_path))

    for name, param in model.named_parameters():
        logger.info(f"{name}: {param.requires_grad}")

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)

    # mean-squared error loss
    mse = nn.MSELoss()
    logsigmoid = nn.LogSigmoid()

    logger.info("Loading dataset...")

    # Some file-handling logic first.
    train_nlcomp_index_file = os.path.join(data_dir, "train/nlcomp_indexes_{}.npy".format(bert_model))
    train_unique_nlcomp_file = os.path.join(data_dir, "train/unique_nlcomps_{}.json".format(bert_model))
    val_nlcomp_index_file = os.path.join(data_dir, "val/nlcomp_indexes_{}.npy".format(bert_model))
    val_unique_nlcomp_file = os.path.join(data_dir, "val/unique_nlcomps_{}.json".format(bert_model))

    train_traj_a_index_file = os.path.join(data_dir, "train/traj_a_indexes.npy")
    train_traj_b_index_file = os.path.join(data_dir, "train/traj_b_indexes.npy")
    train_traj_file = os.path.join(data_dir, "train/trajs.npy")
    val_traj_a_index_file = os.path.join(data_dir, "val/traj_a_indexes.npy")
    val_traj_b_index_file = os.path.join(data_dir, "val/traj_b_indexes.npy")
    val_traj_file = os.path.join(data_dir, "val/trajs.npy")

    train_dataset = NLTrajComparisonDataset(train_nlcomp_index_file, train_traj_a_index_file,
                                            train_traj_b_index_file, tokenizer=tokenizer,
                                            preprocessed_nlcomps=preprocessed_nlcomps, id_mapped=id_mapped,
                                            unique_nlcomp_file=train_unique_nlcomp_file, traj_file=train_traj_file)
    val_dataset = NLTrajComparisonDataset(val_nlcomp_index_file, val_traj_a_index_file, val_traj_b_index_file,
                                          tokenizer=tokenizer,
                                          preprocessed_nlcomps=preprocessed_nlcomps, id_mapped=id_mapped,
                                          unique_nlcomp_file=val_unique_nlcomp_file, traj_file=val_traj_file)

    # NOTE: this creates a dataset that doesn't have trajectories separated across datasets. DEPRECATED.
    # generator = torch.Generator().manual_seed(seed)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[0.9, 0.1], generator=generator)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    if initial_loss_check:
        logger.info("Initial loss sanity check...")
        temp_loss = 0
        num_correct = 0
        for val_datapoint in val_loader:
            with torch.no_grad():
                traj_a, traj_b, lang_tokens, lang_attention_maks = val_datapoint
                traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
                traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
                lang_tokens = torch.as_tensor(lang_tokens, dtype=torch.long, device=device)
                lang_attention_maks = torch.as_tensor(lang_attention_maks, device=device).long()
                # lang = torch.as_tensor(lang, device=device)
                val_datapoint = (traj_a, traj_b, lang_tokens, lang_attention_maks)
                pred = model(val_datapoint)

                encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = pred
                reconstruction_loss = mse(decoded_traj_a, torch.mean(traj_a, dim=-2)) + mse(decoded_traj_b,
                                                                                            torch.mean(traj_b, dim=-2))

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
    for epoch in range(epochs):
        loss = 0
        with tqdm(total=len(train_loader), unit='batch') as pbar:
            for train_datapoint in train_loader:
                pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
                traj_a, traj_b, lang_tokens, lang_attention_maks = train_datapoint

                # load it to the active device
                # also cast down (from float64 in np) to float32, since PyTorch's matrices are float32.
                traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
                traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
                lang_tokens = torch.as_tensor(lang_tokens, dtype=torch.long, device=device)
                lang_attention_maks = torch.as_tensor(lang_attention_maks, dtype=torch.long, device=device)
                # lang = torch.as_tensor(lang, device=device)

                # train_datapoint = train_datapoint.to(device)  # Shouldn't be needed, since already on device
                train_datapoint = (traj_a, traj_b, lang_tokens, lang_attention_maks)

                # reset the gradients back to zero
                optimizer.zero_grad()

                # compute reconstructions
                output = model(train_datapoint)
                encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = output

                # compute training reconstruction loss
                # MSELoss already takes the mean over the batch.
                reconstruction_loss = mse(decoded_traj_a, torch.mean(traj_a, dim=-2)) + mse(decoded_traj_b,
                                                                                            torch.mean(traj_b, dim=-2))
                # print("reconstruction_loss:", reconstruction_loss.shape)

                # F.cosine_similarity only reduces along the feature dimension, so we take the mean over the batch later.
                cos_sim = F.cosine_similarity(encoded_traj_b - encoded_traj_a, encoded_lang)
                cos_sim = torch.mean(cos_sim)  # Take the mean over the batch.
                distance_loss = 1 - cos_sim  # Then convert the value to a loss.
                # print("distance_loss:", distance_loss.shape)

                # print("encoded_traj_b - encoded_traj_a:", (encoded_traj_b - encoded_traj_a).detach().cpu().numpy()[0])
                # print("encoded_lang:", (encoded_lang).detach().cpu().numpy()[0])
                dot_prod = torch.einsum('ij,ij->i', encoded_traj_b - encoded_traj_a, encoded_lang)
                # print("dot_prod:", dot_prod.detach().cpu().numpy()[0])
                log_likelihood = logsigmoid(dot_prod)
                log_likelihood = torch.mean(log_likelihood)  # Take the mean over the batch.
                log_likelihood_loss = -1 * log_likelihood  # Then convert the value to a loss.

                # By now, train_loss is a scalar.
                # train_loss = reconstruction_loss + distance_loss
                train_loss = reconstruction_loss + log_likelihood_loss
                # print("train_loss:", train_loss.shape)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

                pbar.set_postfix({"log_likelihood_loss": log_likelihood_loss.item(),
                                  "reconstruction_loss": reconstruction_loss.item(),
                                  "cosine_similarity": cos_sim.item()})
                pbar.update()

        # compute the epoch training loss
        # Note: this is the per-BATCH loss. len(train_loader) gives number of batches.
        loss = loss / len(train_loader)

        # Evaluation
        val_loss = 0
        val_reconstruction_loss = 0
        val_cosine_similarity = 0
        val_log_likelihood = 0
        num_correct = 0
        for val_datapoint in val_loader:
            with torch.no_grad():
                traj_a, traj_b, lang_tokens, lang_attention_maks = val_datapoint
                traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
                traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
                lang_tokens = torch.as_tensor(lang_tokens, dtype=torch.long, device=device)
                lang_attention_maks = torch.as_tensor(lang_attention_maks, dtype=torch.long, device=device)
                # lang = torch.as_tensor(lang, device=device)
                val_datapoint = (traj_a, traj_b, lang_tokens, lang_attention_maks)
                pred = model(val_datapoint)

                encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = pred
                reconstruction_loss = mse(decoded_traj_a, torch.mean(traj_a, dim=-2)) + mse(decoded_traj_b,
                                                                                            torch.mean(traj_b, dim=-2))
                val_reconstruction_loss += reconstruction_loss.item()  # record

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

        # display the epoch training loss
        logger.info(
            "epoch : {}/{}, [train] loss = {:.4f}, [val] loss = {:.4f}, [val] reconstruction_loss = {:.4f}, [val] cos_similarity = {:.4f}, [val] log_likelihood = {:.4f}, [val] accuracy = {:.6f}".format(
                epoch + 1, epochs, loss, val_loss, val_reconstruction_loss, val_cosine_similarity, val_log_likelihood,
                val_acc))
        # Don't forget to save the model (as we go)!
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_state_dict_{}.pth'.format(epoch)))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_state_dict.pth'))
        train_losses.append(loss)
        val_losses.append(val_loss)
        val_reconstruction_losses.append(val_reconstruction_loss)
        val_cosine_similarities.append(val_cosine_similarity)
        val_log_likelihoods.append(val_log_likelihood)
        accuracies.append(val_acc)
        np.save(os.path.join(save_dir, 'train_losses.npy'), np.asarray(train_losses))
        np.save(os.path.join(save_dir, 'val_losses.npy'), np.asarray(val_losses))
        np.save(os.path.join(save_dir, 'val_reconstruction_losses.npy'), np.asarray(val_reconstruction_losses))
        np.save(os.path.join(save_dir, 'val_cosine_similarities.npy'), np.asarray(val_cosine_similarities))
        np.save(os.path.join(save_dir, 'val_log_likelihoods.npy'), np.asarray(val_log_likelihoods))
        np.save(os.path.join(save_dir, 'accuracies.npy'), np.asarray(accuracies))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--exp-name', type=str, default='feature_learning', help='The name of experiment')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--data-dir', type=str, default='data/', help='')
    parser.add_argument('--epochs', type=int, default=10, help='')
    parser.add_argument('--batch-size', type=int, default=1024, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--weight-decay', type=float, default=0, help='')
    parser.add_argument('--encoder-hidden-dim', type=int, default=128, help='')
    parser.add_argument('--decoder-hidden-dim', type=int, default=128, help='')
    parser.add_argument('--preprocessed-nlcomps', action="store_true", help='')
    parser.add_argument('--id-mapped', action="store_true", help='whether the data is id mapped')
    parser.add_argument('--initial-loss-check', action="store_true", help='whether to check initial loss')
    parser.add_argument('--finetune-bert', action="store_true", help='whether to finetune BERT')
    parser.add_argument('--model-save-dir', type=str, default='feature_learning/', help='where to save the model')
    parser.add_argument('--bert-model', type=str, default='bert-base', help='which BERT model to use')

    args = parser.parse_args()

    # Create exp directory and logger
    exp_dir = os.path.join('exp', timeStamped(args.exp_name))
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)

    trained_model = train(logger, args.seed, args.data_dir, exp_dir, args.epochs, args.batch_size,
                          learning_rate=args.lr, weight_decay=args.weight_decay,
                          encoder_hidden_dim=args.encoder_hidden_dim, decoder_hidden_dim=args.decoder_hidden_dim,
                          preprocessed_nlcomps=args.preprocessed_nlcomps, id_mapped=args.id_mapped,
                          initial_loss_check=args.initial_loss_check,
                          finetune_bert=args.finetune_bert, bert_model=args.bert_model)
