import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import bert.extract_features as b
from feature_learning.nl_traj_dataset import NLTrajComparisonDataset
from feature_learning.model import NLTrajAutoencoder
import argparse
import os

INITIAL_LOSS_SANITY_CHECK = True


def train(seed, data_dir, epochs, save_dir, learning_rate=1e-3, weight_decay=0, encoder_hidden_dim=128,
          decoder_hidden_dim=128, bert_model='bert-base', remove_lang_encoder_hidden=False,
          preprocessed_nlcomps=False, id_mapped=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # load it to the specified device, either gpu or cpu
    print("Initializing model and loading to device...")
    model = NLTrajAutoencoder(encoder_hidden_dim=encoder_hidden_dim, decoder_hidden_dim=decoder_hidden_dim,
                              bert_model=bert_model, remove_lang_encoder_hidden=remove_lang_encoder_hidden,
                              preprocessed_nlcomps=preprocessed_nlcomps).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # mean-squared error loss
    mse = nn.MSELoss()
    logsigmoid = nn.LogSigmoid()

    print("Loading dataset...")

    # Some file-handling logic first.
    if id_mapped:
        train_nlcomp_index_file = os.path.join(data_dir, "train/nlcomp_indexes_{}.npy".format(bert_model))
        train_unique_nlcomp_file = os.path.join(data_dir, "train/unique_nlcomps_{}.npy".format(bert_model))
        val_nlcomp_index_file = os.path.join(data_dir, "val/nlcomp_indexes_{}.npy".format(bert_model))
        val_unique_nlcomp_file = os.path.join(data_dir, "val/unique_nlcomps_{}.npy".format(bert_model))

        train_traj_a_index_file = os.path.join(data_dir, "train/traj_a_indexes.npy")
        train_traj_b_index_file = os.path.join(data_dir, "train/traj_b_indexes.npy")
        train_traj_file = os.path.join(data_dir, "train/trajs.npy")
        val_traj_a_index_file = os.path.join(data_dir, "val/traj_a_indexes.npy")
        val_traj_b_index_file = os.path.join(data_dir, "val/traj_b_indexes.npy")
        val_traj_file = os.path.join(data_dir, "val/trajs.npy")
    else:
        if preprocessed_nlcomps:
            train_nlcomp_file = os.path.join(data_dir, "train/nlcomps.npy")
            val_nlcomp_file = os.path.join(data_dir, "val/nlcomps.npy")
        else:
            train_nlcomp_file = os.path.join(data_dir, "train/nlcomps.json")
            val_nlcomp_file = os.path.join(data_dir, "val/nlcomps.json")

        train_traj_a_file = os.path.join(data_dir, "train/traj_as.npy")
        train_traj_b_file = os.path.join(data_dir, "train/traj_bs.npy")
        val_traj_a_file = os.path.join(data_dir, "val/traj_as.npy")
        val_traj_b_file = os.path.join(data_dir, "val/traj_bs.npy")

    if id_mapped:
        train_dataset = NLTrajComparisonDataset(train_nlcomp_index_file, train_traj_a_index_file,
                                                train_traj_b_index_file,
                                                preprocessed_nlcomps=preprocessed_nlcomps, id_mapped=id_mapped,
                                                unique_nlcomp_file=train_unique_nlcomp_file, traj_file=train_traj_file)
        val_dataset = NLTrajComparisonDataset(val_nlcomp_index_file, val_traj_a_index_file, val_traj_b_index_file,
                                              preprocessed_nlcomps=preprocessed_nlcomps, id_mapped=id_mapped,
                                              unique_nlcomp_file=val_unique_nlcomp_file, traj_file=val_traj_file)
    else:
        train_dataset = NLTrajComparisonDataset(train_nlcomp_file, train_traj_a_file, train_traj_b_file,
                                                preprocessed_nlcomps=preprocessed_nlcomps)
        val_dataset = NLTrajComparisonDataset(val_nlcomp_file, val_traj_a_file, val_traj_b_file,
                                              preprocessed_nlcomps=preprocessed_nlcomps)

    # NOTE: this creates a dataset that doesn't have trajectories separated across datasets. DEPRECATED.
    # generator = torch.Generator().manual_seed(seed)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[0.9, 0.1], generator=generator)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True
    )

    if INITIAL_LOSS_SANITY_CHECK:
        print("Initial loss sanity check...")
        # temp_loss = 0
        # for train_datapoint in train_loader:
        #     with torch.no_grad():
        #         traj_a, traj_b, lang = train_datapoint
        #         traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
        #         traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
        #         if preprocessed_nlcomps:
        #             lang = torch.as_tensor(lang, dtype=torch.float32, device=device)
        #         # lang = torch.as_tensor(lang, device=device)
        #         train_datapoint = (traj_a, traj_b, lang)
        #         pred = model(train_datapoint)
        #
        #         encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = pred
        #         reconstruction_loss = mse(decoded_traj_a, torch.mean(traj_a, dim=-2)) + mse(decoded_traj_b,
        #                                                                                     torch.mean(traj_b, dim=-2))
        #
        #         dot_prod = torch.einsum('ij,ij->i', encoded_traj_b - encoded_traj_a, encoded_lang)
        #         log_likelihood = torch.mean(logsigmoid(dot_prod))
        #         log_likelihood_loss = -1 * log_likelihood
        #
        #         temp_loss += (reconstruction_loss + log_likelihood_loss).item()
        # temp_loss /= len(train_loader)
        # print("initial train loss:", temp_loss)
        temp_loss = 0
        num_correct = 0
        for val_datapoint in val_loader:
            with torch.no_grad():
                traj_a, traj_b, lang = val_datapoint
                traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
                traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
                if preprocessed_nlcomps:
                    lang = torch.as_tensor(lang, dtype=torch.float32, device=device)
                # lang = torch.as_tensor(lang, device=device)
                val_datapoint = (traj_a, traj_b, lang)
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
        print("initial val loss: {:.4f}, accuracy: {:.4f}".format(temp_loss, accuracy))

    print("Beginning training...")
    train_losses = []
    val_losses = []
    val_reconstruction_losses = []
    val_cosine_similarities = []
    val_log_likelihoods = []
    accuracies = []
    for epoch in range(epochs):
        loss = 0
        for train_datapoint in train_loader:
            traj_a, traj_b, lang = train_datapoint

            # load it to the active device
            # also cast down (from float64 in np) to float32, since PyTorch's matrices are float32.
            traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
            traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
            if preprocessed_nlcomps:
                lang = torch.as_tensor(lang, dtype=torch.float32, device=device)
            # lang = torch.as_tensor(lang, device=device)

            # train_datapoint = train_datapoint.to(device)  # Shouldn't be needed, since already on device
            train_datapoint = (traj_a, traj_b, lang)

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
                traj_a, traj_b, lang = val_datapoint
                traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
                traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
                if preprocessed_nlcomps:
                    lang = torch.as_tensor(lang, dtype=torch.float32, device=device)
                # lang = torch.as_tensor(lang, device=device)
                val_datapoint = (traj_a, traj_b, lang)
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
        accuracy = num_correct / len(val_dataset)

        # display the epoch training loss
        print(
            "epoch : {}/{}, [train] loss = {:.6f}, [val] loss = {:.6f}, [val] reconstruction_loss = {:.6f}, [val] cosine_similarity = {:.6f}, [val] log_likelihood = {:.6f}, [val] accuracy = {:.6f}".format(
                epoch + 1, epochs, loss, val_loss, val_reconstruction_loss, val_cosine_similarity, val_log_likelihood,
                accuracy))
        # Don't forget to save the model (as we go)!
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_state_dict.pth'))
        torch.save(model, os.path.join(save_dir, 'model.pth'))
        train_losses.append(loss)
        val_losses.append(val_loss)
        val_reconstruction_losses.append(val_reconstruction_loss)
        val_cosine_similarities.append(val_cosine_similarity)
        val_log_likelihoods.append(val_log_likelihood)
        accuracies.append(accuracy)
        np.save(os.path.join(save_dir, 'train_losses.npy'), np.asarray(train_losses))
        np.save(os.path.join(save_dir, 'val_losses.npy'), np.asarray(val_losses))
        np.save(os.path.join(save_dir, 'val_reconstruction_losses.npy'), np.asarray(val_reconstruction_losses))
        np.save(os.path.join(save_dir, 'val_cosine_similarities.npy'), np.asarray(val_cosine_similarities))
        np.save(os.path.join(save_dir, 'val_log_likelihoods.npy'), np.asarray(val_log_likelihoods))
        np.save(os.path.join(save_dir, 'accuracies.npy'), np.asarray(accuracies))

    # # Evaluation
    # num_correct = 0
    # log_likelihood = 0
    # for val_datapoint in val_loader:
    #     with torch.no_grad():
    #         traj_a, traj_b, lang = val_datapoint
    #         traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
    #         traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
    #         # lang = torch.as_tensor(lang, device=device)
    #
    #         val_datapoint = (traj_a, traj_b, lang)
    #         pred = model(val_datapoint)
    #         encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = pred
    #
    #         encoded_traj_a = encoded_traj_a.detach().cpu().numpy()
    #         encoded_traj_b = encoded_traj_b.detach().cpu().numpy()
    #         encoded_lang = encoded_lang.detach().cpu().numpy()
    #
    #         dot_prod = np.dot(encoded_traj_b-encoded_traj_a, encoded_lang)
    #         if dot_prod > 0:
    #             num_correct += 1
    #         log_likelihood += np.log(1/(1 + np.exp(-dot_prod)))
    #
    # accuracy = num_correct / len(val_loader)
    # print("final accuracy:", accuracy)
    # print("final log likelihood:", log_likelihood)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--data-dir', type=str, default='data/', help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--save-dir', type=str, default='', help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--weight-decay', type=float, default=0, help='')
    parser.add_argument('--encoder-hidden-dim', type=int, default=128, help='')
    parser.add_argument('--decoder-hidden-dim', type=int, default=128, help='')
    parser.add_argument('--remove-lang-encoder-hidden', action="store_true", help='')
    parser.add_argument('--preprocessed-nlcomps', action="store_true", help='')
    parser.add_argument('--id-mapped', action="store_true", help='whether the data is id mapped')
    parser.add_argument('--bert-model', type=str, default='bert-base', help='Which BERT model to use')

    args = parser.parse_args()

    trained_model = train(args.seed, args.data_dir, args.epochs, args.save_dir,
                          learning_rate=args.lr, weight_decay=args.weight_decay,
                          encoder_hidden_dim=args.encoder_hidden_dim, decoder_hidden_dim=args.decoder_hidden_dim,
                          bert_model=args.bert_model,
                          remove_lang_encoder_hidden=args.remove_lang_encoder_hidden,
                          preprocessed_nlcomps=args.preprocessed_nlcomps, id_mapped=args.id_mapped)
