import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
from tqdm import tqdm
import json


from lang_pref_learning.feature_learning.nl_traj_dataset import NLTrajComparisonDataset
from lang_pref_learning.model.encoder import NLTrajAutoencoder
from lang_pref_learning.feature_learning.utils import (
    timeStamped,
    HF_LANG_MODEL_NAME,
    LANG_OUTPUT_DIM,
    create_logger,
    AverageMeter,
)

from data.utils import env_dims

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"


def load_data(args, split="train"):
    # Load the data.
    # nlcomp_index_file is a .npy file with the indexes of the nlcomps in the dataset.
    nlcomp_index_file = os.path.join(args.data_dir, "{}/nlcomp_indexes.npy".format(split))

    unique_nlcomp_file = os.path.join(args.data_dir, "{}/unique_nlcomps.json".format(split))

    # traj_a_index_file is a .npy file with the indexes of the first trajectory in the dataset.
    # traj_b_index_file is a .npy file with the indexes of the second trajectory in the dataset.
    # traj_file is a .npy file with the trajectories of shape (n_trajectories, n_timesteps, STATE_DIM+ACTION_DIM)
    traj_a_index_file = os.path.join(args.data_dir, "{}/traj_a_indexes.npy".format(split))
    traj_b_index_file = os.path.join(args.data_dir, "{}/traj_b_indexes.npy".format(split))
    traj_file = os.path.join(args.data_dir, "{}/trajs.npy".format(split))

    return_files_dict = {
        "trajs": traj_file,
        "traj_a_index": traj_a_index_file,
        "traj_b_index": traj_b_index_file,
        "unique_nlcomp": unique_nlcomp_file,
        "nlcomp_index": nlcomp_index_file,
    }

    if args.use_img_obs:
        if args.use_stack_img_obs:
            if args.use_visual_features:
                traj_img_obs_file = os.path.join(
                    args.data_dir,
                    f"{split}/traj_img_features_{args.feature_extractor}_stack_{args.n_frames}.npy",
                )
            else:
                traj_img_obs_file = os.path.join(args.data_dir, f"{split}/traj_img_obs_stack_{args.n_frames}.npy")
        else:
            if args.use_visual_features:
                traj_img_obs_file = os.path.join(
                    args.data_dir,
                    f"{split}/traj_img_features_{args.feature_extractor}.npy",
                )
            else:
                traj_img_obs_file = os.path.join(args.data_dir, f"{split}/traj_img_obs.npy")
        action_file = os.path.join(args.data_dir, f"{split}/actions.npy")
        return_files_dict["traj_img_obs"] = traj_img_obs_file
        return_files_dict["actions"] = action_file

    return return_files_dict


def evaluate(model, data_loader, device):
    total_loss = 0
    total_reconstruction_loss = 0
    total_norm_loss = 0
    total_cosine_similarity = 0
    total_log_likelihood = 0
    total_num_correct = 0
    logsigmoid = nn.LogSigmoid()
    for data in tqdm(data_loader):
        with torch.no_grad():
            data = {key: value.to(device) for key, value in data.items()}

            pred = model(data, train=False)
            (
                encoded_traj_a,
                encoded_traj_b,
                encoded_lang,
                decoded_traj_a,
                decoded_traj_b,
            ) = pred

            reconstruction_loss = torch.tensor(0.0).to(device)
            if decoded_traj_a is not None and decoded_traj_b is not None:
                reconstruction_loss = F.mse_loss(decoded_traj_a, torch.mean(data["traj_a"], dim=-2))
                reconstruction_loss += F.mse_loss(decoded_traj_b, torch.mean(data["traj_b"], dim=-2))
                total_reconstruction_loss += reconstruction_loss.detach().cpu().item()

            norm_loss = F.mse_loss(
                torch.norm(encoded_lang, dim=-1),
                torch.ones(encoded_lang.shape[0]).to(device),
            )
            total_norm_loss += norm_loss.detach().cpu().item()

            cos_sim = torch.mean(F.cosine_similarity(encoded_traj_b - encoded_traj_a, encoded_lang))
            total_cosine_similarity += cos_sim.detach().cpu().item()

            dot_prod = torch.einsum("ij,ij->i", encoded_traj_b - encoded_traj_a, encoded_lang)
            log_likelihood = torch.mean(logsigmoid(dot_prod))
            log_likelihood_loss = -1 * log_likelihood
            total_log_likelihood += -log_likelihood_loss.detach().cpu().item()

            total_loss += (reconstruction_loss + log_likelihood_loss).detach().cpu().item()
            total_num_correct += np.sum(dot_prod.detach().cpu().numpy() > 0)

    metrics = {
        "loss": total_loss / len(data_loader),
        "reconstruction_loss": total_reconstruction_loss / len(data_loader),
        "norm_loss": total_norm_loss / len(data_loader),
        "cosine_similarity": total_cosine_similarity / len(data_loader),
        "log_likelihood": total_log_likelihood / len(data_loader),
        "accuracy": total_num_correct / len(data_loader.dataset),
    }

    return metrics


def train_one_epoch(args, model, data_loader, optimizer, device, epoch, use_lr_scheduler=False):
    ep_loss = AverageMeter("loss")
    ep_log_likelihood_loss = AverageMeter("log_likelihood_loss")
    ep_reconstruction_loss = AverageMeter("reconstruction_loss")
    ep_cosine_sim = AverageMeter("cosine_similarity")
    ep_norm_loss = AverageMeter("norm_loss")
    ep_train_acc = AverageMeter("accuracy")
    ep_traj_reg_loss = AverageMeter("traj_reg_loss")

    logsigmoid = nn.LogSigmoid()
    if use_lr_scheduler:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(data_loader))

    with tqdm(total=len(data_loader), unit="batch", position=0, leave=True) as pbar:
        pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")
        for train_data in data_loader:
            # load it to the active device
            # train_data = {key: value.to(device) for key, value in train_data.items()}

            # reset the gradients back to zero
            optimizer.zero_grad()

            # compute reconstructions
            output = model(train_data, train=True)
            (
                encoded_traj_a,
                encoded_traj_b,
                encoded_lang,
                decoded_traj_a,
                decoded_traj_b,
            ) = output

            # compute training reconstruction loss
            # MSELoss already takes the mean over the batch.
            reconstruction_loss = torch.tensor(0.0).to(device)
            if decoded_traj_a is not None and decoded_traj_b is not None:
                reconstruction_loss = F.mse_loss(decoded_traj_a, torch.mean(train_data["traj_a"], dim=-2))
                reconstruction_loss += F.mse_loss(decoded_traj_b, torch.mean(train_data["traj_b"], dim=-2))

            # F.cosine_similarity only reduces along the feature dimension, so we take the mean over the batch later.
            cos_sim = F.cosine_similarity(encoded_traj_b - encoded_traj_a, encoded_lang)
            cos_sim = torch.mean(cos_sim)  # Take the mean over the batch.
            distance_loss = 1 - cos_sim  # Then convert the value to a loss.

            dot_prod = torch.einsum("ij,ij->i", encoded_traj_b - encoded_traj_a, encoded_lang)
            log_likelihood = logsigmoid(dot_prod).mean()  # Take the mean over the batch.
            log_likelihood_loss = -1 * log_likelihood  # Then convert the value to a loss.

            # Norm loss, to make sure the encoded lang vectors are unit vectors
            norm_loss = F.mse_loss(
                torch.norm(encoded_lang, dim=-1),
                torch.ones(encoded_lang.shape[0]).to(device),
            )

            # Add L2 loss to the trajectory embeddings
            traj_reg_loss = 0.0
            traj_reg_loss += (torch.norm(encoded_traj_a, dim=-1) - args.traj_reg_margin).clamp(min=0.0).mean()
            traj_reg_loss += (torch.norm(encoded_traj_b, dim=-1) - args.traj_reg_margin).clamp(min=0.0).mean()

            # By now, train_loss is a scalar.
            # train_loss = reconstruction_loss + distance_loss
            train_loss = reconstruction_loss + log_likelihood_loss + args.traj_reg_coeff * traj_reg_loss

            if args.add_norm_loss:
                train_loss += norm_loss

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()
            if use_lr_scheduler:
                lr_scheduler.step()

            # add the mini-batch training loss to epoch loss
            ep_loss.update(train_loss.item(), train_data["traj_a"].shape[0])
            ep_reconstruction_loss.update(reconstruction_loss.item(), train_data["traj_a"].shape[0])
            ep_log_likelihood_loss.update(log_likelihood_loss.item(), train_data["traj_a"].shape[0])
            ep_cosine_sim.update(cos_sim.item(), train_data["traj_a"].shape[0])
            ep_norm_loss.update(norm_loss.item(), train_data["traj_a"].shape[0])
            ep_traj_reg_loss.update(traj_reg_loss.item(), train_data["traj_a"].shape[0])
            ep_train_acc.update(
                np.sum(dot_prod.detach().cpu().numpy() > 0) / len(dot_prod),
                train_data["traj_a"].shape[0],
            )

            tqdm_postfix = {
                "log_likelihood_loss": ep_log_likelihood_loss.avg,
                "reconstruction_loss": ep_reconstruction_loss.avg,
                "cosine_similarity": ep_cosine_sim.avg,
                "norm_loss": ep_norm_loss.avg,
                "traj_reg_loss": ep_traj_reg_loss.avg,
                "accuracy": ep_train_acc.avg,
            }

            pbar.set_postfix(tqdm_postfix)
            pbar.update()

    metrics = {
        "loss": ep_loss.avg,
        "reconstruction_loss": ep_reconstruction_loss.avg,
        "norm_loss": ep_norm_loss.avg,
        "cosine_similarity": ep_cosine_sim.avg,
        "log_likelihood": ep_log_likelihood_loss.avg,
        "accuracy": ep_train_acc.avg,
    }
    return metrics


def train(logger, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # Load model to the specified device, either gpu or cpu
    logger.info("Initializing model and loading to device...")
    if "t5" in args.lang_model_name:
        lang_encoder = T5EncoderModel.from_pretrained(args.lang_model_name)
    else:
        lang_encoder = AutoModel.from_pretrained(HF_LANG_MODEL_NAME[args.lang_model_name])

    tokenizer = AutoTokenizer.from_pretrained(HF_LANG_MODEL_NAME[args.lang_model_name])
    feature_dim = LANG_OUTPUT_DIM[args.lang_model_name]

    STATE_OBS_DIM, ACTION_DIM, PROPRIO_STATE_DIM, OBJECT_STATE_DIM = env_dims.get(args.env, None)

    model = NLTrajAutoencoder(
        STATE_OBS_DIM,
        ACTION_DIM,
        PROPRIO_STATE_DIM,
        OBJECT_STATE_DIM,
        encoder_hidden_dim=args.encoder_hidden_dim,
        feature_dim=feature_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        lang_encoder=lang_encoder,
        lang_embed_dim=LANG_OUTPUT_DIM[args.lang_model_name],
        traj_encoder=args.traj_encoder,
        use_stack_img_obs=args.use_stack_img_obs,
        n_frames=args.n_frames,
        use_visual_features=args.use_visual_features,
        visual_feature_dim=(
            args.visual_feature_dim * args.n_frames if args.use_stack_img_obs else args.visual_feature_dim
        ),
        seq_len=int(args.seq_len * args.resample_factor),
    )

    if not args.finetune_lang_model:
        # Freeze the languge model in the first training stage
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

    logger.info("Loading dataset...")

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    # save the args
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f)

    train_files_dict = load_data(args, split="train")
    val_files_dict = load_data(args, split="val")
    test_files_dict = load_data(args, split="test")

    train_img_obs_file = train_files_dict.get("traj_img_obs", None)
    val_img_obs_file = val_files_dict.get("traj_img_obs", None)
    test_img_obs_file = test_files_dict.get("traj_img_obs", None)
    train_action_file = train_files_dict.get("actions", None)
    val_action_file = val_files_dict.get("actions", None)
    test_action_file = test_files_dict.get("actions", None)

    # apply downsample in train and val, but not in test
    train_dataset = NLTrajComparisonDataset(
        train_files_dict["nlcomp_index"],
        train_files_dict["traj_a_index"],
        train_files_dict["traj_b_index"],
        seq_len=args.seq_len,
        tokenizer=tokenizer,
        unique_nlcomp_file=train_files_dict["unique_nlcomp"],
        traj_file=train_files_dict["trajs"],
        use_img_obs=args.use_img_obs,
        img_obs_file=train_img_obs_file,
        action_file=train_action_file,
        use_visual_features=args.use_visual_features,
        resample=args.resample,
        resample_factor=args.resample_factor,
        device=device,
    )

    val_dataset = NLTrajComparisonDataset(
        val_files_dict["nlcomp_index"],
        val_files_dict["traj_a_index"],
        val_files_dict["traj_b_index"],
        seq_len=args.seq_len,
        tokenizer=tokenizer,
        unique_nlcomp_file=val_files_dict["unique_nlcomp"],
        traj_file=val_files_dict["trajs"],
        use_img_obs=args.use_img_obs,
        img_obs_file=val_img_obs_file,
        action_file=val_action_file,
        use_visual_features=args.use_visual_features,
        resample=args.resample,
        resample_factor=args.resample_factor,
        device=device,
    )

    test_dataset = NLTrajComparisonDataset(
        test_files_dict["nlcomp_index"],
        test_files_dict["traj_a_index"],
        test_files_dict["traj_b_index"],
        seq_len=args.seq_len,
        tokenizer=tokenizer,
        unique_nlcomp_file=test_files_dict["unique_nlcomp"],
        traj_file=test_files_dict["trajs"],
        use_img_obs=args.use_img_obs,
        img_obs_file=test_img_obs_file,
        action_file=test_action_file,
        use_visual_features=args.use_visual_features,
        resample=args.resample,
        resample_factor=args.resample_factor,
        device=device,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,

    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    if args.initial_loss_check:
        logger.info("Initial loss sanity check...")

        val_metrics = evaluate(model, val_loader, device)
        logger.info("initial val loss: {:.4f}, accuracy: {:.4f}".format(val_metrics["loss"], val_metrics["accuracy"]))

    logger.info("Beginning training...")
    train_losses = []
    val_losses = []
    val_reconstruction_losses = []
    val_cosine_similarities = []
    val_log_likelihoods = []
    accuracies = []
    best_val_cos_sim = -1

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            args,
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            use_lr_scheduler=args.use_lr_scheduler,
        )

        # Evaluation
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_reconstruction_loss = val_metrics["reconstruction_loss"]
        val_norm_loss = val_metrics["norm_loss"]
        val_cosine_similarity = val_metrics["cosine_similarity"]
        val_log_likelihood = val_metrics["log_likelihood"]
        val_acc = val_metrics["accuracy"]

        # display the epoch training and evaluation metrics
        logger.info(
            "epoch : {}/{}, [train] loss = {:.4f}, [train] acc= {:.4f}".format(
                epoch + 1,
                args.epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
            )
        )

        logger.info(
            "[val] loss = {:.4f}, [val] reconstruction_loss = {:.4f}, "
            "[val] norm loss = {:.4f}, [val] cos_similarity = {:.4f}, [val] log_likelihood = {:.4f}, [val] accuracy = {:.4f}".format(
                val_metrics["loss"],
                val_metrics["reconstruction_loss"],
                val_metrics["norm_loss"],
                val_metrics["cosine_similarity"],
                val_metrics["log_likelihood"],
                val_metrics["accuracy"],
            )
        )
        # Don't forget to save the model (as we go)!
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, "model_state_dict_{}.pth".format(epoch)),
        )
        if val_cosine_similarity > best_val_cos_sim:
            best_val_cos_sim = val_cosine_similarity
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, "best_model_state_dict.pth"),
            )

        train_losses.append(train_metrics["loss"])
        val_losses.append(val_loss)
        val_reconstruction_losses.append(val_reconstruction_loss)
        val_cosine_similarities.append(val_cosine_similarity)
        val_log_likelihoods.append(val_log_likelihood)
        accuracies.append(val_acc)
        np.save(os.path.join(args.save_dir, "train_losses.npy"), np.asarray(train_losses))
        np.save(os.path.join(args.save_dir, "val_losses.npy"), np.asarray(val_losses))
        np.save(
            os.path.join(args.save_dir, "val_reconstruction_losses.npy"),
            np.asarray(val_reconstruction_losses),
        )
        np.save(
            os.path.join(args.save_dir, "val_cosine_similarities.npy"),
            np.asarray(val_cosine_similarities),
        )
        np.save(
            os.path.join(args.save_dir, "val_log_likelihoods.npy"),
            np.asarray(val_log_likelihoods),
        )
        np.save(os.path.join(args.save_dir, "accuracies.npy"), np.asarray(accuracies))

    # Test the model on the test set.
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model_state_dict.pth")))
    model.eval()

    test_metrics = evaluate(model, test_loader, device)

    test_loss = test_metrics["loss"]
    test_reconstruction_loss = test_metrics["reconstruction_loss"]
    test_norm_loss = test_metrics["norm_loss"]
    test_cosine_similarity = test_metrics["cosine_similarity"]
    test_log_likelihood = test_metrics["log_likelihood"]
    test_acc = test_metrics["accuracy"]

    # display the epoch training loss
    logger.info(
        "[test] loss = {:.4f}, [test] reconstruction_loss = {:.4f}, [test] norm loss = {:.4f}, "
        "[test] cos_similarity = {:.4f}, [test] log_likelihood = {:.4f}, [test] accuracy = {:.4f}".format(
            test_loss,
            test_reconstruction_loss,
            test_norm_loss,
            test_cosine_similarity,
            test_log_likelihood,
            test_acc,
        )
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--exp-name", type=str, default="feature_learning",
        help="the name of experiment",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="robosuite",
        choices=["robosuite", "widowx", "metaworld", "franka-kitchen"],
        help="which environment to use",
    )
    parser.add_argument("--seed", type=int, default=0, help="")
    parser.add_argument("--data-dir", type=str, default="data/", help="")
    parser.add_argument("--epochs", type=int, default=2, help="")
    parser.add_argument("--batch-size", type=int, default=1024, help="")
    parser.add_argument("--lr", type=float, default=1e-3, help="")
    parser.add_argument("--finetune-lr", type=float, default=1e-4, help="")
    parser.add_argument("--weight-decay", type=float, default=0, help="")
    parser.add_argument("--use_lr_scheduler", action="store_true", help="use lr scheduler")
    parser.add_argument("--encoder-hidden-dim", type=int, default=128, help="")
    parser.add_argument("--decoder-hidden-dim", type=int, default=128, help="")
    parser.add_argument("--initial-loss-check", action="store_true",help="whether to check initial loss")
    parser.add_argument("--finetune-lang-model", action="store_true", help="whether to finetune BERT")
    parser.add_argument("--save-dir", type=str, default="feature_learning/", help="where to save the model")
    parser.add_argument(
        "--lang-model-name",
        type=str,
        default="t5-base",
        choices=["t5-small", "t5-base", "bert-base", "bert-mini", "bert-tiny", ],
        help="which language model to use",
    )
    parser.add_argument(
        "--traj-encoder",
        default="mlp",
        choices=["mlp", "cnn"],
        help="which trajectory encoder to use",
    )
    parser.add_argument(
        "--add-norm-loss",
        action="store_true",
        help="whether to add norm loss to the total loss",
    )

    parser.add_argument("--seq-len", type=int, default=500, help="sequence length for the trajectory")
    parser.add_argument("--use-img-obs", action="store_true", help="whether to use image observations")
    parser.add_argument("--use-stack-img-obs", action="store_true", help="whether to use stacked image observations",)
    parser.add_argument("--n-frames", type=int, default=3, help="number of frames to stack")
    parser.add_argument("--use-visual-features", action="store_true", help="whether to use visual features",)
    parser.add_argument(
        "--feature-extractor",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnetb3", "s3d"],
        help="feature extractor for image observations",
    )
    parser.add_argument(
        "--visual-feature-dim", type=int, default=256,
        help="dimension of visual features",
    )
    parser.add_argument(
        "--resample", action="store_true", help="whether to resample the image observations",
    )
    parser.add_argument("--resample-factor", type=float, default=1.0, help="resample ratio for the image observations")
    parser.add_argument(
        "--traj-reg-coeff", type=float, default=1e-3, 
        help="coefficient for the trajectory regularization loss",
    )
    parser.add_argument(
        "--traj-reg-margin", type=float, default=1.0, 
        help="margin for the trajectory regularization loss",
    )

    args = parser.parse_args()

    # Create exp directory and logger
    exp_dir = os.path.join(
        "exp",
        f"{timeStamped(args.exp_name)}_lr_{args.lr}_schedule_{args.use_lr_scheduler}",
    )
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)
    args.save_dir = exp_dir


    # Two-stage training
    ## stage 1: train the trajectory encoder with BERT frozen
    logger.info("\n------------------ Freeze Language Model ------------------")
    train(logger, args)

    ## stage 2: co-finetune BERT and the trajectory encoder
    logger.info("\n------------------ Co-finetune Language Model ------------------")
    args.finetune_lang_model = True
    args.lr = args.finetune_lr
    train(logger, args)
