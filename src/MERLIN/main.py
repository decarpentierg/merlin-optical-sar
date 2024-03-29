import argparse
from glob import glob
import os
from pathlib import Path
import time
import numpy as np
import torch

from dataset import Dataset, ValDataset
from model import AE
from utils import save_checkpoint, save_model, init_weights
from generate_dataset import generate_patches

ROOT_DIR = Path(__file__).parent.parent.parent.absolute().as_posix()  # get path to project root
resultsdir = ROOT_DIR + "/results"
datasetdir = ROOT_DIR + "/data"

torch.manual_seed(2)

# ---------------
# Parse arguments
# ---------------

parser = argparse.ArgumentParser(description="")

# Batches and patches
parser.add_argument(
    "--batch_size", dest="batch_size", type=int, default=12, help="# images in batch"
)
parser.add_argument(
    "--val_batch_size",
    dest="val_batch_size",
    type=int,
    default=1,
    help="# images in batch",
)
parser.add_argument(
    "--patch_size", dest="patch_size", type=int, default=256, help="# size of a patch"
)
parser.add_argument(
    "--stride_size",
    dest="stride_size",
    type=int,
    default=32,
    help="# size of the stride",
)
parser.add_argument(
    "--n_data_augmentation",
    dest="n_data_augmentation",
    type=int,
    default=1,
    help="# data aug techniques",
)

# Optimizer
parser.add_argument("--epoch", dest="epoch", type=int, default=30, help="# of epoch")
parser.add_argument(
    "--lr", dest="lr", type=float, default=0.001, help="initial learning rate for adam"
)
parser.add_argument(
    "--weight_decay",
    dest="weight_decay",
    type=float,
    default=0.001,
    help="weight decay for adam",
)

# GPU or CPU
parser.add_argument(
    "--use_gpu",
    dest="use_gpu",
    type=int,
    default=1,
    help="gpu flag, 1 for GPU and 0 for CPU",
)
parser.add_argument(
    "--device",
    dest="device",
    default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    help="gpu or cpu",
)

# Test or train
parser.add_argument("--phase", dest="phase", default="train", help="train or test")

# Directories
parser.add_argument(
    "--checkpoint_dir",
    dest="ckpt_dir",
    default=resultsdir + "/saved_model",
    help="models are saved here",
)
parser.add_argument(
    "--sample_dir",
    dest="sample_dir",
    default=resultsdir + "/sample",
    help="sample are saved here",
)
parser.add_argument(
    "--test_dir",
    dest="test_dir",
    default=resultsdir + "/test",
    help="test sample are saved here",
)

# Dataset and method
parser.add_argument(
    "--dataset", dest="dataset", default="Saclay", help="dataset to use : Saclay or Sendai"
)
parser.add_argument(
    "--method",
    dest="method",
    default="SAR",
    help="method to use : SAR, SAR+OPT, SAR+SAR or SAR+OPT+SAR",
)
parser.add_argument(
    "--index",
    dest="index",
    default=0,
    type=int,
    help="Index of SAR image to denoise. Either 0, 1 or 2."
)

args = parser.parse_args()
if args.method not in ["SAR", "SAR+OPT", "SAR+SAR", "SAR+OPT+SAR"]:
    raise ValueError("Method must be either SAR, SAR+OPT, SAR+SAR or SAR+OPT+SAR")

TRAINING_SET = f"{datasetdir}/{args.dataset}_{args.method}/train/spotlight/"
TEST_SET = f"{datasetdir}/{args.dataset}_{args.method}/validation/spotlight/"
EVAL_SET = f"{datasetdir}/{args.dataset}_{args.method}/validation/spotlight/"

torch.autograd.set_detect_anomaly(True)

# ----------------
# Define functions
# ----------------

def fit(
    model,
    train_loader,
    val_loader,
    epochs,
    lr_list,
    gn_list,
    eval_files,
    eval_set,
    checkpoint_folder,
):
    """Fit the model according to the given evaluation data and parameters.

    Parameters
    ----------
    model : model as defined in main
    train_loader : Pytorch's DataLoader of training data
    val_loader : Pytorch's DataLoader of validation data
    lr_list : list of learning rates
    eval_files : .npy files used for evaluation in training
    eval_set : directory of dataset used for evaluation in training

    Returns
    ----------
    self : object
      Fitted estimator.
    """

    train_losses = []
    val_losses = []
    history = {}

    # Initialize weights
    ckpt_files = glob(checkpoint_folder + "/checkpoint_*")
    if len(ckpt_files) == 0:
        epoch_num = 0
        model.apply(init_weights)
        loss = 0.0
        print("[*] Pre-trained model not found! Starting training from scratch.")
    else:
        max_file = max(ckpt_files, key=os.path.getctime)
        checkpoint = torch.load(max_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.train()
        epoch_num = checkpoint["epoch_num"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[epoch_num - 1])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss = checkpoint["loss"]
        print("[*] Model restored! Resume training from latest checkpoint at " + max_file)

    # Compute validation loss
    with torch.no_grad():
        image_num = 0
        for batch in val_loader:
            val_loss = model.validation_step(
                batch, image_num, epoch_num, eval_files, eval_set, args.sample_dir
            )
            image_num = image_num + 1

    # Start training
    start_time = time.time()
    for epoch in range(epoch_num, epochs):
        epoch_num = epoch_num + 1
        print("\nEpoch", epoch_num)
        print("\nLearning rate", lr_list[epoch])
        print("\nGradient norm", gn_list[epoch])
        print("*****************\n")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[epoch])

        # Run one epoch
        for i, batch in enumerate(train_loader):
            running_loss = 0.0

            optimizer.zero_grad()
            loss = model.training_step(batch, i)
            train_losses.append(loss)

            loss.backward()

            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item()**2
            total_norm = total_norm**0.5
            print(total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gn_list[epoch])

            optimizer.step()

            running_loss += loss.item()     # extract the loss value
            print(
                "[%d, %5d] time: %4.4f, loss: %.3f"
                % (epoch_num, i + 1, time.time() - start_time, loss)
            )
            # zero the loss
            running_loss = 0.0

        # save current checkpoint
        save_checkpoint(model, checkpoint_folder, epoch_num, optimizer, loss)

        # Compute validation loss
        with torch.no_grad():
            image_num = 0
            for batch in val_loader:
                val_loss = model.validation_step(
                    batch, image_num, epoch_num, eval_files, eval_set, args.sample_dir
                )
                image_num = image_num + 1

    history["train_loss"] = train_losses
    history["validation_loss"] = val_losses

    return history


def denoiser_train(model, lr_list, gn_list):
    """Runs the denoiser algorithm for the training and evaluation dataset

    Parameters
    ----------
    model : model as defined in main
    lr_list : list of learning rates

    Returns
    -------
    history : list of both training and validation loss

    """
    # Prepare train DataLoader
    train_data = generate_patches(
        TRAINING_SET,
        args.index,
        args.patch_size,
        0,
        args.stride_size,
        args.batch_size,
        args.n_data_augmentation,
        args.method
    )

    train_dataset = Dataset(train_data, args.method)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Prepare Validation DataLoader
    eval_dataset = ValDataset(EVAL_SET, args.method, args.index)  # range [0; 1]
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=True
    )
    eval_files = glob(EVAL_SET + f"val_data_{args.index}*.npy")

    # Train the model
    history = fit(
        model,
        train_loader,
        eval_loader,
        args.epoch,
        lr_list,
        gn_list,
        eval_files,
        EVAL_SET,
        args.ckpt_dir,
    )

    # Save the model
    save_model(model, args.ckpt_dir)
    print("\n model saved at :", args.ckpt_dir)
    return history


def denoiser_test(model):
    """Runs the test denoiser algorithm

    Parameters
    ----------
    denoiser : model as defined in main

    Returns
    -------

    """
    # Prepare Validation DataLoader
    test_dataset = ValDataset(TEST_SET, args.method, args.index)  # range [0; 1]
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=True
    )
    test_files = glob(TEST_SET + "*.npy")

    val_losses = []
    ckpt_files = glob(args.ckpt_dir + "/checkpoint_*")
    if len(ckpt_files) == 0:
        print("[*] Pre-trained model not found!")
        return None

    else:
        max_file = max(ckpt_files, key=os.path.getctime)
        checkpoint = torch.load(max_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        # model.train()

        print("[*] Model restored! Start testing...")

        with torch.no_grad():
            image_num = 0
            for batch in test_loader:
                print(image_num)
                model.test_step(batch, image_num, test_files, TEST_SET, args.test_dir)
                image_num = image_num + 1


def main():
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    # learning rate list
    lr = args.lr * np.ones([args.epoch])
    lr[4:20] = lr[0] / 10
    lr[20:] = lr[0] / 100
    # gradient norm list
    gn = 1.0 * np.ones([args.epoch])

    model = AE(args.batch_size, args.val_batch_size, args.device, args.method)
    model.to(args.device)

    if args.phase == "train":
        denoiser_train(model, lr, gn)
    elif args.phase == "test":
        denoiser_test(model)
    else:
        print("[!] Unknown phase")
        exit(0)

# --------
# Run main
# --------

if __name__ == "__main__":
    main()
