# ==============================
# train_pd.py - Pre-training Normalize Network
# ==============================

import os
import math
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from NAFNet_arch import NAFNet_arch
from utils.dataset_PNNet import ISBI_Loader

# -------------------- Reproducibility --------------------
seed_value = 3407
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True

# -------------------- Training Function --------------------
def train_net(net, device, train_data_path, csv_name, criterion, epochs=300, batch_size=8, lr=1e-3, model_path='model/PDNet.pth'):
    """
    Train the normalization network using L1 loss.
    """
    print(f"Training data path: {train_data_path}")

    # -------------------- Load Dataset --------------------
    if isinstance(train_data_path, list):
        if len(train_data_path) == 2:
            simul_dataset = ISBI_Loader(train_data_path[0])
            real_dataset = ISBI_Loader(train_data_path[1])
            all_dataset = torch.utils.data.dataset.ConcatDataset([simul_dataset, real_dataset])
            print(f"Dataset length (simu+real): {len(all_dataset)}")
        else:
            raise ValueError("train_data_path list length must be 2.")
    else:
        all_dataset = ISBI_Loader(train_data_path)

    total_size = len(all_dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    print(f"Total samples: {total_size}, Train: {train_size}, Val: {val_size}")

    train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4
    )

    # -------------------- Optimizer & Scheduler --------------------
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    # Initialize CSV log file
    if not os.path.exists(csv_name):
        pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss']).to_csv(csv_name, index=False)

    # -------------------- Training Loop --------------------
    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}] - Time: {datetime.datetime.now()}")

        # ======== Training ========
        net.train()
        train_losses = []

        for image, label in train_loader:
            image = image.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            pred = net(image)
            loss = F.l1_loss(pred, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()
        train_loss_mean = np.mean(train_losses)
        print(f"  Train Loss: {train_loss_mean:.6f}")

        # ======== Validation ========
        net.eval()
        val_losses = []

        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)
                pred = net(image)
                val_loss = criterion(pred, label)
                val_losses.append(val_loss.item())

        val_loss_mean = np.mean(val_losses)
        print(f"  Val   Loss: {val_loss_mean:.6f}")

        # Save best model
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            torch.save(net.state_dict(), model_path)

        # Log to CSV
        csv_row = pd.DataFrame([[epoch, train_loss_mean, val_loss_mean]])
        csv_row.to_csv(csv_name, mode='a', header=False, index=False)

# -------------------- Main --------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model definition (NAFNet)
    img_channel = 2
    net = NAFNet_arch(
        in_channel=img_channel,
        out_channel=img_channel,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1]
    ).to(device)

    # Training entry
    train_data_path = 'autodl-fs/simu_train'
    train_net(
        net,
        device,
        train_data_path,
        criterion=F.l1_loss,
        csv_name="excel/PDNet.csv",
        batch_size=8,
        lr=1e-3,  # Learning rate can be adjusted depending on training needs; goal: smooth convergence
        model_path='model/PDNet.pth'
    )
