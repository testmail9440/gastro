#!/usr/bin/env python
# coding: utf-8

"""
GastroVision FINAL FIX
- Fixed SWA update_bn device error
- Restored detailed epoch metrics (Precision/Recall/F1/MCC)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms, datasets
from torch import optim
from torch.utils import data
from torch.optim.swa_utils import AveragedModel, SWALR

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics as mtc
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import time
import os
import itertools
import collections
import logging
import warnings

# --- Optional Dependency for FLOPs ---
try:
    from thop import profile

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='Train GastroVision Final Fix')
    parser.add_argument('-e', '--epochs', type=int, default=60, help='Total Epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=320, help='Input image resolution')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--mixup_alpha', type=float, default=0.1, help='Mixup alpha value')
    parser.add_argument('--backbone1', default='b0')
    parser.add_argument('--backbone2', default='b4')
    parser.add_argument('--fusion',
                        default='sum_cbam',
                        choices=['concat','sum','sum_cbam','late'])
    parser.add_argument('--data_dir', type=str, default="/home/srinivas/Documents/debesh_gastrovision/dataset",
                        help='Root data directory')
    return parser.parse_args()


# ==========================================
# 0. Metrics Helper
# ==========================================
def calculate_metrics(y_true, y_pred, loss_val):
    """Calculates comprehensive metrics for detailed logging"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}
    metrics['loss'] = loss_val

    # Micro Average
    metrics['micro_precision'] = mtc.precision_score(y_true, y_pred, average="micro", zero_division=0)
    metrics['micro_recall'] = mtc.recall_score(y_true, y_pred, average="micro", zero_division=0)
    metrics['micro_f1'] = mtc.f1_score(y_true, y_pred, average="micro", zero_division=0)

    # Macro Average
    metrics['macro_precision'] = mtc.precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics['macro_recall'] = mtc.recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics['macro_f1'] = mtc.f1_score(y_true, y_pred, average="macro", zero_division=0)

    # MCC
    metrics['mcc'] = mtc.matthews_corrcoef(y_true, y_pred)

    return metrics


def print_epoch_metrics(phase, metrics):
    """Prints metrics in the requested format"""
    print(f"{phase}...")
    output_str = (f"loss:{metrics['loss']:.4f},"
                  f"micro_precision:{metrics['micro_precision']:.4f},"
                  f"micro_recall:{metrics['micro_recall']:.4f},"
                  f"micro_f1:{metrics['micro_f1']:.4f},"
                  f"macro_precision:{metrics['macro_precision']:.4f},"
                  f"macro_recall:{metrics['macro_recall']:.4f},"
                  f"macro_f1:{metrics['macro_f1']:.4f},"
                  f"mcc:{metrics['mcc']:.4f}")
    print(output_str)


# ==========================================
# 1. MixUp & CBAM Utilities
# ==========================================
def mixup_data(x, y, alpha=0.1):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

from torchvision.models import (
    efficientnet_b0, efficientnet_b1,
    efficientnet_b2, efficientnet_b3,
    efficientnet_b4,
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights
)

def load_efficientnet(name):

    if name == "b0":
        net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    elif name == "b1":
        net = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    elif name == "b2":
        net = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    elif name == "b3":
        net = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    elif name == "b4":
        net = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
    else:
        raise ValueError("Unsupported backbone")

    return net.features



# ==========================================
# 2. Hybrid Model
# ==========================================

# Channel sizes for each EfficientNet variant at each of the 5 stages
CH_SIZES = {
    'b0': [16, 24, 40, 112, 320],
    'b1': [16, 24, 40, 112, 320],
    'b2': [16, 24, 48, 120, 352],
    'b3': [24, 32, 48, 136, 384],
    'b4': [24, 32, 56, 160, 448],
}

class DualEncoderNet(nn.Module):

    def __init__(self,
                 backbone1="b0",
                 backbone2="b4",
                 fusion="sum_cbam",
                 n_classes=22):

        super().__init__()

        self.fusion = fusion

        c1 = CH_SIZES[backbone1]  # channels for backbone 1
        c2 = CH_SIZES[backbone2]  # channels for backbone 2

        net1 = load_efficientnet(backbone1)
        net2 = load_efficientnet(backbone2)

        # SAME STAGE SPLIT
        self.B1_1 = net1[0:2]
        self.B1_2 = net1[2:3]
        self.B1_3 = net1[3:4]
        self.B1_4 = net1[4:6]
        self.B1_5 = net1[6:8]

        self.B2_1 = net2[0:2]
        self.B2_2 = net2[2:3]
        self.B2_3 = net2[3:4]
        self.B2_4 = net2[4:6]
        self.B2_5 = net2[6:8]

        # Cross-branch projection layers (backbone1 -> backbone2 dim and back)
        self.conv1x1_1 = nn.Conv2d(c1[0], c2[0], kernel_size=1)
        self.bn1_1 = nn.BatchNorm2d(c2[0])
        self.conv1x1_1_1 = nn.Conv2d(c2[0], c1[0], kernel_size=1)
        self.bn1_1_1 = nn.BatchNorm2d(c1[0])

        self.conv1x1_2 = nn.Conv2d(c1[1], c2[1], kernel_size=1)
        self.bn2 = nn.BatchNorm2d(c2[1])
        self.conv1x1_2_1 = nn.Conv2d(c2[1], c1[1], kernel_size=1)
        self.bn2_1 = nn.BatchNorm2d(c1[1])

        self.conv1x1_3 = nn.Conv2d(c1[2], c2[2], kernel_size=1)
        self.bn3 = nn.BatchNorm2d(c2[2])
        self.conv1x1_3_1 = nn.Conv2d(c2[2], c1[2], kernel_size=1)
        self.bn3_1 = nn.BatchNorm2d(c1[2])

        self.conv1x1_4 = nn.Conv2d(c1[3], c2[3], kernel_size=1)
        self.bn4 = nn.BatchNorm2d(c2[3])
        self.conv1x1_4_1 = nn.Conv2d(c2[3], c1[3], kernel_size=1)
        self.bn4_1 = nn.BatchNorm2d(c1[3])

        self.conv1x1_5 = nn.Conv2d(c1[4], c2[4], kernel_size=1)
        self.bn5 = nn.BatchNorm2d(c2[4])

        self.cbam1 = CBAM(c1[0])
        self.cbam2 = CBAM(c1[1])
        self.cbam3 = CBAM(c1[2])
        self.cbam4 = CBAM(c1[3])
        self.cbam5 = CBAM(c2[4])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(c2[4], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, xb):
        # Stage 1
        s1_1 = self.B1_1(xb)
        s2_1 = self.B2_1(xb)
        s1_c1 = self.bn1_1(self.conv1x1_1(s1_1))
        C1 = s1_c1 + s2_1
        C1_1 = self.bn1_1_1(self.conv1x1_1_1(C1))
        C1_1 = self.cbam1(C1_1)

        # Stage 2
        s1_2 = self.B1_2(C1_1 + s1_1)
        s2_2 = self.B2_2(C1 + s2_1)
        s1_c2 = self.bn2(self.conv1x1_2(s1_2))
        C2 = s1_c2 + s2_2
        C2_1 = self.bn2_1(self.conv1x1_2_1(C2))
        C2_1 = self.cbam2(C2_1)

        # Stage 3
        s1_3 = self.B1_3(C2_1 + s1_2)
        s2_3 = self.B2_3(C2 + s2_2)
        s1_c3 = self.bn3(self.conv1x1_3(s1_3))
        C3 = s1_c3 + s2_3
        C3_1 = self.bn3_1(self.conv1x1_3_1(C3))
        C3_1 = self.cbam3(C3_1)

        # Stage 4
        s1_4 = self.B1_4(C3_1 + s1_3)
        s2_4 = self.B2_4(C3 + s2_3)
        s1_c4 = self.bn4(self.conv1x1_4(s1_4))
        C4 = s1_c4 + s2_4
        C4_1 = self.bn4_1(self.conv1x1_4_1(C4))
        C4_1 = self.cbam4(C4_1)

        # Stage 5
        s1_5 = self.B1_5(C4_1 + s1_4)
        s2_5 = self.B2_5(C4 + s2_4)
        s1_c5 = self.bn5(self.conv1x1_5(s1_5))
        C5 = s1_c5 + s2_5
        C5 = self.cbam5(C5)

        Out = self.avg_pool(C5)
        Out = Out.view(Out.size(0), -1)
        out = self.classifier(Out)
        return out


# ==========================================
# 3. Helpers
# ==========================================
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()


def training_curve(epochs, lossesT, lossesV):
    plt.figure()
    plt.plot(epochs, lossesT, 'c-', label='Train Loss')
    plt.plot(epochs, lossesV, 'm-', label='Val Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('train_val_curve.png')
    plt.close()


class ModelBenchmark:
    def __init__(self, model, input_size, device='cpu'):
        self.model = model
        self.input_size = input_size
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def get_model_stats(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        flops = 0.0
        if THOP_AVAILABLE:
            input_tensor = torch.randn(self.input_size).to(self.device)
            try:
                macs, _ = profile(self.model, inputs=(input_tensor,), verbose=False)
                flops = macs / 1e9 * 2
            except Exception as e:
                pass
        return total_params, trainable_params, flops

    def measure_inference_speed(self, num_warmup=5, num_steps=50):
        input_tensor = torch.randn(self.input_size).to(self.device)
        with torch.no_grad():
            for _ in range(num_warmup): _ = self.model(input_tensor)
        start_time = time.time()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                for _ in range(num_steps): _ = self.model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0
        else:
            with torch.no_grad():
                for _ in range(num_steps): _ = self.model(input_tensor)
            elapsed = time.time() - start_time
        return num_steps / elapsed, (elapsed / num_steps) * 1000


def print_benchmark_report(model, img_size):
    print("\n" + "=" * 50)
    print("      COMPUTATIONAL PERFORMANCE REPORT      ")
    print("=" * 50)
    input_dims = (1, 3, img_size, img_size)
    bench_cpu = ModelBenchmark(model, input_dims, device=torch.device('cpu'))
    total_p, train_p, gflops = bench_cpu.get_model_stats()
    print(f"Model Complexity (Input: {img_size}x{img_size}):")
    print(f" - Total Parameters:    {total_p / 1e6:.2f} Million")
    print(f" - Trainable Params:    {train_p / 1e6:.2f} Million")
    print(f" - GFLOPs (Est.):       {gflops:.2f} G")
    print("-" * 50)
    fps_cpu, lat_cpu = bench_cpu.measure_inference_speed()
    print(f"Inference Speed (CPU):  {fps_cpu:.2f} FPS | {lat_cpu:.2f} ms")
    if torch.cuda.is_available():
        bench_gpu = ModelBenchmark(model, input_dims, device=torch.device('cuda'))
        fps_gpu, lat_gpu = bench_gpu.measure_inference_speed()
        print(f"Inference Speed (GPU):  {fps_gpu:.2f} FPS | {lat_gpu:.2f} ms")
    print("=" * 50 + "\n")


# ==========================================
# 4. Trainer with SWA and Detailed Logs
# ==========================================
class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_dir = os.path.join(args.data_dir, "train")
        self.val_dir = os.path.join(args.data_dir, "valid")
        self.test_dir = os.path.join(args.data_dir, "test")
        self.checkpoint_dir = f'./checkpoints_{args.backbone1}_{args.backbone2}_{args.fusion}'

        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)

        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        self.transforms = {
            'train': transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomRotation(degrees=45),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ]),
            'val': transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ]),
            'test': transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
        }

        print("[INFO] Initializing Datasets...")
        self.train_set = datasets.ImageFolder(self.train_dir, transform=self.transforms['train'])
        self.val_set = datasets.ImageFolder(self.val_dir, transform=self.transforms['val'])
        self.test_set = datasets.ImageFolder(self.test_dir, transform=self.transforms['test'])

        self.train_loader = data.DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        self.val_loader = data.DataLoader(self.val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
        self.test_loader = data.DataLoader(self.test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

        self.n_classes = len(self.train_set.classes)
        print(f"[INFO] Classes: {self.n_classes} | Image Size: {args.img_size}x{args.img_size}")

        class_counts = dict(collections.Counter(self.train_set.targets))
        sorted_counts = [class_counts[i] for i in range(self.n_classes)]
        total_samples = sum(sorted_counts)
        weights = [min(total_samples / (self.n_classes * count), 10.0) for count in sorted_counts]
        self.class_weights = torch.FloatTensor(weights).to(device)

    def run(self):
        model = DualEncoderNet(
            backbone1=self.args.backbone1,
            backbone2=self.args.backbone2,
            fusion=self.args.fusion,
            n_classes=self.n_classes
        ).to(device)
        # SWA Setup
        swa_model = AveragedModel(model)
        swa_start_epoch = int(self.args.epochs * 0.75)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=1e-2)

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        swa_scheduler = SWALR(optimizer, swa_lr=0.00005)

        best_f1 = 0.0
        train_losses, val_losses = [], []
        epoch_indices = []

        print(f"\n[INFO] Starting training for {self.args.epochs} epochs (SWA starts at {swa_start_epoch})...")
        total_start = time.time()

        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch + 1}/{self.args.epochs}")
            print("-" * 10)

            # --- TRAIN ---
            model.train()
            running_loss = 0.0
            train_preds, train_targets = [], []

            if epoch < 5:
                warmup_lr = self.args.learning_rate * (epoch + 1) / 5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                if np.random.random() < 0.5:
                    images, targets_a, targets_b, lam = mixup_data(images, labels, self.args.mixup_alpha)
                    outputs = model(images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(labels.cpu().numpy())

            if epoch >= swa_start_epoch:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            elif epoch >= 5:
                scheduler.step()

            epoch_loss = running_loss / len(self.train_set)
            train_losses.append(epoch_loss)

            # Calculate and Print Train Metrics
            train_metrics = calculate_metrics(train_targets, train_preds, epoch_loss)
            print_epoch_metrics("Training", train_metrics)

            # --- VALIDATION ---
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            val_epoch_loss = val_loss / len(self.val_set)
            val_losses.append(val_epoch_loss)
            epoch_indices.append(epoch + 1)

            # Calculate and Print Val Metrics
            val_metrics = calculate_metrics(val_targets, val_preds, val_epoch_loss)
            print(".....")
            print_epoch_metrics("Validating", val_metrics)

            if val_metrics['macro_f1'] > best_f1:
                print(f"--> BEST MODEL SAVED ({best_f1:.4f} -> {val_metrics['macro_f1']:.4f})")
                best_f1 = val_metrics['macro_f1']
                torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, 'best_model.pth'))

            print()  # Empty line for readability

        print(f"\n[INFO] Training Complete in {(time.time() - total_start) // 60:.0f} mins.")

        # --- FIX: Pass DEVICE to update_bn ---
        print("[INFO] Updating Batch Norm stats for SWA model on GPU...")
        torch.optim.swa_utils.update_bn(self.train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), os.path.join(self.checkpoint_dir, 'swa_model.pth'))

        training_curve(epoch_indices, train_losses, val_losses)
        return swa_model.module

    def test_with_tta(self, model_to_use=None):
        print("\n[INFO] Loading SWA model for TTA Testing...")
        if model_to_use is None:
            model = DualEncoderNet(
                backbone1=self.args.backbone1,
                backbone2=self.args.backbone2,
                fusion=self.args.fusion,
                n_classes=self.n_classes
            ).to(device)
            state_dict = torch.load(os.path.join(self.checkpoint_dir, 'swa_model.pth'))
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model = model_to_use

        model.eval()
        test_preds, test_targets = [], []

        print("[INFO] Running Test-Time Augmentation (4x views)...")
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(device)
                out1 = model(images)
                out2 = model(torch.flip(images, dims=[3]))
                out3 = model(torch.flip(images, dims=[2]))
                out4 = model(torch.rot90(images, 1, [2, 3]))
                avg_output = (out1 + out2 + out3 + out4) / 4.0
                _, preds = torch.max(avg_output, 1)
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())

        print("\n" + "=" * 20 + " SWA + TTA TEST RESULTS " + "=" * 20)
        print(classification_report(test_targets, test_preds, target_names=self.test_set.classes, digits=4))
        cm = confusion_matrix(test_targets, test_preds)
        plot_confusion_matrix(cm, classes=self.test_set.classes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    print(f"[INFO] Device: {device}")
    trainer = Trainer(args)
    final_swa_model = trainer.run()
    trainer.test_with_tta(final_swa_model)
    print("\n[INFO] Starting Benchmarking...")
    bench_model = DualEncoderNet(
            backbone1=args.backbone1,
            backbone2=args.backbone2,
            fusion=args.fusion,
            n_classes=trainer.n_classes
        )
    print_benchmark_report(bench_model, args.img_size)
