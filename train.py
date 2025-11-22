import argparse
import os
import random
import time
import json

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# Utility
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CIFAR-10 mean/std（必要ならあなたのコードの値に合わせて変更）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_dataloaders(batch_size: int = 128, use_augment: bool = False):
    """train / val の DataLoader を返す。"""
    if use_augment:
        # ← ここをあなたの「データ拡張施策」の transform に差し替えてOK
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, trainset.classes


def build_model(num_classes: int = 10, use_pretrained: bool = False):
    """ResNet50 を構築。転移学習かどうかを切り替える。"""
    if use_pretrained:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        # 転移学習らしく特徴抽出部を凍結（必要に応じて外してもOK）
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        model = models.resnet50(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_optimizer_and_scheduler(model, lr, weight_decay, use_scheduler, epochs):
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    if use_scheduler:
        # ← あなたのノートの LR 調整（StepLR, CosineAnnealing など）に合わせて変更OK
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None
    return optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds_all, targets_all = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds_all.extend(out.argmax(1).detach().cpu().numpy())
        targets_all.extend(y.detach().cpu().numpy())

    return np.mean(losses), accuracy_score(targets_all, preds_all)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all, targets_all = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            losses.append(loss.item())
            preds_all.extend(out.argmax(1).detach().cpu().numpy())
            targets_all.extend(y.detach().cpu().numpy())

    return np.mean(losses), accuracy_score(targets_all, preds_all), np.array(targets_all), np.array(preds_all)


def plot_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves_acc.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()


def save_misclassified_images(model, loader, class_names, device, out_dir, n=25):
    """誤分類サンプルを画像として保存。"""
    model.eval()
    images = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            mask = preds != y
            if mask.any():
                mis_x = x[mask]
                mis_y = y[mask]
                mis_p = preds[mask]
                for xi, yi, pi in zip(mis_x, mis_y, mis_p):
                    images.append(xi.cpu())
                    true_labels.append(int(yi.cpu()))
                    pred_labels.append(int(pi.cpu()))
                    if len(images) >= n:
                        break
            if len(images) >= n:
                break

    if not images:
        return

    g = int(np.ceil(np.sqrt(len(images))))
    plt.figure(figsize=(2*g, 2*g))
    for i, (img, t, p) in enumerate(zip(images, true_labels, pred_labels)):
        ax = plt.subplot(g, g, i+1)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = CIFAR10_STD * img_np + CIFAR10_MEAN
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.axis("off")
        ax.set_title(f"T:{class_names[t]}\nP:{class_names[p]}", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "misclassified.png"))
    plt.close()


def get_exp_config(exp_name: str):
    """
    6種類の施策をここでまとめて定義しておく。
    Baseline, データ拡張, 学習率調整, 転移学習, Weight Decay,
    転移学習＋データ拡張
    """
    base_cfg = dict(
        use_augment=False,
        use_pretrained=False,
        use_scheduler=False,
        weight_decay=0.0,
        lr=3e-4,
        epochs=10,
        batch_size=256,
    )
    cfg = base_cfg.copy()

    if exp_name == "baseline":
        pass
    elif exp_name == "augment":
        base_cfg["use_augment"] = True
    elif exp_name == "lr":
        base_cfg["use_scheduler"] = True
    elif exp_name == "transfer":
        base_cfg["use_pretrained"] = True
    elif exp_name == "wd":
        base_cfg["weight_decay"] = 1e-4
    elif exp_name == "transfer_augment":
        base_cfg["use_pretrained"] = True
        base_cfg["use_augment"] = True
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")

    return base_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str,
                        choices=["baseline", "augment", "lr", "transfer", "wd", "transfer_augment"],
                        default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", type=str, default="artifacts")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    cfg = get_exp_config(args.exp)

    # run directory: artifacts/exp_name/yyyymmdd-hhmmss
    run_name = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.output_root, args.exp, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # config を保存（再現性のため）
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dict(exp=args.exp, seed=args.seed, **cfg), f, indent=2)

    trainloader, valloader, class_names = get_dataloaders(
        batch_size=cfg["batch_size"],
        use_augment=cfg["use_augment"],
    )

    model = build_model(num_classes=len(class_names),
                        use_pretrained=cfg["use_pretrained"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        use_scheduler=cfg["use_scheduler"], epochs=cfg["epochs"]
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = evaluate(model, valloader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[{args.exp}] Epoch {epoch}/{cfg['epochs']}"
              f" train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
              f" train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # 一番 val_acc が良かったモデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            np.save(os.path.join(out_dir, "y_true.npy"), y_true)
            np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)

    # ログを CSV で保存（再現性要件）
    import pandas as pd
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # 可視化用に best モデルを読み直し
    best_model = build_model(num_classes=len(class_names),
                             use_pretrained=cfg["use_pretrained"]).to(device)
    best_model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))

    y_true = np.load(os.path.join(out_dir, "y_true.npy"))
    y_pred = np.load(os.path.join(out_dir, "y_pred.npy"))

    # 混同行列、誤分類サンプル、学習曲線など
    plot_curves(history, out_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, out_dir)
    save_misclassified_images(best_model, valloader, class_names, device, out_dir, n=25)

    # text レポートも保存
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
