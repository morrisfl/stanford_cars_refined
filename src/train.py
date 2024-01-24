import argparse
import math
import os.path

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from src.dataset import VCoRDataset
from src.model import ConvNeXtB, ViTB16, CLIPConvNeXtB, EfficientNetB1
from src.utils import plot_losses, plot_accuracies


def parse_args():
    parser = argparse.ArgumentParser(description="Train classification model.")
    parser.add_argument("mode_name", choices=["convnext", "efficientnet", "clip-convnext", "dinov2", "siglip"],
                        help="Name of the model to train.")
    parser.add_argument("data_root", help="Path to the data directory.")
    parser.add_argument("--output_dir", default="results/", help="Path to the output directory.")
    parser.add_argument("--batch_size", default=64, help="Batch size.")
    parser.add_argument("--train_style", default="lp", choices=["lp", "ft", "lp-ft"],
                        help="How to train the model, linear probing (lp), fine-tuning (ft), or both (lp-ft).")
    parser.add_argument("--lp_epochs", default=None, help="Number of linear probing epochs if train_style is lp-ft.")
    parser.add_argument("--epochs", default=10, help="Number of training epochs.")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"], type=str,
                        help="Optimizer to use.")
    parser.add_argument("--lr", default=1e-3, help="Learning rate.")
    parser.add_argument("--ft_lr_factor", default=0.0, help="Fine-tuning learning rate factor for lp-ft.")
    parser.add_argument("--weight_decay", default=1e-4, help="Weight decay.")
    parser.add_argument("--scheduler", default="cosine", help="Scheduler to use.")
    parser.add_argument("--min_lr", default=1e-4, help="Minimum learning rate for cosine scheduler.")
    parser.add_argument("--warmup_steps", default=82, help="Number of warmup steps.")
    parser.add_argument("--warmup_factor", default=0.1, help="Warmup factor.")

    return parser.parse_args()


def main():
    args = parse_args()

    train_cfg = {
        "model_name": args.model_name,
        "data_root": args.data_root,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "train_style": args.train_style,
        "lp_epochs": args.lp_epochs,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "ft_lr_factor": args.ft_lr_factor,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "warmup_factor": args.warmup_factor
    }

    train(train_cfg)


def train(cfg):
    output_dir = os.path.join(cfg["output_dir"], f"{cfg['train_style']}_{cfg['model_name']}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    if cfg["model_name"] == "convnext":
        model = ConvNeXtB(num_classes=10)
    elif cfg["model_name"] == "efficientnet":
        model = EfficientNetB1(num_classes=10)
    elif cfg["model_name"] == "clip-convnext":
        model = CLIPConvNeXtB(num_classes=10)
    elif cfg["model_name"] == "dinov2":
        model = ViTB16(model_name="vit_base_patch14_dinov2.lvd142m", num_classes=10)
    elif cfg["model_name"] == "siglip":
        model = ViTB16(model_name="vit_base_patch16_siglip_256", num_classes=10)
    else:
        raise NotImplementedError(f"Unsupported model name: {cfg['model_name']}")

    if cfg["train_style"] in ["lp", "lp-ft"]:
        model.freeze_backbone()

    model.to(device)

    # Data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(model.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(model.mean, model.std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((model.img_size, model.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(model.mean, model.std)
    ])

    excluded_cls = ["beige", "gold", "pink", "purple", "tan"]

    train_dataset = VCoRDataset(os.path.join(cfg["data_root"], "train"), transform=train_transform,
                                excluded_cls=excluded_cls)
    val_dataset = VCoRDataset(os.path.join(cfg["data_root"], "val"), transform=test_transform,
                              excluded_cls=excluded_cls)
    test_dataset = VCoRDataset(os.path.join(cfg["data_root"], "test"), transform=test_transform,
                               excluded_cls=excluded_cls)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if cfg["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    else:
        raise NotImplementedError(f"Unsupported optimizer: {cfg['optimizer']}")

    # Scheduler
    if cfg["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["min_lr"])
    elif cfg["scheduler"] is None:
        scheduler = None
    else:
        raise NotImplementedError(f"Unsupported scheduler: {cfg['scheduler']}")

    # Training
    results = train_and_evaluate_model(cfg, model, train_loader, val_loader, criterion, optimizer, device, output_dir,
                                       scheduler)

    # Plot results
    plot_losses(results["train_losses"], results["val_losses"], os.path.join(output_dir, "losses.png"))
    plot_accuracies(results["top1_accuracies"], os.path.join(output_dir, "accuracies.png"))

    # Test
    max_top1 = max(results["top1_accuracies"])
    test_epoch = results["top1_accuracies"].index(max_top1) + 1
    model_path = os.path.join(output_dir, f"model_epoch{test_epoch}.pt")

    test_model(model_path, test_loader, device)


def train_and_evaluate_model(cfg, model, train_loader, val_loader, criterion, optimizer, device, output_dir,
                             scheduler=None):
    train_losses = []
    val_losses = []
    val_top1_accuracies = []
    model.to(device)

    for epoch in range(cfg["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg["warmup_steps"],
                                     cfg["warmup_factor"])
        val_loss, val_top1_accuracy = evaluate_one_epoch(model, val_loader, criterion, device, epoch)

        model_scripted = torch.jit.script(model)
        model_scripted.save(os.path.join(output_dir, f"model_epoch{epoch + 1}.pt"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_top1_accuracies.append(val_top1_accuracy)

        if scheduler:
            scheduler.step()

        if cfg["train_style"] == "lp-ft":
            if cfg["lp_epochs"] <= epoch + 1:
                cfg["train_style"] = "ft"
                model.unfreeze_backbone()
                lr = cfg["lr"] * cfg["ft_lr_factor"]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

        print(f"Epoch {epoch + 1}/{cfg['epochs']}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
              f"Validation Top-1 Accuracy: {val_top1_accuracy:.4f}")

    train_results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "top1_accuracies": val_top1_accuracies
    }

    return train_results


def train_one_epoch(model, data_loader, criterion, optimizer, device, curr_epoch, warmup_steps=None,
                    warmup_factor=None):
    scheduler = None
    if curr_epoch == 0:
        if warmup_steps is not None and warmup_factor is not None:
            warmup_iters = min(warmup_steps, len(data_loader) - 1)
            scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    model.train()
    running_loss = 0.0

    for images, labels in tqdm(data_loader, desc=f"Training epoch {curr_epoch + 1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

    epoch_loss = running_loss / len(data_loader)

    return epoch_loss


def evaluate_one_epoch(model, data_loader, criterion, device, curr_epoch):
    model.eval()
    running_loss = 0.0
    top1_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Evaluating epoch {curr_epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, top1_predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            top1_correct += (top1_predicted == labels).sum().item()

    epoch_loss = running_loss / len(data_loader)
    top1_accuracy = top1_correct / total

    return epoch_loss, top1_accuracy


def test_model(model_path, dataloader, device):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    top1_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Test model"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, top1_predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            top1_correct += (top1_predicted == labels).sum().item()

    top1_accuracy = top1_correct / total
    print(f"Test Top-1 Accuracy: {top1_accuracy:.4f}")


if __name__ == "__main__":
    main()
