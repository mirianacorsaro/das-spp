import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.dataset import get_dataset
from models.tunet import TUNet as DASSPP
from models.unet import UNet
from utils.picking_metrics import evaluate_plot_and_metrics
from models.swin_transformer_v2 import SwinTransformer
from training.train import train_model  
import wandb

def build_model(name: str, num_heads: int, forward_expansion: int, device: torch.device) -> torch.nn.Module:
    """Instantiate the selected model and move it to device."""
    if name == "das-spp":
        model = DASSPP(n_channels=1, n_classes=3, heads=num_heads, forward_expansion=forward_expansion)
    elif name == "unet":
        model = UNet()
    elif name == "swint":
        model = SwinTransformer()
    else:
        raise ValueError(f"Unsupported model_name: {name}")
    return model.to(device)


def build_criterion(kind: str, class_weights: Optional[np.ndarray], device: torch.device) -> torch.nn.Module:
    """Return CrossEntropy with or without per-class weights."""
    if kind == "weighted_classes":
        if class_weights is None:
            raise ValueError("class_weights required for 'weighted_classes'.")
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(device))
        return criterion
    if kind == "not_weighted_classes":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported criterion: {kind}")


def build_optimizer(name: str, model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """Return optimizer instance."""
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    raise ValueError(f"Unsupported optimizer: {name}")


def choose_device(gpu: Optional[str]) -> torch.device:
    """Pick a device; if GPU id is provided and CUDA present, use it."""
    if torch.cuda.is_available():
        if gpu is not None:
            return torch.device(f"cuda:{gpu}")
        return torch.device("cuda")
    return torch.device("cpu")


def safe_wandb_init(enabled: bool, project: str, config: dict) -> None:
    """Initialize W&B if enabled; otherwise no-op."""
    if not enabled:
        return
    try:
        wandb.init(project=project, config=config)
    except Exception:
        pass


def safe_wandb_finish() -> None:
    """Finish W&B run if present (no-op on failure)."""
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train or run inference for DAS picking (directories only, no ZIP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser):
        sp.add_argument("--train-data", default="data/campi_flegrei/labeled-data/data/train", help="Training directory")
        sp.add_argument("--val-data",   default="data/campi_flegrei/labeled-data/data/val",   help="Validation directory")
        sp.add_argument("--test-data",  default="data/campi_flegrei/labeled-data/data/test",  help="Test directory")
        sp.add_argument("--targets",    default="data/campi_flegrei/labeled-data/masks", help="Targets directory")
        sp.add_argument("--model-name", choices=["das-spp", "unet", "swint"], default="das-spp")
        sp.add_argument("--num-heads", type=int, default=2, help="Only for das-spp")
        sp.add_argument("--forward-expansion", type=int, default=2, help="Only for das-spp")
        sp.add_argument("--criterion", choices=["weighted_classes", "not_weighted_classes"], default="weighted_classes")
        sp.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
        sp.add_argument("--batch-size", type=int, default=2)
        sp.add_argument("--lr", type=float, default=1e-4)
        sp.add_argument("--epochs", type=int, default=100)
        sp.add_argument("--gpu", default=None, help="GPU id, e.g., 0")
        sp.add_argument("--num-workers", type=int, default=4)
        sp.add_argument("--seed", type=int, default=1337)
        sp.add_argument("--project", default="ssp-picking-final", help="Weights & Biases project")
        sp.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

        sp.add_argument("--exp-id", type=int, default=0)
        sp.add_argument("--model-root", default="runs/models/trained_models")
        sp.add_argument("--csv-root",   default="runs/results/csv")
        sp.add_argument("--figs-root",  default="runs/results/figures")
        
        sp.add_argument("--das-val-dir", default="data/das-picking/labeled-data/val",
                        help="Validation DAS directory with .npy (for metrics/time axis)")
        sp.add_argument("--das-test-dir", default="data/das-picking/labeled-data/test",
                        help="Test DAS directory with .npy (for metrics/time axis)")

    # train subcommand
    sp_train = sub.add_parser("train", help="Run training")
    add_common(sp_train)
    sp_train.add_argument("--eval-every", type=int, default=10, help="Run eval+plots every N epochs")
    sp_train.add_argument("--save-ckpt-every", type=int, default=10, help="Save state_dict every N epochs")
    sp_train.add_argument("--gate-val-acc", type=float, default=0.80, help="Only eval/snapshot if Val Acc ≥ gate")
    sp_train.add_argument("--num-plot-images", type=int, default=8)
    sp_train.add_argument("--plot-original-das", action="store_true", help="Plot figures using original DAS arrays")
    sp_train.add_argument("--orig-h", type=int, default=4324, help="Original H (channels) if not using original plots")
    sp_train.add_argument("--orig-w", type=int, default=12000, help="Original W (time) if not using original plots")
    sp_train.add_argument("--finetuning", type=str, default=None, help="Path to a checkpoint to fine-tune from (model or state_dict)")
    sp_train.add_argument("--dpi", type=int, default=150)

    # infer subcommand
    sp_infer = sub.add_parser("infer", help="Run inference/plots/metrics on TEST set from a checkpoint")
    add_common(sp_infer)
    sp_infer.add_argument("--checkpoint", required=True, help="Path to best model .pth saved with torch.save(model, ...)")
    sp_infer.add_argument("--epoch", type=int, default=0, help="Epoch tag for output folder naming")
    sp_infer.add_argument("--num-plot-images", type=int, default=16)
    sp_infer.add_argument("--plot-original-das", action="store_true")
    sp_infer.add_argument("--orig-h", type=int, default=4324)
    sp_infer.add_argument("--orig-w", type=int, default=12000)
    sp_infer.add_argument("--dpi", type=int, default=150)

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = choose_device(args.gpu)
    print(f"[Info] Using device: {device}")

    train_ds, class_w, val_ds, test_ds = get_dataset(
        args.train_data, args.val_data, args.test_data, args.targets
    )

    pin_memory = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=pin_memory)

    model = build_model(args.model_name, args.num_heads, args.forward_expansion, device)
    criterion = build_criterion(args.criterion, class_w, device)
    optimizer = build_optimizer(args.optimizer, model, args.lr)

    use_wandb = not args.no_wandb

    if args.command == "train":
        safe_wandb_init(
            enabled=use_wandb,
            project=args.project,
            config={
                "command": "train",
                "model_name": args.model_name,
                "num_heads": args.num_heads,
                "forward_expansion": args.forward_expansion,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "optimizer": args.optimizer,
                "criterion": args.criterion,
                "epochs": args.epochs,
                "exp_id": args.exp_id,
            },
        )

        best_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=args.epochs,
            exp_id=args.exp_id,
            model_root=args.model_root,
            csv_root=args.csv_root,
            figs_root=args.figs_root,
            das_val_dir=args.das_val_dir,
            eval_every=args.eval_every,
            save_ckpt_every=args.save_ckpt_every,
            gate_val_acc=args.gate_val_acc,
            num_plot_images=args.num_plot_images,
            plot_original_das=args.plot_original_das,
            original_hw=(args.orig_h, args.orig_w),
            finetuning=args.finetuning,
            dpi=args.dpi,
        )
        print(f"[Done] Best model at: {best_path}")
        safe_wandb_finish()
        return

    if args.command == "infer":
        print(f"[Info] Loading checkpoint: {args.checkpoint}")
        model = torch.load(args.checkpoint, map_location=device)
        model.eval()

        safe_wandb_init(
            enabled=use_wandb,
            project=args.project,
            config={
                "command": "infer",
                "checkpoint": args.checkpoint,
                "batch_size": args.batch_size,
                "criterion": args.criterion,
                "exp_id": args.exp_id,
            },
        )

        (p_prec, p_rec, p_f1, dt_p,
         s_prec, s_rec, s_f1, dt_s) = evaluate_plot_and_metrics(
            model=model,
            data_loader=test_loader,
            device=device,
            epoch=args.epoch,
            figures_dir=str(Path(args.figs_root) / f"figures_exp_{args.exp_id}" / "test"),
            csv_dir=str(Path(args.csv_root) / f"csv_exp_{args.exp_id}" / "test"),
            num_plot_images=args.num_plot_images,
            save_every=1,
            dpi=args.dpi,
        )

        print(
            f"[TEST] P: P={p_prec:.4f} R={p_rec:.4f} F1={p_f1:.4f} | "
            f"S: P={s_prec:.4f} R={s_rec:.4f} F1={s_f1:.4f}"
        )
        if wandb.run is not None:
            wandb.log({
                "test/p_precision": float(p_prec),
                "test/p_recall": float(p_rec),
                "test/p_f1": float(p_f1),
                "test/p_dt_mean": float(dt_p),
                "test/s_precision": float(s_prec),
                "test/s_recall": float(s_rec),
                "test/s_f1": float(s_f1),
                "test/s_dt_mean": float(dt_s),
            })
        safe_wandb_finish()
        return


if __name__ == "__main__":
    main()
