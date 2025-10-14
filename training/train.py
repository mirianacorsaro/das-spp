import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from alive_progress import alive_bar
import wandb
from .utils.utils import calculate_accuracy, compute_dice_loss
from utils.picking_metrics import evaluate_plot_and_metrics 


def _ensure_dir(path: str | os.PathLike) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _wb_log(payload: dict) -> None:
    try:
        if wandb.run is not None:
            wandb.log(payload)
    except Exception:
        pass


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    num_epochs: int,
    exp_id: int = 0,
    model_root: str = "runs/models/trained",
    csv_root: str = "runs/results/training/csv",
    figs_root: str = "runs/results/training/figures",
    das_val_dir: Optional[str] = "data/das-picking/labeled-data/val",
    eval_every: int = 10,           
    save_ckpt_every: int = 10,       
    gate_val_acc: float = 0.80,      
    num_plot_images: int = 8,
    plot_original_das: bool = True,
    original_hw: Optional[Tuple[int, int]] = (4324, 12000),
    finetuning: str | None = None,
    dpi: int = 150,
) -> str:
    """
    Train loop with checkpointing and integrated qualitative plots + metrics.

    Returns:
        best_model_path: path to the best full-model checkpoint (torch.save(model, ...)).
    """
    model = model.to(device)

    run_dir = Path(model_root) / f"exp_{exp_id}"
    ckpt_dir = Path(_ensure_dir(run_dir / "checkpoints"))
    best_path = run_dir / "best_picking_model.pth"

    best_f1_mean = -1.0
    best_epoch = 0

    if finetuning is not None:
        print(f"[Fine-tune] Loading weights from: {finetuning}")
        model = torch.load(finetuning, map_location=device)
        if isinstance(model, torch.nn.Module):
            print("[Fine-tune] Loaded full model checkpoint.")
        else:
            raise RuntimeError("Unsupported checkpoint format for finetuning.")

    for epoch in range(1, int(num_epochs) + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        with alive_bar(len(train_loader), title=f"Epoch {epoch}/{num_epochs}") as bar:
            for signals, targets, _ in train_loader:
                signals = signals.to(device)
                targets = targets.to(device)

                logits = model(signals)                 
                probs = logits.sigmoid()                

                num_classes = logits.shape[1]
                one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

                loss_ce = criterion(logits, targets.long())
                loss_dice = compute_dice_loss(one_hot, probs)
                loss = loss_ce + loss_dice

                acc = calculate_accuracy(probs, one_hot)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                running_acc += float(acc.item())
                bar()

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = running_acc / max(1, len(train_loader))

        val_loss, val_acc = _validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}/{num_epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )
        _wb_log({"train/loss": train_loss, "train/acc_dice": train_acc,
                 "val/loss": val_loss, "val/acc_dice": val_acc, "epoch": epoch})

        if (epoch % max(1, save_ckpt_every) == 0) and (val_acc >= gate_val_acc):
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                },
                ckpt_dir / f"picking_model_{epoch}.pth",
            )

        should_eval = (epoch % max(1, eval_every) == 0) and (val_acc >= gate_val_acc)

        if should_eval and das_val_dir:
            csv_dir = Path(csv_root) / f"csv_exp_{exp_id}" / f"epoch_{epoch}"
            figs_dir = Path(figs_root) / f"figures_exp_{exp_id}"

            (p_prec, p_rec, p_f1, dt_p,
             s_prec, s_rec, s_f1, dt_s) = evaluate_plot_and_metrics(
                model=model,
                data_loader=val_loader,
                device=device,
                epoch=epoch,
                csv_dir=str(csv_dir),
                figures_dir=str(figs_dir),
                num_plot_images=int(num_plot_images),
                save_every=1,     
                dpi=int(dpi),
            )

            mean_f1 = (float(p_f1) + float(s_f1)) / 2.0
            _wb_log({
                "val/p_precision": float(p_prec),
                "val/p_recall": float(p_rec),
                "val/p_f1": float(p_f1),
                "val/p_dt_mean": float(dt_p),
                "val/s_precision": float(s_prec),
                "val/s_recall": float(s_rec),
                "val/s_f1": float(s_f1),
                "val/s_dt_mean": float(dt_s),
                "val/mean_f1_ps": float(mean_f1),
                "epoch": epoch,
            })

            print(f"p_precision: {float(p_prec)},\
                p_recall: {float(p_rec)},\
                p_f1: {float(p_f1)},\
                p_dt_mean: {float(dt_p)},\
                s_precision: {float(s_prec)},\
                s_recall: {float(s_rec)},\
                s_f1: {float(s_f1)},\
                s_dt_mean: {float(dt_s)}, \
                mean_f1_ps: {float(mean_f1)}")
            
            if mean_f1 > best_f1_mean:
                best_f1_mean = mean_f1
                best_epoch = epoch
                torch.save(model, best_path)
                print(f"[Best] epoch={epoch} meanF1={best_f1_mean:.4f} → saved {best_path}")

    print(f"\nBest model by mean F1 at epoch {best_epoch} (meanF1={best_f1_mean:.4f}). Path: {best_path}")
    return best_path.as_posix()

@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    ):
    """Validation loop: returns (avg_loss, avg_dice_accuracy)."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for signals, targets, _ in loader:
        signals = signals.to(device)
        targets = targets.to(device)

        logits = model(signals)
        probs = logits.sigmoid()

        num_classes = logits.shape[1]
        one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

        loss = criterion(logits, targets.long()) + compute_dice_loss(one_hot, probs)
        acc = calculate_accuracy(probs, one_hot)

        total_loss += float(loss.item())
        total_acc += float(acc.item())

    n = max(1, len(loader))
    return total_loss / n, total_acc / n
