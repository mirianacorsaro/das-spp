from tqdm import tqdm
import os
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.save_pick import save_csv_results
from training.utils.utils import get_first_consecutive_positions

@torch.no_grad()
def evaluate_plot_and_metrics(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    epoch: int,                                
    figures_dir: Optional[str] = None,             
    csv_dir: Optional[str] = None,                 
    num_plot_images: int = 8,                      
    save_every: int = 10,
    dpi: int = 150,
) -> Tuple[float, float, float, List[float], float, float, float, List[float]]:
    """
    Model evaluation

    Return:
        P: prec, recall, f1, time difference
        S: prec, recall, f1, time difference
    """
    model.eval()

    results_by_id: Dict[int, List[dict]] = {}

    precision_p_list, recall_p_list, f1_score_p_list, td_p_list = [], [], [], []
    precision_s_list, recall_s_list, f1_score_s_list, td_s_list = [], [], [], []

    will_save_figs = (
        figures_dir is not None
        and (epoch % max(1, save_every) == 0)
        and num_plot_images > 0
    )
    if will_save_figs:
        fig_dir = Path(figures_dir) / f"epoch_{epoch}"
        fig_dir.mkdir(parents=True, exist_ok=True)
    plots_written = 0

    for batch_idx, (signals, targets, info) in enumerate(data_loader):
        signals = signals.to(device)
        targets = targets.to(device)

        outputs = model(signals)                
        mask_pred = torch.argmax(outputs, dim=1) 

        b_mask_pred = (mask_pred == 0).cpu().float()
        p_mask_pred = (mask_pred == 1).cpu().float()
        s_mask_pred = (mask_pred == 2).cpu().float()

        b_mask_target = (targets == 0).cpu().float()
        p_mask_target = (targets == 1).cpu().float()
        s_mask_target = (targets == 2).cpu().float()

        B = signals.size(0)

        for i in range(B):
            sample_id = int(info[i])
            if sample_id not in results_by_id:
                results_by_id[sample_id] = []

            p_pred_i = p_mask_pred[i]  
            s_pred_i = s_mask_pred[i]
            p_tgt_i = p_mask_target[i]
            s_tgt_i = s_mask_target[i]

            p_x, p_y = get_first_consecutive_positions(p_pred_i, 1)
            s_x, s_y = get_first_consecutive_positions(s_pred_i, 1)


            for k in range(len(p_x)):
                results_by_id[sample_id].append({
                    'channel': int(p_x[k]),
                    'time_index': int(p_y[k]),
                    'type': 'P'
                })
            for k in range(len(s_x)):
                results_by_id[sample_id].append({
                    'channel': int(s_x[k]),
                    'time_index': int(s_y[k]),
                    'type': 'S'
                })

            all_pd_p = list(zip(p_x, p_y))
            all_pd_s = list(zip(s_x, s_y))

            gt_p_x, gt_p_y = get_first_consecutive_positions(p_tgt_i, 1)
            gt_s_x, gt_s_y = get_first_consecutive_positions(s_tgt_i, 1)
            all_true_p = list(zip(gt_p_x, gt_p_y))
            all_true_s = list(zip(gt_s_x, gt_s_y))

            precision_p, recall_p, f1_score_p, td_p = calculate_catalog_overlap(all_true_p, all_pd_p, margin=7)
            precision_s, recall_s, f1_score_s, td_s = calculate_catalog_overlap(all_true_s, all_pd_s, margin=7)

            precision_p_list.append(precision_p)
            recall_p_list.append(recall_p)
            f1_score_p_list.append(f1_score_p)
            td_p_list.append(td_p)

            precision_s_list.append(precision_s)
            recall_s_list.append(recall_s)
            f1_score_s_list.append(f1_score_s)
            td_s_list.append(td_s)

            if will_save_figs and plots_written < num_plot_images:
                sig_img = signals[i].detach().cpu()
                if sig_img.dim() == 3 and sig_img.size(0) in (1, 3):
                    sig_img_np = sig_img.permute(1, 2, 0).numpy().squeeze()
                else:
                    sig_img_np = sig_img.squeeze().numpy()

                fig = plt.figure(figsize=(20, 8), dpi=dpi)
                gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

                grid_left = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0])

                images = [
                    (sig_img_np, 'Dati DAS'),
                    (p_mask_target[i].numpy(), 'GT P Mask'),
                    (s_mask_target[i].numpy(), 'GT S Mask'),
                    (b_mask_target[i].numpy(), 'GT Bg Mask'),
                    (sig_img_np, 'Dati DAS'),
                    (p_mask_pred[i].numpy(), 'Pred P Mask'),
                    (s_mask_pred[i].numpy(), 'Pred S Mask'),
                    (b_mask_pred[i].numpy(), 'Pred Bg Mask'),
                ]

                for idx, (img, title) in enumerate(images):
                    ax = fig.add_subplot(grid_left[idx // 4, idx % 4])
                    if 'Mask' in title:
                        ax.imshow(img, cmap='gray', aspect="auto", interpolation="none")
                    else:
                        ax.imshow(img, aspect="auto", interpolation="none")
                    ax.set_title(title)
                    ax.axis("off")

                ax_right = fig.add_subplot(gs[1])
                ax_right.imshow(sig_img_np, aspect="auto", interpolation="none")
                ax_right.scatter(list(map(int, p_y)), list(map(int, p_x)), s=2, label="P")
                ax_right.scatter(list(map(int, s_y)), list(map(int, s_x)), s=2, label="S")
                ax_right.legend()
                ax_right.set_title("DAS con arrivi P/S predetti")
                ax_right.axis("off")

                fig.tight_layout()
                fig_path = (Path(figures_dir) / f"epoch_{epoch}" / f"{sample_id}.png")
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_path, bbox_inches="tight")
                plt.close(fig)
                plots_written += 1

    p_prec = np.mean(precision_p_list)
    p_rec = np.mean(recall_p_list)
    p_f1 = np.mean(f1_score_p_list)
    p_dt = np.mean(td_p_list)/200

    s_prec = np.mean(precision_s_list)
    s_rec = np.mean(recall_s_list)
    s_f1 = np.mean(f1_score_s_list)
    s_dt = np.mean(td_s_list)/200

    if csv_dir is not None:
        save_csv_results(results_by_id, path=csv_dir, clean=True)

    return (
        float(p_prec), float(p_rec), float(p_f1), float(p_dt),
        float(s_prec), float(s_rec), float(s_f1), float(s_dt),
    )

def calculate_catalog_overlap(true_positions, pred_positions, margin=100):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    distances = []

    for r, c in true_positions:
        matched = False
        for rp, cp in pred_positions:
            if r == rp:
                if abs(c - cp) <= margin:
                    distances.append(abs(c - cp))
                    true_positives += 1
                    matched = True
                    break
        if not matched:
            false_negatives += 1

    false_positives = len(pred_positions) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1_score = (2 * (precision * recall) )/ (precision + recall) if (precision + recall) != 0 else 0
    
    td_mean = np.mean(distances) if len(distances) != 0 else 0
    
    return precision, recall, f1_score, td_mean