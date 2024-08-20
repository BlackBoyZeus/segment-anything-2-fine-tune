import time
from torchvision.transforms.functional import resize, to_tensor, to_pil_image  

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss
from losses import FocalLoss
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter, calc_iou, best_score_mask, show_box, show_points

torch.set_float32_matmul_precision('high')
import os
import matplotlib.pyplot as plt
import numpy as np


def save_segmentation(images, pred_masks, pred_scores, gt_masks, name, centers, bboxes):
    """Function to save segmentation results as JPG files"""
    output_dir = cfg.segmentated_validation_images_dir
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta', 'orange', 'lime', 'pink']

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    tensor_img = to_tensor(images)
    image_array = tensor_img.cpu().permute(1, 2, 0).numpy()

    axes[0].imshow(image_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    pred_overlay = image_array.copy()

    gt_overlay = image_array.copy()
    for mask_idx in range(pred_masks[0].size(0)):
        best_mask, score, _ = best_score_mask(pred_masks[0][mask_idx].cpu(), pred_scores[0][mask_idx])
        prd_mask = best_mask.cpu().numpy()
        gt_mask = gt_masks[0][mask_idx].cpu().numpy()
        color = plt.get_cmap('tab10')(mask_idx % len(colors))

        pred_overlay[prd_mask > 0.5] = (1 - 0.5) * pred_overlay[prd_mask > 0.5] + 0.5 * np.array(color[:3])
        gt_overlay[gt_mask > 0.5] = (1 - 0.5) * gt_overlay[gt_mask > 0.5] + 0.5 * np.array(color[:3])

    axes[1].imshow(pred_overlay)
    input_label = np.array([1])

    if (cfg.prompt_type == "bounding_box"):
        for i in range(len(bboxes[0])):
            show_box(bboxes[0][i].cpu(), axes[1])
    if (cfg.prompt_type == "points"):
        show_points(centers[0][0].cpu().permute(1, 2, 0), input_label, axes[1])

    axes[1].set_title('Predicted Mask with the prompt')
    axes[1].axis('off')

    axes[2].imshow(pred_overlay)

    axes[2].set_title('Predicted Mask Overlay')
    axes[2].axis('off')

    axes[3].imshow(gt_overlay)
    axes[3].set_title('Ground Truth Mask Overlay')
    axes[3].axis('off')
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{name[0]}.jpg')
    plt.savefig(filename)
    plt.close(fig)

def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int=0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks, name, centers = data

            if (cfg.prompt_type == "bounding_box"):
                pred_masks, pred_scores,_ = model(images, name, bboxes=bboxes)
                for idx, (pred_mask,pred_score, gt_mask) in enumerate(zip(pred_masks[0],pred_scores[0], gt_masks[0])):
                    prd_mask,score,_=best_score_mask(pred_mask,pred_score)
                    prd_mask=prd_mask.cpu()
                    gt_mask_cpu=gt_mask.cpu()

                    batch_stats = smp.metrics.get_stats(
                        prd_mask,
                        gt_mask_cpu.int(),
                        mode='binary',
                        threshold=0.5,
                    )
                    batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                    batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                    ious.update(batch_iou, 1)
                    f1_scores.update(batch_f1, 1)

            if(cfg.prompt_type=="points"):
                pred_masks, pred_scores ,_= model(images,name,centers=centers)

                for idx, (pred_mask,pred_score, gt_mask) in enumerate(zip(pred_masks[0],pred_scores[0], gt_masks[0])):
                    prd_mask,score,_=best_score_mask(pred_mask,pred_score)
                    prd_mask=prd_mask.cpu()
                    gt_mask_cpu=gt_mask.cpu()

                    batch_stats = smp.metrics.get_stats(
                        prd_mask,
                        gt_mask_cpu.int(),
                        mode='binary',
                        threshold=0.5,
                    )
                    batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                    batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                    ious.update(batch_iou, 1)
                    f1_scores.update(batch_f1, 1)

                    # Save the segmentation for the images
            if(cfg.save_validation_images_result==True):
              save_segmentation(images, pred_masks,pred_scores, gt_masks,name,centers,bboxes)

            fabric.print(
                f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    fabric.print(f"Saving checkpoint to {cfg.out_checkpoint_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(cfg.out_checkpoint_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()


def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop with Laplacian Smoothing."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    segmentation_loss_weight = cfg.loss.segmentation_weight  # Weight for segmentation loss
    temporal_loss_weight = cfg.loss.temporal_weight  # Weight for temporal loss
    tv_loss_weight = cfg.loss.tv_weight  # Weight for total variation loss
    laplacian_weight = cfg.loss.laplacian_weight  # Weight for Laplacian smoothing loss

    # --- Define Laplacian Kernel ---
    laplacian_kernel = torch.tensor([
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]]
    ], dtype=torch.float32, device=fabric.device)  # Define on the correct device

    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        temporal_losses = AverageMeter()  # Add temporal loss metric
        tv_losses = AverageMeter()  # Add total variation loss metric
        laplacian_losses = AverageMeter()  # Add Laplacian smoothing loss metric

        end = time.time()
        validated = False
        iter = 0
        for num, data in enumerate(train_dataloader):

            if epoch % cfg.eval_interval == 0 and not validated:
                validate(fabric, model, val_dataloader, epoch)
                validated = True

            data_time.update(time.time() - end)
            images, bboxes, gt_masks, name, centers = data

            batch_size = images.shape[0]  

            if cfg.iterative_sampling:
                iter_sampling_num = cfg.correction_clicks
            else:
                iter_sampling_num = 1

            iter_total = len(train_dataloader) * iter_sampling_num

            for click_num in range(iter_sampling_num):
                iter = iter + 1

                loss_focal = torch.tensor(0., device=fabric.device)
                loss_dice = torch.tensor(0., device=fabric.device)
                loss_iou = torch.tensor(0., device=fabric.device)
                loss_temporal = torch.tensor(0., device=fabric.device)
                loss_tv = torch.tensor(0., device=fabric.device)
                loss_laplacian = torch.tensor(0., device=fabric.device) # Initialize Laplacian loss

                if click_num == 0:
                    if cfg.prompt_type == "bounding_box":
                        pred_masks, iou_predictions = model(images, name, bboxes=bboxes)
                    elif cfg.prompt_type == "points":
                        pred_masks, iou_predictions = model(images, name, centers=centers)
                else:
                    if cfg.prompt_type == "bounding_box":
                        pred_masks, iou_predictions = model(images, name, bboxes=bboxes, previous_masks=previous_best_mask[0])
                    elif cfg.prompt_type == "points":
                        pred_masks, iou_predictions = model(images, name, centers=centers, previous_masks=previous_best_mask[0])

                previous_best_mask = []
                num_masks = sum(len(prd_mask) for prd_mask in pred_masks[0])

                for prd_mask, gt_mask, pred_score in zip(pred_masks[0], gt_masks[0], iou_predictions[0]):
                    num_columns = prd_mask.shape[0]
                    temp_focal_loss = torch.tensor(0., device=fabric.device)
                    temp_dice_loss = torch.tensor(0., device=fabric.device)
                    min_seg_loss = torch.tensor(float('inf'), device=fabric.device)  # Initialize with a large value

                    for col_idx in range(num_columns):
                        mask = prd_mask[col_idx, :, :]
                        score = pred_score[col_idx]
                        batch_iou = calc_iou(mask, gt_mask)

                        if col_idx == 0:
                            best_score = score
                            num = col_idx
                        else:
                            if score > best_score:
                                best_score = score
                                num = col_idx

                        loss_iou += F.l1_loss(score, batch_iou, reduction='sum') / num_masks
                        seg_loss = focal_loss(mask, gt_mask) + dice_loss(mask, gt_mask)

                        if seg_loss < min_seg_loss:
                            min_seg_loss = seg_loss
                            temp_focal_loss = focal_loss(mask, gt_mask)
                            temp_dice_loss = dice_loss(mask, gt_mask)

                    loss_focal += temp_focal_loss
                    loss_dice += temp_dice_loss
                    previous_best_mask.append(prd_mask[num].detach())

                # --- Temporal Loss (MSE between frames) ---
                if click_num > 0:  # Calculate temporal loss after the first click
                    loss_temporal += F.mse_loss(mask, previous_best_mask[-1], reduction='mean')

                # --- Total Variation Loss ---
                loss_tv += torch.sum(torch.image.total_variation(mask.unsqueeze(0))) / batch_size

                # --- Laplacian Smoothing Loss ---
                smoothed_mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0), laplacian_kernel, padding=1)
                loss_laplacian = torch.mean(torch.abs(smoothed_mask)) / batch_size  # Calculate mean absolute value


                # --- Combine Losses ---
                loss_total = (
                    segmentation_loss_weight * seg_loss +
                    temporal_loss_weight * loss_temporal +
                    tv_loss_weight * loss_tv +
                    loss_iou +
                    laplacian_weight * loss_laplacian  # Add Laplacian loss with a weight
                )

                # --- Apply Gradients and Update ---
                optimizer.zero_grad()
                fabric.backward(loss_total)
                optimizer.step()
                scheduler.step()

                # --- Update Metrics ---
                batch_time.update(time.time() - end)
                end = time.time()

                focal_losses.update(loss_focal.item(), batch_size)
                dice_losses.update(loss_dice.item(), batch_size)
                iou_losses.update(loss_iou.item(), batch_size)
                total_losses.update(loss_total.item(), batch_size)
                temporal_losses.update(loss_temporal.item(), batch_size)
                tv_losses.update(loss_tv.item(), batch_size)
                laplacian_losses.update(loss_laplacian.item(), batch_size)  # Update Laplacian loss metric

                # --- Print/Log Information (Including Laplacian Loss) ---
                fabric.print(f'Epoch: [{epoch}][{iter}/{iter_total}]:'
                             f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                             f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                             f' | Focal Loss [{segmentation_loss_weight * focal_losses.val:.4f} ({segmentation_loss_weight * focal_losses.avg:.4f})]'
                             f' | Dice Loss [{segmentation_loss_weight * dice_losses.val:.4f} ({segmentation_loss_weight * dice_losses.avg:.4f})]'
                             f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                             f' | Temporal Loss [{temporal_loss_weight * temporal_losses.val:.4f} ({temporal_loss_weight * temporal_losses.avg:.4f})]'
                             f' | TV Loss [{tv_loss_weight * tv_losses.val:.4f} ({tv_loss_weight * tv_losses.avg:.4f})]'
                             f' | Laplacian Loss [{laplacian_weight * laplacian_losses.val:.4f} ({laplacian_weight * laplacian_losses.avg:.4f})]'
                             f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

                steps = epoch * len(train_dataloader) + iter
                log_info = {
                    'Loss': total_losses.val,
                    'Focal Loss': segmentation_loss_weight * focal_losses.val,
                    'Dice Loss': segmentation_loss_weight * dice_losses.val,
                    'IoU Loss': iou_losses.val,
                    'Temporal Loss': temporal_loss_weight * temporal_losses.val,
                    'TV Loss': tv_loss_weight * tv_losses.val,
                    'Laplacian Loss': laplacian_weight * laplacian_losses.val,
                }
                fabric.log_dict(log_info, step=steps)

            if cfg.save_embeddings_only_for_iterative_sampling:
                features_file_name = os.path.join(cfg.image_features_embeddings_dir,
                                                  (str(name) + "_image_embeddings_cache.pklz"))
                if os.path.exists(features_file_name):
                    os.remove(features_file_name)


def configure_opt(cfg: Box, model: Model):
    """Configure optimizer and scheduler."""

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor ** 2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate,
                                 weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def main(cfg: Box) -> None:
    """Main training function."""
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_checkpoint_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_checkpoint_dir, exist_ok=True)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    train_data, val_data = load_datasets(cfg, cfg.dataset.image_resize)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    validate(fabric, model, val_data, epoch=cfg.num_epochs)


if __name__ == "__main__":
    main(cfg)

'''
Explanation:

Imports: Necessary libraries are imported, including PyTorch, Lightning, segmentation models, and your custom modules.
save_segmentation() Function: This function takes predicted masks, scores, ground truth masks, image names, prompt centers, and bounding boxes, and visualizes them, saving the results as JPG images in the specified directory.
validate() Function:
Sets the model to evaluation mode (model.eval()).
Initializes metrics (IoU and F1 score).
Iterates through the validation dataloader, making predictions and calculating metrics.
Saves segmentation visualization if enabled in the config.
Prints validation metrics and saves the model checkpoint.
train_sam() Function (Main Training Loop):
Initializes loss functions (Focal, Dice), loss weights from the config, and metrics.
The main training loop iterates through epochs.
Validation is performed at specified intervals.
Data is loaded and batch size is determined.
The inner loop handles iterative sampling (corrective clicks).
In each iteration:
Losses are initialized.
Predictions are made based on the prompt type (bounding boxes or points).
The best-scoring mask is selected.
IoU, focal loss, dice loss, temporal loss, and total variation loss are calculated.
Losses are combined with their respective weights.
Gradients are calculated and applied (optimizer.step()).
The learning rate is updated using the scheduler.
Metrics are updated and printed.
Training progress is logged.
If enabled, image embeddings are saved for iterative sampling.
configure_opt() Function:
Defines a learning rate scheduler function (lr_lambda).
Creates an Adam optimizer and a LambdaLR scheduler based on the configuration.
main() Function:
Initializes Lightning Fabric with specified settings (accelerator, devices, strategy, logging).
Sets the random seed for reproducibility.
Creates the output directory for checkpoints.
Instantiates the Model and sets it up.
Loads training and validation datasets and creates dataloaders.
Configures the optimizer and scheduler.
Sets up the model and optimizer with Lightning Fabric.
Runs the training loop (train_sam()).
Performs final validation after all epochs.
This script provides a complete implementation of the training process, incorporating the new loss components for improved spatial and temporal consistency in mask predictions. You should have a config.py file to define all the necessary configuration parameters, a dataset.py file to load your data, a model.py file with your SAM model architecture, and the custom losses.py and utils.py files for the loss functions and utility functions.
'''
