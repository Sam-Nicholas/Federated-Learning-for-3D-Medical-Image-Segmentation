import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
from tqdm import tqdm
from pathlib import Path

try:
    from dataset import save_visualisation_2d
except ImportError:
    logger = logging.getLogger("utils")
    logger.error("Could not import save_visualisation_2d from dataset. Visualisation will be disabled.")
    def save_visualisation_2d(*args, **kwargs):
        logger.warning("Visualisation disabled due to import error.")
        return None

logger = logging.getLogger("utils")



def dice_coefficient(pred, target, smooth=1e-5):
    """Compute the Dice coefficient."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum(dim=0) + target.sum(dim=0)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def dice_loss(pred, target, smooth=1e-5):
    """Compute Dice loss averaged over foreground classes."""
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    dice_per_class = []
    for i in range(num_classes):
        dice_class = dice_coefficient(pred[:, i], target[:, i], smooth)
        dice_per_class.append(dice_class)
    if num_classes > 1:
         mean_fg_dice = np.mean(dice_per_class[1:])
         return 1.0 - mean_fg_dice
    else:
        return 1.0 - np.mean(dice_per_class)

def combined_loss(pred, target, alpha=0.5, ce_weight=None):
    """Combine cross-entropy and Dice loss."""
    target_indices = target.argmax(dim=1)
    ce = F.cross_entropy(pred, target_indices, weight=ce_weight)
    dice = dice_loss(pred, target)
    return alpha * dice + (1 - alpha) * ce

def compute_dice_metrics(pred, target):
    """Compute Dice scores for each class."""
    dice_scores = {}
    classes = ['background', 'edema', 'non_enhancing', 'enhancing']
    num_classes = pred.shape[1]
    pred_classes = pred.argmax(dim=1)
    pred_one_hot = F.one_hot(pred_classes, num_classes=num_classes).permute(0, 3, 1, 2).float()

    for i in range(num_classes):
        class_name = classes[i] if i < len(classes) else f'class_{i}'
        dice = dice_coefficient(pred_one_hot[:, i], target[:, i])
        dice_scores[class_name] = dice

    if num_classes > 1:
        tumor_dice_scores = [dice_scores[classes[i]] for i in range(1, min(num_classes, len(classes))) if classes[i] in dice_scores]
        dice_scores['tumor_mean'] = np.mean(tumor_dice_scores) if tumor_dice_scores else 0.0
    else:
        dice_scores['tumor_mean'] = dice_scores.get(classes[0], 0.0)

    return dice_scores


# FedProx Training Epoch
def train_epoch_fedprox(model, loader, optimizer, device, epoch, save_vis, client_id, global_params_dict, mu):
    """Train for one epoch using FedProx"""
    model.train()
    running_loss = 0.0
    running_prox_loss = 0.0 # Track proximal term separately
    agg_dice_scores = {}
    num_batches = len(loader)
    start_time = time.time()
    vis_saved = False

    progress_bar = tqdm(loader, desc=f"Client {client_id} Train Epoch {epoch} (mu={mu})", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        patient_ids = batch['patient_id']
        slice_indices = batch['slice_idx']

        optimizer.zero_grad()
        outputs = model(images)

        # Calculate Standard Loss
        standard_loss = combined_loss(outputs, masks)

        # Calculate FedProx Loss Term
        prox_loss = 0.0
        if mu > 0 and global_params_dict is not None:
            # Iterate over parameters and compute L2 norm squared difference
            for param_name, param in model.named_parameters():
                if param.requires_grad: # Only consider trainable parameters
                    global_param = global_params_dict.get(param_name)
                    if global_param is not None:
                        # Ensure global param is on the same device
                        global_param = global_param.to(device)
                        # Calculate squared L2 norm difference
                        param_diff_norm_sq = torch.sum(torch.pow(param - global_param, 2))
                        prox_loss += param_diff_norm_sq
                    else:
                         logger.warning(f"Global parameter '{param_name}' not found for prox loss calculation.")

            prox_loss = (mu / 2.0) * prox_loss

        # Total Loss
        total_loss = standard_loss + prox_loss

        # Backward Pass and Optimization
        total_loss.backward()
        optimizer.step()

        # Metrics Calculation (based on standard loss/outputs)
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            batch_dice = compute_dice_metrics(probs, masks)

        # Accumulate Metrics
        running_loss += total_loss.item() # Accumulate total loss
        running_prox_loss += prox_loss.item() if isinstance(prox_loss, torch.Tensor) else prox_loss # Accumulate prox loss
        for k, v in batch_dice.items():
            agg_dice_scores[k] = agg_dice_scores.get(k, 0.0) + v

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'avg_loss': f"{running_loss / (batch_idx + 1):.4f}",
            'dice': f"{batch_dice.get('tumor_mean', 0.0):.4f}"
        })

        # Save Visualisation
        if save_vis and not vis_saved and len(images) > 0 and save_visualisation_2d is not None:
            try:
                vis_path = save_visualisation_2d(
                    images[0], outputs[0], masks[0],
                    epoch, client_id, patient_ids[0], slice_indices[0].item()
                )
                if vis_path: vis_saved = True # Only set if saving succeeded
            except Exception as e: logger.error(f"Error saving training visualisation: {e}")

    # Compute average metrics for the epoch
    epoch_loss = running_loss / num_batches
    epoch_prox_loss = running_prox_loss / num_batches
    epoch_dice = {k: v / num_batches for k, v in agg_dice_scores.items()}

    duration = time.time() - start_time
    logger.info(f"Client {client_id} Epoch {epoch} training completed in {duration:.2f}s")
    logger.info(f"  Avg Total Loss: {epoch_loss:.4f} (incl. Avg Prox Loss: {epoch_prox_loss:.4f})")
    logger.info(f"  Mean Tumor Dice: {epoch_dice.get('tumor_mean', 0.0):.4f}")

    # Return total average loss and dice dictionary
    return epoch_loss, epoch_dice


# Standard Evaluation Function (no prox term)
def evaluate(model, loader, device, epoch, save_vis=True, client_id="1"):
    """Evaluate the model (standard evaluation, no prox term)"""
    model.eval()
    running_loss = 0.0
    agg_dice_scores = {}
    num_batches = len(loader)
    start_time = time.time()
    vis_saved = False

    progress_bar = tqdm(loader, desc=f"Client {client_id} Evaluate Epoch {epoch}", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            patient_ids = batch['patient_id']
            slice_indices = batch['slice_idx']

            outputs = model(images)
            # Use standard combined loss for evaluation metric
            loss = combined_loss(outputs, masks)

            probs = F.softmax(outputs, dim=1)
            batch_dice = compute_dice_metrics(probs, masks)

            running_loss += loss.item()
            for k, v in batch_dice.items():
                agg_dice_scores[k] = agg_dice_scores.get(k, 0.0) + v

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{running_loss / (batch_idx + 1):.4f}",
                'dice': f"{batch_dice.get('tumor_mean', 0.0):.4f}"
            })

            if save_vis and not vis_saved and len(images) > 0 and save_visualisation_2d is not None:
                 try:
                     vis_path = save_visualisation_2d(
                         images[0], outputs[0], masks[0],
                         epoch, client_id, patient_ids[0], slice_indices[0].item(),
                         save_dir="./val_visualisations"
                     )
                     if vis_path: vis_saved = True
                 except Exception as e: logger.error(f"Error saving validation visualisation: {e}")

    val_loss = running_loss / num_batches
    val_dice = {k: v / num_batches for k, v in agg_dice_scores.items()}

    duration = time.time() - start_time
    logger.info(f"Client {client_id} Validation Epoch {epoch} completed in {duration:.2f}s")
    logger.info(f"Validation Loss: {val_loss:.4f}, Mean Tumor Dice: {val_dice.get('tumor_mean', 0.0):.4f}")

    return val_loss, val_dice


# Checkpoint Functions (remain the same, but maybe update suffix)
def save_checkpoint(model, optimizer, epoch, loss, dice, client_id, checkpoint_dir="./checkpoints"):
    """Save a model checkpoint"""
    checkpoint_dir = Path(checkpoint_dir) # Ensure it's a Path object
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filename = checkpoint_dir / f"client_{client_id}_epoch_{epoch}_fedprox_2d.pth"

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'dice': dice
    }
    try:
        torch.save(checkpoint, str(filename)) # Save path as string
        logger.info(f"Checkpoint saved to {filename}")
    except Exception as e: logger.error(f"Failed to save checkpoint {filename}: {e}")
    return str(filename)


def load_checkpoint(model, optimizer, filename):
    """Load a model checkpoint"""
    ckpt_path = Path(filename)
    if not ckpt_path.exists():
        logger.warning(f"Checkpoint {filename} does not exist")
        return None, None, None # epoch, loss, dice

    try:
        checkpoint = torch.load(str(ckpt_path), map_location='cpu') # Load to CPU first
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) # Use strict=False for flexibility
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 logger.info(f"Loaded optimiser state from {filename}")
            except ValueError as e:
                 logger.warning(f"Could not load optimiser state from {filename}, possibly due to parameter mismatch: {e}. Skipping optimiser load.")
        elif optimizer is not None:
             logger.warning(f"Optimiser state not found in checkpoint {filename}, optimiser not loaded.")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', None)
        dice = checkpoint.get('dice', None)
        logger.info(f"Loaded model state from checkpoint {filename} (Epoch: {epoch})")
        return epoch, loss, dice
    except Exception as e:
        logger.error(f"Failed to load checkpoint {filename}: {e}", exc_info=True)
        return None, None, None
