import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
from tqdm import tqdm
from pathlib import Path
from dataset import save_visualisation_2d

logger = logging.getLogger("utils")


def dice_coefficient(pred, target, smooth=1e-5):
    """
    Compute the Dice coefficient between prediction and target.
    Works for both 2D and 3D by flattening tensors.

    Args:
        pred: Prediction tensor (binary or probabilities for one class).
        target: Target tensor (binary ground truth for one class).
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice coefficient (float).
    """
    # Ensure tensors are contiguous in memory and flatten them
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    # Use dim=0 for sum on flattened tensors
    union = pred.sum(dim=0) + target.sum(dim=0)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item() # Return as Python float

def dice_loss(pred, target, smooth=1e-5):
    """
    Compute the Dice loss (1 - Dice coefficient) averaged over classes.
    Expects pred shape (B, C, H, W) and target shape (B, C, H, W) (one-hot).

    Args:
        pred: Prediction tensor (logits or probabilities).
        target: Target tensor (one-hot encoded).
        smooth: Smoothing factor.

    Returns:
        Dice loss (float).
    """
    # Compute softmax over the classes dimension (C)
    pred = F.softmax(pred, dim=1)

    # Compute per-class Dice coefficients
    dice_per_class = []
    num_classes = pred.shape[1]

    for i in range(num_classes):  # Iterate over classes
        # Calculate Dice for each class across the batch
        dice_class = dice_coefficient(pred[:, i], target[:, i], smooth)
        dice_per_class.append(dice_class)


    if num_classes > 1:
         # Average Dice score for foreground classes
        mean_fg_dice = np.mean(dice_per_class[1:])
        return 1.0 - mean_fg_dice
    else:
        # If only one class, return 1 - Dice of that class
        return 1.0 - np.mean(dice_per_class)


def combined_loss(pred, target, alpha=0.5, ce_weight=None):
    """
    Combine cross-entropy and Dice loss for 2D segmentation.
    Expects pred shape (B, C, H, W) and target shape (B, C, H, W) (one-hot).

    Args:
        pred: Prediction tensor (logits).
        target: Target tensor (one-hot encoded).
        alpha: Weight for Dice loss (1-alpha is weight for CE loss).
        ce_weight: Optional weights for cross-entropy classes.

    Returns:
        Combined loss tensor.
    """
    # Cross-entropy loss expects class indices as target (B, H, W)
    # Convert one-hot target back to class indices
    target_indices = target.argmax(dim=1) # Shape: (B, H, W)

    # Compute cross-entropy loss
    ce = F.cross_entropy(pred, target_indices, weight=ce_weight)

    # Compute Dice loss (expects probabilities and one-hot target)
    dice = dice_loss(pred, target) # dice_loss applies softmax internally

    # Combine losses
    return alpha * dice + (1 - alpha) * ce


def compute_dice_metrics(pred, target):
    """
    Compute Dice scores for each class for a batch of 2D slices.

    Args:
        pred: Prediction tensor after softmax (B, C, H, W).
        target: Target tensor (B, C, H, W) (one-hot).

    Returns:
        Dictionary with Dice scores for each class and mean over tumor classes.
    """
    dice_scores = {}
    # Standard BraTS classes (adjust if different)
    classes = ['background', 'edema', 'non_enhancing', 'enhancing']
    num_classes = pred.shape[1]

    # Convert prediction probabilities to one-hot class predictions
    pred_classes = pred.argmax(dim=1)  # Shape: (B, H, W)
    # Create one-hot tensor from predicted classes
    pred_one_hot = F.one_hot(pred_classes, num_classes=num_classes).permute(0, 3, 1, 2).float() # Shape: (B, C, H, W)


    # Compute Dice for each class
    for i in range(num_classes):
        class_name = classes[i] if i < len(classes) else f'class_{i}'
        # Calculate Dice for this class across the whole batch
        dice = dice_coefficient(pred_one_hot[:, i], target[:, i])
        dice_scores[class_name] = dice

    # Compute mean Dice (excluding background - class 0)
    if num_classes > 1:
        tumor_dice_scores = [dice_scores[classes[i]] for i in range(1, min(num_classes, len(classes)))]
        # Handle cases where there might be no tumor classes present in the batch for target
        if tumor_dice_scores:
             dice_scores['tumor_mean'] = np.mean(tumor_dice_scores)
        else:
             dice_scores['tumor_mean'] = 0.0 # Or None, or handle as appropriate
    else:
        dice_scores['tumor_mean'] = dice_scores[classes[0]] # If only one class, it's the mean

    return dice_scores


def train_epoch(model, loader, optimizer, device, epoch, save_vis=True, client_id="1"):
    """Train for one epoch using 2D slices"""
    model.train()
    running_loss = 0.0
    # Initialize aggregate dice scores dictionary
    agg_dice_scores = {}
    num_batches = len(loader)
    start_time = time.time()

    # Visualisation flag - save only one visualisation per epoch
    vis_saved = False

    progress_bar = tqdm(loader, desc=f"Client {client_id} Train Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device) # Shape: (B, C, H, W)
        masks = batch['mask'].to(device)   # Shape: (B, n_classes, H, W)
        patient_ids = batch['patient_id'] # List of patient IDs in the batch
        slice_indices = batch['slice_idx'] # List of slice indices

        optimizer.zero_grad()
        outputs = model(images) # Shape: (B, n_classes, H, W)
        loss = combined_loss(outputs, masks) # Use combined loss

        loss.backward()
        optimizer.step()

        # Metrics Calculation
        with torch.no_grad():
            # Apply softmax to get probabilities for metrics calculation
            probs = F.softmax(outputs, dim=1)
            # Compute dice scores for this batch
            batch_dice = compute_dice_metrics(probs, masks)

        # Accumulate Metrics
        running_loss += loss.item()
        # Accumulate dice scores correctly
        for k, v in batch_dice.items():
            agg_dice_scores[k] = agg_dice_scores.get(k, 0.0) + v

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}", # Current batch loss
            'avg_loss': f"{running_loss / (batch_idx + 1):.4f}", # Average loss so far
            'dice': f"{batch_dice.get('tumor_mean', 0.0):.4f}" # Current batch mean tumor dice
        })

        # Save Visualisation
        if save_vis and not vis_saved and len(images) > 0:
            try:
                vis_path = save_visualisation_2d(
                    images[0], outputs[0], masks[0], # Use logits for pred_mask
                    epoch, client_id, patient_ids[0], slice_indices[0].item() # Get first item info
                )
                vis_saved = True
            except Exception as e:
                logger.error(f"Error saving training visualisation: {e}")


    # Compute average metrics for the epoch
    epoch_loss = running_loss / num_batches
    # Average the accumulated dice scores
    epoch_dice = {k: v / num_batches for k, v in agg_dice_scores.items()}

    # Log results
    duration = time.time() - start_time
    logger.info(f"Client {client_id} Epoch {epoch} training completed in {duration:.2f}s")
    # Use .get for safety in case a metric wasn't computed
    logger.info(f"Training Loss: {epoch_loss:.4f}, Mean Tumor Dice: {epoch_dice.get('tumor_mean', 0.0):.4f}")

    return epoch_loss, epoch_dice

def evaluate(model, loader, device, epoch, save_vis=True, client_id="1"):
    """Evaluate the model using 2D slices"""
    model.eval()
    running_loss = 0.0
    agg_dice_scores = {}
    num_batches = len(loader)
    start_time = time.time()

    # Visualisation flag - save only one visualisation per evaluation
    vis_saved = False

    progress_bar = tqdm(loader, desc=f"Client {client_id} Evaluate Epoch {epoch}", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device) # Shape: (B, C, H, W)
            masks = batch['mask'].to(device)   # Shape: (B, n_classes, H, W)
            patient_ids = batch['patient_id']
            slice_indices = batch['slice_idx']

            outputs = model(images) # Shape: (B, n_classes, H, W)
            loss = combined_loss(outputs, masks) # Use combined loss


            probs = F.softmax(outputs, dim=1)
            batch_dice = compute_dice_metrics(probs, masks)


            running_loss += loss.item()
            for k, v in batch_dice.items():
                agg_dice_scores[k] = agg_dice_scores.get(k, 0.0) + v

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{running_loss / (batch_idx + 1):.4f}",
                'dice': f"{batch_dice.get('tumor_mean', 0.0):.4f}"
            })



            if save_vis and not vis_saved and len(images) > 0:
                 try:
                     vis_path = save_visualisation_2d(
                         images[0], outputs[0], masks[0], # Use logits for pred_mask
                         epoch, client_id, patient_ids[0], slice_indices[0].item(),
                         save_dir="./val_visualisations" # Specify validation save directory
                     )
                     vis_saved = True
                 except Exception as e:
                      logger.error(f"Error saving validation visualisation: {e}")


    # Compute average metrics
    val_loss = running_loss / num_batches
    val_dice = {k: v / num_batches for k, v in agg_dice_scores.items()}

    # Log results
    duration = time.time() - start_time
    logger.info(f"Client {client_id} Validation Epoch {epoch} completed in {duration:.2f}s")
    logger.info(f"Validation Loss: {val_loss:.4f}, Mean Tumor Dice: {val_dice.get('tumor_mean', 0.0):.4f}")

    return val_loss, val_dice


def save_checkpoint(model, optimizer, epoch, loss, dice, client_id, checkpoint_dir="./checkpoints"):
    """Save a model checkpoint (unchanged, works for 2D model)"""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{checkpoint_dir}/client_{client_id}_epoch_{epoch}_2d.pth" # Add _2d suffix

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'dice': dice # Store the whole dice dictionary
    }

    try:
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint {filename}: {e}")

    return filename


def load_checkpoint(model, optimizer, filename):
    """Load a model checkpoint (unchanged, works for 2D model)"""
    if not Path(filename).exists():
        logger.warning(f"Checkpoint {filename} does not exist")
        return None, None, None # Return epoch, loss, dice

    try:
        # Load checkpoint onto the CPU first to avoid GPU memory issues
        # weights_only=False is deprecated, use map_location='cpu'
        checkpoint = torch.load(filename, map_location='cpu')

        # Load model state dict
        # Add strict=False if loading weights from a slightly different architecture
        # or if loading a global model without optimizer state
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Load optimizer state dict if it exists in the checkpoint
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded optimizer state from {filename}")
        elif optimizer is not None:
             logger.warning(f"Optimizer state not found in checkpoint {filename}, optimizer not loaded.")


        epoch = checkpoint.get('epoch', 0) # Default to 0 if not found
        loss = checkpoint.get('loss', None)
        dice = checkpoint.get('dice', None) # Load dice dict if available

        logger.info(f"Loaded model state from checkpoint {filename} (Epoch: {epoch})")

        return epoch, loss, dice

    except Exception as e:
        logger.error(f"Failed to load checkpoint {filename}: {e}")
        return None, None, None
