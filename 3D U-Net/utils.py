import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from model import UNet3D

logger = logging.getLogger("utils")

# --- Metrics and Loss Functions ---

def dice_coefficient(pred_binary: torch.Tensor, target_binary: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Computes the Dice Similarity Coefficient (DSC) between two binary tensors.
    Commonly used for evaluating segmentation performance.

    Args:
        pred_binary (torch.Tensor): Binary prediction tensor (0s and 1s).
        target_binary (torch.Tensor): Binary ground truth tensor (0s and 1s).
        smooth (float, optional): Smoothing factor to prevent division by zero. Defaults to 1e-5.

    Returns:
        float: The Dice coefficient, ranging from 0 (no overlap) to 1 (perfect overlap).
    """
    assert pred_binary.shape == target_binary.shape, "Prediction and target tensors must have the same shape."
    assert pred_binary.dtype == target_binary.dtype, "Prediction and target tensors must have the same dtype."

    # Ensure tensors are flattened and contiguous for efficient calculation
    pred_flat = pred_binary.contiguous().view(-1)
    target_flat = target_binary.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()

    # Calculate Dice score
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    return dice.item() # Return as a Python float

def dice_loss(pred_probs: torch.Tensor, target_one_hot: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """
    Computes the Dice loss based on predicted probabilities and one-hot encoded targets.
    Calculates Dice coefficient per class (excluding background) and returns 1 - mean Dice.

    Args:
        pred_probs (torch.Tensor): Predicted probability tensor (Batch, Classes, D, H, W). Assumes softmax applied.
        target_one_hot (torch.Tensor): Ground truth tensor in one-hot format (Batch, Classes, D, H, W).
        smooth (float, optional): Smoothing factor for Dice calculation. Defaults to 1e-5.

    Returns:
        torch.Tensor: The Dice loss (scalar tensor). Lower is better.
    """
    assert pred_probs.shape == target_one_hot.shape, "Prediction probabilities and target mask must have the same shape."
    num_classes = pred_probs.shape[1]
    assert num_classes > 1, "Dice loss requires at least 2 classes."

    dice_per_class = []
    # Iterate over classes, typically excluding the background class (index 0)
    for i in range(1, num_classes): # Exclude background class (index 0)
        pred_class = pred_probs[:, i] # Probabilities for current class
        target_class = target_one_hot[:, i] # Ground truth for current class
        # Note: dice_coefficient expects binary inputs, but works reasonably well with probabilities here
        # for loss calculation. For strict metric calculation, convert probs to binary first.
        dice_val = dice_coefficient(pred_class, target_class, smooth)
        dice_per_class.append(dice_val)

    # Calculate the mean Dice score across the relevant classes
    mean_dice = np.mean(dice_per_class) if dice_per_class else 0.0

    # Dice loss is 1 minus the mean Dice score
    return torch.tensor(1.0 - mean_dice, dtype=torch.float32)


def combined_loss(pred_logits: torch.Tensor, target_one_hot: torch.Tensor, alpha: float = 0.5, ce_weight=None) -> torch.Tensor:
    """
    Combines Cross-Entropy (CE) loss and Dice loss for segmentation tasks.
    Helps balance voxel-wise accuracy (CE) with overlap performance (Dice).

    Args:
        pred_logits (torch.Tensor): Raw output logits from the model (Batch, Classes, D, H, W).
        target_one_hot (torch.Tensor): Ground truth tensor in one-hot format (Batch, Classes, D, H, W).
        alpha (float, optional): Weighting factor for Dice loss (0 <= alpha <= 1).
                                 CE loss weight = (1 - alpha). Defaults to 0.5.
        ce_weight (torch.Tensor, optional): Class weights for Cross-Entropy loss. Defaults to None.


    Returns:
        torch.Tensor: The combined loss value (scalar tensor).
    """
    # --- Cross-Entropy Loss ---
    # CE expects logits and target class indices
    target_indices = torch.argmax(target_one_hot, dim=1) # Convert one-hot to class indices
    ce = F.cross_entropy(pred_logits, target_indices, weight=ce_weight)

    # --- Dice Loss ---
    # Dice loss expects probabilities
    pred_probs = F.softmax(pred_logits, dim=1)
    dice = dice_loss(pred_probs, target_one_hot) # Uses our dice_loss function

    # --- Combine Losses ---
    combined = alpha * dice + (1 - alpha) * ce
    return combined


def compute_dice_metrics(pred_probs: torch.Tensor, target_one_hot: torch.Tensor) -> Dict[str, float]:
    """
    Computes Dice scores for each individual class and the mean Dice score
    across the tumour classes (excluding background).

    Args:
        pred_probs (torch.Tensor): Predicted probability tensor (Batch, Classes, D, H, W) after softmax.
        target_one_hot (torch.Tensor): Ground truth tensor in one-hot format (Batch, Classes, D, H, W).

    Returns:
        Dict[str, float]: A dictionary containing Dice scores for 'background', 'oedema',
                          'non_enhancing', 'enhancing', and 'tumour_mean'.
    """
    assert pred_probs.shape == target_one_hot.shape, "Prediction and target must have the same shape."
    num_classes = pred_probs.shape[1]
    assert num_classes == 4, "Expected 4 classes for BraTS metrics."

    dice_scores = {}
    # Define class names corresponding to the channel indices
    # Assumes: 0: Background, 1: Oedema, 2: Non-Enhancing, 3: Enhancing
    classes = ['background', 'oedema', 'non_enhancing', 'enhancing'] # British spelling 'oedema'

    # Convert probabilities to binary predictions (one-hot format) using argmax
    pred_indices = pred_probs.argmax(dim=1) # Get class index with highest probability
    pred_one_hot = F.one_hot(pred_indices, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    # Shape: (Batch, Classes, D, H, W)

    # Compute Dice for each class individually
    for i, class_name in enumerate(classes):
        dice = dice_coefficient(pred_one_hot[:, i], target_one_hot[:, i])
        dice_scores[class_name] = dice

    # Compute the mean Dice score for the tumour classes only
    # Indices 1, 2, 3 correspond to oedema, non_enhancing, enhancing
    tumour_dice_values = [dice_scores[c] for c in classes[1:]] # Exclude background
    dice_scores['tumour_mean'] = np.mean(tumour_dice_values) if tumour_dice_values else 0.0

    return dice_scores

# --- Training and Evaluation Loops ---

def train_epoch(
    model: UNet3D, # Or torch.nn.Module
    loader: DataLoader,
    optimizer: Optimizer, # Keep torch class name
    device: torch.device,
    epoch: int,
    save_vis: bool = False,
    client_id: str = "unknown"
) -> Tuple[float, Dict[str, float]]:
    """
    Performs one epoch of training for the segmentation model.

    Args:
        model: The PyTorch model to train.
        loader: DataLoader for the training dataset.
        optimizer: The optimiser for updating model weights.
        device: The device (CPU or CUDA) to perform computations on.
        epoch (int): The current epoch number (for logging).
        save_vis (bool, optional): If True, saves one visualisation per epoch. Defaults to False.
        client_id (str, optional): Identifier for the client (for logging/saving). Defaults to "unknown".

    Returns:
        Tuple[float, Dict[str, float]]: Average training loss for the epoch, and dictionary of average Dice metrics.
    """
    model.train() # Set model to training mode
    running_loss = 0.0
    # Initialise dictionary to accumulate Dice scores per class
    accumulated_dice = { 'background': 0.0, 'oedema': 0.0, 'non_enhancing': 0.0, 'enhancing': 0.0, 'tumour_mean': 0.0 }
    num_batches = len(loader)
    start_time = time.time()

    vis_saved_this_epoch = False # Flag to save only one visualisation

    progress_bar = tqdm(loader, desc=f"Client {client_id} Train Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        # Ensure batch is valid (might be None if dataset loading failed)
        if batch is None:
            logger.warning(f"Skipping None batch {batch_idx+1}/{num_batches} in training epoch {epoch}.")
            continue

        images = batch['image'].to(device, non_blocking=True) # B, C, D, H, W
        masks = batch['mask'].to(device, non_blocking=True)   # B, N_Class, D, H, W (one-hot)
        patient_id = batch['patient_id'][0] if 'patient_id' in batch and batch['patient_id'] else "unknown_patient"

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images) # Logits: B, N_Class, D, H, W
        loss = combined_loss(outputs, masks) # Calculate combined loss

        # Backward pass and optimisation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate metrics (on non-gradient graph)
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            batch_dice = compute_dice_metrics(probs, masks)
            # Accumulate dice scores
            for key in accumulated_dice:
                accumulated_dice[key] += batch_dice.get(key, 0.0)

        # Update progress bar description
        progress_bar.set_postfix({
            'loss': f"{running_loss / (batch_idx + 1):.4f}",
            'dice_mean': f"{batch_dice.get('tumour_mean', 0.0):.4f}"
        })

        # Save one visualisation sample per epoch if enabled
        if save_vis and not vis_saved_this_epoch and batch_idx == 0: # Save first batch
            try:
                # Defer import to avoid circular dependency if utils is imported by dataset
                from dataset import save_visualisation
                save_visualisation(
                    images[0], probs[0], masks[0], # Use first item in batch
                    epoch, client_id, patient_id, save_dir="./train_visualisations"
                )
                vis_saved_this_epoch = True
            except ImportError:
                 logger.warning("Could not import save_visualisation function. Skipping visualisation.")
                 save_vis = False # Disable for rest of run if import fails
            except Exception as e:
                 logger.error(f"Error saving training visualisation: {e}")


    # Calculate average metrics for the epoch
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = {k: v / num_batches for k, v in accumulated_dice.items()} if num_batches > 0 else accumulated_dice

    duration = time.time() - start_time
    logger.info(f"Client {client_id} Epoch {epoch} Training Summary: "
                f"Duration={duration:.2f}s, Avg Loss={avg_loss:.4f}, "
                f"Avg Dice (Tumour Mean)={avg_dice.get('tumour_mean', 0.0):.4f}")

    return avg_loss, avg_dice


def evaluate(
    model: UNet3D, # Or torch.nn.Module
    loader: DataLoader,
    device: torch.device,
    epoch: int, # Current epoch context for logging/saving
    save_vis: bool = False,
    client_id: str = "unknown"
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates the segmentation model on a given dataset (typically validation).

    Args:
        model: The PyTorch model to evaluate.
        loader: DataLoader for the evaluation dataset.
        device: The device (CPU or CUDA) to perform computations on.
        epoch (int): The current epoch number (for context in logging/saving).
        save_vis (bool, optional): If True, saves one visualisation. Defaults to False.
        client_id (str, optional): Identifier for the client (for logging/saving). Defaults to "unknown".

    Returns:
        Tuple[float, Dict[str, float]]: Average validation loss, and dictionary of average Dice metrics.
    """
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    accumulated_dice = { 'background': 0.0, 'oedema': 0.0, 'non_enhancing': 0.0, 'enhancing': 0.0, 'tumour_mean': 0.0 }
    num_batches = len(loader)
    start_time = time.time()

    vis_saved = False # Flag to save only one visualisation

    with torch.no_grad(): # Disable gradient calculations
        progress_bar = tqdm(loader, desc=f"Client {client_id} Evaluate Epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                logger.warning(f"Skipping None batch {batch_idx+1}/{num_batches} in evaluation epoch {epoch}.")
                continue

            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            patient_id = batch['patient_id'][0] if 'patient_id' in batch and batch['patient_id'] else "unknown_patient"

            outputs = model(images) # Logits
            loss = combined_loss(outputs, masks) # Calculate loss

            running_loss += loss.item()

            # Calculate metrics
            probs = F.softmax(outputs, dim=1)
            batch_dice = compute_dice_metrics(probs, masks)
            for key in accumulated_dice:
                accumulated_dice[key] += batch_dice.get(key, 0.0)

             # Update progress bar description
            progress_bar.set_postfix({
                'loss': f"{running_loss / (batch_idx + 1):.4f}",
                'dice_mean': f"{batch_dice.get('tumour_mean', 0.0):.4f}"
            })

            # Save one visualisation sample if enabled
            if save_vis and not vis_saved and batch_idx == 0: # Save first batch
                try:
                    # Defer import
                    from dataset import save_visualisation
                    save_visualisation(
                        images[0], probs[0], masks[0],
                        epoch, client_id, patient_id, save_dir="./val_visualisations"
                    )
                    vis_saved = True
                except ImportError:
                     logger.warning("Could not import save_visualisation function. Skipping visualisation.")
                     save_vis = False
                except Exception as e:
                     logger.error(f"Error saving validation visualisation: {e}")


    # Calculate average metrics for the evaluation run
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = {k: v / num_batches for k, v in accumulated_dice.items()} if num_batches > 0 else accumulated_dice

    duration = time.time() - start_time
    logger.info(f"Client {client_id} Epoch {epoch} Evaluation Summary: "
                f"Duration={duration:.2f}s, Avg Loss={avg_loss:.4f}, "
                f"Avg Dice (Tumour Mean)={avg_dice.get('tumour_mean', 0.0):.4f}")

    return avg_loss, avg_dice


# --- Checkpointing ---

def save_checkpoint(
    model: UNet3D, # Or torch.nn.Module
    optimizer: Optimizer, # Keep torch class name
    epoch: int,
    loss: float,
    dice: Dict[str, float], # Store the whole dice dict
    client_id: str,
    checkpoint_dir: str = "./checkpoints"
) -> str:
    """
    Saves a training checkpoint containing model state, optimiser state, epoch, loss, and dice metrics.

    Args:
        model: The model instance.
        optimizer: The optimiser instance.
        epoch (int): The epoch number completed.
        loss (float): The training loss at this epoch.
        dice (Dict[str, float]): The dice metrics at this epoch.
        client_id (str): Identifier of the client saving the checkpoint.
        checkpoint_dir (str, optional): Directory to save the checkpoint file. Defaults to "./checkpoints".

    Returns:
        str: The path to the saved checkpoint file.
    """
    save_path = Path(checkpoint_dir)
    save_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    filename = save_path / f"client_{client_id}_epoch_{epoch}.pth"

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), # Keep original key name for compatibility
        'loss': loss, # Store training loss
        'dice_metrics': dice # Store full dice metrics dictionary
    }

    try:
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved for client {client_id}, epoch {epoch} to {filename}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint {filename}: {e}")

    return str(filename)


def load_checkpoint(model: UNet3D, optimizer: Optimizer, filename: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Loads model and optimiser state from a checkpoint file.

    Args:
        model: The model instance to load state into.
        optimizer: The optimiser instance to load state into.
        filename (str): Path to the checkpoint file.

    Returns:
        Tuple[Optional[int], Optional[float]]: The epoch and loss stored in the checkpoint, or (None, None) if loading fails.
    """
    checkpoint_path = Path(filename)
    if not checkpoint_path.is_file():
        logger.warning(f"Checkpoint file not found: {filename}")
        return None, None

    try:
        # Load checkpoint; weights_only=False is safer if non-tensor data might be present
        # Use map_location to ensure tensors are loaded onto the correct device model expects
        map_location = next(model.parameters()).device
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)

        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
             optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Keep original key name
        else:
             logger.warning(f"Optimiser state not found or optimiser not provided for checkpoint {filename}.")


        epoch = checkpoint.get('epoch', -1) # Default to -1 if epoch not found
        loss = checkpoint.get('loss', None)
        dice_metrics = checkpoint.get('dice_metrics', {})

        logger.info(f"Successfully loaded checkpoint from {filename} (Epoch: {epoch}, Loss: {loss}, DiceMean: {dice_metrics.get('tumour_mean', 'N/A'):.4f})")

        # Return epoch and loss for potential resumption logic
        return epoch, loss

    except KeyError as e:
         logger.error(f"Missing key in checkpoint file {filename}: {e}")
         return None, None
    except Exception as e:
        logger.error(f"Error loading checkpoint from {filename}: {e}")
        return None, None
