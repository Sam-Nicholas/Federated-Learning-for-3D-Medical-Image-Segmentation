import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import nibabel as nib
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger("dataset")


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing BraTS 3D MRI scans.
    Handles data partitioning for federated learning scenarios.
    """
    def __init__(self, data_dir, client_id=None, num_clients=1, is_train=True, transform=None):
        """
        Initialises the dataset.

        Args:
            data_dir (str): Path to the directory containing patient subdirectories.
            client_id (int, optional): ID of the current client (0 to num_clients-1) for data partitioning. Defaults to None (non-federated).
            num_clients (int, optional): Total number of clients for partitioning. Defaults to 1.
            is_train (bool, optional): If True, uses the training split; otherwise, uses the validation split. Defaults to True.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        self.modalities = ['t1n', 't1c', 't2w', 't2f'] # T1, T1ce, T2, FLAIR


        try:
            patient_dirs_all = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        except FileNotFoundError:
            logger.error(f"Data directory not found: {data_dir}")
            raise
        if not patient_dirs_all:
            logger.error(f"No patient subdirectories found in {data_dir}")
            raise FileNotFoundError(f"No patient data found in {data_dir}")

        logger.info(f"Found {len(patient_dirs_all)} patients in {data_dir}")

        # Split patient directories into training and validation sets (80/20 split)
        # Use a fixed random state for reproducible splits
        train_dirs, val_dirs = train_test_split(patient_dirs_all, test_size=0.2, random_state=42)

        # Select the appropriate list of directories based on is_train flag
        self.patient_dirs = train_dirs if is_train else val_dirs

        # Partition the selected directories if running in a federated setting
        if client_id is not None and num_clients > 1:
            num_patients = len(self.patient_dirs)
            # Basic partitioning: divide patients equally among clients
            patients_per_client = num_patients // num_clients
            start_idx = int(client_id) * patients_per_client
            # Ensure the last client gets any remaining patients
            end_idx = start_idx + patients_per_client if int(client_id) < num_clients - 1 else num_patients

            self.patient_dirs = self.patient_dirs[start_idx:end_idx]
            logger.info(
                f"Client {client_id}: Assigned {len(self.patient_dirs)} {('training' if is_train else 'validation')} patients (indices {start_idx}-{end_idx})")
        elif client_id is not None:
             logger.info(f"Client {client_id}: Using all {len(self.patient_dirs)} available {('training' if is_train else 'validation')} patients (num_clients=1).")
        else:
             logger.info(f"Using {len(self.patient_dirs)} {('training' if is_train else 'validation')} patients (non-federated).")


    def __len__(self):
        """Returns the number of patients assigned to this dataset instance."""
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        """
        Loads and returns a single sample (multi-modal image and segmentation mask)
        for the patient at the given index.
        """
        patient_id = self.patient_dirs[idx]
        patient_dir_path = self.data_dir / patient_id

        modality_images = []
        img_shape = None # Store shape for mask creation if needed

        # Load each required modality for the patient
        for modality in self.modalities:
            try:
                # Find the NIfTI file corresponding to the current modality
                modality_file = next(patient_dir_path.glob(f'*{modality}*.nii.gz'))
                nib_img = nib.load(modality_file)
                img_data = nib_img.get_fdata()

                if img_shape is None:
                    img_shape = img_data.shape
                elif img_shape != img_data.shape:
                    # Basic check for consistent shapes across modalities
                    logger.warning(f"Inconsistent shape for modality {modality} in patient {patient_id}. Expected {img_shape}, got {img_data.shape}. Skipping patient.")
                    # Return None or raise error? Returning None might be handled by collate_fn
                    return None # Or handle differently

                # Normalise image data to [0, 1] range
                img_data_normalised = self._normalise(img_data)
                modality_images.append(img_data_normalised)
            except StopIteration:
                logger.error(f"Modality '{modality}' file not found for patient {patient_id} in {patient_dir_path}. Skipping patient.")
                return None # Or handle differently
            except Exception as e:
                logger.error(f"Error loading modality '{modality}' for patient {patient_id}: {e}. Skipping patient.")
                return None # Or handle differently

        # Stack modalities along the channel dimension (C, D, H, W)
        image = np.stack(modality_images)

        # Load the segmentation mask
        try:
            seg_file = next(patient_dir_path.glob('*seg*.nii.gz'))
            seg_img = nib.load(seg_file)
            seg_data = seg_img.get_fdata()

            if img_shape != seg_data.shape:
                 logger.warning(f"Segmentation mask shape mismatch for patient {patient_id}. Image: {img_shape}, Mask: {seg_data.shape}. Skipping patient.")
                 return None

            # Convert segmentation labels to one-hot encoding
            # BraTS classes: 0: Background, 1: Necrotic/Non-Enhancing Tumour, 2: Edema, 4: Enhancing Tumour
            # Target format: [Background, Edema, Non-Enhancing, Enhancing]
            mask = np.zeros((4, *seg_data.shape), dtype=np.float32)
            mask[0, seg_data == 0] = 1  # Background
            mask[1, seg_data == 2] = 1  # Edema (Label 2 -> Index 1)
            mask[2, seg_data == 1] = 1  # Non-enhancing tumour (Label 1 -> Index 2)
            mask[3, seg_data == 4] = 1  # Enhancing tumour (Label 4 -> Index 3)

        except StopIteration:
            # If no segmentation file is found, create an empty mask (all background)
            logger.warning(f"No segmentation file found for patient {patient_id}. Creating empty mask.")
            if img_shape is None:
                logger.error(f"Cannot create empty mask for patient {patient_id} as image shape is unknown.")
                return None
            mask = np.zeros((4, *img_shape), dtype=np.float32)
            mask[0] = 1 # Assign all voxels to the background class
        except Exception as e:
             logger.error(f"Error loading segmentation for patient {patient_id}: {e}. Skipping patient.")
             return None

        # Convert numpy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # Apply any custom transformations if provided
        if self.transform:
            # Assuming transform takes image and mask and returns transformed versions
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        sample = {
            'image': image_tensor,
            'mask': mask_tensor,
            'patient_id': patient_id # Include patient ID for reference/logging
        }

        return sample

    def _normalise(self, img):
        """Performs min-max normalisation on the input image array."""
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val > min_val:
            # Normalise to [0, 1]
            return (img - min_val) / (max_val - min_val)
        elif max_val == min_val and max_val != 0:
             # Handle case where all values are the same but non-zero
             return np.ones_like(img)
        else:
            # Handle case where all values are zero
            return np.zeros_like(img)


def create_dataloaders(data_dir, val_data_dir=None, client_id=None, num_clients=1, batch_size=2, max_samples=None):
    """
    Creates PyTorch DataLoaders for training and validation datasets.
    Optionally limits the number of samples used, especially for faster experimentation.

    Args:
        data_dir (str): Path to the main data directory (used for training and potentially validation).
        val_data_dir (str, optional): Path to a separate validation data directory. If None, validation data is split from data_dir.
        client_id (int, optional): Client ID for federated partitioning.
        num_clients (int, optional): Total number of clients.
        batch_size (int, optional): Batch size for the DataLoaders. Defaults to 2.
        max_samples (int, optional): Maximum number of training samples to use per client. If None, uses all available samples. Defaults to None.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create the training dataset instance
    train_dataset_full = BraTSDataset(data_dir, client_id, num_clients, is_train=True)

    # Create the validation dataset instance
    if val_data_dir and Path(val_data_dir).exists():
        logger.info(f"Using separate validation dataset from: {val_data_dir}")
        val_dataset_full = BraTSDataset(val_data_dir, client_id, num_clients, is_train=False)
    else:
        logger.info("Using validation split derived from the main data directory.")
        # Create validation set from the same source dir, but with is_train=False
        val_dataset_full = BraTSDataset(data_dir, client_id, num_clients, is_train=False)

    train_dataset = train_dataset_full
    val_dataset = val_dataset_full

    # Limit the number of training samples if max_samples is specified
    if max_samples is not None and len(train_dataset_full) > max_samples:
        logger.info(f"Limiting training samples from {len(train_dataset_full)} to {max_samples} for client {client_id}.")
        # Use a fixed seed based on client_id for reproducible subsets
        rng = np.random.default_rng(seed=42 + int(client_id) if client_id is not None else 42)
        indices = rng.choice(len(train_dataset_full), max_samples, replace=False)
        train_dataset = Subset(train_dataset_full, indices)
    elif max_samples is not None:
         logger.info(f"Client {client_id} has {len(train_dataset_full)} training samples, which is not more than max_samples ({max_samples}). Using all.")


    # Optionally limit validation samples (e.g., for faster validation during training)
    # Calculate a proportional limit, ensuring at least a few samples
    val_max_samples = max(3, max_samples // 5) if max_samples is not None else None
    if val_max_samples is not None and len(val_dataset_full) > val_max_samples:
        logger.info(f"Limiting validation samples from {len(val_dataset_full)} to {val_max_samples} for client {client_id}.")
        # Use a different fixed seed for validation subset selection
        rng_val = np.random.default_rng(seed=100 + int(client_id) if client_id is not None else 100)
        indices_val = rng_val.choice(len(val_dataset_full), val_max_samples, replace=False)
        val_dataset = Subset(val_dataset_full, indices_val)
    elif val_max_samples is not None:
         logger.info(f"Client {client_id} has {len(val_dataset_full)} validation samples, which is not more than the limit ({val_max_samples}). Using all.")


    # Create DataLoaders
    # Consider adding a collate_fn to handle potential None values from __getitem__ if errors occur
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    logger.info(f"Created DataLoaders for client {client_id}: "
                f"{len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    return train_loader, val_loader


def save_visualisation(image, pred_mask_probs, true_mask, epoch, client_id, patient_id, save_dir="./visualisations"):
    """
    Saves a visual comparison of a central slice from the input image (T1ce),
    the predicted segmentation overlay, and the ground truth segmentation overlay.

    Args:
        image (torch.Tensor): Input image tensor (C, D, H, W), expects 4 modalities.
        pred_mask_probs (torch.Tensor): Predicted segmentation probability map (N_Class, D, H, W) after softmax.
        true_mask (torch.Tensor): Ground truth one-hot encoded mask (N_Class, D, H, W).
        epoch (int): Current epoch number (for filename).
        client_id (str/int): Client identifier (for filename/directory).
        patient_id (str): Patient identifier (for filename).
        save_dir (str): Base directory to save the visualisation plots.
    """
    save_dir_path = Path(save_dir)
    client_save_dir = save_dir_path / f"client_{client_id}"
    client_save_dir.mkdir(parents=True, exist_ok=True)

    # Move tensors to CPU and convert to NumPy arrays
    image_np = image.cpu().numpy()
    pred_probs_np = pred_mask_probs.cpu().numpy()
    true_mask_np = true_mask.cpu().numpy()

    # Determine the index for the T1ce modality (assuming standard order)
    t1c_idx = 1 # Index 1 corresponds to 't1c' in self.modalities

    # Find a representative slice: prefer a slice containing tumour, otherwise use the middle slice.
    # Sum tumour class masks (Edema, Non-Enhancing, Enhancing) along the depth axis
    # Indices [1, 2, 3] correspond to Edema, Non-Enhancing, Enhancing
    tumour_presence_per_slice = np.sum(true_mask_np[1:], axis=(0, 2, 3)) # Sum over C(tumour), H, W
    slices_with_tumour = np.where(tumour_presence_per_slice > 0)[0]

    if len(slices_with_tumour) > 0:
        # If tumour is present, select the middle slice among those containing tumour
        middle_slice_idx = slices_with_tumour[len(slices_with_tumour) // 2]
    else:
        # If no tumour is found in the ground truth, use the geometric middle slice
        middle_slice_idx = image_np.shape[1] // 2 # Depth is the second dimension (C, D, H, W)

    # --- Prepare data for plotting ---
    # Base image slice (T1ce)
    base_image_slice = image_np[t1c_idx, middle_slice_idx]

    # Predicted segmentation slice (convert probabilities to class labels)
    pred_classes_slice = np.argmax(pred_probs_np[:, middle_slice_idx], axis=0) # Argmax along class dim
    pred_overlay = np.zeros((*pred_classes_slice.shape, 4)) # RGBA overlay
    # Colour mapping: Yellow (Edema), Red (Non-Enhancing), Blue (Enhancing)
    pred_overlay[pred_classes_slice == 1, :] = [1, 1, 0, 0.5] # Edema (Index 1)
    pred_overlay[pred_classes_slice == 2, :] = [1, 0, 0, 0.5] # Non-Enhancing (Index 2)
    pred_overlay[pred_classes_slice == 3, :] = [0, 0, 1, 0.5] # Enhancing (Index 3)

    # True segmentation slice (convert one-hot to class labels for overlay)
    true_classes_slice = np.argmax(true_mask_np[:, middle_slice_idx], axis=0) # Argmax along class dim
    true_overlay = np.zeros((*true_classes_slice.shape, 4)) # RGBA overlay
    true_overlay[true_classes_slice == 1, :] = [1, 1, 0, 0.5] # Edema (Index 1)
    true_overlay[true_classes_slice == 2, :] = [1, 0, 0, 0.5] # Non-Enhancing (Index 2)
    true_overlay[true_classes_slice == 3, :] = [0, 0, 1, 0.5] # Enhancing (Index 3)

    # --- Create Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Patient: {patient_id}, Epoch: {epoch}, Slice: {middle_slice_idx}', fontsize=14)

    # Plot 1: Base T1ce Image
    axes[0].imshow(base_image_slice, cmap='gray')
    axes[0].set_title(f'Input T1ce')
    axes[0].axis('off')

    # Plot 2: Predicted Segmentation Overlay
    axes[1].imshow(base_image_slice, cmap='gray')
    axes[1].imshow(pred_overlay) # Overlay coloured mask
    axes[1].set_title('Predicted Segmentation')
    axes[1].axis('off')

    # Plot 3: True Segmentation Overlay
    axes[2].imshow(base_image_slice, cmap='gray')
    axes[2].imshow(true_overlay) # Overlay coloured mask
    axes[2].set_title('Ground Truth Segmentation')
    axes[2].axis('off')

    # Save the figure
    save_path = client_save_dir / f"epoch_{epoch}_patient_{patient_id}_slice_{middle_slice_idx}.png"
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.debug(f"Saved visualisation to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save visualisation {save_path}: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory

    return str(save_path)

