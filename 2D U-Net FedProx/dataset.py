import os
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
import random
import pickle
import numbers

logger = logging.getLogger("dataset")


class BraTSDataset2D(Dataset):
    def __init__(self, data_dir, client_id=None, num_clients=1, is_train=True, transform=None, slices_per_volume=None, require_tumor=False, n_classes=4):
        """
        BraTS dataset loader for 2D slices with slice map caching.

        Args:
            data_dir: Path to BraTS dataset directory.
            client_id: ID of the client for federated learning (None for non-federated).
            num_clients: Total number of clients for data partitioning.
            is_train: Whether to use training or validation split.
            transform: Additional transformations to apply (applied slice-wise).
            slices_per_volume: Number of slices to sample from each volume (None to use all).
            require_tumor: If True, only sample slices containing tumor pixels (in train mode).
            n_classes: Number of output classes for segmentation. Default 4 for BraTS.
        """
        self.data_dir = Path(data_dir) # Ensure data_dir is a Path object
        self.transform = transform
        self.is_train = is_train
        self.modalities = ['t1n', 't1c', 't2w', 't2f']
        self.slices_per_volume = slices_per_volume
        self.require_tumor = require_tumor and is_train
        self.n_classes = n_classes
        self.client_id = client_id
        self.num_clients = num_clients

        # --- Patient ID Assignment ---
        # Use Pathlib for robust directory listing
        all_patient_dirs = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        if not all_patient_dirs:
             raise FileNotFoundError(f"No patient directories found in {self.data_dir}")
        logger.info(f"Found {len(all_patient_dirs)} patients in {self.data_dir}")

        # Split patient IDs for train/validation
        train_patient_ids, val_patient_ids = train_test_split(all_patient_dirs, test_size=0.2, random_state=42)
        self.patient_ids = train_patient_ids if is_train else val_patient_ids

        # Partition patient IDs for the current client in federated learning
        if client_id is not None and num_clients > 1:
            num_patients = len(self.patient_ids)
            patients_per_client = num_patients // num_clients
            start_idx = int(client_id) * patients_per_client
            end_idx = start_idx + patients_per_client if int(client_id) < num_clients - 1 else num_patients
            self.patient_ids = self.patient_ids[start_idx:end_idx]
            logger.info(
                f"Client {client_id}: Assigned {len(self.patient_ids)} patients (indices {start_idx}-{end_idx})")
        # --- End Patient ID Assignment ---


        # --- Slice Map Loading/Generation ---
        # Define a unique filename for the slice map cache based on configuration
        split_name = "train" if self.is_train else "val"
        # Handle None case for slices_per_volume in filename
        sampling_info = f"spv{self.slices_per_volume}" if self.slices_per_volume is not None else "spvAll"
        tumor_req_info = "reqT" if self.require_tumor else "noReqT"
        # Ensure client_id and num_clients are part of the filename for FL partitioning
        client_info = f"c{self.client_id}of{self.num_clients}" if self.client_id is not None else "allClients"

        cache_dir = Path("./slice_map_cache") # Directory to store cache files
        cache_dir.mkdir(parents=True, exist_ok=True) # Create cache directory if it doesn't exist

        # Construct the full cache file path
        cache_filename = cache_dir / f"slice_map_{split_name}_{client_info}_{sampling_info}_{tumor_req_info}.pkl"

        try:
            if cache_filename.exists():
                # If cache file exists, load it
                logger.info(f"Attempting to load slice map from cache: {cache_filename}")
                with open(cache_filename, 'rb') as f:
                    self.slice_map = pickle.load(f)
                logger.info(f"Successfully loaded slice map from cache ({len(self.slice_map)} slices).")
            else:
                # If cache file doesn't exist, generate the map
                logger.info(f"Slice map cache not found at {cache_filename}. Generating...")
                self.slice_map = self._create_slice_map()
                # Save the newly generated map to the cache file
                logger.info(f"Attempting to save newly generated slice map ({len(self.slice_map)} slices) to cache: {cache_filename}")
                with open(cache_filename, 'wb') as f:
                    pickle.dump(self.slice_map, f)
                logger.info(f"Saved slice map to cache.")

        except (pickle.UnpicklingError, EOFError, FileNotFoundError, OSError, AttributeError) as e:
             # Handle potential errors during loading/saving (e.g., corrupted file, permission issues)
             logger.warning(f"Error handling slice map cache file {cache_filename}: {e}. Regenerating...")
             self.slice_map = self._create_slice_map()
             # Attempt to save again, catching potential errors
             try:
                 logger.info(f"Attempting to save regenerated slice map ({len(self.slice_map)} slices) to cache: {cache_filename}")
                 with open(cache_filename, 'wb') as f:
                     pickle.dump(self.slice_map, f)
                 logger.info(f"Saved regenerated slice map to cache.")
             except OSError as save_e:
                 logger.error(f"Could not save regenerated slice map cache {cache_filename}: {save_e}")
        # --- End Slice Map Loading/Generation ---

        if not hasattr(self, 'slice_map') or not self.slice_map: # Check if slice_map exists and is not empty
             logger.warning(f"Client {client_id} ({split_name}): Slice map is empty or could not be loaded/generated. Check data directory and partitioning.")
             # Initialise as empty list to prevent errors later
             self.slice_map = []
        else:
             logger.info(f"Client {client_id} ({split_name}): Dataset initialised with {len(self.slice_map)} slices.")


    def _create_slice_map(self):
        """ Creates a list of tuples (patient_id, slice_idx) for indexing. """
        slice_map = []
        logger.info(f"Generating slice map for {len(self.patient_ids)} assigned patients...")
        # Set a seed for random sampling *within* this function for reproducibility
        # Use a seed based on client_id for different sampling per client if desired
        if self.client_id is not None:
             random.seed(42 + int(self.client_id)) # Seed for reproducible sampling per client
        else:
             random.seed(42) # Default seed if no client ID

        processed_patient_count = 0
        for patient_id in self.patient_ids:
            patient_path = self.data_dir / patient_id
            if not patient_path.is_dir():
                 logger.warning(f"Patient directory not found: {patient_path}, skipping.")
                 continue

            try:
                # Load one modality to get dimensions (e.g., T1n)
                t1n_files = list(patient_path.glob('*t1n*.nii.gz'))
                if not t1n_files:
                    logger.warning(f"No T1n file found for patient {patient_id}, skipping.")
                    continue
                t1n_path = t1n_files[0]

                nib_img = nib.load(t1n_path)
                 # Check image dimensions, expecting 3D
                if nib_img.ndim != 3:
                    logger.warning(f"Expected 3D image but got {nib_img.ndim} dimensions for {t1n_path}. Skipping patient.")
                    continue
                depth = nib_img.shape[-1] # Assume depth is the last dimension

                # --- Determine eligible slices ---
                eligible_slices = list(range(depth)) # Start with all slices
                if self.require_tumor:
                    seg_files = list(patient_path.glob('*seg*.nii.gz'))
                    if not seg_files:
                        logger.warning(f"Segmentation file not found for patient {patient_id}. Cannot apply tumor requirement. Skipping patient.")
                        continue # Skip if tumor required but no seg file
                    seg_path = seg_files[0]

                    seg_img = nib.load(seg_path)
                    if seg_img.ndim != 3:
                         logger.warning(f"Expected 3D segmentation but got {seg_img.ndim} dimensions for {seg_path}. Skipping patient.")
                         continue
                    seg_data = seg_img.get_fdata()

                    tumor_slices = []
                    for s_idx in range(depth):
                        # Ensure slice index is valid before accessing data
                        if s_idx < seg_data.shape[-1]:
                             seg_slice_data = seg_data[..., s_idx]
                             # Check for tumor labels 1, 2, or 4
                             if np.any(np.isin(seg_slice_data, [1, 2, 4])):
                                 tumor_slices.append(s_idx)
                        else:
                             logger.warning(f"Slice index {s_idx} out of bounds for seg data depth {seg_data.shape[-1]} in patient {patient_id}")


                    if not tumor_slices:
                        logger.debug(f"Patient {patient_id} has no tumor slices, skipping for training map.")
                        continue # Skip patient if no tumor slices found when required

                    eligible_slices = tumor_slices # Only consider tumor slices now

                # --- Apply slices_per_volume sampling ---
                if self.slices_per_volume is not None and len(eligible_slices) > self.slices_per_volume:
                     # Sample from the eligible slices (either all slices or just tumor slices)
                     final_slice_indices = random.sample(eligible_slices, self.slices_per_volume)
                else:
                     # Use all eligible slices if no sampling needed or not enough slices
                     final_slice_indices = eligible_slices

                # Add selected slices to the map
                for s_idx in final_slice_indices:
                    slice_map.append((patient_id, s_idx))

                processed_patient_count += 1

            except FileNotFoundError as e:
                 logger.warning(f"File not found error during slice map creation for patient {patient_id}: {e}, skipping.")
                 continue
            except Exception as e:
                 logger.error(f"Unexpected error processing patient {patient_id} for slice map: {e}", exc_info=True) # Log full traceback for unexpected errors
                 continue

        logger.info(f"Finished generating slice map. Processed {processed_patient_count}/{len(self.patient_ids)} assigned patients. Total slices: {len(slice_map)}.")
        return slice_map

    def __len__(self):
        """ Returns the total number of slices across all assigned patients. """
        # Ensure slice_map exists before accessing its length
        return len(self.slice_map) if hasattr(self, 'slice_map') else 0


    def __getitem__(self, idx):
        """ Loads and returns a single 2D slice and its mask. """
        # Check if slice_map exists and is populated
        if not hasattr(self, 'slice_map') or not self.slice_map:
             logger.error("Slice map is not initialised or is empty, cannot get item.")
             # Return dummy data or raise a more specific error
             h, w = (240, 240)
             dummy_image = torch.zeros((len(self.modalities), h, w), dtype=torch.float32)
             dummy_mask = torch.zeros((self.n_classes, h, w), dtype=torch.float32)
             dummy_mask[0] = 1
             return {'image': dummy_image, 'mask': dummy_mask, 'patient_id': 'dummy', 'slice_idx': -1, 'affine': np.eye(4)}

        # Check if idx is a valid integer type (Python int or NumPy integer)
        # Use numbers.Integral for broader compatibility or explicitly check np.integer
        if not isinstance(idx, (int, np.integer)): # MODIFIED check
             logger.error(f"Invalid index type received: {type(idx)}. Value: {idx}. Using index 0 as fallback.")
             idx = 0 # Fallback or raise error

        # Convert potential NumPy integer to standard Python int for list indexing
        idx = int(idx)

        # Check index bounds
        if idx < 0 or idx >= len(self.slice_map):
             logger.error(f"Index {idx} out of bounds for slice_map length {len(self.slice_map)}. Using index 0 as fallback.")
             idx = 0 # Use index 0 as fallback (make sure slice_map is not empty here)
             if not self.slice_map: # Double check after potential fallback
                  logger.error("Slice map is empty even after fallback, cannot get item.")
                  h, w = (240, 240)
                  dummy_image = torch.zeros((len(self.modalities), h, w), dtype=torch.float32)
                  dummy_mask = torch.zeros((self.n_classes, h, w), dtype=torch.float32)
                  dummy_mask[0] = 1
                  return {'image': dummy_image, 'mask': dummy_mask, 'patient_id': 'dummy', 'slice_idx': -1, 'affine': np.eye(4)}


        patient_id, slice_idx = self.slice_map[idx]
        patient_dir = self.data_dir / patient_id # Use Pathlib

        modality_slices = []
        affine = np.eye(4) # Default affine
        # Use a reasonable default, check actual size later
        h, w = (240, 240)


        try:
            # --- Load Modalities ---
            first_modality_loaded = False
            for modality in self.modalities:
                modality_files = list(patient_dir.glob(f'*{modality}*.nii.gz'))
                if not modality_files:
                    raise FileNotFoundError(f"Modality {modality} file not found for patient {patient_id}")
                modality_path = modality_files[0]

                nib_img = nib.load(modality_path)
                img_data_3d = nib_img.get_fdata()

                # Get spatial dimensions and check slice index validity
                current_h, current_w = img_data_3d.shape[0], img_data_3d.shape[1]
                depth = img_data_3d.shape[-1]
                if slice_idx >= depth:
                     raise IndexError(f"Slice index {slice_idx} out of bounds ({depth} slices) for modality {modality} in patient {patient_id}")

                # Store affine/shape from the first successfully loaded modality
                if not first_modality_loaded:
                    affine = nib_img.affine
                    # header = nib_img.header # Don't store header
                    h, w = current_h, current_w # Update actual H, W
                    first_modality_loaded = True
                elif (current_h, current_w) != (h, w):
                     logger.warning(f"Inconsistent spatial dimensions for patient {patient_id}. Modality {modality}: {(current_h, current_w)}, Expected: {(h, w)}. Check data integrity.")
                     # Handle inconsistency: skip patient, resize, or raise error


                img_slice = img_data_3d[..., slice_idx]

                # Normalise the 2D slice
                img_slice_norm = self._normalize(img_slice)
                modality_slices.append(img_slice_norm)

            # Stack modalities: check if list is not empty
            if not modality_slices:
                 raise RuntimeError(f"No modality slices loaded for patient {patient_id}, slice {slice_idx}")
            image_slice = np.stack(modality_slices, axis=0) # Shape: [C, H, W]

            # --- Load Segmentation Mask ---
            seg_files = list(patient_dir.glob('*seg*.nii.gz'))
            if not seg_files:
                 raise FileNotFoundError(f"Segmentation file not found for patient {patient_id}")
            seg_path = seg_files[0]

            seg_img = nib.load(seg_path)
            seg_data_3d = seg_img.get_fdata()
            seg_depth = seg_data_3d.shape[-1]
            if slice_idx >= seg_depth:
                 raise IndexError(f"Slice index {slice_idx} out of bounds ({seg_depth} slices) for segmentation in patient {patient_id}")

            # Check spatial consistency with image modalities
            seg_h, seg_w = seg_data_3d.shape[0], seg_data_3d.shape[1]
            if (seg_h, seg_w) != (h, w):
                 logger.warning(f"Inconsistent spatial dimensions between image and mask for patient {patient_id}. Mask: {(seg_h, seg_w)}, Image: {(h, w)}. Check data integrity.")
                 # Handle inconsistency: skip, resize, or raise error

            seg_slice = seg_data_3d[..., slice_idx]

            # Convert 2D mask slice to one-hot encoding [n_classes, H, W]
            mask_slice = np.zeros((self.n_classes, h, w), dtype=np.float32) # Use actual h, w
            mask_slice[0, seg_slice == 0] = 1  # Background
            # Use explicit checks for class indices to avoid errors if n_classes < 4
            if self.n_classes > 1: mask_slice[1, seg_slice == 1] = 1  # Edema (NCR/NET) - Label 1
            if self.n_classes > 2: mask_slice[2, seg_slice == 2] = 1  # Non-enhancing tumor (ED) - Label 2
            if self.n_classes > 3: mask_slice[3, seg_slice == 4] = 1  # Enhancing tumor (ET) - Label 4

        except Exception as e:
            # Log the specific error and patient/slice causing it
            logger.error(f"Error in __getitem__ for index {idx} (Patient: {patient_id}, Slice: {slice_idx}): {e}", exc_info=False)
            # Return zero tensors as a fallback
            logger.warning(f"Returning dummy data for index {idx}")
            image_slice_np = np.zeros((len(self.modalities), h, w), dtype=np.float32)
            mask_slice_np = np.zeros((self.n_classes, h, w), dtype=np.float32)
            mask_slice_np[0] = 1 # Set background class
            image = torch.from_numpy(image_slice_np)
            mask = torch.from_numpy(mask_slice_np)
            # Return dummy sample dictionary, excluding header
            return {'image': image, 'mask': mask, 'patient_id': patient_id, 'slice_idx': slice_idx, 'affine': affine}


        # Convert final numpy arrays to torch tensors
        image = torch.from_numpy(image_slice.astype(np.float32))
        mask = torch.from_numpy(mask_slice.astype(np.float32))

        # Apply any additional 2D transforms
        if self.transform:
            try:
                image, mask = self.transform(image, mask)
            except Exception as e:
                 logger.error(f"Error applying transform for patient {patient_id} slice {slice_idx}: {e}")
                 # Return untransformed tensors if transform fails


        sample = {
            'image': image,      # Shape: [C, H, W]
            'mask': mask,        # Shape: [n_classes, H, W]
            'patient_id': patient_id, # -> str (OK)
            'slice_idx': slice_idx,   # -> int (OK)
            'affine': affine,      # -> numpy.ndarray (OK)
            # 'header': header     # REMOVED: Nifti1Header causes collate error
        }

        return sample

    def _normalize(self, img_slice):
        """Normalise a 2D image slice to the range [0, 1]"""
        # Ensure input is numpy array
        img_slice = np.asarray(img_slice)
        # Handle potential non-numeric types if necessary
        if not np.issubdtype(img_slice.dtype, np.number):
             logger.warning(f"Non-numeric dtype found in slice: {img_slice.dtype}. Attempting conversion.")
             try:
                  img_slice = img_slice.astype(np.float32)
             except ValueError:
                  logger.error("Could not convert slice to numeric type. Returning zeros.")
                  return np.zeros_like(img_slice, dtype=np.float32)

        # Check for NaNs or Infs
        if not np.isfinite(img_slice).all():
             logger.warning("Non-finite values (NaN or Inf) found in slice. Replacing with 0.")
             img_slice = np.nan_to_num(img_slice, nan=0.0, posinf=0.0, neginf=0.0)


        min_val = np.min(img_slice)
        max_val = np.max(img_slice)
        if max_val > min_val:
            # Apply windowing or clipping here if needed before normalisation
            # e.g., percentile clipping: p01 = np.percentile(img_slice, 1); p99 = np.percentile(img_slice, 99)
            # img_slice = np.clip(img_slice, p01, p99)
            # min_val, max_val = p01, p99 # Update min/max after clipping
            normalized_slice = (img_slice - min_val) / (max_val - min_val)
            # Final check for NaNs after division
            if not np.isfinite(normalized_slice).all():
                 logger.warning("Non-finite values after normalisation. Returning zeros.")
                 return np.zeros_like(img_slice, dtype=np.float32)
            return normalized_slice
        elif max_val == min_val and max_val != 0: # Handle constant non-zero slices
             return np.ones_like(img_slice) * (1.0 if max_val > 0 else 0.0)
        return np.zeros_like(img_slice) # Handle all-zero slices


# --- Dataloader Creation ---
def create_dataloaders(data_dir, val_data_dir=None, client_id=None, num_clients=1, batch_size=16,
                       max_slices=None,
                       slices_per_volume=10,
                       require_tumor_train=True,
                       num_workers=1):
    """Create train and validation dataloaders with 2D slices.

    Args:
        data_dir: Path to training data directory.
        val_data_dir: Path to validation data directory (if None, use data_dir with train/val split).
        client_id: ID of the client.
        num_clients: Total number of clients.
        batch_size: Batch size for training (number of slices).
        max_slices: Maximum number of total slices to use per client dataset (train or val).
        slices_per_volume: How many slices to sample per 3D volume. None means use all.
        require_tumor_train: Sample only tumor-containing slices for training.
        num_workers: Number of worker processes for DataLoader.
    """
    logger.info(f"Creating dataloaders for client {client_id}...")
    logger.info(f"  Training data dir: {data_dir}")
    logger.info(f"  Validation data dir: {val_data_dir if val_data_dir else data_dir}")
    logger.info(f"  Batch size: {batch_size}, Num workers: {num_workers}")
    logger.info(f"  Slices per volume: {slices_per_volume}, Max slices: {max_slices}")
    logger.info(f"  Require tumor for training: {require_tumor_train}")


    # Instantiate dataset with default n_classes=4
    try:
        train_dataset = BraTSDataset2D(
            data_dir, client_id, num_clients, is_train=True,
            slices_per_volume=slices_per_volume, require_tumor=require_tumor_train
        )
    except FileNotFoundError as e:
         logger.error(f"Error initialising training dataset: {e}. Check data_dir path.")
         raise # Reraise the exception to stop execution

    # Use separate validation dataset if provided
    val_dir_to_use = val_data_dir if (val_data_dir and os.path.exists(val_data_dir)) else data_dir
    if val_data_dir and os.path.exists(val_data_dir):
         logger.info(f"Using separate validation dataset from: {val_data_dir}")
    else:
         logger.info("Using validation split from training dataset directory")

    try:
        val_dataset = BraTSDataset2D(
            val_dir_to_use, client_id, num_clients, is_train=False,
            slices_per_volume=slices_per_volume, require_tumor=False # Usually evaluate on all sampled val slices
        )
    except FileNotFoundError as e:
         logger.error(f"Error initialising validation dataset: {e}. Check val_data_dir path.")
         raise # Reraise the exception


    # --- Apply max_slices limit ---
    train_dataset_final = train_dataset
    if max_slices is not None and len(train_dataset) > 0: # Check if dataset is not empty
        if len(train_dataset) > max_slices:
            original_len = len(train_dataset)
            logger.info(f"Limiting training slices from {original_len} to {max_slices}")
            # Ensure reproducibility of subset selection if needed
            np.random.seed(42 + (int(client_id) if client_id is not None else 0))
            indices = np.random.choice(original_len, max_slices, replace=False)
            train_dataset_final = Subset(train_dataset, indices)
        else:
             logger.info(f"Training dataset has {len(train_dataset)} slices, which is not more than max_slices ({max_slices}). Using all.")
    elif len(train_dataset) == 0:
         logger.warning("Training dataset is empty after initialisation.")


    val_dataset_final = val_dataset
    if max_slices is not None and len(val_dataset) > 0: # Check if dataset is not empty
        # Limit validation slices proportionally or to a fixed number
        val_max_slices = max(batch_size * 2, max_slices // 5) # Ensure at least a few batches for validation
        if len(val_dataset) > val_max_slices:
             original_len = len(val_dataset)
             logger.info(f"Limiting validation slices from {original_len} to {val_max_slices}")
             # Ensure reproducibility
             np.random.seed(100 + (int(client_id) if client_id is not None else 0))
             indices = np.random.choice(original_len, val_max_slices, replace=False)
             val_dataset_final = Subset(val_dataset, indices)
        else:
             logger.info(f"Validation dataset has {len(val_dataset)} slices, which is not more than limit ({val_max_slices}). Using all.")
    elif len(val_dataset) == 0:
         logger.warning("Validation dataset is empty after initialisation.")
    # --- End max_slices limit ---


    # Create DataLoaders
    # Set persistent_workers=True if num_workers > 0 to speed up after first epoch
    # Handle num_workers=0 case where persistent_workers is not applicable
    persistent_workers = num_workers > 0 if num_workers else False
    pin_memory = torch.cuda.is_available() # Only pin memory if CUDA is available

    # Check if datasets are empty before creating loaders
    if len(train_dataset_final) == 0:
         logger.warning("Cannot create DataLoader: Training dataset is empty.")
         train_loader = None # Or an empty list/dummy loader
    else:
         train_loader = DataLoader(
             train_dataset_final, batch_size=batch_size, shuffle=True,
             num_workers=num_workers, pin_memory=pin_memory,
             persistent_workers=persistent_workers,
             # worker_init_fn=seed_worker # Optional: for perfect reproducibility with workers
             # collate_fn=custom_collate # Optional: if custom batching needed
         )

    if len(val_dataset_final) == 0:
         logger.warning("Cannot create DataLoader: Validation dataset is empty.")
         val_loader = None
    else:
         val_loader = DataLoader(
             val_dataset_final, batch_size=batch_size, shuffle=False, # No shuffle for validation
             num_workers=num_workers, pin_memory=pin_memory,
             persistent_workers=persistent_workers
         )

    # Get actual number of samples in the final loaders (after Subset)
    num_train_samples = len(train_loader.dataset) if train_loader else 0
    num_val_samples = len(val_loader.dataset) if val_loader else 0

    logger.info(f"Created dataloaders: {num_train_samples} training slices, {num_val_samples} validation slices")

    # Handle case where one or both loaders could not be created
    if train_loader is None or val_loader is None:
         logger.error("One or both DataLoaders are None because the underlying dataset was empty.")
         # Depending on requirements, you might want to raise an error here
         # raise ValueError("Cannot proceed with empty DataLoader(s)")

    return train_loader, val_loader


# --- Visualisation ---
def save_visualisation_2d(image_slice, pred_mask_slice, true_mask_slice, epoch, client_id, patient_id, slice_idx, save_dir="./visualisations"):
    """
    Save a visualisation of a 2D image slice, predicted mask, and true mask.

    Args:
        image_slice: 3D tensor [C, H, W] with modalities for the slice.
        pred_mask_slice: 3D tensor [n_classes, H, W] with predicted segmentation probabilities/logits.
        true_mask_slice: 3D tensor [n_classes, H, W] with ground truth segmentation (one-hot).
        epoch: Current epoch number.
        client_id: Client ID.
        patient_id: Patient ID.
        slice_idx: Index of the slice within the original volume.
        save_dir: Directory to save visualisations.
    """
    save_dir = Path(save_dir)
    client_save_dir = save_dir / f"client_{client_id}"
    client_save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Convert tensors to numpy arrays
        image_np = image_slice.detach().cpu().numpy() # Shape (C, H, W)

        # Apply softmax if necessary (if input is logits) and argmax to get class predictions
        # Ensure pred_mask_slice is on CPU before numpy conversion
        pred_mask_cpu = pred_mask_slice.detach().cpu()
        if pred_mask_cpu.shape[0] == true_mask_slice.shape[0]: # Check if it's probabilities/logits
            pred_classes = torch.softmax(pred_mask_cpu, dim=0).argmax(dim=0).numpy() # Shape (H, W)
        else: # Assume it's already class indices if shapes differ
            pred_classes = pred_mask_cpu.numpy() # Shape (H, W)

        true_classes = true_mask_slice.detach().cpu().argmax(dim=0).numpy() # Shape (H, W)

        # Select T1c modality for background image (assuming index 1)
        t1c_idx = 1
        if image_np.shape[0] > t1c_idx:
            background_img = image_np[t1c_idx]
        else:
            background_img = image_np[0] # Fallback to first modality

        # Create a figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Patient: {patient_id}, Slice: {slice_idx}, Epoch: {epoch}, Client: {client_id}')

        # 1. Plot T1c image slice
        axes[0].imshow(background_img, cmap='gray')
        axes[0].set_title(f'T1c Image Slice')
        axes[0].axis('off')

        # 2. Plot Predicted Segmentation Overlay
        axes[1].imshow(background_img, cmap='gray') # Background
        pred_overlay = np.zeros((*pred_classes.shape, 4), dtype=float) # RGBA overlay, use float
        # Color mapping (same as before, but applied to 2D class map)
        # Ensure indices match the class order: 1:Edema, 2:Non-enhancing, 3:Enhancing
        pred_overlay[pred_classes == 1, :] = [1.0, 1.0, 0.0, 0.6]  # Edema (Yellow) RGBA
        pred_overlay[pred_classes == 2, :] = [1.0, 0.0, 0.0, 0.6]  # Non-enhancing (Red) RGBA
        pred_overlay[pred_classes == 3, :] = [0.0, 0.0, 1.0, 0.6]  # Enhancing (Blue) RGBA
        axes[1].imshow(pred_overlay) # Apply overlay with transparency
        axes[1].set_title('Predicted Segmentation')
        axes[1].axis('off')

        # 3. Plot True Segmentation Overlay
        axes[2].imshow(background_img, cmap='gray') # Background
        true_overlay = np.zeros((*true_classes.shape, 4), dtype=float) # RGBA overlay, use float
        true_overlay[true_classes == 1, :] = [1.0, 1.0, 0.0, 0.6]  # Edema (Yellow) RGBA
        true_overlay[true_classes == 2, :] = [1.0, 0.0, 0.0, 0.6]  # Non-enhancing (Red) RGBA
        true_overlay[true_classes == 3, :] = [0.0, 0.0, 1.0, 0.6]  # Enhancing (Blue) RGBA
        axes[2].imshow(true_overlay) # Apply overlay with transparency
        axes[2].set_title('True Segmentation')
        axes[2].axis('off')

        # Save figure
        save_path = client_save_dir / f"epoch_{epoch}_patient_{patient_id}_slice_{slice_idx}.png"
        plt.savefig(save_path, bbox_inches='tight')
        logger.debug(f"Saved visualisation to {save_path}") # Use debug level for frequent logs

    except Exception as e:
        logger.error(f"Failed to save visualisation for patient {patient_id} slice {slice_idx}: {e}", exc_info=True)
    finally:
        # Ensure the plot is closed even if saving fails
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig) # Close the figure to free memory

    # Return the path even if saving failed, or None
    return str(save_path) if 'save_path' in locals() else None


