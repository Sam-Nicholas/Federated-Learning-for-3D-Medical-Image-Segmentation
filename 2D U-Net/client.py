import torch
import os
import logging
import flwr as fl
import numpy as np
from collections import OrderedDict
from torch.optim import AdamW
from model import UNet2D
from dataset import create_dataloaders
from utils import train_epoch, evaluate, save_checkpoint, load_checkpoint
from pathlib import Path

logger = logging.getLogger("client")

class BrainSegmentationClient(fl.client.NumPyClient):
    def __init__(self, args):
        """Initialise the client with 2D model, data, and training parameters"""
        self.args = args
        self.client_id = args.client_id
        # Ensure device selection respects CUDA availability
        if args.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            if args.device == "cuda":
                 logger.warning("CUDA selected but not available, using CPU.")
            self.device = torch.device("cpu")

        logger.info(f"Client {self.client_id} using device: {self.device}")

        # Create 2D model
        self.model = UNet2D(n_channels=4, n_classes=4).to(self.device)
        logger.info(f"Initialised UNet2D with {self.model.get_model_parameters()} parameters.")


        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

        # Create 2D data loaders
        # Pass relevant args for 2D dataloading
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=args.data_dir,
            val_data_dir=args.val_data_dir,
            client_id=args.client_id,
            num_clients=args.num_clients,
            batch_size=args.batch_size, # Batch size now refers to slices
            max_slices=args.max_slices, # Max slices per dataset
            slices_per_volume=args.slices_per_volume, # Slices to sample per volume
            require_tumor_train=args.require_tumor_train # Whether to focus training on tumor slices
        )

        # Initialise epoch counter
        self.current_epoch = 0

        # Load checkpoint if specified
        if args.load_checkpoint:
            self._load_checkpoint(args.load_checkpoint)

        logger.info(f"Client {self.client_id} initialised")
        logger.info(f"Training slices: {len(self.train_loader.dataset)}")
        logger.info(f"Validation slices: {len(self.val_loader.dataset)}")

    def _load_checkpoint(self, checkpoint_path):
        """Load a checkpoint - either client-specific or global model"""
        try:
            if os.path.isfile(checkpoint_path):
                 # Pass model and optimiser to load_checkpoint
                 # load_checkpoint handles moving model to appropriate device after loading
                 loaded_epoch, loaded_loss, loaded_dice = load_checkpoint(
                     self.model, self.optimizer, checkpoint_path
                 )

                 if loaded_epoch is not None:
                     # Check if it was likely a client checkpoint (contains optimiser state)
                     # load_checkpoint function now handles loading optimiser state if present
                     # We can infer based on whether optimiser state was loaded (check optimiser's param_groups)
                     if any(pg['params'] for pg in self.optimizer.param_groups):
                         self.current_epoch = loaded_epoch # Resume epoch count
                         logger.info(f"Client {self.client_id} loaded client checkpoint: {checkpoint_path}")
                         logger.info(f"Resuming from epoch {self.current_epoch}, Loss: {loaded_loss}, Dice: {loaded_dice}")
                     else:
                         # Likely a global model (only model weights loaded)
                         self.current_epoch = 0 # Start epochs from 0
                         logger.info(f"Client {self.client_id} loaded global model weights: {checkpoint_path}")
                         # Reset optimiser state if loading only global weights
                         self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
                         logger.info("Optimiser state reset.")

                     # Ensure model is on the correct device after loading
                     self.model.to(self.device)

                 else:
                     logger.warning(f"Failed to load checkpoint: {checkpoint_path}")

            else:
                logger.warning(f"Checkpoint file not found: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_path}: {e}", exc_info=True)


    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays"""
        # Ensure model is on CPU before extracting parameters
        self.model.cpu()
        params = [val.numpy() for _, val in self.model.state_dict().items()]
        # Move model back to the original device
        self.model.to(self.device)
        return params


    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
             logger.error(f"Error loading state dict: {e}. Keys might not match.")
             logger.info("Attempting to load with strict=False")
             # Try loading with strict=False if keys mismatch (e.g., loading global model)
             self.model.load_state_dict(state_dict, strict=False)

        # Ensure the model is on the correct device after setting parameters
        self.model.to(self.device)


    def fit(self, parameters, config):
        """Train the 2D model on the locally held dataset of slices"""
        logger.info(f"Client {self.client_id}: Starting local training round {config.get('round', 0)}")

        # Set model parameters received from the server
        self.set_parameters(parameters)

        # Get training config
        epochs = self.args.epochs # Local epochs per round
        current_round = config.get("round", 0)

        # Train for specified number of local epochs
        round_losses = []
        round_dice_scores = [] # Store mean tumor dice per epoch

        for epoch in range(epochs):
            # Increment global epoch counter
            self.current_epoch += 1
            logger.info(f"--- Client {self.client_id} Round {current_round} Local Epoch {epoch+1}/{epochs} (Global Epoch {self.current_epoch}) ---")

            # Perform one epoch of training
            epoch_loss, epoch_dice_dict = train_epoch(
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimizer,
                device=self.device,
                epoch=self.current_epoch, # Log with global epoch number
                save_vis=self.args.save_vis, # Control visualisation saving
                client_id=self.client_id
            )

            round_losses.append(epoch_loss)
            # Store the mean tumor dice score for the epoch
            round_dice_scores.append(epoch_dice_dict.get("tumor_mean", 0.0))

            # Log metrics after each local epoch
            logger.info(f"Client {self.client_id}, Round {current_round}, Local Epoch {epoch+1} completed.")
            logger.info(f"  Training Loss: {epoch_loss:.4f}, Mean Tumor Dice: {epoch_dice_dict.get('tumor_mean', 0.0):.4f}")

            # Save checkpoint after each local epoch (optional)
            if self.args.save_freq > 0 and (epoch + 1) % self.args.save_freq == 0:
                checkpoint_dir = Path(f"./checkpoints/round_{current_round}")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=self.current_epoch, # Save with global epoch
                    loss=epoch_loss,
                    dice=epoch_dice_dict, # Save the whole dice dictionary
                    client_id=self.client_id,
                    checkpoint_dir=str(checkpoint_dir)
                )

        # Evaluate on the local validation set after finishing local epochs for the round
        logger.info(f"Client {self.client_id}: Evaluating model after round {current_round} training.")
        val_loss, val_dice_dict = evaluate(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            epoch=self.current_epoch, # Use current global epoch for eval log/vis name
            save_vis=self.args.save_vis, # Control visualisation saving
            client_id=self.client_id
        )

        logger.info(f"Client {self.client_id} completed training round {current_round}")
        logger.info(f"  Avg Round Training Loss: {np.mean(round_losses):.4f}, Avg Round Training Dice: {np.mean(round_dice_scores):.4f}")
        logger.info(f"  Validation Loss: {val_loss:.4f}, Validation Mean Tumor Dice: {val_dice_dict.get('tumor_mean', 0.0):.4f}")

        # Get updated model parameters to send back to the server
        updated_parameters = self.get_parameters(config={}) # Pass empty config

        # Return updated parameters, number of samples used in training, and metrics
        num_train_samples = len(self.train_loader.dataset)
        results = {
            "client_id": self.client_id,
            # Report average metrics over the local epochs for this round
            "train_loss": np.mean(round_losses),
            "train_dice": np.mean(round_dice_scores),
            # Report validation metrics from the end of the round
            "val_loss": val_loss,
            "val_dice": val_dice_dict.get("tumor_mean", 0.0), # Return the mean tumor dice
            "num_samples": num_train_samples # Number of slices used
        }

        return updated_parameters, num_train_samples, results

    def evaluate(self, parameters, config):
        """Evaluate the 2D model on the locally held validation set of slices"""
        logger.info(f"Client {self.client_id}: Starting evaluation round {config.get('round', -1)}")

        # Set model parameters received from the server
        self.set_parameters(parameters)

        # Evaluate the model
        val_loss, val_dice_dict = evaluate(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            epoch=self.current_epoch, # Use current global epoch, or round number from config
            save_vis=self.args.save_vis,
            client_id=self.client_id
        )

        logger.info(f"Client {self.client_id} evaluation completed.")
        logger.info(f"  Validation Loss: {val_loss:.4f}, Mean Tumor Dice: {val_dice_dict.get('tumor_mean', 0.0):.4f}")

        # Return evaluation results: loss, number of samples, metrics dict
        num_val_samples = len(self.val_loader.dataset)
        results = {
            "client_id": self.client_id,
            "val_loss": val_loss,
            "val_dice": val_dice_dict.get("tumor_mean", 0.0) # Return mean tumor dice
            # Can add other dice scores from val_dice_dict if needed
            # "val_dice_edema": val_dice_dict.get("edema", 0.0),
            # ...
        }
        # Flower evaluate expects: loss, num_examples, metrics_dict
        return float(val_loss), num_val_samples, results


# Helper function to get server address (remains the same)
def get_server_address(file_path="./server_address.txt", default_address="[::]:8080"):
    """Read server address from file, or use default if file not found"""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                address = f.read().strip()
            logger.info(f"Using server address from {file_path}: {address}")
            return address
        else:
            logger.warning(f"Server address file {file_path} not found, using default: {default_address}")
            return default_address
    except Exception as e:
        logger.error(f"Error reading server address: {e}")
        return default_address


# Helper function to find latest checkpoint (updated suffix)
def find_latest_checkpoint(args):
    """Find the latest checkpoint based on args, looking for '_2d.pth' suffix"""
    suffix = "_2d.pth" # Suffix for 2D model checkpoints
    if args.load_latest_global:
        # Find latest global model
        global_dir = "./global_models"
        if not os.path.exists(global_dir):
            logger.warning(f"Global model directory not found: {global_dir}")
            return None

        # Look for files ending with the specific suffix
        checkpoints = [f for f in os.listdir(global_dir) if f.startswith("model_round_") and f.endswith(suffix)]
        if not checkpoints:
            logger.warning(f"No global model checkpoints found with suffix {suffix}")
            return None

        # Sort by round number
        try:
            checkpoints.sort(key=lambda x: int(x.split("_round_")[1].split(suffix)[0]), reverse=True)
            latest = os.path.join(global_dir, checkpoints[0])
            logger.info(f"Found latest global model: {latest}")
            return latest
        except (IndexError, ValueError) as e:
             logger.error(f"Error parsing round number from global checkpoints: {e}")
             return None


    elif args.load_latest_client:
        # Find latest checkpoint for this client
        checkpoint_base_dir = "./checkpoints"
        if not os.path.exists(checkpoint_base_dir):
             logger.warning(f"Base checkpoint directory not found: {checkpoint_base_dir}")
             return None

        round_dirs = [d for d in os.listdir(checkpoint_base_dir) if d.startswith("round_") and os.path.isdir(os.path.join(checkpoint_base_dir, d))]
        if not round_dirs:
            logger.warning("No round checkpoint directories found")
            return None

        # Sort round directories by round number (descending)
        try:
            round_dirs.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
        except (IndexError, ValueError):
             logger.error("Error parsing round number from checkpoint directories.")
             return None


        # Search for the latest client checkpoint in the most recent rounds
        for round_dir_name in round_dirs:
            latest_round_dir = os.path.join(checkpoint_base_dir, round_dir_name)
            client_checkpoints = [
                f for f in os.listdir(latest_round_dir)
                if f.startswith(f"client_{args.client_id}_epoch_") and f.endswith(suffix)
            ]

            if client_checkpoints:
                # Sort by epoch number (descending)
                try:
                    client_checkpoints.sort(key=lambda x: int(x.split("_epoch_")[1].split(suffix)[0]), reverse=True)
                    latest = os.path.join(latest_round_dir, client_checkpoints[0])
                    logger.info(f"Found latest client checkpoint: {latest}")
                    return latest
                except (IndexError, ValueError) as e:
                    logger.error(f"Error parsing epoch number from client checkpoints in {latest_round_dir}: {e}")
                    continue # Try the next older round directory

        logger.warning(f"No checkpoints found for client {args.client_id} with suffix {suffix}")
        return None

    return None # Return None if neither flag is set


def start_client(args):
    """Start a federated learning client for 2D segmentation"""
    # Basic checks (same as before)
    if args.num_clients > 1 and args.server_address and not args.server_address.startswith("[::]:"):
        logger.info(f"Attempting to connect to server {args.server_address} as client ID {args.client_id}")
    else:
        logger.info(f"Running in standalone mode or server not specified, client ID {args.client_id}")
        if args.num_clients > 1:
            logger.warning(f"num_clients is {args.num_clients} but running standalone. Setting num_clients=1 for data loading.")
            args.num_clients = 1 # Adjust for data partitioning

    logger.info(f"Client {args.client_id} participating with {args.num_clients} total clients expected by server.")

    # Check for automatic checkpoint loading
    if args.load_latest_global or args.load_latest_client:
        checkpoint_path = find_latest_checkpoint(args)
        if checkpoint_path:
            args.load_checkpoint = checkpoint_path # Set the specific path to load
        else:
             logger.warning("Could not find latest checkpoint to load.")


    # Get server address
    server_address = get_server_address(default_address=args.server_address)
    logger.info(f"Client {args.client_id} will attempt to connect to server at: {server_address}")


    # Create the 2D client instance
    client = BrainSegmentationClient(args)

    # Start Flower client connection
    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client,
            # Add root certificates if using TLS/SSL
            # root_certificates=Path(".cache/certificates/ca.crt").read_bytes()
        )
        logger.info(f"Client {args.client_id} finished.")
    except ConnectionRefusedError:
         logger.error(f"Connection refused by server at {server_address}. Is the server running?")
    except Exception as e:
         logger.error(f"Client {args.client_id} failed: {e}", exc_info=True)

