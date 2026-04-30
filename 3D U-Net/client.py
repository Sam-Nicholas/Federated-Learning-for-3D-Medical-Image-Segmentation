import torch
import os
import logging
import flwr as fl
import numpy as np
from collections import OrderedDict
from torch.optim import Adam
from model import UNet3D
from dataset import create_dataloaders
from utils import train_epoch, evaluate, save_checkpoint, load_checkpoint

logger = logging.getLogger("client")


class BrainSegmentationClient(fl.client.NumPyClient):
    """
    Flower client for 3D brain tumour segmentation using federated learning.
    Manages local data, model training, and evaluation.
    """
    def __init__(self, args):
        """Initialises the client with configuration, model, optimiser, and data."""
        self.args = args
        self.client_id = args.client_id
        self.device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

        logger.info(f"Using device: {self.device}")

        self.model = UNet3D(n_channels=4, n_classes=4).to(self.device)
        self.optimiser = Adam(self.model.parameters(), lr=1e-4)

        # Initialise data loaders, potentially limiting samples per client based on args.max_samples
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=args.data_dir,
            val_data_dir=args.val_data_dir,
            client_id=args.client_id,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )

        self.current_epoch = 0 # Tracks local training epochs across rounds

        # Load previous state if a checkpoint path is provided
        if args.load_checkpoint:
            self._load_checkpoint(args.load_checkpoint)

        logger.info(f"Client {self.client_id} initialised")
        logger.info(f"Model parameters: {self.model.get_model_parameters()}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

    def _load_checkpoint(self, checkpoint_path):
        """Loads model and optimiser state from a checkpoint file."""
        try:
            if os.path.isfile(checkpoint_path):
                # Set weights_only=False if checkpoint might contain non-tensor objects (like NumPy arrays)
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

                # Check if it's a client-specific checkpoint (includes optimiser state)
                if 'optimizer_state_dict' in checkpoint: # Keep key as is if it matches torch save format
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.current_epoch = checkpoint['epoch']
                    logger.info(f"Client {self.client_id} loaded client checkpoint: {checkpoint_path}")
                    logger.info(f"Resuming from epoch {self.current_epoch}")
                # Otherwise, assume it's a global model checkpoint (only model weights)
                else:
                    self.model.load_state_dict(checkpoint)
                    logger.info(f"Client {self.client_id} loaded global model weights: {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint file not found: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")

    def get_parameters(self, config):
        """Returns the current local model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Updates the local model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Trains the local model using the provided parameters and local data.

        Args:
            parameters (NDArrays): Model parameters received from the server.
            config (Dict): Configuration parameters from the server (e.g., current round).

        Returns:
            Tuple[NDArrays, int, Dict]: Updated parameters, number of training examples, and metrics.
        """
        logger.info(f"Client {self.client_id}: Starting local training round {config.get('round', 0)}")

        self.set_parameters(parameters)

        epochs = self.args.epochs
        current_round = config.get("round", 0)

        epoch_losses = []
        epoch_dice_scores = []

        for epoch in range(epochs):
            self.current_epoch += 1
            epoch_loss, epoch_dice = train_epoch(
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimiser, # Use British spelling variable
                device=self.device,
                epoch=self.current_epoch,
                save_vis=True, # Enable saving visualisations during training
                client_id=self.client_id
            )

            epoch_losses.append(epoch_loss)
            # Track the mean Dice score across tumour classes
            epoch_dice_scores.append(epoch_dice.get("tumor_mean", 0.0))

            logger.info(f"Client {self.client_id}, Round {current_round}, Epoch {self.current_epoch}")
            logger.info(f"Training Loss: {epoch_loss:.4f}, Dice (Tumour Mean): {epoch_dice.get('tumor_mean', 0.0):.4f}")

            # Save a checkpoint after each local epoch
            checkpoint_dir = f"./checkpoints/round_{current_round}"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimiser, # Use British spelling variable
                epoch=self.current_epoch,
                loss=epoch_loss,
                dice=epoch_dice,
                client_id=self.client_id,
                checkpoint_dir=checkpoint_dir
            )

        # Evaluate the updated model on the local validation set
        val_loss, val_dice = evaluate(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            epoch=self.current_epoch, # Use current epoch for context in logging/saving
            save_vis=True, # Enable saving visualisations during evaluation
            client_id=self.client_id
        )

        logger.info(f"Client {self.client_id} completed training round {current_round}")
        logger.info(f"Validation Loss: {val_loss:.4f}, Dice (Tumour Mean): {val_dice.get('tumor_mean', 0.0):.4f}")

        updated_parameters = self.get_parameters(config={}) # Pass empty config

        # Return results to the server
        results = {
            "client_id": self.client_id,
            "train_loss": np.mean(epoch_losses),
            "train_dice": np.mean(epoch_dice_scores), # Mean Dice over local epochs
            "val_loss": val_loss,
            "val_dice": val_dice.get("tumor_mean", 0.0), # Final validation Dice
            "num_samples": len(self.train_loader.dataset)
        }

        return updated_parameters, len(self.train_loader.dataset), results

    def evaluate(self, parameters, config):
        """
        Evaluates the provided model parameters on the local validation dataset.

        Args:
            parameters (NDArrays): Model parameters received from the server.
            config (Dict): Configuration parameters from the server.

        Returns:
            Tuple[float, int, Dict]: Validation loss, number of validation examples, and metrics.
        """
        logger.info(f"Client {self.client_id}: Starting evaluation round {config.get('round', 0)}")

        self.set_parameters(parameters)

        val_loss, val_dice = evaluate(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            epoch=self.current_epoch, # Use current epoch for context
            save_vis=True, # Enable saving visualisations during evaluation
            client_id=self.client_id
        )

        logger.info(f"Evaluation Loss: {val_loss:.4f}, Dice (Tumour Mean): {val_dice.get('tumor_mean', 0.0):.4f}")

        # Return evaluation results to the server
        return float(val_loss), len(self.val_loader.dataset), {
            "client_id": self.client_id,
            "val_loss": val_loss,
            "val_dice": val_dice.get("tumor_mean", 0.0) # Report the mean tumour Dice score
        }


def get_server_address(file_path="./server_address.txt", default_address="[::]:8080"):
    """Reads the server address from a specified file, falling back to a default."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                address = f.read().strip()
                if address: # Ensure the file wasn't empty
                     logger.info(f"Using server address from {file_path}: {address}")
                     return address
                else:
                     logger.warning(f"Server address file {file_path} is empty, using default: {default_address}")
                     return default_address
        else:
            logger.warning(f"Server address file {file_path} not found, using default: {default_address}")
            return default_address
    except Exception as e:
        logger.error(f"Error reading server address from {file_path}: {e}. Using default: {default_address}")
        return default_address


def find_latest_checkpoint(args):
    """
    Finds the path to the latest checkpoint based on command-line arguments.
    Prioritises latest global model, then latest client-specific checkpoint if specified.
    """
    if args.load_latest_global:
        # Search for the global model with the highest round number
        global_dir = "./global_models"
        if not os.path.exists(global_dir):
            logger.warning(f"Global model directory not found: {global_dir}")
            return None

        checkpoints = [f for f in os.listdir(global_dir) if f.startswith("model_round_") and f.endswith(".pth")]
        if not checkpoints:
            logger.warning("No global model checkpoints found in ./global_models")
            return None

        # Sort checkpoints by round number (extracted from filename) in descending order
        checkpoints.sort(key=lambda x: int(x.split("_round_")[1].split(".")[0]), reverse=True)
        latest = os.path.join(global_dir, checkpoints[0])
        logger.info(f"Found latest global model checkpoint: {latest}")
        return latest

    elif args.load_latest_client:
        # Search for the latest checkpoint specific to this client
        base_checkpoint_dir = "./checkpoints"
        if not os.path.exists(base_checkpoint_dir):
             logger.warning(f"Base checkpoint directory not found: {base_checkpoint_dir}")
             return None

        # Find round directories (e.g., "round_5")
        checkpoint_dirs = [d for d in os.listdir(base_checkpoint_dir) if d.startswith("round_") and os.path.isdir(os.path.join(base_checkpoint_dir, d))]
        if not checkpoint_dirs:
            logger.warning(f"No round checkpoint directories found in {base_checkpoint_dir}")
            return None

        # Sort round directories by round number in descending order
        checkpoint_dirs.sort(key=lambda x: int(x.split("_")[1]), reverse=True)

        # Look for client checkpoints within the latest round directory
        latest_round_dir = os.path.join(base_checkpoint_dir, checkpoint_dirs[0])
        client_checkpoints = [f for f in os.listdir(latest_round_dir)
                              if f.startswith(f"client_{args.client_id}_") and f.endswith(".pth")]

        if not client_checkpoints:
            logger.warning(f"No checkpoints found for client {args.client_id} in the latest round directory: {latest_round_dir}")
            # Optionally, could search previous rounds here if desired
            return None

        # Sort client checkpoints by epoch number in descending order
        client_checkpoints.sort(key=lambda x: int(x.split("_epoch_")[1].split(".")[0]), reverse=True)
        latest = os.path.join(latest_round_dir, client_checkpoints[0])
        logger.info(f"Found latest client-specific checkpoint: {latest}")
        return latest

    # If neither flag is set, or no checkpoints found as specified
    return None


def start_client(args):
    """Configures and starts a Flower client instance."""
    logger.info(f"Configuring client {args.client_id} for FL with {args.num_clients} total clients.")

    # Automatically determine checkpoint path if latest loading flags are set
    if args.load_latest_global or args.load_latest_client:
        checkpoint_path = find_latest_checkpoint(args)
        if checkpoint_path:
            args.load_checkpoint = checkpoint_path # Override explicit path if latest found

    # Determine server address (read from file or use default)
    server_address = get_server_address(default_address=args.server_address)

    # Instantiate the custom Flower client
    client = BrainSegmentationClient(args)

    logger.info(f"Attempting to connect to server at {server_address}")

    # Start the Flower client connection to the server
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

    logger.info(f"Client {args.client_id} finished.")

