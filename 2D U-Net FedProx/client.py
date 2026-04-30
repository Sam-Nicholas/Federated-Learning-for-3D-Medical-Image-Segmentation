import torch
import os
import logging
import flwr as fl
import numpy as np
from collections import OrderedDict
from torch.optim import AdamW
from pathlib import Path
from model import UNet2D
from dataset import create_dataloaders
from utils import train_epoch_fedprox, evaluate, save_checkpoint, load_checkpoint

logger = logging.getLogger("client")

class BrainSegmentationClient(fl.client.NumPyClient):
    def __init__(self, args):
        """Initialise the client with 2D model, data, and training parameters"""
        self.args = args
        self.client_id = args.client_id
        if args.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            if args.device == "cuda": logger.warning("CUDA selected but not available, using CPU.")
            self.device = torch.device("cpu")
        logger.info(f"Client {self.client_id} using device: {self.device}")

        # Create 2D model
        self.model = UNet2D(n_channels=4, n_classes=4).to(self.device)
        logger.info(f"Initialised UNet2D with {self.model.get_model_parameters()} parameters.")

        # Create optimiser
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

        # Create 2D data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=args.data_dir,
            val_data_dir=args.val_data_dir,
            client_id=args.client_id,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            max_slices=args.max_slices,
            slices_per_volume=args.slices_per_volume,
            require_tumor_train=args.require_tumor_train,
            num_workers=args.num_workers
        )

        self.current_epoch = 0
        self.initial_global_params = None # To store global params for FedProx loss

        if args.load_checkpoint: self._load_checkpoint(args.load_checkpoint)

        logger.info(f"Client {self.client_id} initialised")
        if self.train_loader: logger.info(f"Training slices: {len(self.train_loader.dataset)}")
        else: logger.warning("Training DataLoader is None.")
        if self.val_loader: logger.info(f"Validation slices: {len(self.val_loader.dataset)}")
        else: logger.warning("Validation DataLoader is None.")

    def _load_checkpoint(self, checkpoint_path):
        """Load a checkpoint - either client-specific or global model"""
        try:
            if os.path.isfile(checkpoint_path):
                 loaded_epoch, loaded_loss, loaded_dice = load_checkpoint(
                     self.model, self.optimizer, checkpoint_path
                 )
                 if loaded_epoch is not None:
                     # Check if optimiser state was loaded
                     if any(pg['params'] for pg in self.optimizer.param_groups if pg['params']):
                         self.current_epoch = loaded_epoch
                         logger.info(f"Client {self.client_id} loaded client checkpoint: {checkpoint_path}")
                         logger.info(f"Resuming from epoch {self.current_epoch}, Loss: {loaded_loss}, Dice: {loaded_dice}")
                     else:
                         self.current_epoch = 0
                         logger.info(f"Client {self.client_id} loaded global model weights: {checkpoint_path}")
                         self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
                         logger.info("Optimiser state reset.")
                     self.model.to(self.device)
                 else: logger.warning(f"Failed to load checkpoint: {checkpoint_path}")
            else: logger.warning(f"Checkpoint file not found: {checkpoint_path}")
        except Exception as e: logger.error(f"Error loading checkpoint {checkpoint_path}: {e}", exc_info=True)

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays"""
        self.model.cpu()
        params = [val.numpy() for _, val in self.model.state_dict().items()]
        self.model.to(self.device)
        return params

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays"""
        # Store the received parameters as the 'global model' for FedProx loss calculation
        # Convert numpy arrays to tensors and store in a dictionary matching model state_dict keys
        keys = self.model.state_dict().keys()
        self.initial_global_params = {k: torch.tensor(v).to(self.device) for k, v in zip(keys, parameters)}
        logger.debug(f"Client {self.client_id}: Stored initial global parameters for FedProx.")

        # Load parameters into the local model
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
             logger.error(f"Error loading state dict: {e}. Keys might not match.")
             logger.info("Attempting to load with strict=False")
             try: self.model.load_state_dict(state_dict, strict=False)
             except Exception as e_false: logger.error(f"Failed to load state dict even with strict=False: {e_false}")
        self.model.to(self.device)


    def fit(self, parameters, config):
        """Train the 2D model using FedProx"""
        if self.train_loader is None:
             logger.error(f"Client {self.client_id}: Training loader is None. Cannot fit.")
             return self.get_parameters(config={}), 0, {"train_loss": None, "train_dice": None, "val_loss": None, "val_dice": None}

        current_round = config.get("round", 0)
        logger.info(f"Client {self.client_id}: Starting local training round {current_round} (FedProx)")

        # Set model parameters received from the server AND store them for prox loss
        self.set_parameters(parameters)

        # Get FedProx mu from config
        mu = config.get("proximal_mu", 0.0) # Default to 0 if not provided (acts like FedAvg)
        if mu > 0:
             logger.info(f"Client {self.client_id}: Using FedProx with mu={mu}")
        else:
             logger.info(f"Client {self.client_id}: FedProx mu=0, training behaves like FedAvg.")


        # Get other training config
        epochs = self.args.epochs

        # Train for specified number of local epochs
        round_losses = []
        round_dice_scores = []

        for epoch in range(epochs):
            self.current_epoch += 1
            logger.info(f"--- Client {self.client_id} Round {current_round} Local Epoch {epoch+1}/{epochs} (Global Epoch {self.current_epoch}) ---")

            # Perform one epoch of training using the FedProx-specific function
            epoch_loss, epoch_dice_dict = train_epoch_fedprox(
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimizer,
                device=self.device,
                epoch=self.current_epoch,
                save_vis=self.args.save_vis,
                client_id=self.client_id,
                global_params_dict=self.initial_global_params, # Pass initial global params
                mu=mu                                          # Pass mu
            )

            round_losses.append(epoch_loss)
            round_dice_scores.append(epoch_dice_dict.get("tumor_mean", 0.0))

            logger.info(f"Client {self.client_id}, Round {current_round}, Local Epoch {epoch+1} completed.")
            logger.info(f"  Training Loss (incl. prox): {epoch_loss:.4f}, Mean Tumor Dice: {epoch_dice_dict.get('tumor_mean', 0.0):.4f}")

            # Save checkpoint (optional)
            if self.args.save_freq > 0 and (epoch + 1) % self.args.save_freq == 0:
                checkpoint_dir = Path(f"./checkpoints/round_{current_round}")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    model=self.model, optimizer=self.optimizer, epoch=self.current_epoch,
                    loss=epoch_loss, dice=epoch_dice_dict, client_id=self.client_id,
                    checkpoint_dir=str(checkpoint_dir)
                )

        # Evaluate on the local validation set
        val_loss, val_dice_dict = None, {"tumor_mean": None}
        if self.val_loader:
            logger.info(f"Client {self.client_id}: Evaluating model after round {current_round} training.")
            # Standard evaluate function is fine here, prox term is only for training
            val_loss, val_dice_dict = evaluate(
                model=self.model, loader=self.val_loader, device=self.device,
                epoch=self.current_epoch, save_vis=self.args.save_vis, client_id=self.client_id
            )
        else: logger.warning(f"Client {self.client_id}: Validation loader is None. Skipping evaluation.")

        logger.info(f"Client {self.client_id} completed training round {current_round}")
        logger.info(f"  Avg Round Training Loss: {np.mean(round_losses):.4f}, Avg Round Training Dice: {np.mean(round_dice_scores):.4f}")
        if val_loss is not None: logger.info(f"  Validation Loss: {val_loss:.4f}, Validation Mean Tumor Dice: {val_dice_dict.get('tumor_mean', 0.0):.4f}")
        else: logger.info("  Validation skipped.")

        # Get updated model parameters to send back
        updated_parameters = self.get_parameters(config={})

        num_train_samples = len(self.train_loader.dataset)
        results = {
            "client_id": self.client_id,
            "train_loss": np.mean(round_losses), # Note: This loss includes the prox term if mu > 0
            "train_dice": np.mean(round_dice_scores),
            "val_loss": val_loss,
            "val_dice": val_dice_dict.get("tumor_mean"),
            "num_samples": num_train_samples
        }
        return updated_parameters, num_train_samples, results

    def evaluate(self, parameters, config):
        """Evaluate the 2D model (standard evaluation, no prox term)"""
        if self.val_loader is None:
             logger.error(f"Client {self.client_id}: Validation loader is None. Cannot evaluate.")
             return 0.0, 0, {"val_loss": None, "val_dice": None}

        logger.info(f"Client {self.client_id}: Starting evaluation round {config.get('round', -1)}")
        self.set_parameters(parameters) # Load model weights

        val_loss, val_dice_dict = evaluate(
            model=self.model, loader=self.val_loader, device=self.device,
            epoch=self.current_epoch, save_vis=self.args.save_vis, client_id=self.client_id
        )

        logger.info(f"Client {self.client_id} evaluation completed.")
        logger.info(f"  Validation Loss: {val_loss:.4f}, Mean Tumor Dice: {val_dice_dict.get('tumor_mean', 0.0):.4f}")

        num_val_samples = len(self.val_loader.dataset)
        results = {
            "client_id": self.client_id,
            "val_loss": val_loss,
            "val_dice": val_dice_dict.get("tumor_mean", 0.0)
        }
        return float(val_loss), num_val_samples, results


# Helper functions (get_server_address, find_latest_checkpoint remain the same)
def get_server_address(file_path="./server_address.txt", default_address="[::]:8080"):
    try:
        server_addr_file = Path(file_path)
        if server_addr_file.exists():
            address = server_addr_file.read_text().strip()
            logger.info(f"Using server address from {file_path}: {address}")
            return address
        else:
            logger.warning(f"Server address file {file_path} not found, using default: {default_address}")
            return default_address
    except Exception as e:
        logger.error(f"Error reading server address: {e}")
        return default_address

def find_latest_checkpoint(args):
    """Find the latest checkpoint based on args, looking for '_fedprox_2d.pth' suffix"""
    # Update suffix based on strategy if needed, or keep generic _2d
    suffix = "_fedprox_2d.pth" # Or maybe just "_2d.pth" if checkpoints are compatible
    if args.load_latest_global:
        global_dir = Path("./global_models")
        if not global_dir.exists(): logger.warning(f"Global model directory not found: {global_dir}"); return None
        checkpoints = [f.name for f in global_dir.glob(f"model_round_*{suffix}")]
        if not checkpoints: logger.warning(f"No global model checkpoints found with suffix {suffix} in {global_dir}"); return None
        try:
            checkpoints.sort(key=lambda x: int(x.split("_round_")[1].split(suffix)[0]), reverse=True)
            latest = global_dir / checkpoints[0]
            logger.info(f"Found latest global model: {latest}")
            return str(latest)
        except (IndexError, ValueError) as e: logger.error(f"Error parsing round number from global checkpoints: {e}"); return None
    elif args.load_latest_client:
        checkpoint_base_dir = Path("./checkpoints")
        if not checkpoint_base_dir.exists(): logger.warning(f"Base checkpoint directory not found: {checkpoint_base_dir}"); return None
        round_dirs = [d for d in checkpoint_base_dir.iterdir() if d.is_dir() and d.name.startswith("round_")]
        if not round_dirs: logger.warning("No round checkpoint directories found"); return None
        try: round_dirs.sort(key=lambda x: int(x.name.split("_")[1]), reverse=True)
        except (IndexError, ValueError): logger.error("Error parsing round number from checkpoint directories."); return None
        for round_dir_path in round_dirs:
            # Update client checkpoint filename pattern if needed
            client_checkpoint_pattern = f"client_{args.client_id}_epoch_*{suffix}" # Or generic "_2d.pth"
            client_checkpoints = [f.name for f in round_dir_path.glob(client_checkpoint_pattern)]
            if client_checkpoints:
                try:
                    client_checkpoints.sort(key=lambda x: int(x.split("_epoch_")[1].split(suffix)[0]), reverse=True)
                    latest = round_dir_path / client_checkpoints[0]
                    logger.info(f"Found latest client checkpoint: {latest}")
                    return str(latest)
                except (IndexError, ValueError) as e: logger.error(f"Error parsing epoch number from client checkpoints in {round_dir_path}: {e}"); continue
        logger.warning(f"No checkpoints found for client {args.client_id} with suffix {suffix}")
        return None
    return None


def start_client(args):
    """Start a FedProx federated learning client"""
    if args.num_clients > 1 and args.server_address and not args.server_address.startswith("[::]:") and not args.server_address.startswith("0.0.0.0"):
        logger.info(f"Attempting to connect to server {args.server_address} as client ID {args.client_id}")
    else:
        logger.info(f"Running in standalone mode or server address implies local server, client ID {args.client_id}")
        if args.num_clients > 1:
            logger.warning(f"num_clients is {args.num_clients} but running standalone/local. Setting num_clients=1 for data loading.")
            args.num_clients = 1
    logger.info(f"Client {args.client_id} participating with {args.num_clients} total clients expected by server.")

    if args.load_latest_global or args.load_latest_client:
        checkpoint_path = find_latest_checkpoint(args)
        if checkpoint_path: args.load_checkpoint = checkpoint_path
        else: logger.warning("Could not find latest checkpoint to load.")

    server_address = get_server_address(default_address=args.server_address)
    logger.info(f"Client {args.client_id} will attempt to connect to server at: {server_address}")

    try:
        client = BrainSegmentationClient(args)
    except Exception as e:
         logger.error(f"Failed to initialise BrainSegmentationClient: {e}", exc_info=True)
         return

    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client,
        )
        logger.info(f"Client {args.client_id} finished.")
    except ConnectionRefusedError:
         logger.error(f"Connection refused by server at {server_address}. Is the server running?")
    except Exception as e:
         logger.error(f"Client {args.client_id} failed during Flower communication: {e}", exc_info=True)


