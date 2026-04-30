import argparse
import logging
from pathlib import Path


log_file = "federated_segmentation_2d_fedprox.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() # Also print logs to console
    ]
)
logger = logging.getLogger("main")
logger.info(f"Logging to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Federated Learning (FedProx) for Brain Tumor Segmentation (2D)")

    # Mode
    parser.add_argument("--mode", type=str, choices=["server", "client"], required=True,
                        help="Run as server or client")

    # Network
    parser.add_argument("--client_id", type=str, default="0",
                        help="Client ID (required for client mode)")
    parser.add_argument("--server_address", type=str, default="localhost:8080",
                        help="Server address (host:port) for clients to connect to, or address for server to bind to ([::]:port or 0.0.0.0:port)")

    # Data
    parser.add_argument("--data_dir", type=str, default="./BraTS_data/train",
                        help="Path to main BraTS dataset directory (containing patient folders)")
    parser.add_argument("--val_data_dir", type=str, default=None,
                        help="Path to separate validation BraTS dataset (optional)")
    parser.add_argument("--max_slices", type=int, default=None,
                        help="Maximum number of total slices to load per client dataset (train or val).")
    parser.add_argument("--slices_per_volume", type=int, default=20,
                        help="Number of slices to sample per 3D volume (None uses all).")
    parser.add_argument("--require_tumor_train", action='store_true',
                        help="If set, only sample/use slices containing tumor for the training set.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for DataLoader")


    # FL
    parser.add_argument("--num_clients", type=int, default=2,
                        help="Total number of clients expected by the server.")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of federated rounds (server only)")
    parser.add_argument("--fraction_fit", type=float, default=1.0,
                        help="Fraction of clients to use for training per round (server only)")
    parser.add_argument("--fraction_evaluate", type=float, default=1.0,
                        help="Fraction of clients to use for evaluation per round (server only)")
    # --- FedProx Specific ---
    parser.add_argument("--proximal_mu", type=float, default=0.01, # FedProx mu parameter
                        help="Proximal term coefficient (mu) for FedProx strategy (server only)")


    # Local Training (Client)
    parser.add_argument("--epochs", type=int, default=3, # Fewer local epochs often better in FL
                        help="Number of local training epochs per client per round")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training/evaluation (number of slices)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimiser")

    # Hardware & Checkpointing
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for training ('cuda' if available, otherwise 'cpu')")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint file (.pth) to load (client or global model)")
    parser.add_argument("--load_latest_global", action="store_true",
                        help="Automatically load the latest global model checkpoint from ./global_models")
    parser.add_argument("--load_latest_client", action="store_true",
                        help="Automatically load the latest checkpoint for this specific client from ./checkpoints")
    parser.add_argument("--save_freq", type=int, default=1, # Save checkpoint every local epoch
                        help="Frequency (in local epochs) to save client checkpoints (0 to disable)")
    parser.add_argument("--save_vis", action='store_true', # Default: False
                        help="If set, save visualisation images during training/validation")


    args = parser.parse_args()

    # Create Output Directories
    Path("./checkpoints").mkdir(parents=True, exist_ok=True)
    Path("./global_models").mkdir(parents=True, exist_ok=True)
    Path("./visualisations").mkdir(parents=True, exist_ok=True)
    Path("./val_visualisations").mkdir(parents=True, exist_ok=True)
    Path("./slice_map_cache").mkdir(parents=True, exist_ok=True) # For dataset cache


    # Start Server or Client
    if args.mode == "server":
        # Ensure server address uses a bindable format if default is used
        if args.server_address == "localhost:8080":
             args.server_address = "0.0.0.0:8080" # Bind to all interfaces
             logger.info("Server mode: Defaulting server_address to bind to 0.0.0.0:8080")

        logger.info(f"Starting FedProx server on {args.server_address}...")
        from server import start_server
        start_server(args) # Pass all args

    elif args.mode == "client":
        logger.info(f"Starting FedProx client {args.client_id}...")
        from client import start_client
        start_client(args) # Pass all args
    else:
        # This case should not be reachable due to 'choices' in argparse
        logger.error("Invalid mode specified.")


if __name__ == "__main__":
    main()
