import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("federated_segmentation.log"), # Log to file
        logging.StreamHandler(sys.stdout) # Log to standard output
    ]
)

logger = logging.getLogger("main")

def main():
    """Parses command-line arguments and starts the server or client process."""
    parser = argparse.ArgumentParser(
        description="Federated Learning for 3D Brain Tumour Segmentation (BraTS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    # Core arguments
    parser.add_argument("--mode", type=str, choices=["server", "client"], required=True,
                        help="Specify whether to run as 'server' or 'client'.")
    parser.add_argument("--server_address", type=str, default="[::]:8080",
                        help="Server address (host:port). Use '[::]:port' for IPv6 or '0.0.0.0:port' for IPv4 listening on all interfaces.")
    parser.add_argument("--data_dir", type=str, default="./BraTS_Data/train", # Example default path
                        help="Path to the main BraTS dataset directory (containing patient subfolders).")
    parser.add_argument("--val_data_dir", type=str, default=None,
                        help="Path to a separate validation dataset directory. If None, validation split is derived from --data_dir.")

    # Client-specific arguments
    parser.add_argument("--client_id", type=str, default="0",
                        help="Unique identifier for the client (required in client mode). Usually an integer index (0, 1, 2...).")

    # Federated Learning parameters (Server controls rounds, Client controls local epochs)
    parser.add_argument("--rounds", type=int, default=10,
                        help="Total number of federated learning rounds (Server only).")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of local training epochs per client per round (Client only).")
    parser.add_argument("--num_clients", type=int, default=2, # Example default
                        help="Total number of clients expected to participate in the federation. Used for data partitioning and server strategy.")
    parser.add_argument("--fraction_fit", type=float, default=1.0,
                        help="Fraction of available clients used for training in each round (Server only). 1.0 means all available clients.")
    # Note: fraction_evaluate is set equal to fraction_fit in the server strategy

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=1, # Smaller batch size typical for 3D UNets
                        help="Batch size for local training and evaluation.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for PyTorch computations ('cuda' or 'cpu').")
    parser.add_argument("--max_samples", type=int, default=None, # Default to using all samples
                        help="Maximum number of training samples to use per client (for faster runs/debugging). If None, uses all assigned samples.")

    # Checkpoint loading arguments
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint file (.pth) to load initial weights from (client or global model format).")
    parser.add_argument("--load_latest_global", action="store_true",
                        help="If set, automatically finds and loads the latest global model checkpoint from './global_models'. Overrides --load_checkpoint.")
    parser.add_argument("--load_latest_client", action="store_true",
                        help="If set, automatically finds and loads the latest client-specific checkpoint from './checkpoints'. Overrides --load_checkpoint and --load_latest_global.")


    args = parser.parse_args()

    if args.mode == "server":
        from server import start_server
        logger.info(f"Starting Federated Learning Server...")
        logger.info(f"Configuration: Rounds={args.rounds}, Expected Clients={args.num_clients}, FractionFit={args.fraction_fit}")
        start_server(args)
        logger.info("Server finished.")
    elif args.mode == "client":
        from client import start_client
        logger.info(f"Starting Federated Learning Client ID: {args.client_id}")
        logger.info(f"Configuration: Local Epochs={args.epochs}, Batch Size={args.batch_size}, Device={args.device}, Max Samples={args.max_samples}")
        start_client(args)
        logger.info(f"Client {args.client_id} finished.")
    else:
        # This case should not be reachable due to 'choices' in parser argument
        logger.error(f"Invalid mode specified: {args.mode}. Use 'server' or 'client'.")
        sys.exit(1) # Exit with error code


if __name__ == "__main__":
    main()
