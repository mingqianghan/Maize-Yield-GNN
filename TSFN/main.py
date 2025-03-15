import argparse
import sys
import os

# Add project root to the Python path for module imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run_model import run

def parse_arguments():
    """
    Parse command-line arguments for running the model training and evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the TSFN model for Maize Yield Prediction with specified configuration."
    )
    parser.add_argument("--model_type", type=str, default="TSFN", help="Model type to use")
    parser.add_argument("--data_path", type=str, default="dataset/image_dataset.pkl", help="Absolute path to the data file")
    parser.add_argument("--timepoints", type=str, nargs="+", default=["R1", "R2"], help="Data collection time points (e.g., R1, R2, R3, ...)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size (fraction)")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size (fraction)")
    parser.add_argument("--dist_scale", type=float, default=0.5, help="Distance scale multiplier for graph weight construction")
    parser.add_argument("--n_bins", type=int, default=5, help="Number of bins for yield stratification")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="Factor for learning rate scheduler")
    parser.add_argument("--lr_patience", type=int, default=20, help="Patience for learning rate scheduler")
    parser.add_argument("--loss_method", type=str, default="logcosh", help="Loss method to use")

    return parser.parse_args()


def main():
    """
    Main function to parse arguments, print them, and run the model training.
    """
    args = parse_arguments()

    # Print all parsed arguments
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Run the model training and evaluation pipeline.
    run(args)


if __name__ == "__main__":
    main()
