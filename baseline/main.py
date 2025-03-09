import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run_model import run

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="baseline", help="Model type to use")
parser.add_argument("--method", type=str, default="ensemble", help="Model to use")
parser.add_argument("--data_path", type=str, default="dataset/stat_dataset.pkl", help="Path to the data file")
parser.add_argument("--timepoint", type=str, default="R1", help="Data collection time point:R1, R2, R3, R4, R5, R6")
parser.add_argument("--threshold", type=float, default=0.4, help="Threshold to choose features based one Pearson Correlation Coefficient")
parser.add_argument("--n_iter", type=int, default=100, help="Iterations for optimizing parameters using BayesSearch")
parser.add_argument("--cv", type=int, default=5, help="Number of folds of cross-validation for optimizing parameters using BayesSearch")
parser.add_argument("--scoring", type=str, default="neg_mean_squared_error", help="scoring method for optimizing parameters using BayesSearch")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
parser.add_argument("--test_size", type=float, default=0.2, help="Test set size (fraction)")
parser.add_argument("--n_bins", type=int, default=5, help="Number of bins for yield stratification")


args = parser.parse_args()

for arg, value in vars(args).items():
    print(f"{arg}: {value}")


if __name__ == "__main__":
    run(args)





