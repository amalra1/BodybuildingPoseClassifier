# main.py

import argparse
import sys

def run_train(args):
    """Imports and runs the training script."""
    from src.scripts.training import train
    print("--> Running Training Script...")
    train()

def run_predict(args):
    """Imports and runs the prediction script."""
    from src.scripts.prediction import predict
    print(f"--> Running Prediction for: {args.image_path}")
    predict(args.image_path)

def run_visualize(args):
    """Imports and runs the visualization script."""
    from src.scripts.visualization import visualize
    print(f"--> Running Visualization for: {args.image_path}")
    visualize(args.image_path)

def run_clean(args):
    """Imports and runs the cleanup script."""
    from src.scripts.cleanup import clean
    print("--> Running Cleanup Script...")
    clean()
    
def run_validate(args):
    """Imports and runs the full validation script."""
    from src.scripts.validation import validate_all
    print("--> Running Full Validation Script...")
    validate_all()

def run_real_time(args):
    """Imports and runs the real-time pose detection script."""
    from src.scripts.real_time import detect_real_time
    # The print statements are inside the real_time script itself
    detect_real_time()

def main():
    """
    Main function to parse command-line arguments and run the chosen command.
    """
    parser = argparse.ArgumentParser(
        description="Bodybuilding Pose Classifier - Main control script.",
        epilog="Example: python main.py predict path/to/image.jpg"
    )
    
    # A container for all available commands
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Command 'train' ---
    parser_train = subparsers.add_parser('train', help='Run the training process on the dataset in data/raw/.')
    parser_train.set_defaults(func=run_train)

    # --- Command 'predict' ---
    parser_predict = subparsers.add_parser('predict', help='Predict the pose for a single image.')
    parser_predict.add_argument('image_path', type=str, help='Path to the image to be classified.')
    parser_predict.set_defaults(func=run_predict)

    # --- Command 'visualize' ---
    parser_visualize = subparsers.add_parser('visualize', help='Generate a skeleton visualization for an image.')
    parser_visualize.add_argument('image_path', type=str, help='Path to the image to be visualized.')
    parser_visualize.set_defaults(func=run_visualize)

    # --- Command 'validate' ---
    parser_validate = subparsers.add_parser('validate', help='Run prediction on all images in the data/validation/ folder.')
    parser_validate.set_defaults(func=run_validate)

    # --- Command 'real-time' ---
    parser_realtime = subparsers.add_parser('real-time', help='Run real-time pose detection using the webcam.')
    parser_realtime.set_defaults(func=run_real_time)

    # --- Command 'clean' ---
    parser_clean = subparsers.add_parser('clean', help='Clean all generated files (prototypes, skeletons, visualizations).')
    parser_clean.set_defaults(func=run_clean)
    
    # If no command is given, print the help message and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Execute the function associated with the chosen command
    args.func(args)

if __name__ == "__main__":
    main()