# Saved Trained Models

This directory contains trained brain-to-text models and evaluation results.

## Files Structure

- `baseline_rnn/checkpoint/args.yaml` - Training configuration
- `baseline_rnn/train_val_trials.json` - Training/validation split info  
- `baseline_rnn/training_log` - Complete training history
- `rnn_baseline_submission_file_valsplit.csv` - Evaluation results

## Large Model Files (Not in Git)

The following large files are stored locally but not tracked in Git:

- `baseline_rnn/checkpoint/best_checkpoint` (508MB) - Trained model weights
- `baseline_rnn/checkpoint/val_metrics.pkl` (98MB) - Validation metrics

## Usage

To use these models on Lambda Labs or other systems:

1. Train your own model using the provided training scripts, OR
2. Copy the large checkpoint files separately if needed

The configuration and training logs are preserved here to reproduce the training.
