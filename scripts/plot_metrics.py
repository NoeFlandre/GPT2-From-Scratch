import re
import os
import pandas as pd
import matplotlib.pyplot as plt

def save_plot(filepath, title, ylabel, xlabel="Training step"):
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

def parse_log(log_file_path):
    # Initialize lists to store the data
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    hellaswag_accuracies = []
    hellaswag_steps = []

    # Regular expressions to match the different log entries
    train_loss_pattern = re.compile(r'^(\d+): Loss (\d+\.\d+)')
    val_loss_pattern = re.compile(r'^(\d+): validation loss: (\d+\.\d+)')
    hellaswag_pattern = re.compile(r'^(\d+): Hellaswag accuracy normalized: (\d+\.\d+)')

    # Parse the log file
    with open(log_file_path, 'r') as f:
        for line in f:
            # Check for training loss
            train_match = train_loss_pattern.match(line)
            if train_match:
                step = int(train_match.group(1))
                loss = float(train_match.group(2))
                train_steps.append(step)
                train_losses.append(loss)
                continue

            # Check for validation loss
            val_match = val_loss_pattern.match(line)
            if val_match:
                step = int(val_match.group(1))
                loss = float(val_match.group(2))
                val_steps.append(step)
                val_losses.append(loss)
                continue

            # Check for Hellaswag accuracy
            hellaswag_match = hellaswag_pattern.match(line)
            if hellaswag_match:
                step = int(hellaswag_match.group(1))
                accuracy = float(hellaswag_match.group(2))
                hellaswag_steps.append(step)
                hellaswag_accuracies.append(accuracy)
                continue

    print(f"Parsed {len(train_steps)} training steps")
    print(f"Parsed {len(val_steps)} validation steps")
    print(f"Parsed {len(hellaswag_steps)} Hellaswag steps")
    print(f"Parsed {len(hellaswag_accuracies)} Hellaswag accuracy entries")

    return train_steps, train_losses, val_steps, val_losses, hellaswag_steps, hellaswag_accuracies

def create_dataframes(log_file_path):

    train_steps, train_losses, val_steps, val_losses, hellaswag_steps, hellaswag_accuracies = parse_log(log_file_path)

    # Create DataFrames
    train_df = pd.DataFrame({
        'Step': train_steps,
        'Loss': train_losses
    })

    val_df = pd.DataFrame({
        'Step': val_steps,
        'Loss': val_losses,
    })

    hellaswag_df = pd.DataFrame({
        'Step': hellaswag_steps,
        'Hellaswag Accuracy': hellaswag_accuracies,
    })

    val_df = val_df.merge(hellaswag_df, on="Step", how="left")

    # Display the first few rows of each DataFrame
    print("Training Data:")
    print(train_df.head())

    print("\nValidation Data:")
    print(val_df.head())

    return train_df, val_df

def plot_figures(log_file_path):

    os.makedirs("plots", exist_ok=True)

    train_df, val_df = create_dataframes(log_file_path)

    plt.figure(figsize=(14, 6))
    plt.plot(train_df['Step'], train_df['Loss'], color='blue', alpha=0.5, label='Raw Training Loss')

    # Add smoothed version of the loss curve
    smoothed_loss = train_df['Loss'].rolling(100).mean()
    plt.plot(train_df['Step'], smoothed_loss, color='darkblue', linewidth=2, label='Smoothed')

    save_plot("plots/training_loss.png", title="Training loss over time", ylabel="Loss")

    plt.figure(figsize=(14, 6))
    plt.plot(val_df['Step'], val_df['Loss'], color='red', marker='o', linestyle='-', linewidth=2, label='Validation Loss')

    # Add GPT-2 baseline line
    gpt2_baseline = 3.2924
    plt.axhline(y=gpt2_baseline, color='blue', linestyle='--', linewidth=2, label='GPT-2 (124M) Baseline')

    save_plot("plots/validation_loss.png", title="Validation loss over time", ylabel="Loss")

    plt.figure(figsize=(14, 6))
    plt.plot(val_df['Step'], val_df['Hellaswag Accuracy'], color='green', marker='o', linestyle='-', linewidth=2, label='Hellaswag Accuracy')

    # Add GPT-2 baseline line
    gpt2_baseline_accuracy = 0.294463
    plt.axhline(y=gpt2_baseline_accuracy, color='blue', linestyle='--', linewidth=2, label='GPT-2 (124M) Baseline')

    save_plot("plots/hellaswag_accuracy.png", title="Hellaswag accuracy over time", ylabel="Accuracy")

if __name__ == "__main__":
    plot_figures('logs/log.txt')
