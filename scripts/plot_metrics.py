import re
import pandas as pd

def create_dataframes(log_file_path):

    # Initialize lists to store the data
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    hellaswag_accuracies = []

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
                hellaswag_accuracies.append(accuracy)
                continue

    print(f"Parsed {len(train_steps)} training steps")
    print(f"Parsed {len(val_steps)} validation steps")
    print(f"Parsed {len(hellaswag_accuracies)} Hellaswag accuracy entries")

    # Create DataFrames
    train_df = pd.DataFrame({
        'Step': train_steps,
        'Loss': train_losses
    })

    val_df = pd.DataFrame({
        'Step': val_steps,
        'Loss': val_losses,
        'Hellaswag Accuracy': hellaswag_accuracies
    })

    # Display the first few rows of each DataFrame
    print("Training Data:")
    display(train_df.head())

    print("\nValidation Data:")
    display(val_df.head())

    return train_df, val_df


if __name__ == "__main__":
    create_dataframes('../logs/log.txt')
