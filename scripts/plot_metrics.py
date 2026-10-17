import re

def parse_log(log_file_path):

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

    return train_steps, train_losses, val_steps, val_losses, hellaswag_accuracies


if __name__ == "__main__":
    parse_log('../logs/log.txt')
