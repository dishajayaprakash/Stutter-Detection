import matplotlib.pyplot as plt

def plot_training_results(train_bin_losses, val_bin_losses, train_multi_losses, val_multi_losses, train_f1_bins, train_f1_multis, val_f1_bins, val_f1_multis):
    """
    Plots the training and validation loss and F1 scores.

    Args:
        train_bin_losses (list): List of training losses for binary classification.
        val_bin_losses (list): List of validation losses for binary classification.
        train_multi_losses (list): List of training losses for multiclass classification.
        val_multi_losses (list): List of validation losses for multiclass classification.
        train_f1_bins (list): List of training F1 scores for binary classification.
        train_f1_multis (list): List of training F1 scores for multiclass classification.
        val_f1_bins (list): List of validation F1 scores for binary classification.
        val_f1_multis (list): List of validation F1 scores for multiclass classification.
    """

    epochs = range(1, len(train_bin_losses) + 1)

    plt.figure(figsize=(12, 10))

    # Plot training and validation loss for binary classification
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_bin_losses, label='Train Binary Loss')
    plt.plot(epochs, val_bin_losses, label='Validation Binary Loss')
    plt.title('Training and Validation Loss (Binary)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation loss for multiclass classification
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_multi_losses, label='Train Multiclass Loss')
    plt.plot(epochs, val_multi_losses, label='Validation Multiclass Loss')
    plt.title('Training and Validation Loss (Multiclass)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot F1 scores for binary classification
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_f1_bins, label='Train Binary F1')
    plt.plot(epochs, val_f1_bins, label='Validation Binary F1')
    plt.title('Training and Validation F1 Score (Binary)')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # Plot F1 scores for multiclass classification
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_f1_multis, label='Train Multiclass F1')
    plt.plot(epochs, val_f1_multis, label='Validation Multiclass F1')
    plt.title('Training and Validation F1 Score (Multiclass)')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Usage example:
# plot_training_results(train_bin_losses, val_bin_losses, train_multi_losses, val_multi_losses, train_f1_bins, train_f1_multis, val_f1_bins, val_f1_multis)
