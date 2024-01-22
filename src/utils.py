from matplotlib import pyplot as plt


def plot_losses(train_losses, val_losses, save_path):
    num_epochs = len(train_losses)
    epochs = list(range(1, num_epochs + 1))
    plt.plot(epochs, train_losses, label="Training Loss", marker="o", linestyle="-")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o", linestyle="-")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()


def plot_accuracies(val_top1_accuracies, save_path):
    num_epochs = len(val_top1_accuracies)
    epochs = list(range(1, num_epochs + 1))
    plt.plot(epochs, val_top1_accuracies, label="Validation Top-1 Accuracy", marker="o", linestyle="-")

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Top-1 Accuracy')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()



