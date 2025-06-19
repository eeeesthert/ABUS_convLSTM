import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(train_losses, val_losses, save_path='loss_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def print_test_results(test_results):
    fdr = np.mean([r['fdr'] for r in test_results])
    adr = np.mean([r['adr'] for r in test_results])
    md = np.mean([r['md'] for r in test_results])
    sd = np.mean([r['sd'] for r in test_results])
    hd = np.mean([r['hd'] for r in test_results])

    print("\nTest Results:")
    print(f"Final Drift Rate (FDR): {fdr:.4f}%")
    print(f"Average Drift Rate (ADR): {adr:.4f}%")
    print(f"Maximum Drift (MD): {md:.4f} mm")
    print(f"Sum of Drift (SD): {sd:.4f} mm")
    print(f"Bidirectional Hausdorff Distance (HD): {hd:.4f} mm")