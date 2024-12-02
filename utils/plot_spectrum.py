import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_eigenvalues(eigenvalues, title, file_prefix, log=False):
    plt.figure(figsize=(7, 5))
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    sns.set_palette("deep")
    x = np.arange(len(eigenvalues))
    # Create stem plot with custom styling
    markerline, stemlines, baseline = plt.stem(x, eigenvalues, label='Eigenvalues/Singular values', basefmt=' ')
    plt.setp(markerline, markersize=4, color=sns.color_palette()[0])
    plt.setp(stemlines, linewidth=1, color=sns.color_palette()[0], alpha=0.7)
 
    # Set labels and title
    plt.xlabel('Index' ) 
    plt.ylabel('Eigenvalue/Singular value' + (' (log scale)' if log else ''))
    plt.title(f'{title}',
            pad=20, fontsize=14)
    # Set y-axis to log scale
    if log: 
        plt.yscale('log')
        # plt.xscale('log')
    # Add legend with better positioning
    plt.legend(frameon=True, fancybox=True, framealpha=0.9, 
            loc='upper right', bbox_to_anchor=(0.99, 0.99))
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_spectrum.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    # Reset seaborn style (optional, if you don't want it to affect subsequent plots)
    sns.reset_defaults()
