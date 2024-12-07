import os
import click
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # For custom legend handles
from sklearn.manifold import TSNE
from collections import defaultdict
from pathlib import Path
import sys
import seaborn as sns


sys.path.append(str(Path(__file__).parent.parent))
from util import generate_embeddings_filename 
from utils.custom_colormap import generate_colormap
from utils.pca import pca, sklearn_pca
from utils.plot_spectrum import plot_eigenvalues
from utils.uniform_points import random_uniform_points


@click.command()
@click.option('--model_type', type=click.Choice(['SimCLR', 'SupCon', 'TargetVector'], case_sensitive=False), required=True, help='Type of model to use.')
@click.option('--model_architecture', type=click.Choice(['resnet18', 'resnet34', 'resnet50'], case_sensitive=False), required=True, help='Model architecture to use.')
@click.option('--dataset', type=click.Choice(['cifar10', 'cifar100'], case_sensitive=False), required=True, help='Dataset to use.')
@click.option('--num_embeddings_per_class', type=int, default=-1, help='Number of embeddings to select per class. Use -1 for all.')
@click.option('--embeddings_dir', type=str, default='./embeddings',help='Directory containing the embeddings.')
@click.option('--output_dir', type=str, default='./analyses/plots/spectra', help='Directory to save the t-SNE plots.')
@click.option('--perplexity', type=int, default=30, help='Perplexity parameter for t-SNE.')
@click.option('--n_components', type=int, default=2, help='Number of components for t-SNE.')
@click.option('--head', is_flag=True, help='Use the head of the model.')
@click.option('--epoch', type=int, default=1500, help='Epoch number of the pre-trained model checkpoint.')
@click.option('--normalize', is_flag=True, help='Normalize the embeddings.')
@click.option('--trial', type=str, default="0", help='Trial (Run identifier in a way)')

def main(model_type, model_architecture, dataset, num_embeddings_per_class, 
         embeddings_dir, output_dir,
         perplexity, n_components, head, epoch, normalize, trial):
    # Load embeddings and labels
    dataset_name = dataset
    try:
        embeddings_filename =  generate_embeddings_filename(model_type, dataset_name, model_architecture, -1, head, epoch, trial)
    except FileNotFoundError:
        embeddings_filename =  generate_embeddings_filename(model_type, dataset_name, model_architecture, num_embeddings_per_class, head, epoch, trial)
    embeddings_path = os.path.join(embeddings_dir, embeddings_filename)
    embeddings_data = torch.load(embeddings_path)
    print(f"Loaded embeddings from {embeddings_path}")
    embeddings = embeddings_data['embeddings']
    labels = embeddings_data['labels']  # Extract labels directly from the loaded dictionary

    # Check if the number of embeddings matches the number of labels
    if len(embeddings) != len(labels):
        raise ValueError("The number of embeddings and labels must match.")

    # Create a dictionary to hold indices for each class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label.item()].append(idx)

    # Select a class-balanced subset
    selected_indices = []
    for class_label, indices in class_indices.items():
        if num_embeddings_per_class == -1 or len(indices) < num_embeddings_per_class:
            selected_indices.extend(indices)  # Use all if -1 or not enough samples
        else:
            selected_indices.extend(np.random.choice(indices, num_embeddings_per_class, replace=False))

    # Create subset of embeddings and labels
    subset_embeddings = embeddings[selected_indices]
    subset_labels = labels[selected_indices] 
    if normalize:
        if head:
            assert torch.allclose(torch.norm(subset_embeddings, dim=1, keepdim=True),
                                   torch.ones(subset_embeddings.shape[0], 1))
        else:
            subset_embeddings = subset_embeddings / torch.norm(subset_embeddings, dim=1, keepdim=True)

    eigenvectors, eigenvalues = pca(subset_embeddings)
    _, eigenvalues_sklearn = sklearn_pca(subset_embeddings)
    eigenvalues_sklearn = eigenvalues_sklearn * (len(subset_embeddings) - 1) 
    # debug_both_methods(subset_embeddings)
    # debug_pca(subset_embeddings)
    # if model_type != "SimCLR":
    #     if dataset_name == 'cifar10':
    #         eigenvalues = eigenvalues[:10]
    #     elif dataset_name == 'cifar100':
    #         eigenvalues = eigenvalues[:100]

    os.makedirs(output_dir, exist_ok=True)
    emb_dim = embeddings.shape[1]
    file_prefix =  f'{model_type}_{model_architecture}_{dataset_name}_ep-{epoch}{"_head" if head else ""}_N-{len(subset_embeddings)}_emb-dim-{emb_dim}{"_normalized" if normalize else ""}_trial-{trial}'
    # plot_eigenvalues(eigenvalues,
    #                  f'Eigenvalues of {model_type} ({model_architecture}) {dataset_name} embeddings', 
    #                  os.path.join(output_dir, file_prefix),
    #                  log=False)
    # print(f"Eigenvalues plot saved to {os.path.join(output_dir, f'{file_prefix}_spectrum.png')}")



    # Create a figure with three rows and two columns
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    axs = axs.flatten()
    # Plot eigenvalues
    # sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    sns.set_palette("deep")
    x = np.arange(len(eigenvalues))

    # Create scatter plot for eigenvalues
    axs[0].scatter(x, eigenvalues, label='Eigenvalues')#, basefmt=' ')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Eigenvalue')
    axs[0].set_title(f'Eigenvalues of {model_type}\n({model_architecture}) {dataset_name} embeddings', pad=20, fontsize=14)
    # Logscale
    axs[1].scatter(x, eigenvalues, label='Eigenvalues (log scale)')#, basefmt=' ')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Eigenvalue (log scale)')
    axs[1].set_title(f'Eigenvalues of {model_type}\n({model_architecture}) {dataset_name} embeddings', pad=20, fontsize=14)

    # Create scatter plot for eigenvalues
    axs[4].scatter(x, eigenvalues_sklearn, label='Eigenvalues (sklearn)')#, basefmt=' ')
    axs[4].set_xlabel('Index')
    axs[4].set_ylabel('Eigenvalue')
    axs[4].set_title(f'Eigenvalues of {model_type}\n({model_architecture}) {dataset_name} embeddings', pad=20, fontsize=14)
    # Logscale
    axs[5].scatter(x, eigenvalues_sklearn, label='Eigenvalues (sklearn, log scale)')#, basefmt=' ')
    axs[5].set_xlabel('Index')
    axs[5].set_ylabel('Eigenvalue (log scale)')
    axs[5].set_title(f'Eigenvalues of {model_type}\n({model_architecture}) {dataset_name} embeddings', pad=20, fontsize=14)

    # Set y-axis to log scale if needed
    # Uncomment the next line if you want to use log scale
    axs[1].set_yscale('log')
    axs[1].grid(True)
    axs[5].set_yscale('log')
    axs[5].grid(True)

    # Add legend
    axs[0].legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    axs[1].legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    axs[4].legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    axs[5].legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper right', bbox_to_anchor=(0.99, 0.99))

    # Generate random points for comparison
    if model_type == "SimCLR":
        N_random = 2048
    else:
        N_random = 10 if dataset_name == 'cifar10' else 100
    points_random_t = random_uniform_points(N_random, n=emb_dim)
    points_random_t = torch.tensor(points_random_t)
    _, eigenvalues_random = pca(points_random_t)
    x_random = np.arange(len(eigenvalues_random))
    axs[2].scatter(x_random, eigenvalues_random, label='Eigenvalues/Singular values')#, basefmt=' ')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel('Eigenvalue')
    axs[2].set_title(f'Eigenvalues of {N_random} random uniform points', pad=20, fontsize=14)
    axs[3].scatter(x_random, eigenvalues_random, label='Eigenvalues/Singular values')#, basefmt=' ')
    axs[3].set_xlabel('Index')
    axs[3].set_ylabel('Eigenvalue (log scale)')
    axs[3].set_title(f'Eigenvalues of {N_random} random uniform points', pad=20, fontsize=14)
    axs[3].set_yscale('log')
    # dont show the very last eigenvalue, since its much smaller 
    axs[3].set_ylim(bottom=eigenvalues_random[-2])
    axs[3].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_spectrum.png"), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Eigenvalues plot saved to {os.path.join(output_dir, f'{file_prefix}_spectrum.png')}")


def debug_pca(t: torch.Tensor):
    from sklearn.decomposition import PCA
    # Your PCA
    t_centered = t - t.mean(dim=0, keepdim=True)
    print("Centered data stats:")
    print("Mean of centered data:", t_centered.mean(dim=0)[:5])
    print("Norm of centered data:", torch.norm(t_centered))
    
    u, s, vh = torch.linalg.svd(t_centered)
    print("\nDirect SVD values:")
    print("Singular values:", s[:5])
    print("Squared singular values:", s[:5]**2)
    
    # Sklearn PCA
    t_np = t.cpu().numpy()
    pca_ = PCA(n_components=min(t.shape[1], t.shape[0]))
    pca_.fit(t_np)
    print("\nSklearn values:")
    print("Singular values:", pca_.singular_values_[:5])
    print("Explained variance:", pca_.explained_variance_[:5])
    print("Explained variance * (n-1):", pca_.explained_variance_[:5] * (t.shape[0]-1))

def debug_both_methods(t: torch.Tensor):
    from sklearn.decomposition import PCA
    # Your PCA
    t_centered = t - t.mean(dim=0, keepdim=True)
    print("Direct method centered data:")
    print("Norm:", torch.norm(t_centered))
    print("Mean:", t_centered.mean(dim=0)[:5])
    
    # Sklearn
    t_np = t.cpu().numpy()
    pca_ = PCA(n_components=min(t.shape[1], t.shape[0]))
    pca_.fit(t_np)
    t_centered_sklearn = t_np - pca_.mean_
    print("\nSklearn centered data:")
    print("Norm:", np.linalg.norm(t_centered_sklearn))
    print("Mean:", t_centered_sklearn.mean(axis=0)[:5])

def debug_centering(t: torch.Tensor):
    # Your method
    mean1 = t.mean(dim=0, keepdim=True)
    t_centered1 = t - mean1
    
    # Sklearn's method (but implemented manually)
    t_np = t.cpu().numpy()
    mean2 = np.mean(t_np, axis=0)
    t_centered2 = t_np - mean2
    
    print("Your method:")
    print("Mean shape:", mean1.shape)
    print("First few means:", mean1[0, :5])
    
    print("\nSklearn method:")
    print("Mean shape:", mean2.shape)
    print("First few means:", mean2[:5])
    
    # Convert to same format for comparison
    mean1_np = mean1.cpu().numpy()
    print("\nAbsolute difference in means:", np.abs(mean1_np - mean2).max())
    
    # Check if there's any numerical precision difference
    print("Your centered data dtype:", t_centered1.dtype)
    print("Sklearn centered data dtype:", t_centered2.dtype)


if __name__ == '__main__':
    main()