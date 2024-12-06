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

sys.path.append(str(Path(__file__).parent.parent))
from util import generate_embeddings_filename 
from utils.custom_colormap import generate_colormap

@click.command()
@click.option('--model_type', type=click.Choice(['SimCLR', 'SupCon', 'TargetVector'], case_sensitive=False), required=True, help='Type of model to use.')
@click.option('--model_architecture', type=click.Choice(['resnet18', 'resnet34', 'resnet50'], case_sensitive=False), required=True, help='Model architecture to use.')
@click.option('--dataset', type=click.Choice(['cifar10', 'cifar100'], case_sensitive=False), required=True, help='Dataset to use.')
@click.option('--num_embeddings_per_class', type=int, default=1000, help='Number of embeddings to select per class. Use -1 for all.')
@click.option('--embeddings_dir', type=str, default='./embeddings',help='Directory containing the embeddings.')
@click.option('--output_dir', type=str, default='./analyses/plots/tSNE', help='Directory to save the t-SNE plots.')
@click.option('--perplexity', type=int, default=30, help='Perplexity parameter for t-SNE.')
@click.option('--n_components', type=int, default=2, help='Number of components for t-SNE.')
@click.option('--head', is_flag=True, help='Use the head of the model.')
@click.option('--epoch', type=int, default=1500, help='Epoch number of the pre-trained model checkpoint.')
@click.option('--trial', type=str, default="0", help='Trial number.')
def main(model_type, model_architecture, dataset, num_embeddings_per_class, embeddings_dir, output_dir, perplexity, n_components, head, epoch, trial):
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

    # Compute t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(subset_embeddings.numpy())

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    cmap = generate_colormap((10 if dataset_name == 'cifar10' else 100)) 
    # colors = plt.cm.hsv(np.linspace(0, 1, 100))  # 10 colors from the viridis / hsv colormap
    # markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X']  # 10 different markers

    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=subset_labels, cmap=cmap, s=4, alpha=0.2, marker='o')
    # Create a scatter plot with different colors and markers
    # for i in range(100):  # Assuming you have 100 different labels
    #     # Select points for the current label
    #     label_indices = np.where(subset_labels == i)[0]
    #     plt.scatter(tsne_results[label_indices, 0], tsne_results[label_indices, 1],
    #                 color=colors[i ],  # Color based on label # // 10
    #                 marker=markers[i % 10],  # Marker based on label % 10
    #                 alpha=0.2, s=4, label=f'Label {i}' if i % 10 == 0 else "")  # Add label only for the first of each color
    # Create custom legend handles
    legend_handles = [mlines.Line2D([], [], color=cmap(i), marker='o', linestyle='',
                                  markersize=10, label=f'Label {i + 0}') 
        for i in range(10)  ]
 
    plt.legend(handles=legend_handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f't-SNE of Class-Balanced Embeddings\n(N per class = {num_embeddings_per_class})\n({model_type}, {model_architecture}, {dataset_name}, Epoch {epoch})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    emb_dim = embeddings.shape[1]
    # Generate the plot filename using the utility function
    tsne_filename = f'tSNE_{model_type}_{dataset_name}_dim-{emb_dim}_{model_architecture}_perplexity_{perplexity}_n_components_{n_components}_{"num_embeddings_" + str(num_embeddings_per_class) if num_embeddings_per_class != -1 else "all"}{"_head" if head else ""}_ep-{epoch}_trial-{trial}.png'
    plot_file = os.path.join(output_dir, tsne_filename)
    plt.savefig(plot_file)
    plt.close()
    
    print(f"t-SNE plot saved to {plot_file}")

if __name__ == '__main__':
    main()