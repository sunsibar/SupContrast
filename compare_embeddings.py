import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.uniform_points import random_uniform_points
from losses import SupConLoss

def get_cosine_schedule(optimizer, num_epochs, initial_lr):
    """Create cosine learning rate schedule."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0.0
    )

class SimpleEmbeddingNet(nn.Module):
    """Two-layer network to transform embeddings."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x):
        # Input shape: [batch_size, n_views, embedding_dim]
        batch_size, n_views, embedding_dim = x.shape
        
        # Reshape to [batch_size * n_views, embedding_dim]
        x = x.view(-1, embedding_dim)
        
        # Apply network
        x = self.net(x)
        
        # Normalize output
        x = nn.functional.normalize(x, dim=-1)
        
        # Reshape back
        x = x.view(batch_size, n_views, embedding_dim)
        return x

def optimize_embeddings_contrastive(initial_embeddings, labels=None, temperature=0.5, 
                                  num_epochs=100, lr=.1, device='cuda'):
    """Optimize embeddings using SimCLR/SupCon loss."""
    # Initialize network and wrap embeddings
    net = SimpleEmbeddingNet(initial_embeddings.shape[-1]).to(device)
    embeddings = nn.Parameter(initial_embeddings.clone())
    
    # Initialize loss and optimizer
    loss_fn = SupConLoss(temperature=temperature).to(device)
    optimizer = optim.Adam([
        {'params': net.parameters()},
        {'params': [embeddings]}
    ], lr=lr)
    scheduler = get_cosine_schedule(optimizer, num_epochs, lr)
    
    losses = []
    learning_rates = []
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Forward pass through network
        transformed_embeddings = net(embeddings)
        loss = loss_fn(transformed_embeddings, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Normalize base embeddings after each step
        with torch.no_grad():
            embeddings.data = nn.functional.normalize(embeddings.data, dim=-1)
            
        losses.append(loss.item())
        learning_rates.append(scheduler.get_last_lr()[0])
        pbar.set_description(f"Loss: {loss.item():.4e}, LR: {scheduler.get_last_lr()[0]:.4e}")
    
    # Get final transformed embeddings
    with torch.no_grad():
        final_embeddings = net(embeddings)
    
    return final_embeddings.detach(), losses, learning_rates

def plot_loss_comparison(losses_dict, losses_history, learning_rates, title, filename):
    """Create comparison plots: bar plot and training curves.
    
    Args:
        losses_dict: OrderedDict with keys 'Uniform', 'Optimized' containing loss values
        losses_history: List of loss values during training
        learning_rates: List of learning rates during training
        title: Title for the plot
        filename: Where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar plot
    x = np.arange(2)  # always 2 bars: uniform and optimized
    ax1.bar(x, [losses_dict['Uniform'], losses_dict['Optimized']], width=0.6)
    
    # Customize bar plot
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Uniform', 'Optimized'], rotation=45)
    ax1.set_title(f'{title} - Final Values')
    ax1.set_ylabel('Loss Value')
    
    # Add value labels on top of bars
    for i, v in enumerate([losses_dict['Uniform'], losses_dict['Optimized']]):
        ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # Training curve with two y-axes
    color1, color2 = 'tab:blue', 'tab:orange'
    ax2_twin = ax2.twinx()
    
    # Plot loss
    line1 = ax2.plot(losses_history, color=color1, label='Training loss')
    ax2.axhline(y=losses_dict['Uniform'], color='r', linestyle='--', 
                label='Uniform embeddings loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss', color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)
    
    # Plot learning rate
    line2 = ax2_twin.plot(learning_rates, color=color2, label='Learning rate')
    ax2_twin.set_ylabel('Learning Rate', color=color2)
    ax2_twin.tick_params(axis='y', labelcolor=color2)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels)
    
    ax2.set_title(f'{title} - Training Progress')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def compare_simclr_vs_uniform(N=512, embedding_dim=128, num_epochs=100):
    """Compare random uniform embeddings to SimCLR-optimized embeddings."""
    print(f"\nComparing {N} SimCLR-optimized embeddings vs random uniform")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate random uniform embeddings
    uniform_points = random_uniform_points(N, embedding_dim)
    uniform_embeddings = torch.from_numpy(uniform_points).float().unsqueeze(1).to(device)
    
    # Create initial random embeddings for optimization
    initial_embeddings = nn.Parameter(torch.randn(N, 2, embedding_dim).to(device))
    initial_embeddings.data = nn.functional.normalize(initial_embeddings.data, dim=-1)
    
    # Initialize loss function
    simclr_loss = SupConLoss(temperature=0.5).to(device)
    
    # Compute SimCLR loss for uniform embeddings (without transformation)
    uniform_loss = simclr_loss(uniform_embeddings.repeat(1, 2, 1))
    print(f"\nSimCLR Loss for {N} uniform embeddings: {uniform_loss.item():.4f}")
    
    # Optimize embeddings using SimCLR loss
    optimized_embeddings, losses, learning_rates = optimize_embeddings_contrastive(
        initial_embeddings, labels=None, temperature=0.5, num_epochs=num_epochs, 
        lr=.1, device=device
    )
    
    final_loss = losses[-1]
    print(f"Final SimCLR Loss: {final_loss:.4f}")
    
    # Create comparison plots
    from collections import OrderedDict
    losses_dict = OrderedDict([
        ('Uniform', uniform_loss.item()),
        ('Optimized', final_loss)
    ])
    
    plot_loss_comparison(
        losses_dict,
        losses,
        learning_rates,
        'SimCLR Loss',
        'simclr_comparison.png'
    )
    
    return uniform_embeddings, optimized_embeddings

def compare_supcon_vs_uniform(N=512, num_classes=10, embedding_dim=128, num_epochs=100):
    """Compare random uniform embeddings to SupCon-optimized embeddings."""
    print(f"\nComparing {N} SupCon-optimized embeddings ({num_classes} classes) vs random uniform")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate random uniform embeddings
    uniform_points = random_uniform_points(num_classes, embedding_dim)
    uniform_embeddings = torch.from_numpy(uniform_points).float().unsqueeze(1).to(device)

    # Create balanced labels
    samples_per_class = N // num_classes
    remainder = N % num_classes
    labels = []
    for i in range(num_classes):
        count = samples_per_class + (1 if i < remainder else 0)
        labels.extend([i] * count)
    labels = torch.tensor(labels[:N]).to(device)

    # Repeat uniform embeddings to fit to labels
    repeated_embeddings = []
    for i in range(num_classes):
        count = samples_per_class + (1 if i < remainder else 0)
        repeated_embeddings.append(uniform_embeddings[i].repeat(count, 1, 1))
    uniform_embeddings = torch.cat(repeated_embeddings, dim=0)
    uniform_embeddings = uniform_embeddings[:N]
    
    # Create initial random embeddings for optimization
    initial_embeddings = nn.Parameter(torch.randn(N, 2, embedding_dim).to(device))
    initial_embeddings.data = nn.functional.normalize(initial_embeddings.data, dim=-1)
    
    # Initialize loss function
    supcon_loss = SupConLoss(temperature=0.1).to(device)
    
    # Compute SupCon loss for uniform embeddings
    uniform_loss = supcon_loss(uniform_embeddings.repeat(1, 2, 1), labels)
    print(f"\nSupCon Loss for uniform embeddings: {uniform_loss.item():.4f}")
    
    # Optimize embeddings using SupCon loss
    optimized_embeddings, losses, learning_rates = optimize_embeddings_contrastive(
        initial_embeddings, labels=labels, temperature=0.1, num_epochs=num_epochs,
        lr=.1, device=device
    )
    
    final_loss = losses[-1]
    print(f"Final SupCon Loss: {final_loss:.4f}")
    
    # Create comparison plots
    from collections import OrderedDict
    losses_dict = OrderedDict([
        ('Uniform', uniform_loss.item()),
        ('Optimized', final_loss)
    ])
    
    plot_loss_comparison(
        losses_dict,
        losses,
        learning_rates,
        f'SupCon Loss ({num_classes} classes)',
        f'supcon_comparison_{num_classes}classes.png'
    )
    
    return uniform_embeddings, optimized_embeddings, labels

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Compare SimCLR vs uniform
    compare_simclr_vs_uniform(N=2048, embedding_dim=128, num_epochs=1000)
    
    # Compare SupCon vs uniform for different numbers of classes
    compare_supcon_vs_uniform(N=2048, num_classes=10, embedding_dim=128, num_epochs=1000)
    compare_supcon_vs_uniform(N=2048, num_classes=100, embedding_dim=128, num_epochs=1000)
    # compare_supcon_vs_uniform(N=512, num_classes=10, embedding_dim=128, num_epochs=100)
    # compare_supcon_vs_uniform(N=512, num_classes=100, embedding_dim=128, num_epochs=100)

if __name__ == "__main__":
    main() 