import torch
import numpy as np
from utils.uniform_points import greedy_uniform_points, random_uniform_points
from losses import SupConLoss
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_points_with_losses(points, target_batch_size=2048, temperature_simclr=0.5, temperature_supcon=0.1):
    """Evaluate points using SimCLR and SupCon losses.
    Args:
        points: numpy array of shape [num_points, dim]
        target_batch_size: desired total number of points after repeating
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to torch
    points_tensor = torch.from_numpy(points).float()
    num_points = len(points)
    
    # Calculate number of repeats needed to get close to target_batch_size
    num_repeats = target_batch_size // num_points
    
    # Create repeated points and labels
    points_repeated = points_tensor.repeat(num_repeats, 1)
    labels = torch.arange(num_points).repeat_interleave(num_repeats)
    
    # Add any remaining points needed to reach exactly target_batch_size
    remaining = target_batch_size - len(points_repeated)
    if remaining > 0:
        extra_points = points_tensor[:remaining]
        extra_labels = torch.arange(min(remaining, num_points))
        points_repeated = torch.cat([points_repeated, extra_points])
        labels = torch.cat([labels, extra_labels])
    
    # Reshape for SupConLoss format [batch_size, n_views, dim]
    points_reshaped = points_repeated.unsqueeze(1)
    # Add second view (identical for this test)
    points_2view = torch.cat([points_reshaped, points_reshaped], dim=1)
    
    # Initialize losses
    simclr_loss = SupConLoss(temperature=temperature_simclr).to(device)
    supcon_loss = SupConLoss(temperature=temperature_supcon).to(device)
    
    # Move to device
    points_2view = points_2view.to(device)
    labels = labels.to(device)
    
    # Compute losses
    simclr_value = simclr_loss(points_2view)
    supcon_value = supcon_loss(points_2view, labels)
    
    # Compute similarity statistics on original points
    sim_matrix = cosine_similarity(points)
    np.fill_diagonal(sim_matrix, -1)  # exclude self-similarity
    max_sim = np.max(sim_matrix)
    mean_sim = np.mean(sim_matrix)
    min_sim = np.min(sim_matrix)
    
    return {
        'simclr_loss': simclr_value.item(),
        'supcon_loss': supcon_value.item(),
        'max_similarity': max_sim,
        'mean_similarity': mean_sim,
        'min_similarity': min_sim,
        'num_repeats': num_repeats,
        'total_points': len(points_repeated)
    }

def test_uniform_points(num_points=100, dim=512, n_iter=1000):
    print(f"\nTesting uniform points generation with {num_points} points in {dim} dimensions")
    print("=" * 80)
    
    # Generate random uniform points as baseline
    random_points = random_uniform_points(num_points, dim)
    print("\nRandom uniform points metrics:")
    random_metrics = evaluate_points_with_losses(random_points)
    for key, value in random_metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Generate greedy uniform points
    print(f"\nGenerating greedy uniform points with {n_iter} iterations...")
    greedy_points, mean_similarity = greedy_uniform_points(num_points, dim, n_iter=n_iter)
    print("\nGreedy uniform points metrics:")
    greedy_metrics = evaluate_points_with_losses(greedy_points)
    for key, value in greedy_metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Compare similarity distributions
    random_sims = cosine_similarity(random_points).flatten()
    greedy_sims = cosine_similarity(greedy_points).flatten()
    
    plt.figure(figsize=(10, 5))
    plt.hist(random_sims, bins=50, alpha=0.5, label='Random uniform')
    plt.hist(greedy_sims, bins=50, alpha=0.5, label='Greedy uniform')
    plt.title(f'Distribution of cosine similarities ({num_points} points)')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f'similarity_dist_{num_points}points.png')
    plt.close()

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test with both 10 and 100 points
    test_uniform_points(num_points=10, n_iter=1000)
    test_uniform_points(num_points=100, n_iter=1000)

if __name__ == "__main__":
    main() 