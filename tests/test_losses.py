import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path 
import sys
sys.path.append(str(Path(__file__).parent.parent))
from losses import TargetVectorMSELoss, SupConLoss


def create_random_embeddings(batch_size, embedding_dim, num_classes, n_views=2):
    """Create random embeddings and corresponding labels."""
    # Calculate samples per class to get close to even distribution
    samples_per_class = batch_size // num_classes
    remainder = batch_size % num_classes
    
    # Create labels
    labels = []
    for i in range(num_classes):
        count = samples_per_class + (1 if i < remainder else 0)
        labels.extend([i] * count)
    
    # Ensure we have exactly batch_size labels
    labels = labels[:batch_size]
    labels = torch.tensor(labels)
    
    # Create random embeddings and normalize them
    embeddings = torch.randn(batch_size, n_views, embedding_dim)
    embeddings = nn.functional.normalize(embeddings, dim=-1)
    
    return embeddings, labels

def evaluate_losses(embeddings, labels, target_vectors, supcon_loss, mse_loss, simclr_loss):
    """Evaluate both losses on given embeddings."""
    # For SimCLR/SupCon, we need two views
    if embeddings.shape[1] == 1:
        # If we only have one view, duplicate it 
        embeddings_2view = torch.cat([
            embeddings, embeddings
        ], dim=1)
    else:
        embeddings_2view = embeddings

    # Evaluate SupConLoss
    supcon_value = supcon_loss(embeddings_2view, labels)
    
    # Evaluate SimCLR loss (unsupervised)
    simclr_value = simclr_loss(embeddings_2view)
    
    # Evaluate MSE Loss
    mse_value = mse_loss(embeddings, labels)

    # mean cosine similarity of embeddings to their respective class target vector
    cosine_similarities = []
    for i in range(len(labels.unique())):
        class_embeddings = embeddings[labels == i]
        class_target_vector = target_vectors[i:i+1]
        cosine_similarities.append(nn.functional.cosine_similarity(class_embeddings, class_target_vector, dim=-1)) 
    cosine_similarities = torch.cat(cosine_similarities).mean()

    return {
        'supcon': supcon_value.item(),
        'simclr': simclr_value.item(),
        'mse': mse_value.item(),
        'cosine': cosine_similarities.item()
    }

def optimize_embeddings(initial_embeddings, labels, mse_loss, num_epochs=1000, lr=0.01):
    """Optimize embeddings using TargetVectorMSELoss."""
    embeddings = nn.Parameter(initial_embeddings.clone())
    optimizer = optim.Adam([embeddings], lr=lr)
    
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        loss = mse_loss(embeddings, labels)
        loss.backward()
        optimizer.step()
        
        # Normalize embeddings after each step
        with torch.no_grad():
            embeddings.data = nn.functional.normalize(embeddings.data, dim=-1)
            
        pbar.set_description(f"Loss: {loss.item():.4e}")
    
    return embeddings.detach()

def run_test(num_classes):
    print(f"\nRunning test with {num_classes} classes")
    print("=" * 50)
    
    # Parameters
    batch_size = 2048
    embedding_dim = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize losses
    mse_loss = TargetVectorMSELoss(num_classes, embedding_dim).to(device)
    supcon_loss = SupConLoss(temperature=0.1).to(device)
    simclr_loss = SupConLoss(temperature=0.5).to(device)
    
    # Create initial random embeddings
    initial_embeddings, labels = create_random_embeddings(batch_size, embedding_dim, num_classes)
    initial_embeddings = initial_embeddings.to(device)
    labels = labels.to(device)
    
    # Evaluate initial state
    print("\nInitial state:")
    metrics_init = evaluate_losses(initial_embeddings, labels, 
                                 mse_loss.target_vectors, supcon_loss, mse_loss, simclr_loss)
    print(f"SupConLoss: {metrics_init['supcon']:.6f}")
    print(f"SimCLR Loss: {metrics_init['simclr']:.6f}")
    print(f"MSE Loss: {metrics_init['mse']:.4e}")
    print(f"Cosine Similarity: {metrics_init['cosine']:.6f}")
    
    # Optimize embeddings
    print("\nOptimizing embeddings...")
    optimized_embeddings = optimize_embeddings(initial_embeddings, labels, mse_loss, lr=0.01)
    
    # Evaluate optimized embeddings
    print("\nAfter optimization:")
    metrics_opt = evaluate_losses(optimized_embeddings, labels, 
                                mse_loss.target_vectors, supcon_loss, mse_loss, simclr_loss)
    print(f"SupConLoss: {metrics_opt['supcon']:.6f}")
    print(f"SimCLR Loss: {metrics_opt['simclr']:.6f}")
    print(f"MSE Loss: {metrics_opt['mse']:.4e}")
    print(f"Cosine Similarity: {metrics_opt['cosine']:.6f}")
    
    # Evaluate target vectors
    # Repeat target vectors to match batch size
    repeats_needed = batch_size // num_classes
    remainder = batch_size % num_classes
    
    target_embeddings = []
    target_labels = []
    for i in range(num_classes):
        count = repeats_needed + (1 if i < remainder else 0)
        target_embeddings.extend([mse_loss.target_vectors[i]] * count)
        target_labels.extend([i] * count)
    
    target_embeddings = torch.stack(target_embeddings).unsqueeze(1).to(device)
    target_labels = torch.tensor(target_labels).to(device)
    
    print("\nTarget vectors evaluation:")
    metrics_target = evaluate_losses(target_embeddings[:batch_size], 
                                   target_labels[:batch_size],
                                   mse_loss.target_vectors, supcon_loss, mse_loss, simclr_loss)
    print(f"SupConLoss: {metrics_target['supcon']:.6f}")
    print(f"SimCLR Loss: {metrics_target['simclr']:.6f}")
    print(f"MSE Loss: {metrics_target['mse']:.4e}")
    print(f"Cosine Similarity: {metrics_target['cosine']:.6f}")


    # Optimize embeddings
    print("\nOptimizing embeddings further...")
    optimized_embeddings = optimize_embeddings(optimized_embeddings, labels, supcon_loss, lr=5)
    metrics_opt = evaluate_losses(optimized_embeddings, labels, 
                                mse_loss.target_vectors, supcon_loss, mse_loss, simclr_loss)
    print(f"SupConLoss: {metrics_opt['supcon']:.6f}")
    print(f"SimCLR Loss: {metrics_opt['simclr']:.6f}")
    print(f"MSE Loss: {metrics_opt['mse']:.4e}")
    print(f"Cosine Similarity: {metrics_opt['cosine']:.6f}")



def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests for both 10 and 100 classes
    run_test(10)
    run_test(100)

if __name__ == "__main__":
    main() 