import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path 
import sys
import math
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


def test_label_smoothing_supcon():
    """Test that label smoothing is correctly implemented for SupCon."""
    print("\nTesting Label Smoothing for SupCon")
    print("=" * 50)
    raise AssertionError("Adjust this based on the simclr version, I think some of my assumptions were wrong about how weighting happens "
                         "(you weight after taking the log or something)")
    
    # Create a simple test case with perfect alignment
    batch_size = 4
    n_views = 2
    embedding_dim = 2
    temperature = 0.1
    
    # Create identical embeddings for each class (2 classes, 2 samples each)
    # Class 0: [1, 0], Class 1: [0, 1]
    embeddings = torch.zeros(batch_size, n_views, embedding_dim)
    embeddings[0:2, :, 0] = 1.0  # First two samples are [1, 0]
    embeddings[2:4, :, 1] = 1.0  # Last two samples are [0, 1]
    
    labels = torch.tensor([0, 0, 1, 1])
    
    # Test with different label smoothing values
    smoothing_values = [0.0, 0.1, 0.3, 0.5]
    
    for smoothing in smoothing_values:
        # Create loss function with current smoothing value
        loss_fn = SupConLoss(temperature=temperature, label_smoothing=smoothing)
        
        # Calculate loss
        loss = loss_fn(embeddings, labels)
        
        # Calculate expected loss with label smoothing
        # With smoothing, positive pairs get weight (1-smoothing)
        # Negative pairs get weight smoothing/(batch_size-1)

        pos_weight = 1.0 - smoothing 
        neg_weight = smoothing / (batch_size - 1) / 2

        # Simplified calculation for this specific case
        exp_pos = math.exp(1.0 / temperature)
        exp_neg = math.exp(0.0 / temperature)
        
        # other view of self, plus two views of other of same label, = 3
        numerator = pos_weight * exp_pos * 3 + neg_weight * exp_neg * 4
        
        # plus two views of other of different label
        denominator = 4 * exp_neg + 3 * exp_pos
        
        # Expected loss is -log(numerator/denominator)
        expected_loss = -math.log(numerator / denominator)
        
        print(f"Label smoothing: {smoothing}")
        print(f"Actual loss: {loss.item():.6f}")
        print(f"Expected loss: {expected_loss:.6f}")
        print(f"Difference: {abs(loss.item() - expected_loss):.6f}")
        
        # Check if the loss is close to expected
        assert abs(loss.item() - expected_loss) < 1e-5, f"Label smoothing test failed for smoothing={smoothing}"
        
    print("Label smoothing test for SupCon passed!")


def test_label_smoothing_simclr():
    """Test that label smoothing is correctly implemented for SimCLR."""
    print("\nTesting Label Smoothing for SimCLR")
    print("=" * 50)
    
    # Create a simple test case with perfect alignment
    batch_size = 4
    n_views = 2
    embedding_dim = 4
    temperature = 0.1
    
    # Create identical embeddings for each sample
    # Sample 0: [1, 0, 0, 0]
    # Sample 1: [0, 1, 0, 0]
    # Sample 2: [0, 0, 1, 0]
    # Sample 3: [0, 0, 0, 1]
    embeddings = torch.zeros(batch_size, n_views, embedding_dim)
    embeddings[0, :, 0] = 1.0
    embeddings[1, :, 1] = 1.0
    embeddings[2, :, 2] = 1.0
    embeddings[3, :, 3] = 1.0
    
    # Test with different label smoothing values
    smoothing_values = [0.0, 0.1, 0.3, 0.5]
    
    for smoothing in smoothing_values:
        # Create loss function with current smoothing value
        loss_fn = SupConLoss(temperature=temperature, label_smoothing=smoothing)
        
        # Calculate loss (SimCLR mode - no labels)
        loss = loss_fn(embeddings)
        
        # Calculate expected loss with label smoothing
        # In SimCLR, only the same sample with different views is positive
        # With smoothing, positive pairs (diagonal) get weight (1-smoothing)
        # Negative pairs (off-diagonal) get weight smoothing/(batch_size-1)
        pos_weight = 1.0 - smoothing
        neg_weight = smoothing / (batch_size - 1) / 2
        
        # For each anchor, calculate expected loss
        # For sample 0 (view 1):
        # - Positive: sample 0 (view 2)
        # - Negative: all other samples
        
        # Simplified calculation for this specific case
        # Samples 0 and 2 have embedding [1,0]
        # Samples 1 and 3 have embedding [0,1]
        # So dot products are:
        # - Same embedding: 1
        # - Different embedding: 0
        
        # For sample 0 (view 1):
        # - Positive: sample 0 (view 2) - similarity = 1
        # - Negative: samples 1, 2, 3 (all views) - similarity = 0 or -1
        
        # For SimCLR, the mask is the identity matrix
        # With smoothing, diagonal elements are (1-smoothing)
        # Off-diagonal elements are smoothing/(batch_size-1)
        
        # Expected loss calculation
        exp_pos = math.exp(1.0 / temperature)  # exp(similarity/temp) for positive pair
        exp_neg = math.exp(0.0)  # exp(similarity/temp) for negative pair
        
        log_numerator = pos_weight * math.log(exp_pos) + neg_weight * math.log(exp_neg) * (batch_size - 1) * 2
        # Denominator is sum of all exp terms (excluding self-contrast)
        denominator = exp_pos + 6 * exp_neg
        
        # Expected loss is -log(numerator/denominator) 
        expected_loss = -log_numerator +math.log( denominator)
        # almost the same

        # for some reason there's this scaling factor in the loss function
        expected_loss = temperature / loss_fn.base_temperature * expected_loss 
        
        print(f"Label smoothing (SimCLR): {smoothing}")
        print(f"Actual loss: {loss.item():.6f}")
        print(f"Expected loss: {expected_loss:.6f}")
        print(f"Difference: {abs(loss.item() - expected_loss):.6f}")
        
        # Check if the loss is close to expected
        assert abs(loss.item() - expected_loss) < 1e-5, f"SimCLR label smoothing test failed for smoothing={smoothing}"
        
    print("Label smoothing test for SimCLR passed!")

 


def test_clipping_simclr():
    """Test that clipping is correctly implemented for SimCLR."""
    print("\nTesting Clipping for SimCLR")
    print("=" * 50)
    
    # Create a simple test case
    batch_size = 4
    n_views = 2
    embedding_dim = 4
    temperature = 0.1
    
    # Create identical embeddings for each sample
    # Sample 0: [1, 0, 0, 0]
    # Sample 1: [0, 1, 0, 0]
    # Sample 2: [0, 0, 1, 0]
    # Sample 3: [0, 0, 0, 1]
    embeddings = torch.zeros(batch_size, n_views, embedding_dim)
    embeddings[0, :, 0] = 1.0
    embeddings[1, :, 1] = 1.0
    embeddings[2, :, 2] = 1.0
    embeddings[3, :, 3] = 1.0
    
    # Test positive clipping
    print("\nTesting Positive Clipping for SimCLR")
    clip_pos_values = [0.0, 0.1, 0.3, 0.5]
    
    for clip_pos in clip_pos_values:
        # Create loss function with current clipping value
        loss_fn = SupConLoss(temperature=temperature, clip_pos=clip_pos, base_temperature=temperature)
        
        # Calculate loss (SimCLR mode - no labels)
        loss = loss_fn(embeddings)
        
        # Calculate expected loss with positive clipping
        # With clip_pos, positive similarities are capped at (1-clip_pos)
        max_pos_sim = 1.0 - clip_pos
        
        # For SimCLR, only the same sample with different views is positive
        # For each sample, the similarity between its two views is 1
        # With clipping, this becomes min(1, max_pos_sim)
        
        # Simplified calculation for this specific case
        exp_pos = math.exp(max_pos_sim / temperature)
        exp_pos_in_neg = math.exp(1/temperature)
        exp_neg = math.exp(0.0)  # Orthogonal vectors have 0 similarity
        
        # For each sample, we have:
        # - 1 positive pair (other view of same sample): similarity = max_pos_sim
        # - 6 negative pairs (all views of other samples): similarity = 0
        
        # Expected loss is -log(exp_pos / (exp_pos + 6*exp_neg))
        expected_loss = -math.log(exp_pos / (exp_pos_in_neg + 6*exp_neg))
        
        print(f"Positive clipping (SimCLR): {clip_pos}")
        print(f"Actual loss: {loss.item():.6f}")
        print(f"Expected loss: {expected_loss:.6f}")
        print(f"Difference: {abs(loss.item() - expected_loss):.6f}")
        
        # Check if the loss is close to expected
        assert abs(loss.item() - expected_loss) < 1e-5, f"SimCLR positive clipping test failed for clip_pos={clip_pos}"
    
    print("Positive clipping test for SimCLR passed!")
    
    # Test negative clipping
    print("\nTesting Negative Clipping for SimCLR")
    clip_neg_values = [0.0, 0.1, 0.3, 0.5]
    
    embedding_dim = 2
    embeddings = torch.zeros(batch_size, n_views, embedding_dim)
    embeddings[0, :, 0] = 1.0
    embeddings[1, :, 0] = -1.0
    embeddings[2, :, 1] = 1.0
    embeddings[3, :, 1] = -1.0

    for clip_neg in clip_neg_values:
        # Create loss function with current clipping value
        loss_fn = SupConLoss(temperature=temperature, clip_neg=clip_neg, base_temperature=temperature)
        
        # Calculate loss (SimCLR mode - no labels)
        loss = loss_fn(embeddings)
        
        # Calculate expected loss with negative clipping
        # With clip_neg, negative similarities are bounded below by -(1-clip_neg)
        min_neg_sim = -(1.0 - clip_neg)
        
        # For SimCLR, only the same sample with different views is positive
        # For each sample, the similarity between its two views is 1
        # All other pairs have similarity 0 (orthogonal vectors)
        # With negative clipping, this becomes max(0, min_neg_sim)
        # Since min_neg_sim is negative and 0 > min_neg_sim, the clipped value is 0
        
        # Simplified calculation for this specific case
        exp_pos = math.exp(1.0 / temperature)
        exp_neg_min1 = math.exp(min_neg_sim / temperature)
        exp_neg_0 = math.exp(0.0 / temperature)
        
        # For each sample, we have:
        # - 1 positive pair (other view of same sample): similarity = 1
        # - 6 negative pairs (all views of other samples): similarity = 2 x -1, 4x0 (and we clip the -1 to something larger)
        
        # Expected loss is -log(exp_pos / (exp_pos + 6*exp_neg))
        expected_loss = -math.log(exp_pos / (exp_pos + 2*exp_neg_min1 + 4*exp_neg_0))
        
        print(f"Negative clipping (SimCLR): {clip_neg}")
        print(f"Actual loss: {loss.item():.6f}")
        print(f"Expected loss: {expected_loss:.6f}")
        print(f"Difference: {abs(loss.item() - expected_loss):.6f}")
        
        # Check if the loss is close to expected
        assert abs(loss.item() - expected_loss) < 1e-5, f"SimCLR negative clipping test failed for clip_neg={clip_neg}"
    
    print("Negative clipping test for SimCLR passed!")


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests for label smoothing and clipping
    test_label_smoothing_simclr()
    test_clipping_simclr()
    # test_label_smoothing_supcon()
    
    # Run tests for both 10 and 100 classes
    run_test(10)
    run_test(100)

if __name__ == "__main__":
    main() 