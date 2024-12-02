import torch

def pca(t:torch.Tensor):
    n, p = t.shape # expect 2d tensor
    u, s, vh = torch.linalg.svd((t - t.mean(dim=0)))
    eig = s**2
    eig_sorted, ids = torch.sort(eig, descending=True)
    eigenvectors = vh[ids, :]
    # t.T @ t = v s u.T u s vh = v s^2 vh; 
    return eigenvectors, eig_sorted