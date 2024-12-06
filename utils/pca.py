import torch

def pca(t:torch.Tensor):
    n, p = t.shape # expect 2d tensor
    u, s, vh = torch.linalg.svd((t - t.mean(dim=0, keepdim=True)))
    eig = s**2
    eig_sorted, ids = torch.sort(eig, descending=True)
    eigenvectors = vh[ids, :]
    # t.T @ t = v s u.T u s vh = v s^2 vh; 
    return eigenvectors, eig_sorted


# This returns the same results as sklearn's PCA:
# def pca(t:np.ndarray):
#     n, p = t.shape # expect 2d tensor
#     centered = t - t.mean(axis=0)[None, :]
#     u, s, vh = np.linalg.svd(centered)
#     eig = s**2
#     eig_sorted, ids = np.sort(eig)[::-1], np.argsort(eig)[::-1]
#     eigenvectors = vh[ids, :]
#     # t.T @ t = v s u.T u s vh = v s^2 vh; 
#     return eigenvectors, eig_sorted