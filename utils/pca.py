import torch
from sklearn.decomposition import PCA

def pca(t:torch.Tensor):
    n, p = t.shape # expect 2d tensor
    u, s, vh = torch.linalg.svd((t - t.mean(dim=0, keepdim=True)))
    eig = s**2
    eig_sorted, ids = torch.sort(eig, descending=True)
    eigenvectors = vh[ids, :]
    # t.T @ t = v s u.T u s vh = v s^2 vh; 
    return eigenvectors, eig_sorted

def sklearn_pca(t:torch.Tensor):
    pca_ = PCA(n_components=min(t.shape[1], t.shape[0]), svd_solver='full')
    t_np = t.cpu().numpy()
    pca_.fit(t_np)
    eigenvalues = pca_.explained_variance_
    return pca_.components_, eigenvalues

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