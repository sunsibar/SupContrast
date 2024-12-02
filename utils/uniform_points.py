import numpy as np

from scipy.optimize import minimize
import plotly.graph_objects as go
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity

def random_uniform_points(num_points, n):
    points = np.random.randn(num_points, n)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def greedy_uniform_points(num_points, n, n_iter=1000, verbose=False):
    '''For n_iter iterations, compute all similarities between the points,
        pick the most similar pair, and re-sample one of the two. 
        Then recompute all similarities involving that point.
        Report the average similarity for the final set of points, 
        and if verbose=True, print the average similarity for each iteration.'''
    points = random_uniform_points(num_points, n)
    similarities = np.einsum('ik,jk->ij', points, points)
    similarities = similarities - np.eye(num_points)
    assert np.max(similarities) <= 1 and np.min(similarities) >= -1

    for i in range(n_iter): 
        max_sim = np.max(similarities)
        max_sim_idx = np.unravel_index(np.argmax(similarities), similarities.shape)
        new_point = random_uniform_points(1, n)[0]
        points[max_sim_idx[0]] = new_point
        for i in range(1000):
            new_similarities = np.einsum('k,jk->j', new_point, points)
            new_similarities[max_sim_idx[0]] = -1
            if max(new_similarities) < max_sim:
                break
        else:
            print(f'Iteration {i}, max similarity: {max_sim}; unable to improve further. Stopping.')
            break 
        similarities[max_sim_idx[0], :] = new_similarities
        similarities[:, max_sim_idx[0]] = new_similarities 
        if verbose:
            print(f'Iteration {i}, mean similarity: {np.mean(similarities)}, max similarity: {np.max(similarities)}')

    return points, np.mean(similarities)
    
def initialize_points(num_points, n):
    points = np.random.randn(num_points, n)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def energy_function_loop(points, num_points, n):
    points = points.reshape((num_points, n))
    energy = 0.0
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(points[i] - points[j])
            energy += 1.0 / dist
    return energy

def energy_function(points, num_points, n):
    points = points.reshape((num_points, n))
    # cosine_similarity is a function from sklearn.metrics that computes the cosine similarity between all pairs of points
    # It returns a matrix where the entry at row i and column j is the cosine similarity between points[i] and points[j]
    sim_matrix = cosine_similarity(points, points)
    np.fill_diagonal(sim_matrix, -1)  # Set diagonal to -1 to exclude self-similarity
    energy = np.sum(sim_matrix) / 2  # Divide by 2 to avoid double counting
    return energy


def energy_function_infonce(points, num_points, n):
    points = points.reshape((num_points, n))
    sim_matrix = cosine_similarity(points, points)
    pos_sims = np.diag(sim_matrix)
    neg_sims = sim_matrix[~np.eye(num_points, dtype=bool)]
    # we want to minimize energy
    energy = -np.sum(pos_sims) + np.log(np.sum(np.exp(neg_sims))) - np.log(num_points) 
    return energy


def constraint(points, num_points, n):
    points = points.reshape((num_points, n))
    return np.linalg.norm(points, axis=1) - 1

def optimize_points(num_points, n):
    initial_points = initialize_points(num_points, n).flatten()
    constraints = {'type': 'eq', 'fun': constraint, 'args': (num_points, n)}
    result = minimize(energy_function, initial_points, args=(num_points, n), method='SLSQP', constraints=constraints)
    optimized_points = result.x.reshape((num_points, n))
    optimized_points = optimized_points / np.linalg.norm(optimized_points, axis=1, keepdims=True)
    return optimized_points, energy_function(optimized_points, num_points, n)
 