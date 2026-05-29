import numpy as np
import numpy as np

def select_next_node_v2(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """
    Improved constructive heuristic that combines immediate cost, regret, 
    mean/std of distances to other unvisited nodes, and destination proximity.
    
    Weights are fixed but chosen to balance exploitation (immediate cost, destination)
    and exploration (regret, std, mean).
    """
    if len(unvisited_nodes) == 1:
        return next(iter(unvisited_nodes))

    # Fixed weights (tuned for robustness)
    w_imm, w_reg, w_avg, w_std, w_dest = 0.35, 0.25, 0.15, 0.15, 0.10

    best_node = None
    best_score = float('inf')

    for candidate in unvisited_nodes:
        # Immediate cost from current node
        imm = distance_matrix[current_node, candidate]

        # Gather distances from candidate to all other unvisited nodes
        others = [v for v in unvisited_nodes if v != candidate]
        if len(others) >= 2:
            dists = np.array([distance_matrix[candidate, v] for v in others])
            avg = np.mean(dists)
            std = np.std(dists)
            # Regret: difference between second best and best edge from candidate to others
            sorted_dists = np.sort(dists)
            regret = sorted_dists[1] - sorted_dists[0]
        elif len(others) == 1:
            avg = distance_matrix[candidate, others[0]]
            std = 0.0
            regret = 0.0
        else:
            avg = 0.0
            std = 0.0
            regret = 0.0

        # Distance to final destination
        dest_dist = distance_matrix[candidate, destination_node]

        # Score: lower is better
        score = (w_imm * imm
                 - w_reg * regret   # encourage larger regret (better immediate choice)
                 - w_avg * avg      # favor nodes closer to others on average
                 + w_std * std      # favor nodes with more diverse distances to others
                 - w_dest * dest_dist)  # favor nodes closer to destination

        if score < best_score:
            best_score = score
            best_node = candidate

    return best_node
