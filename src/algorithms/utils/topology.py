import networkx as nx
import torch


def generate_adjacency_matrix(args):
    """
    通用的邻域矩阵生成函数，直接从 args 对象中提取参数。
    支持：
    - ring: 环形拓扑
    - complete: 全连接
    - random: 连通的 Erdős-Rényi 随机图 (使用 args.edge_p)
    - small_world: 小世界网络 (使用 args.k, args.edge_p)
    - scale_free: 无标度网络 (使用 args.m)
    - star: 星型拓扑 (0号节点为中心)
    """
    n = args.num_clients
    adj_type = args.adj_type

    if adj_type == "complete":
        g = nx.complete_graph(n)
    elif adj_type == "ring":
        g = nx.cycle_graph(n)
    elif adj_type == "random":
        edge_p = args.edge_p
        g = nx.erdos_renyi_graph(n, edge_p)
        if n > 1:
            retry = 0
            while not nx.is_connected(g) and retry < 100:
                g = nx.erdos_renyi_graph(n, edge_p)
                retry += 1
    elif adj_type == "small_world":
        k = args.k_small_world
        p = args.edge_p
        g = nx.watts_strogatz_graph(n, k, p)
    elif adj_type == "scale_free":
        m = args.m_scale_free
        if n > m:
            g = nx.barabasi_albert_graph(n, m)
        else:
            g = nx.complete_graph(n)
    elif adj_type == "star":
        g = nx.star_graph(n - 1)
    else:
        raise ValueError(
            f"Unsupported adj_type: {adj_type}. "
            "Choices: ['ring', 'complete', 'random', 'small_world', 'scale_free', 'star']"
        )

    A = torch.from_numpy(nx.to_numpy_array(g)).float()
    A.fill_diagonal_(1.0)
    return A


def compute_mh_weights(adj_matrix, device="cpu"):
    """
    计算 Metropolis-Hastings (MH) 混合权重矩阵 (向量化版本)。
    """
    n = adj_matrix.shape[0]
    adj = adj_matrix.to(device).float()

    # 1. 计算度数 d_i (包含自环)
    deg = (adj > 0).sum(dim=1).float()

    # 2. 计算所有 pair 的 1 / max(d_i, d_j)
    max_deg = torch.max(deg.view(n, 1), deg.view(1, n))
    W = adj / max_deg

    # 3. 修正对角线：W_ii = 1 - sum_{j!=i} W_ij
    W.fill_diagonal_(0.0)
    diag_weights = torch.diag(1.0 - W.sum(dim=1))
    W = W + diag_weights

    return W


def sinkhorn_knopp(A, epsilon=1e-3, max_iter=100):
    """将邻接矩阵 W 转化为双随机矩阵 (Doubly Stochastic)"""
    W = A.clone().float()
    for _ in range(max_iter):
        W /= W.sum(dim=1, keepdim=True) + 1e-8
        W /= W.sum(dim=0, keepdim=True) + 1e-8
        if torch.norm(W.sum(dim=1) - 1.0) < epsilon:
            break
    return W
