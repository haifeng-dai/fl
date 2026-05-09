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
    计算 Metropolis-Hastings (MH) 混合权重矩阵。

    基于邻接矩阵的度数信息，为每个节点计算到所有节点的权重。
    每个节点 i 的权重向量 mh_weights[i] 定义了其如何聚合所有节点的参数。

    参数：
        adj_matrix: 邻接矩阵，形状为 [N, N]
        device: 计算设备

    返回：
        mh_weights: MH 权重矩阵，形状为 [N, N]
    """
    n = adj_matrix.shape[0]
    adj_matrix = adj_matrix.to(device)
    mh_weights = torch.zeros(n, n, device=device)

    for i in range(n):
        # 获取节点 i 的度数（邻接矩阵第 i 行的非零元素个数）
        deg_i = (adj_matrix[i] > 0).sum().item()
        self_weight = 1.0

        # 遍历节点 i 的所有邻居（邻接矩阵中非零位置）
        for j in torch.where(adj_matrix[i] > 0)[0].tolist():
            if i == j:
                continue
            # 获取邻居 j 的度数
            deg_j = (adj_matrix[j] > 0).sum().item()
            w_ij = 1.0 / max(deg_i, deg_j)
            mh_weights[i, j] = w_ij
            self_weight -= w_ij

        mh_weights[i, i] = self_weight

    return mh_weights


def sinkhorn_knopp(A, epsilon=1e-3, max_iter=100):
    """将邻接矩阵 W 转化为双随机矩阵 (Doubly Stochastic)"""
    W = A.clone().float()
    for _ in range(max_iter):
        W /= W.sum(dim=1, keepdim=True) + 1e-8
        W /= W.sum(dim=0, keepdim=True) + 1e-8
        if torch.norm(W.sum(dim=1) - 1.0) < epsilon:
            break
    return W
