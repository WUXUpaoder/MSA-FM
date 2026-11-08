import torch

def k_reciprocal_re_ranking(original_dist:torch.Tensor, k1, k2):
    """
    K-reciprocal 重排序，返回重排后的距离矩阵
    :param original_dist: tensor [Nq, Ng], 欧式距离，越小越相似
    :param k1: k-reciprocal 的k1
    :param k2: 局部k近邻
    :return: dist_rerank: [Nq, Ng] 重排后的距离
    """
    Nq, Ng = original_dist.shape
    # 转为相似度：距离越小→相似度越高，这里用负距离
    sim = -original_dist
    # topk index: [Nq, Ng]
    _, rank_idx = torch.topk(sim, k=Ng, dim=1)

    # Step1: k-reciprocal neighbors
    k1_neighbors = rank_idx[:, :k1]
    # reciprocal mask
    reciprocal = torch.zeros_like(original_dist, dtype=torch.bool)
    for i in range(Nq):
        knn_q = set(k1_neighbors[i].cpu().numpy())
        for j in k1_neighbors[i]:
            knn_gj = set(rank_idx[j, :k1].cpu().numpy())
            if i in knn_gj:
                reciprocal[i, j] = True

    # Step2: 计算Jaccard相似度
    jaccard_sim = torch.zeros_like(original_dist)
    for i in range(Nq):
        rnn = torch.nonzero(reciprocal[i], as_tuple=True)[0]
        if len(rnn) ==0:
            continue
        # 局部扩展 k2
        expand = []
        for j in rnn:
            expand.append(rank_idx[j, :k2])
        expand = torch.unique(torch.cat(expand))
        # 集合交集/并集
        A = reciprocal[i, expand].float()
        B = reciprocal[expand, :].float()  # 修复：B转float
        inter = torch.matmul(A, B)
        union = A.sum() + B.sum(dim=0) - inter
        jaccard_sim[i] = inter / (union + 1e-8)

    # Step3: 融合原始距离和Jaccard
    # 归一化原始距离到[0,1]
    dist_min, _ = original_dist.min(dim=1,keepdim=True)
    dist_max, _ = original_dist.max(dim=1,keepdim=True)
    dist_norm = (original_dist - dist_min) / (dist_max - dist_min +1e-8)
    jaccard_dist = 1.0 - jaccard_sim
    dist_rerank = 0.5 * dist_norm + 0.5 * jaccard_dist
    return dist_rerank
