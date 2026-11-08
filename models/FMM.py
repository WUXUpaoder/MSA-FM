import torch
import torch.nn.functional  as F


def getNewFeature_different(x, y, k1):
    dismat = x @ y.T

    if y.size(0) < k1:
        return dismat
    val, rank = dismat.topk(y.size(0))
    dismat[dismat < val[:, k1-1].unsqueeze(1)] = 0
    return dismat


def getNewFeature_similar(x, y,k2, threshold: float = 0.5 ):
    dismat = x @ y.T

    topk_num = min(k2, y.size(0))
    val, rank = dismat.topk(y.size(0))
    dismat = torch.where(dismat >= threshold, 1.0, 0.0)
    dismat = dismat[rank[:, :topk_num]].mean(dim=1)
    return dismat


def FMM(qf: torch.tensor, gf: torch.tensor, k1, k2,threshold_qq: float = 0.5,threshold_gg: float = 0.5,num_shots=1):
    qf = qf.to('cuda')
    gf = gf.to('cuda')

    qf = torch.nn.functional.normalize(qf)
    gf = torch.nn.functional.normalize(gf)

    # new_qf = torch.concat([getNewFeature(qf, gf, k1, k2)], dim=1)
    # new_gf = torch.concat([getNewFeature1(gf, gf, k1, k2, mean=True)], dim=1)
    #
    # new_qf = torch.nn.functional.normalize(new_qf)
    # new_gf = torch.nn.functional.normalize(new_gf)

    # additional use of relationships between query sets
    if num_shots==1:
        new_qf = torch.concat([getNewFeature_similar(qf, qf, k2,threshold_qq), getNewFeature_different(qf, gf, k1)], dim=1)
        new_gf = torch.concat([getNewFeature_different(gf, qf, k1), getNewFeature_similar(gf, gf, k2,threshold_gg)], dim=1)

    # 多发
    if num_shots==10:
        new_qf = torch.concat([getNewFeature_similar(qf, qf, k2, threshold=threshold_qq), getNewFeature_different(qf, gf, k1)],
                              dim=1)
        new_gf = torch.concat([getNewFeature_different(gf, qf, k1), getNewFeature_similar(gf, gf, k2,threshold=threshold_gg)],
                              dim=1)
    new_qf = torch.nn.functional.normalize(new_qf)
    new_gf = torch.nn.functional.normalize(new_gf)

    dist_new_qg=-new_qf @ new_gf.T
    dist_qg=-qf @ gf.T

    dist=dist_new_qg+dist_qg
    return (dist).to('cpu')