import torch
import torch.nn as nn
import numpy as np




class TransformerPool(nn.Module):
    def __init__(self, dim = 2048, part_num = 6, head_num = 8) -> None:
        super().__init__()

        self.part_num = part_num
        self.head_num = head_num
        self.scale = (dim // head_num) ** -0.5

        self.part_tokens = nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, head_num, part_num, dim//head_num), mode='fan_out'))
        self.pos_embeding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(18 * 9, dim), mode='fan_out'))
        self.kv = nn.Linear(dim, dim * 2)
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1) # [B, HW, C]
        x = x + self.pos_embeding

        kv = self.kv(x).reshape(B, H*W, 2, self.head_num, C//self.head_num).permute(2, 0, 3, 1, 4) # [2, B, head_num, HW, C//head_num]
        k, v = kv[0], kv[1] # [B, head_num, H*W, C//head_num]
        
        attn = self.part_tokens @ k.transpose(-1, -2) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, self.part_num, C) 

        return x.view(B, -1)





class SAM(nn.Module):
    def __init__(self, dim=2048, part_num=7) -> None:
        super().__init__()
        # 参数初始化
        self.part_num = part_num  # 局部特征分块数量（如6个身体部位）
        self.part_tokens = nn.Parameter(  # 可学习的局部特征查询向量 [6, 2048]
            nn.init.xavier_normal_(torch.empty(part_num, 2048))
        )
        self.pos_embeding = nn.Parameter(  # 空间位置编码矩阵 [18*9=162, 2048]
            nn.init.xavier_normal_(torch.empty(18 * 9, dim))
        )
        self.active = nn.Sigmoid()  # 注意力激活函数

    def forward(self, x):
        # 输入x形状: [B, C, H, W] (如B=32, C=2048, H=18, W=9)
        B, C, H, W = x.shape
        # 空间维度展平与维度重排 [B, C, H*W] → [B, H*W, C]
        x = x.view(B, C, -1).permute(0, 2, 1)
        # 加入位置编码
        x_pos = x + self.pos_embeding
        # 计算注意力权重 [B, 7, H*W]
        attn = self.part_tokens @ x_pos.transpose(-1, -2)
        attn = self.active(attn)  # 归一化为概率分布
        # 加权聚合特征 [B, 7, C]
        x = attn @ x / (H * W)  # 归一化处理
        # 输出展平特征和注意力图
        return x.view(B, -1), attn,self.part_tokens,self.pos_embeding  # [B, 6*2048], [B, 6, 162]


if __name__ == '__main__':

        # 模块实例化
    model = SAM(dim=2048, part_num=7)

    # 模拟输入数据 (假设来自ResNet-50的最后一层特征图)
    dummy_input = torch.randn(64, 2048, 18, 9)  # [B=2, C=2048, H=18, W=9]

    # 前向传播
    global_feat, attn_weights = model(dummy_input)

    # 输出解析
    print(f"输入特征形状: {dummy_input.shape}")  # torch.Size([64，2048, 18，9])
    print(f"全局特征形状: {global_feat.shape}")  # torch.Size([64, 12288])
    print(f"注意力图形状: {attn_weights.shape}")  # torch.Size([64, 6, 162])









