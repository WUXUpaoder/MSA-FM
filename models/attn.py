import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    2D 图像转换为 Patch 嵌入（Patch Embedding）

    该类旨在将二维图像转换为 Patch
    """

    def __init__(self, img_size=(18,9), patch_size=1, in_chans=2048, embed_dim=2048):
        super().__init__()
        # 将输入的图像大小转换为元组形式
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        # 计算 Patch 的数量
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # 定义卷积层，用于将图像分割成 Patch 并进行嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        前向传播函数

        参数:
        x (torch.Tensor): 输入的图像张量，形状为 (B, C, H, W)

        返回:
        torch.Tensor: 输出的 Patch 嵌入张量，形状为 (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        # 使用卷积层将图像分割成 Patch 并进行嵌入
        x = self.proj(x)
        # print(x.shape)
        # 将卷积层的输出展平，并调整维度
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim=2048, num_heads=8, qkv_bias=False, attn_drop=0.5, proj_drop=0.5):
        super().__init__()
        # 注意力头的数量
        self.num_heads = num_heads
        # 每个注意力头的维度
        head_dim = dim // num_heads
        # 缩放因子，用于缩放点积注意力
        self.scale = head_dim ** -0.5
        # 定义一个线性层，用于生成查询（Q）、键（K）和值（V）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 定义注意力分数的丢弃层
        self.attn_drop = nn.Dropout(attn_drop)
        # 定义投影层，用于将多头注意力的输出投影回原始维度
        self.proj = nn.Linear(dim, dim)
        # 定义投影层后的丢弃层
        self.proj_drop = nn.Dropout(proj_drop)
        self.fc = nn.Linear(dim * 18*9, dim)
        self.relu=nn.ReLU()
        self.active = nn.Sigmoid()  # 注意力激活函数

    def forward(self, x):
        # 获取输入的形状
        B, N, C = x.shape
        # 通过线性层生成 Q、K、V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 分离 Q、K、V
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 对注意力分数进行 softmax 操作，得到注意力权重
        attn = attn.softmax(dim=-1)
        # 对注意力权重进行丢弃操作
        attn = self.attn_drop(attn)
        attn=attn.mean(dim=3)
        attn=self.active(attn)
        x = attn @ x#.transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # x=x.reshape(B,-1)
        # x=self.fc(x)
        # x=self.relu(x)
        x = torch.nn.functional.normalize(x)
        return x.view(B, -1),attn


if __name__ == '__main__':
    model=PatchEmbed(img_size=(18,9),patch_size=1)
    dummy_input = torch.randn(64, 2048, 18, 9)  # [B=64, C=2048, H=18, W=9]
    print(dummy_input.shape)
    x = model(dummy_input)
    attention = Attention(dim=2048, num_heads=8)
    output1,output2 = attention(x)
    print(output1.shape)
    print(output2.shape)





