import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class BinarizedF(Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, threshold)
        a = torch.ones_like(input)
        b = torch.zeros_like(input)
        output = torch.where(input >= threshold, a, b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = 0.2 * grad_output  # 对于输入的梯度
        if ctx.needs_input_grad[1]:
            grad_weight = -grad_output  # 对于阈值的梯度
        return grad_input, grad_weight
    
    
class ThresholdNet(nn.Module):
    def __init__(self):
        super(ThresholdNet, self).__init__()
        
        # 阈值提取模块
        self.Threshold_Module = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.AvgPool2d(15, stride=1, padding=7),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(15, stride=1, padding=7),
        )
        
        # 自定义归一化层和反归一化层
        self.sig = compressedSigmoid()
        self.antisig = reverseCompressedSigmoid()
        
        # 用于反归一化的权重和偏置
        self.weight = nn.Parameter(torch.Tensor(1).fill_(0.5), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(1).fill_(0), requires_grad=True)

    def forward(self, anomaly_scores, mask):
        """
        Forward 方法，输出阈值和二值化结果。

        参数:
        anomaly_scores: (B, 1, H, W), 输入异常分数
        mask: (B, 1, H, W), 掩码，标记有效区域

        返回:
        thresholds: (B,), 每个样本的阈值
        binarized_map: (B, 1, H, W), 二值化结果
        """
        # 自定义归一化，防止数值不稳定
        normalized_scores = self.sig(anomaly_scores)
        # print("normalized_scores=====", normalized_scores, normalized_scores.shape)

        # 下采样异常分数和掩码以匹配特征提取模块的输入大小
        downsampled_scores = F.interpolate(normalized_scores, scale_factor=0.125)
#         downsampled_mask = F.interpolate(mask, scale_factor=0.125)

#         # 提取阈值特征
#         combined_features = downsampled_scores * downsampled_mask
        thresholds_map = self.Threshold_Module( downsampled_scores)
       

        # 归一化为阈值
        thresholds_map = self.sig(thresholds_map)

        # 上采样回原始尺寸
        thresholds_map = F.interpolate(thresholds_map, scale_factor=8)
        

        # 每个样本的最终阈值取整幅图像的平均值
        thresholds = thresholds_map.mean(dim=[2, 3])  # (B,)
     
       

    

        return thresholds

    def compute_loss(self, anomaly_scores, labels, thresholds, mask):
        """
        计算损失函数。

        参数：
            anomaly_scores: torch.Tensor, 异常分数 (B, 1, H, W)
            labels: torch.Tensor, 标注 (B, 1, H, W)
            thresholds: torch.Tensor, 网络输出的异常分数阈值 (B,)
            mask: torch.Tensor, 掩码 (B, 1, H, W)

        返回：
            loss: torch.Tensor, 交叉熵损失
        """
        # 扩展阈值到异常分数的形状
        thresholds_map = thresholds.view(-1, 1, 1, 1).expand_as(anomaly_scores)

        # 二值化异常分数
        anomaly_scores = self.sig(anomaly_scores)
        binary_map = BinarizedF.apply(anomaly_scores, thresholds_map)

        # 在有效区域（mask 指定的区域）计算损失
        valid_labels = labels * mask
        valid_binary_map = binary_map * mask

        # 计算交叉熵损失
        loss = F.binary_cross_entropy(valid_binary_map, valid_labels, reduction='mean')

        return loss
    
class compressedSigmoid(nn.Module):
    def __init__(self, para=2.0, bias=0.3):
        super(compressedSigmoid, self).__init__()

        self.para = para
        self.bias = bias

    def forward(self, x):
        output = 1. / (self.para + torch.exp(-x)) + self.bias
        return output
    
class reverseCompressedSigmoid(nn.Module):
    def __init__(self, para=2.0, bias=0.3):
        super(reverseCompressedSigmoid, self).__init__()
        self.para = para
        self.bias = bias

    def forward(self, y):
        # 防止除零错误，确保y > bias
        epsilon = 1e-6
        y = torch.clamp(y, min=self.bias + epsilon)  # 保证y大于bias + 一个小量，防止 log 函数出错
        x = -torch.log(1. / (y - self.bias) - self.para)
        return x
    
    
