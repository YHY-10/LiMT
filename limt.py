import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
from Embed import PatchEmbedding, DataEmbedding_inverted
from base_model import SequenceModel
import torch.nn.functional as F
from typing import Tuple
import torch
import numpy as np
import random

class MMoE_TaskFusion_RVol2Vola2(nn.Module):
    """volume和volatility的特征是加到return的特征上去"""
    def __init__(self, d_model=256, n_experts=4, return_gate_weight=0):
        super().__init__()
        # MMoE核心组件（论文Section 4.2）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU()
            ) for _ in range(n_experts)])
        
        # 任务特定门控（return/volume/volatility）
        self.gates = nn.ModuleDict({
            'return': nn.Linear(d_model, n_experts),
            'volume': nn.Linear(d_model, n_experts),
            'volatility': nn.Linear(d_model, n_experts)
        })

        self.return_gate_weight = return_gate_weight
        
        # 任务预测层
        self.return_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        self.volume_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
            
        self.volatility_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        # 门控融合组件（新需求）
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.Sigmoid())

    def forward(self, shared_feat):
        # MMoE处理（论文Eq.6-7）
        task_features = {}
        gate_dict = {}
        for task in ['return', 'volume', 'volatility']:
            gates = torch.softmax(self.gates[task](shared_feat), dim=1)
            gate_dict[task] = gates
            expert_outputs = torch.stack([e(shared_feat) for e in self.experts], dim=1)
            task_features[task] = torch.einsum('bi,bij->bj', gates, expert_outputs)
        
        # 中间层融合（新需求）
        return_hidden = self.return_proj[:2](task_features['return'])  # d_model -> d_model/2
        volume_hidden = self.volume_proj[:2](task_features['volume'])
        volatility_hidden = self.volatility_proj[:2](task_features['volatility'])
        
        # 门控融合（论文Section 4.2扩展）
        gate_weights = self.fusion_gate(shared_feat)
        fused = return_hidden + gate_weights[:,0:1] * volume_hidden + gate_weights[:, 1:2] * volatility_hidden
        
        # 最终预测
        return_pred = self.return_proj[2](fused)
        volume_pred = self.volume_proj[2](volume_hidden)
        volatility_pred = self.volatility_proj[2](volatility_hidden)
        
        if self.return_gate_weight == 1:
            return return_pred.squeeze(-1), volume_pred.squeeze(-1), volatility_pred.squeeze(-1), gate_dict
        else:
            return return_pred.squeeze(-1), volume_pred.squeeze(-1), volatility_pred.squeeze(-1)


class MMoE_TaskFusion_RVol2Vola2_RC(nn.Module):
    """残差连接的moe"""
    def __init__(self, d_model=256, n_experts=4, return_gate_weight=0):
        super().__init__()
        # MMoE核心组件（论文Section 4.2）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU()
            ) for _ in range(n_experts)])
        
        # 任务特定门控（return/volume/volatility）
        self.gates = nn.ModuleDict({
            'return': nn.Linear(d_model, n_experts),
            'volume': nn.Linear(d_model, n_experts),
            'volatility': nn.Linear(d_model, n_experts)
        })

        self.return_gate_weight = return_gate_weight
        
        # 任务预测层
        self.return_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        self.volume_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
            
        self.volatility_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        # 门控融合组件（新需求）
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.Sigmoid())

    def forward(self, shared_feat):
        # MMoE处理（论文Eq.6-7）
        task_features = {}
        gate_dict = {}
        for task in ['return', 'volume', 'volatility']:
            gates = torch.softmax(self.gates[task](shared_feat), dim=1)
            gate_dict[task] = gates
            expert_outputs = torch.stack([e(shared_feat) for e in self.experts], dim=1)
            task_features[task] = torch.einsum('bi,bij->bj', gates, expert_outputs) + shared_feat
        
        # 中间层融合（新需求）
        return_hidden = self.return_proj[:2](task_features['return'])  # d_model -> d_model/2
        volume_hidden = self.volume_proj[:2](task_features['volume'])
        volatility_hidden = self.volatility_proj[:2](task_features['volatility'])
        
        # 门控融合（论文Section 4.2扩展）
        gate_weights = self.fusion_gate(shared_feat)
        fused = return_hidden + gate_weights[:,0:1] * volume_hidden + gate_weights[:, 1:2] * volatility_hidden
        
        # 最终预测
        return_pred = self.return_proj[2](fused)
        volume_pred = self.volume_proj[2](volume_hidden)
        volatility_pred = self.volatility_proj[2](volatility_hidden)
        
        if self.return_gate_weight == 1:
            return return_pred.squeeze(-1), volume_pred.squeeze(-1), volatility_pred.squeeze(-1), gate_dict
        else:
            return return_pred.squeeze(-1), volume_pred.squeeze(-1), volatility_pred.squeeze(-1)


class MMoE_TaskFusion_RV(nn.Module):
    """volume和volatility的特征是加到return的特征上去"""
    def __init__(self, d_model=256, n_experts=4, return_gate_weight=0):
        super().__init__()
        # MMoE核心组件（论文Section 4.2）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU()
            ) for _ in range(n_experts)])
        
        # 任务特定门控（return/volume/volatility）
        self.gates = nn.ModuleDict({
            'return': nn.Linear(d_model, n_experts),
            'volume': nn.Linear(d_model, n_experts),
        })

        self.return_gate_weight = return_gate_weight
        
        # 任务预测层
        self.return_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        self.volume_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        # 门控融合组件（新需求）
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid())

    def forward(self, shared_feat):
        # MMoE处理（论文Eq.6-7）
        task_features = {}
        gate_dict = {}
        for task in ['return', 'volume']:
            gates = torch.softmax(self.gates[task](shared_feat), dim=1)
            gate_dict[task] = gates
            expert_outputs = torch.stack([e(shared_feat) for e in self.experts], dim=1)
            task_features[task] = torch.einsum('bi,bij->bj', gates, expert_outputs)
        
        # 中间层融合（新需求）
        return_hidden = self.return_proj[:2](task_features['return'])  # d_model -> d_model/2
        volume_hidden = self.volume_proj[:2](task_features['volume'])
        
        # 门控融合（论文Section 4.2扩展）
        gate_weights = self.fusion_gate(shared_feat)
        fused = return_hidden + gate_weights[:,0:1] * volume_hidden
        
        # 最终预测
        return_pred = self.return_proj[2](fused)
        volume_pred = self.volume_proj[2](volume_hidden)
        
        return return_pred.squeeze(-1), volume_pred.squeeze(-1)


class MMoE_TaskFusion_R(nn.Module):
    """volume和volatility的特征是加到return的特征上去"""
    def __init__(self, d_model=256, n_experts=4, return_gate_weight=0):
        super().__init__()
        # MMoE核心组件（论文Section 4.2）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU()
            ) for _ in range(n_experts)])
        
        # 任务特定门控（return/volume/volatility）
        self.gates = nn.ModuleDict({
            'return': nn.Linear(d_model, n_experts),
        })

        self.return_gate_weight = return_gate_weight
        
        # 任务预测层
        self.return_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))

    def forward(self, shared_feat):
        # MMoE处理（论文Eq.6-7）
        task_features = {}
        gate_dict = {}
        for task in ['return']:
            gates = torch.softmax(self.gates[task](shared_feat), dim=1)
            gate_dict[task] = gates
            expert_outputs = torch.stack([e(shared_feat) for e in self.experts], dim=1)
            task_features[task] = torch.einsum('bi,bij->bj', gates, expert_outputs)
        
        # 中间层融合（新需求）
        return_hidden = self.return_proj[:2](task_features['return'])  # d_model -> d_model/2
        
        # 门控融合（论文Section 4.2扩展）
        fused = return_hidden
        
        # 最终预测
        return_pred = self.return_proj[2](fused)
        
        return return_pred.squeeze(-1)

    
    
class MMoE_TaskFusion_RV1(nn.Module):
    """volume的特征是加到return的特征上去"""
    def __init__(self, d_model=256, n_experts=4):
        super().__init__()
        # MMoE核心组件（论文Section 4.2）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU()
            ) for _ in range(n_experts)])
        
        # 任务特定门控（return/volume/volatility）
        self.gates = nn.ModuleDict({
            'return': nn.Linear(d_model, n_experts),
            'volume': nn.Linear(d_model, n_experts),
            'volatility': nn.Linear(d_model, n_experts)
        })
        
        # 任务预测层
        self.return_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        self.volume_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
            
        self.volatility_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        # 门控融合组件（新需求）
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid())

    def forward(self, shared_feat):
        # MMoE处理（论文Eq.6-7）
        task_features = {}
        for task in ['return', 'volume', 'volatility']:
            gates = torch.softmax(self.gates[task](shared_feat), dim=1)
            expert_outputs = torch.stack([e(shared_feat) for e in self.experts], dim=1)
            task_features[task] = torch.einsum('bi,bij->bj', gates, expert_outputs)
        
        # 中间层融合（新需求）
        return_hidden = self.return_proj[:2](task_features['return'])  # d_model -> d_model/2
        volume_hidden = self.volume_proj[:2](task_features['volume'])
        
        # 门控融合（论文Section 4.2扩展）
        gate_weights = self.fusion_gate(shared_feat)
        fused = return_hidden + gate_weights[:,0:1] * volume_hidden
        
        # 最终预测
        return_pred = self.return_proj[2](fused)
        volume_pred = self.volume_proj[2](volume_hidden)
        
        return return_pred.squeeze(-1), volume_pred.squeeze(-1)

class MMoE_TaskFusion1_RV1(nn.Module):
    """volume的特征是拼接到return的特征上去"""
    def __init__(self, d_model=256, n_experts=4):
        super().__init__()
        # MMoE核心组件（论文Section 4.2）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU()
            ) for _ in range(n_experts)])
        
        # 任务特定门控（return/volume/volatility）
        self.gates = nn.ModuleDict({
            'return': nn.Linear(d_model, n_experts),
            'volume': nn.Linear(d_model, n_experts),
            'volatility': nn.Linear(d_model, n_experts)
        })
        
        # 任务预测层
        self.return_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model, 1))
        
        self.volume_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
            
        self.volatility_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        # 门控融合组件（新需求）
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid())

    def forward(self, shared_feat):
        # MMoE处理（论文Eq.6-7）
        task_features = {}
        for task in ['return', 'volume', 'volatility']:
            gates = torch.softmax(self.gates[task](shared_feat), dim=1)
            expert_outputs = torch.stack([e(shared_feat) for e in self.experts], dim=1)
            task_features[task] = torch.einsum('bi,bij->bj', gates, expert_outputs)
        
        # 中间层融合（新需求）
        return_hidden = self.return_proj[:2](task_features['return'])  # d_model -> d_model/2
        volume_hidden = self.volume_proj[:2](task_features['volume'])
        
        # 门控融合（论文Section 4.2扩展）
        gate_weights = self.fusion_gate(shared_feat)
        fused = torch.concat([return_hidden, gate_weights[:,0:1] * volume_hidden], dim=-1) # 拼接volume的特征到return的特征上
        
        # 最终预测
        return_pred = self.return_proj[2](fused)
        volume_pred = self.volume_proj[2](volume_hidden)
        
        return return_pred.squeeze(-1), volume_pred.squeeze(-1)

class MMoE_TaskFusion2_RV1(MMoE_TaskFusion_RV1):
    """两个并行，不交互"""
    def __init__(self, d_model=256, n_experts=4):
        super().__init__(d_model, n_experts)
        

    def forward(self, shared_feat):
        # MMoE处理（论文Eq.6-7）
        task_features = {}
        for task in ['return', 'volume', 'volatility']:
            gates = torch.softmax(self.gates[task](shared_feat), dim=1)
            expert_outputs = torch.stack([e(shared_feat) for e in self.experts], dim=1)
            task_features[task] = torch.einsum('bi,bij->bj', gates, expert_outputs)
        
        # 中间层融合（新需求）
        return_hidden = self.return_proj[:2](task_features['return'])  # d_model -> d_model/2
        volume_hidden = self.volume_proj[:2](task_features['volume'])
        
        # 门控融合（论文Section 4.2扩展）
        # gate_weights = self.fusion_gate(shared_feat)
        # fused = torch.concat([return_hidden, gate_weights[:,0:1] * volume_hidden], dim=-1) # 拼接volume的特征到return的特征上
        
        # 最终预测
        return_pred = self.return_proj[2](return_hidden)
        volume_pred = self.volume_proj[2](volume_hidden)
        
        return return_pred.squeeze(-1), volume_pred.squeeze(-1)
    
class MMoE_TaskFusion_RV12(nn.Module):
    """volume的特征是加到return的特征上去"""
    def __init__(self, d_model=512, n_experts=4):
        super().__init__()
        # MMoE核心组件（论文Section 4.2）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU()
            ) for _ in range(n_experts)])
        
        # 任务特定门控（return/volume/volatility）
        self.gates = nn.ModuleDict({
            'return': nn.Linear(d_model, n_experts),
            'volume': nn.Linear(d_model, n_experts),
            'volatility': nn.Linear(d_model, n_experts)
        })
        
        # 任务预测层
        self.return_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        self.volume_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
            
        self.volatility_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        # 门控融合组件（新需求）
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.Sigmoid())

    def forward(self, shared_feat):
        # MMoE处理（论文Eq.6-7）
        task_features = {}
        for task in ['return', 'volume', 'volatility']:
            gates = torch.softmax(self.gates[task](shared_feat), dim=1)
            expert_outputs = torch.stack([e(shared_feat) for e in self.experts], dim=1)
            task_features[task] = torch.einsum('bi,bij->bj', gates, expert_outputs)
        
        # 中间层融合（新需求）
        return_hidden = self.return_proj[:2](task_features['return'])  # d_model -> d_model/2
        volume_hidden = self.volume_proj[:2](task_features['volume'])
        volat_hidden = self.volatility_proj[:2](task_features['volatility'])
        
        # 门控融合（论文Section 4.2扩展）
        gate_weights = self.fusion_gate(shared_feat)
        fused = return_hidden + gate_weights[:,0:1] * volume_hidden + gate_weights[:,1:2] * volat_hidden
        
        # 最终预测
        return_pred = self.return_proj[2](fused)
        volume_pred = self.volume_proj[2](volume_hidden)
        volat_pred = self.volatility_proj[2](volat_hidden)
        
        return return_pred.squeeze(-1), volume_pred.squeeze(-1), volat_pred.squeeze(-1)

class MMoE_TaskFusion1_RV12(nn.Module):
    """volume的特征是拼接到return的特征上去"""
    def __init__(self, d_model=512, n_experts=4):
        super().__init__()
        # MMoE核心组件（论文Section 4.2）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU()
            ) for _ in range(n_experts)])
        
        # 任务特定门控（return/volume/volatility）
        self.gates = nn.ModuleDict({
            'return': nn.Linear(d_model, n_experts),
            'volume': nn.Linear(d_model, n_experts),
            'volatility': nn.Linear(d_model, n_experts)
        })
        
        # 任务预测层
        self.return_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model + d_model//2, 1))
        
        self.volume_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
            
        self.volatility_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        # 门控融合组件（新需求）
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.Sigmoid())

    def forward(self, shared_feat):
        # MMoE处理（论文Eq.6-7）
        task_features = {}
        for task in ['return', 'volume', 'volatility']:
            gates = torch.softmax(self.gates[task](shared_feat), dim=1)
            expert_outputs = torch.stack([e(shared_feat) for e in self.experts], dim=1)
            task_features[task] = torch.einsum('bi,bij->bj', gates, expert_outputs)
        
        # 中间层融合（新需求）
        return_hidden = self.return_proj[:2](task_features['return'])  # d_model -> d_model/2
        volume_hidden = self.volume_proj[:2](task_features['volume'])
        volat_hidden = self.volatility_proj[:2](task_features['volatility'])
        
        # 门控融合（论文Section 4.2扩展）
        gate_weights = self.fusion_gate(shared_feat)
        fused = torch.concat([return_hidden, gate_weights[:,0:1] * volume_hidden, gate_weights[:,1:2] * volat_hidden], dim=-1) # 拼接volume和volatility的特征到return特征上
        
        # 最终预测
        return_pred = self.return_proj[2](fused)
        volume_pred = self.volume_proj[2](volume_hidden)
        volat_pred = self.volatility_proj[2](volat_hidden)
        
        return return_pred.squeeze(-1), volume_pred.squeeze(-1), volat_pred.squeeze(-1)

class MMoE_TaskFusion2_RV12(MMoE_TaskFusion_RV12):
    """两个并行，不交互"""
    def __init__(self, d_model=512, n_experts=4):
        super().__init__(d_model, n_experts)

    def forward(self, shared_feat):
        # MMoE处理（论文Eq.6-7）
        task_features = {}
        for task in ['return', 'volume', 'volatility']:
            gates = torch.softmax(self.gates[task](shared_feat), dim=1)
            expert_outputs = torch.stack([e(shared_feat) for e in self.experts], dim=1)
            task_features[task] = torch.einsum('bi,bij->bj', gates, expert_outputs)
        
        # 中间层融合（新需求）
        return_hidden = self.return_proj[:2](task_features['return'])  # d_model -> d_model/2
        volume_hidden = self.volume_proj[:2](task_features['volume'])
        volat_hidden = self.volatility_proj[:2](task_features['volatility'])
        
        # 门控融合（论文Section 4.2扩展）
        # gate_weights = self.fusion_gate(shared_feat)
        # fused = torch.concat([return_hidden, gate_weights[:,0:1] * volume_hidden, gate_weights[:,1:2] * volat_hidden], dim=-1) # 拼接volume和volatility的特征到return特征上
        
        # 最终预测
        return_pred = self.return_proj[2](return_hidden)
        volume_pred = self.volume_proj[2](volume_hidden)
        volat_pred = self.volatility_proj[2](volat_hidden)
        
        return return_pred.squeeze(-1), volume_pred.squeeze(-1), volat_pred.squeeze(-1)

class LiMT_v2_MMoE_RV1(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=4):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        # 原始特征处理层
        self.feature_layer = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model))
        
        # Attention层（保持原结构）
        self.s_attn = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
            # intra-stock aggregation
        self.t_attn = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        
        # MMoE融合模块
        self.mmoe_fusion = MMoE_TaskFusion_RV1(d_model, n_experts=n_experts)
    
    def forward(self, x):
        # 特征提取
        src = self.feature_layer(x)
        s_feat = self.s_attn(src)
        t_feat = self.t_attn(s_feat)
        
        # MMoE多任务预测
        return self.mmoe_fusion(t_feat[:, -1, :])  # 取最后时间步


################################ 消融实验模型 ################################
# 主模型
class LiMT_v2_MMoE_RVol2Vola2(LiMT_v2_MMoE_RV1):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=4, return_gate_weight=0, **kwargs):
        # 调用父类的构造函数，并传递所有必要的参数
        super().__init__(d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts)
        
        # MMoE融合模块
        self.mmoe_fusion = MMoE_TaskFusion_RVol2Vola2(d_model, n_experts=n_experts, return_gate_weight=return_gate_weight)
    
    def forward(self, x):
        # 调用父类的 forward 方法
        return super().forward(x)

# 主模型残差连接版
class LiMT_v2_MMoE_RVol2Vola2_RC(LiMT_v2_MMoE_RV1):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=4, return_gate_weight=0, **kwargs):
        # 调用父类的构造函数，并传递所有必要的参数
        super().__init__(d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts)
        
        # MMoE融合模块
        self.mmoe_fusion = MMoE_TaskFusion_RVol2Vola2_RC(d_model, n_experts=n_experts, return_gate_weight=return_gate_weight)
    
    def forward(self, x):
        # 调用父类的 forward 方法
        return super().forward(x)

# 去除1任务的模型
class LiMT_v2_MMoE_RV(LiMT_v2_MMoE_RV1):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=4, return_gate_weight=0, **kwargs):
        # 调用父类的构造函数，并传递所有必要的参数
        super().__init__(d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts)
        
        # MMoE融合模块
        self.mmoe_fusion = MMoE_TaskFusion_RV(d_model, n_experts=n_experts, return_gate_weight=return_gate_weight)
    
    def forward(self, x):
        # 调用父类的 forward 方法
        return super().forward(x)
    
# 去除2个任务的模型
class LiMT_v2_MMoE_R(LiMT_v2_MMoE_RV1):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=4, return_gate_weight=0, **kwargs):
        # 调用父类的构造函数，并传递所有必要的参数
        super().__init__(d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts)
        
        # MMoE融合模块
        self.mmoe_fusion = MMoE_TaskFusion_R(d_model, n_experts=n_experts, return_gate_weight=return_gate_weight)
    
    def forward(self, x):
        # 调用父类的 forward 方法
        return super().forward(x)

# 去除2个任务+mmoe的模型就是limt_v2
##### 纯为消融MoE设计的
class LiMT_V2_RVol2Vola2_woMoE(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
        
        self.volume_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))
            
        self.vola_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1))

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)[:, -1, :]  # [N, D]
        # Task-Specific Predictions
        return_pred = self.return_head(features).squeeze(-1)  # [N]
        volume_pred = self.volume_head(features).squeeze(-1)  # [N]
        vola_pred = self.vola_head(features).squeeze(-1) # [N]

        return return_pred, volume_pred, vola_pred
# 去除2个任务+mmoe+S_attention的模型就是transformer

# 去除SAttention的模型
class LiMT_v2_MMoE_RVol2Vola2_woSA(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=4, return_gate_weight=0):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        # 原始特征处理层
        self.feature_layer = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model))
        
        # Attention层（保持原结构）
        # self.s_attn = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
            # intra-stock aggregation
        self.t_attn = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        
        # MMoE融合模块
        self.mmoe_fusion = MMoE_TaskFusion_RVol2Vola2(d_model, n_experts=n_experts, return_gate_weight=return_gate_weight)
    
    def forward(self, x):
        # 特征提取
        src = self.feature_layer(x)
        # s_feat = self.s_attn(src)
        t_feat = self.t_attn(src)
        
        # MMoE多任务预测
        return self.mmoe_fusion(t_feat[:, -1, :])  # 取最后时间步


class DeepSeekMoE_v2(nn.Module):
    def __init__(self, hidden_dim=256, d_ff=1024, num_experts=16, k_s=1, dtype=torch.float32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_ff = d_ff
        self.total_experts = num_experts
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, d_ff, dtype=dtype),
                nn.GELU(),
                nn.Linear(d_ff, hidden_dim, dtype=dtype)
            ) for _ in range(self.total_experts - k_s)
        ])
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, d_ff, dtype=dtype),
                nn.GELU(),
                nn.Linear(d_ff, hidden_dim, dtype=dtype)
            ) for _ in range(k_s)
        ])
        # Three independent gates, one per task
        self.gates = nn.ModuleDict({
            "return":  nn.Linear(hidden_dim, len(self.routed_experts), dtype=dtype),
            "vol_hit": nn.Linear(hidden_dim, len(self.routed_experts), dtype=dtype),
            "volatility": nn.Linear(hidden_dim, len(self.routed_experts), dtype=dtype)
        })
        self.return_head = nn.Linear(hidden_dim, 1, dtype=dtype)
        self.vol_hit_head = nn.Linear(hidden_dim, 1, dtype=dtype)
        self.volatility_head = nn.Linear(hidden_dim, 1, dtype=dtype)
        self.active_k = num_experts // 2 - k_s
        self.dtype = dtype
    
    def _moe_forward(self, x: torch.Tensor, gate: nn.Linear) -> torch.Tensor:
        """Shared MoE computation for one task."""
        batch_size = x.size(0)

        # Shared experts
        shared_out = torch.zeros(batch_size, self.hidden_dim, dtype=self.dtype, device=x.device)
        for expert in self.shared_experts:
            shared_out += expert(x)

        # Routed experts
        gate_logits = gate(x)                                 # [B, E_routed]
        gate_scores = F.softmax(gate_logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(gate_scores, k=self.active_k, dim=-1)  # [B, k]

        # One-hot mask
        mask = F.one_hot(top_k_indices, num_classes=len(self.routed_experts)).float()   # [B, k, E_routed]

        # Expert outputs
        routed_input = x.unsqueeze(1).expand(-1, len(self.routed_experts), -1)            # [B, E_routed, D]
        routed_outputs = torch.stack([expert(routed_input[:, i]) for i, expert in enumerate(self.routed_experts)], dim=1)  # [B, E_routed, D]

        # Weighted sum
        top_k_out = torch.bmm(mask, routed_outputs)                                       # [B, k, D]
        top_k_out = (top_k_out * top_k_scores.unsqueeze(-1)).sum(dim=1)                   # [B, D]

        return x + shared_out + top_k_out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Route through each task-specific gate
        return_feat  = self._moe_forward(x, self.gates["return"])
        vol_hit_feat = self._moe_forward(x, self.gates["vol_hit"])
        vol_feat     = self._moe_forward(x, self.gates["volatility"])

        # Final predictions
        return_pred  = self.return_head(return_feat).squeeze(-1)
        vol_hit_pred = self.vol_hit_head(vol_hit_feat).squeeze(-1)
        volatility_pred = self.volatility_head(vol_feat).squeeze(-1)

        return return_pred, vol_hit_pred, volatility_pred
    
class DeepSeekMoE(nn.Module):
    def __init__(self, hidden_dim=256, d_ff=1024, num_experts=8, m=4, k_s=1, dtype=torch.float32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_ff = d_ff
        self.total_experts = num_experts * m  # 32
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, d_ff // m, dtype=dtype),
                nn.ReLU(),
                nn.Linear(d_ff // m, hidden_dim, dtype=dtype)
            ) for _ in range(self.total_experts - k_s)
        ])
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, d_ff // m, dtype=dtype),
                nn.ReLU(),
                nn.Linear(d_ff // m, hidden_dim, dtype=dtype)
            ) for _ in range(k_s)
        ])
        self.gate = nn.Linear(hidden_dim, len(self.routed_experts), dtype=dtype)
        self.return_head = nn.Linear(hidden_dim, 1, dtype=dtype)
        self.vol_hit_head = nn.Linear(hidden_dim, 1, dtype=dtype)
        self.volatility_head = nn.Linear(hidden_dim, 1, dtype=dtype)
        self.active_k = m * 2 - k_s
        self.dtype = dtype
    
    def forward(self, x):
        batch_size = x.shape[0]
        shared_out = torch.zeros(batch_size, self.hidden_dim, dtype=self.dtype, device=x.device)
        for expert in self.shared_experts:
            shared_out += expert(x)
        
        gate_logits = self.gate(x)  # [batch, 31]
        gate_scores = F.softmax(gate_logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(gate_scores, k=self.active_k, dim=-1)  # [batch, 7]
        
        # 生成正确维度的one-hot矩阵
        top_k_mask = F.one_hot(top_k_indices, num_classes=len(self.routed_experts)).float().to(x.device)  # [batch, 7, 31]
        
        # 计算路由专家输出（确保3D张量）
        routed_input = x.unsqueeze(1).expand(-1, len(self.routed_experts), -1)  # [batch, 31, 256]
        all_routed_output = torch.stack([expert(routed_input[:, i]) for i, expert in enumerate(self.routed_experts)], dim=1)  # [batch, 31, 256]
        
        # 批量矩阵乘法（3D × 3D）
        top_k_routed = torch.bmm(top_k_mask, all_routed_output)  # [batch, 7, 256]
        
        # 加权求和
        top_k_scores_3d = top_k_scores.unsqueeze(-1)  # [batch, 7, 1]
        weighted_top_k = top_k_routed * top_k_scores_3d  # [batch, 7, 256]
        top_k_routed = torch.sum(weighted_top_k, dim=1)  # [batch, 256]
        
        # 合并输出
        moe_features = x + shared_out + top_k_routed  # [batch, 256]
        return self.return_head(moe_features).squeeze(-1), self.vol_hit_head(moe_features).squeeze(-1), self.volatility_head(moe_features).squeeze(-1)

class LiMT_v2_DSMOE(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=8):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        # 原始特征处理层
        self.feature_layer = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model))
        
        # Attention层（保持原结构）
        self.s_attn = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
            # intra-stock aggregation
        self.t_attn = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        
        # MMoE融合模块
        self.mmoe_fusion = DeepSeekMoE(d_model, num_experts=n_experts)
    
    def forward(self, x):
        # 特征提取
        src = self.feature_layer(x)
        s_feat = self.s_attn(src)
        t_feat = self.t_attn(s_feat)
        
        # MMoE多任务预测
        return self.mmoe_fusion(t_feat[:, -1, :])  # 取最后时间步

class LiMT_v2_DSMOE2(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=8, k_s=1):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        # 原始特征处理层
        self.feature_layer = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model))
        
        # Attention层（保持原结构）
        self.s_attn = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
            # intra-stock aggregation
        self.t_attn = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        
        # MMoE融合模块
        self.mmoe_fusion = DeepSeekMoE_v2(d_model, num_experts=n_experts, k_s=k_s)
    
    def forward(self, x):
        # 特征提取
        src = self.feature_layer(x)
        s_feat = self.s_attn(src)
        t_feat = self.t_attn(s_feat)
        
        # MMoE多任务预测
        return self.mmoe_fusion(t_feat[:, -1, :])  # 取最后时间步

class LiMT_v2_MMoE1_RV1(LiMT_v2_MMoE_RV1):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=4, **kwargs):
        # 调用父类的构造函数，并传递所有必要的参数
        super().__init__(d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts)
        
        # MMoE融合模块
        self.mmoe_fusion = MMoE_TaskFusion1_RV1(d_model, n_experts=n_experts)
    
    def forward(self, x):
        # 调用父类的 forward 方法
        return super().forward(x)

class LiMT_v2_MMoE2_RV1(LiMT_v2_MMoE_RV1):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=4, **kwargs):
        # 调用父类的构造函数，并传递所有必要的参数
        super().__init__(d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts)
        
        # MMoE融合模块
        self.mmoe_fusion = MMoE_TaskFusion2_RV1(d_model, n_experts=n_experts)
    
    def forward(self, x):
        # 调用父类的 forward 方法
        return super().forward(x)

class DynamicKController(nn.Module):
    """动态k值预测器（基于市场波动性）"""
    def __init__(self, d_model):
        super().__init__()
        self.volatility_net = nn.Sequential(
            nn.Linear(d_model, 16),       # 输入: [B, d_model]
            nn.GELU(),                    # 输出: [B, 16]
            nn.Linear(16, 1)              # 输出: [B, 1]
        )
        self.k_pred = nn.Linear(1, 1)
    
    def forward(self, feat):
        # 输入: feat [B, d_model]
        vol = self.volatility_net(feat.detach())  # 波动性估计，输出: [B, 1]
        k_float = torch.sigmoid(self.k_pred(vol)) * 3 + 1  # k ∈ [1,4]，输出: [B, 1]
        k = k_float.round().int()
        return k + (k_float - k_float.detach())  # 保持梯度流动，输出: [B, 1]

class DynamicTaskMoE(nn.Module):
    def __init__(self, d_model=256, n_experts=8, tasks=['return', 'volume1', 'volume2']):
        super().__init__()
        self.num_task = len(tasks)
        self.tasks = tasks
        # 共享专家池
        self.shared_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
            for _ in range(n_experts//2)])
        
        # 任务特定专家
        self.task_experts = nn.ModuleDict({
            t: nn.ModuleList([
                nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
                for _ in range(n_experts//2)])
            for t in tasks
        })
        
        # 动态路由
        self.task_embeddings = nn.Embedding(self.num_task, d_model)  # [num_task, d_model]
        self.gate_net = nn.Linear(d_model*2, n_experts)  # 输入: [B, d_model*2] 输出: [B, n_experts]
        
        # 预测头
        self.predictors = nn.ModuleDict({
            t: nn.Linear(d_model, 1) for t in tasks
        })
        
        # 动态k值控制
        self.k_controller = DynamicKController(d_model)
        
        # 正则化参数
        self.ortho_lambda = 0.01

    def forward(self, shared_feat):
        """
        输入: shared_feat (B, d_model)
        输出: 各任务预测值和正则损失
        """
        B = shared_feat.size(0)
        task_embs = self.task_embeddings.weight.unsqueeze(0)  # (1, num_task, d_model)
        
        # 并行处理所有任务
        all_gate_input = torch.cat([
            shared_feat.unsqueeze(1).expand(-1, self.num_task, -1),  # (B, self.num_task, d_model)
            task_embs.expand(B, -1, -1)                 # (B, self.num_task, d_model)
        ], dim=-1)                                       # 输出: (B, self.num_task, d_model*2)
        
        # 生成门控权重 (B, self.num_task, n_experts+1)
        gate_logits = self.gate_net(all_gate_input)      # 输入: (B, self.num_task, d_model*2) 输出: (B, self.num_task, n_experts)
        
        # 动态k值获取 (B, self.num_task)
        k_vals = self.k_controller(shared_feat).expand(-1, self.num_task)  # 输入: [B, d_model]，输出: [B, self.num_task]
        
        # 专家计算
        outputs = []
        ortho_loss = 0
        for task_idx in range(self.num_task):
            # 任务特定处理
            task_k = k_vals[:, task_idx]  # [B]
            task_gates = gate_logits[:, task_idx, :]  # [B, n_experts+1]
            
            # Top-k稀疏路由
            gate_vals, gate_idx = torch.topk(task_gates, k=int(task_k.max().item()), dim=1)  # 输出: [B, max_k]
            mask = torch.arange(gate_vals.size(1)).expand(B, -1).to(gate_idx.device) < task_k.unsqueeze(1)  # [B, max_k]
            gate_vals = gate_vals * mask.float()  # 应用掩码
            gate_weights = F.softmax(gate_vals, dim=1)  # [B, max_k]
            
            # 专家执行
            expert_outputs = []
            for idx in gate_idx.unique():  # 按唯一专家索引
                if idx < len(self.shared_experts):
                    expert = self.shared_experts[idx]
                else:
                    expert = self.task_experts[self.tasks[task_idx]][idx - len(self.shared_experts)]
                expert_outputs.append(expert(shared_feat))  # 输入: [B, d_model]，输出: [B, d_model]
            
            # 正交正则
            ortho_loss += self._ortho_reg(torch.stack(expert_outputs, dim=1))  # 输入: [B, k, d_model]
            
            # 加权输出
            combined = torch.stack(expert_outputs, dim=1)  # [B, k, d_model]
            print(gate_weights.shape, combined.shape, gate_idx.unique())
            task_rep = torch.bmm(gate_weights.unsqueeze(1), combined).squeeze(1)  # 输入: [B, 1, max_k] 和 [B, max_k, d_model]，输出: [B, d_model]
            outputs.append(self.predictors[self.tasks[task_idx]](task_rep))  # 输入: [B, d_model]，输出: [B, 1]
        
        return outputs[0].squeeze(-1), outputs[1].squeeze(-1), outputs[2].squeeze(-1)

    def _ortho_reg(self, experts):
        """专家正交约束"""
        # 输入: [B, k, d_model]
        experts = experts - experts.mean(dim=0)  # 中心化
        cov = torch.matmul(experts.transpose(1, 2), experts)  # [B, d_model, d_model]
        eye = torch.eye(cov.size(-1)).to(cov.device)
        return torch.norm(cov - eye, p='fro', dim=(1, 2)).mean()  # 输出: 标量


class LiMT_v2_MMoE_RV12(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=4):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        # 原始特征处理层
        self.feature_layer = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model))
        
        # Attention层（保持原结构）
        self.s_attn = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
            # intra-stock aggregation
        self.t_attn = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        
        # MMoE融合模块
        self.mmoe_fusion = MMoE_TaskFusion_RV12(d_model, n_experts=n_experts)
    
    def forward(self, x):
        # 特征提取
        src = self.feature_layer(x)
        s_feat = self.s_attn(src)
        t_feat = self.t_attn(s_feat)
        
        # MMoE多任务预测
        return self.mmoe_fusion(t_feat[:, -1, :])  # 取最后时间步
    
class LiMT_v2_MMoE1_RV12(LiMT_v2_MMoE_RV12):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=4):
        super().__init__(d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts)
        
        # MMoE融合模块
        self.mmoe_fusion = MMoE_TaskFusion1_RV12(d_model, n_experts=n_experts)
    
    def forward(self, x):
        return super().forward(x)

class LiMT_v2_MMoE2_RV12(LiMT_v2_MMoE_RV12):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=4):
        super().__init__(d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts)
        
        # MMoE融合模块
        self.mmoe_fusion = MMoE_TaskFusion2_RV12(d_model, n_experts=n_experts)
    
    def forward(self, x):
        return super().forward(x)

class LiMT_v2_MMoE3_RV12(LiMT_v2_MMoE_RV12):
    """集成时空注意力的完整模型"""
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts=4):
        super().__init__(d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, n_experts)
        # MoE多任务模块
        self.mmoe_fusion = DynamicTaskMoE(d_model, n_experts, tasks=['return', 'volume1', 'volume2'])
    
    def forward(self, x):
        return super().forward(x)
    
class HierarchicalFusionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),  # 比ReLU更平滑
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, main_feat, aux_feat):
        # 维度处理 (MultiheadAttention需要seq_len在前)
        main_feat = main_feat.unsqueeze(0)  # [1, N, D]
        aux_feat = aux_feat.unsqueeze(0)
        
        # 交叉注意力
        attn_out, _ = self.attn(
            query=main_feat,
            key=aux_feat,
            value=aux_feat
        )
        attn_out = self.dropout(attn_out.squeeze(0))
        
        # 残差连接+Norm
        norm_out = self.norm1(main_feat.squeeze(0) + attn_out)
        
        # FFN
        ffn_out = self.ffn(norm_out)
        output = self.norm2(norm_out + ffn_out)
        return output

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output,  beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output =d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output/self.t, dim=-1)
        return self.d_output*output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z) # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D] --> [N, D]
        return output

class Transformer(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(Transformer, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )
        self.final_pred = nn.Linear(d_model, 1)

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index] # N, T, D
       
        output = self.layers(src)
        output = output[:, -1, :].squeeze(1) # N, D
        output = self.final_pred(output).squeeze(1) # N

        return output
    
class Transformer_RV1(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(Transformer_RV1, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.volume_head = nn.Linear(d_model, 1)  # 辅助任务：Volume预测

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)  # [N, T, D]
        last_features = features[:, -1, :]  # [N, D] (取最后时间步)

        # Task-Specific Predictions
        return_pred = self.return_head(last_features).squeeze(-1)  # [N]
        volume_pred = self.volume_head(last_features).squeeze(-1)  # [N]

        return return_pred, volume_pred

class Transformer_RV12(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(Transformer_RV12, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.volume1_head = nn.Linear(d_model, 1)  # 辅助任务：Volume预测
        self.volume2_head = nn.Linear(d_model, 1)  # 辅助任务：Volume预测

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)  # [N, T, D]
        last_features = features[:, -1, :]  # [N, D] (取最后时间步)

        # Task-Specific Predictions
        return_pred = self.return_head(last_features).squeeze(-1)  # [N]
        volume1_pred = self.volume1_head(last_features).squeeze(-1)  # [N]
        volume2_pred = self.volume2_head(last_features).squeeze(-1)  # [N]

        return return_pred, volume1_pred, volume2_pred

class Transformer_RC(nn.Module):
    """ 收益率回归 + 有序分类多任务模型 """
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, 
                 S_dropout_rate, gate_input_start_index, gate_input_end_index, 
                 beta, num_classes=10):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        # Shared backbone
        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )
        
        # Task-specific heads
        self.return_head = nn.Linear(d_model, 1)
        self.class_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Feature extraction
        src = x[:, :, :self.gate_input_start_index]
        features = self.layers(src)
        last_features = features[:, -1, :]  # [N, D]
        
        # Task predictions
        return_pred = self.return_head(last_features).squeeze(-1)  # [N]
        class_logits = self.class_head(last_features)  # [N, num_classes]
        
        return return_pred, class_logits
    
class Transformer_RVC(nn.Module):
    """ 收益率回归 + volume回归 + 有序分类多任务模型 """
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, 
                 S_dropout_rate, gate_input_start_index, gate_input_end_index, 
                 beta, num_classes=10):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        # Shared backbone
        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )
        
        # Task-specific heads
        self.return_head = nn.Linear(d_model, 1)
        self.volume_head = nn.Linear(d_model, 1) # 辅助任务：Volume预测
        self.class_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Feature extraction
        src = x[:, :, :self.gate_input_start_index]
        features = self.layers(src)
        last_features = features[:, -1, :]  # [N, D]
        
        # Task predictions
        return_pred = self.return_head(last_features).squeeze(-1)  # [N]
        volume_pred = self.volume_head(last_features).squeeze(-1) # [N]
        class_logits = self.class_head(last_features)  # [N, num_classes]
        
        return return_pred, volume_pred, class_logits

class Transformer_RV12C(nn.Module):
    """ 收益率回归 + Volume12回归+有序分类多任务模型 """
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, 
                 S_dropout_rate, gate_input_start_index, gate_input_end_index, 
                 beta, num_classes=10):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        # Shared backbone
        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )
        
        # Task-specific heads
        self.return_head = nn.Linear(d_model, 1)
        self.volume1_head = nn.Linear(d_model, 1) # 辅助任务：Volume预测
        self.volume2_head = nn.Linear(d_model, 1) # 辅助任务：Volume预测
        self.class_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Feature extraction
        src = x[:, :, :self.gate_input_start_index]
        features = self.layers(src)
        last_features = features[:, -1, :]  # [N, D]
        
        # Task predictions
        return_pred = self.return_head(last_features).squeeze(-1)  # [N]
        volume1_pred = self.volume1_head(last_features).squeeze(-1) # [N]
        volume2_pred = self.volume2_head(last_features).squeeze(-1) # [N]
        class_logits = self.class_head(last_features)  # [N, num_classes]
        
        return return_pred, volume1_pred, volume2_pred, class_logits


class LiMT(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(LiMT, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            # decoder
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index] # N, T, D
        # gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        # src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
       
        output = self.layers(src).squeeze(-1)

        return output

class LiMT_RV1(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(LiMT_RV1, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.volume_head = nn.Linear(d_model, 1)  # 辅助任务：Volume预测

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)  # [N, D]
        # Task-Specific Predictions
        return_pred = self.return_head(features).squeeze(-1)  # [N]
        volume_pred = self.volume_head(features).squeeze(-1)  # [N]

        return return_pred, volume_pred
  
class LiMT_RV12(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(LiMT_RV12, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.volume1_head = nn.Linear(d_model, 1)  # 辅助任务：Volume1预测
        self.volume2_head = nn.Linear(d_model, 1)  # 辅助任务：Volume1预测

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)  # [N, D]

        # Task-Specific Predictions
        return_pred = self.return_head(features).squeeze(-1)  # [N]
        volume1_pred = self.volume1_head(features).squeeze(-1)  # [N]
        volume2_pred = self.volume2_head(features).squeeze(-1)  # [N]

        return return_pred, volume1_pred, volume2_pred

class MASTER_RC(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, num_classes=10):
        super(MASTER_RC, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.class_head = nn.Linear(d_model, num_classes)  # 辅助任务：classification预测

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)  # [N, D]
        # Task-Specific Predictions
        return_pred = self.return_head(features).squeeze(-1)  # [N]
        class_logits = self.class_head(features)  # [N, num_classes]

        return return_pred, class_logits
    
class LiMT_RVC(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, num_classes=10):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.volume_head = nn.Linear(d_model, 1) # 辅助任务：Volume预测
        self.class_head = nn.Linear(d_model, num_classes)  # 辅助任务：classification预测

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)  # [N, D]
        # Task-Specific Predictions
        return_pred = self.return_head(features).squeeze(-1)  # [N]
        volume_pred = self.volume_head(features).squeeze(-1) # [N]
        class_logits = self.class_head(features)  # [N, num_classes]

        return return_pred, volume_pred, class_logits

class LiMT_RV12C(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, num_classes=10):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.volume1_head = nn.Linear(d_model, 1) # 辅助任务：Volume预测
        self.volume2_head = nn.Linear(d_model, 1) # 辅助任务：Volume预测
        self.class_head = nn.Linear(d_model, num_classes)  # 辅助任务：classification预测

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)  # [N, D]
        # Task-Specific Predictions
        return_pred = self.return_head(features).squeeze(-1)  # [N]
        volume1_pred = self.volume1_head(features).squeeze(-1) # [N]
        volume2_pred = self.volume2_head(features).squeeze(-1) # [N]
        class_logits = self.class_head(features)  # [N, num_classes]
        
        return return_pred, volume1_pred, volume2_pred, class_logits

class MASTER_Hierarchical(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, 
                 gate_input_start_index, gate_input_end_index, beta, num_classes=10):
        super().__init__()
        # Shared Backbone (与原有MASTER相同)
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model, t_nhead, T_dropout_rate),
            SAttention(d_model, s_nhead, S_dropout_rate),
            TemporalAttention(d_model)
        )
        
        # Task-specific Encoders
        self.volume_head = nn.Linear(d_model, 1)  # Volume预测头
        self.class_head = nn.Linear(d_model, num_classes)  # 分类头
        
        # Hierarchical Fusion Modules
        self.volume_fusion = HierarchicalFusionLayer(d_model, nhead=4)
        self.class_fusion = HierarchicalFusionLayer(d_model, nhead=4)
        
        # 新增分类特征提取层
        self.class_feat_extract = nn.Sequential(
            nn.Linear(num_classes, d_model),  # 将分类logits映射到特征空间
            nn.GELU()
        )

        # Final Prediction
        self.return_head = nn.Sequential(
            nn.Linear(d_model*2, d_model),  # 融合后维度扩展
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]
        shared_feat = self.shared_encoder(src)  # [N, D]
        
        # Auxiliary Task Predictions
        volume_feat = self.shared_encoder[:-1](src)[:, -1, :]  # 共享除Temporal外的层 [N, D]
        volume_pred = self.volume_head(volume_feat).squeeze(-1)
        
        class_logits = self.class_head(shared_feat)

        # 获取分类特征
        class_feat = self.class_feat_extract(class_logits)  # [N, d_model//2]
        
        # 分层融合
        # Step1: Volume特征融合
        fused_vol = self.volume_fusion(shared_feat, volume_feat)
        # Step2: 分类特征融合（使用class_feat而非shared_feat）
        fused_cls = self.class_fusion(fused_vol, class_feat)  # 修正此处
        # 最终预测
        final_feat = torch.cat([fused_vol, fused_cls], dim=-1)
        return_pred = self.return_head(final_feat).squeeze(-1)
        
        return return_pred, volume_pred, class_logits

class LiMT_Hierarchical_v2(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, 
                 gate_input_start_index, gate_input_end_index, beta, num_classes=10):
        super().__init__()
        # Shared Backbone
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model, t_nhead, T_dropout_rate),
            SAttention(d_model, s_nhead, S_dropout_rate),
            TemporalAttention(d_model)
        )
        
        # Task-specific Heads
        self.volume_head = nn.Linear(d_model, 1)
        self.class_head = nn.Linear(d_model, num_classes)
        self.class_feat_extract = nn.Sequential(
            nn.Linear(num_classes, d_model),
            nn.GELU()
        )
        
        # Fusion Modules
        self.volume_fusion = HierarchicalFusionLayer(d_model, nhead=4)
        self.class_fusion = HierarchicalFusionLayer(d_model, nhead=4)
        
        # Enhanced Regression Head
        self.return_head = nn.Sequential(
            nn.Linear(d_model + num_classes, d_model),  # 显式拼接分类概率
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x):
        # Shared Feature
        src = x[:, :, :self.gate_input_start_index]
        shared_feat = self.shared_encoder(src)  # [N, D]
        
        # Auxiliary Predictions
        volume_feat = self.shared_encoder[:-1](src)[:, -1, :] # [N, D]
        volume_pred = self.volume_head(volume_feat).squeeze(-1)
        class_logits = self.class_head(shared_feat)
        class_probs = torch.softmax(class_logits, dim=-1)
        class_feat = self.class_feat_extract(class_logits)
        
        # Hierarchical Fusion
        fused_vol = self.volume_fusion(shared_feat, volume_feat)
        fused_cls = self.class_fusion(fused_vol, class_feat)
        
        # Final Prediction with Explicit Class Probs
        final_feat = torch.cat([fused_cls, class_probs], dim=-1)  # 双重融合
        return_pred = self.return_head(final_feat).squeeze(-1)
        
        return return_pred, volume_pred, class_logits
    
class LiMT_V2(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(LiMT_V2, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )
        # decoder
        self.final_pred = nn.Linear(d_model, 1)
        

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index] # N, T, D
        # gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        # src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
        features = self.layers(src)[:, -1, :]  # [N, D]
        output = self.final_pred(features).squeeze(-1) # [N]

        return output
    
class LiMT_V2_RV1(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(LiMT_V2_RV1, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.volume_head = nn.Linear(d_model, 1)  # 辅助任务：Volume预测

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)[:, -1, :]  # [N, D]
        # Task-Specific Predictions
        return_pred = self.return_head(features).squeeze(-1)  # [N]
        volume_pred = self.volume_head(features).squeeze(-1)  # [N]

        return return_pred, volume_pred
    
class LiMT_V2_RVol2Vola2(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.volume_head = nn.Linear(d_model, 1)  # 辅助任务：Volume预测
        self.vola_head = nn.Linear(d_model, 1)  # 辅助任务：Volatility预测

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)[:, -1, :]  # [N, D]
        # Task-Specific Predictions
        return_pred = self.return_head(features).squeeze(-1)  # [N]
        volume_pred = self.volume_head(features).squeeze(-1)  # [N]
        vola_pred = self.vola_head(features).squeeze(-1) # [N]

        return return_pred, volume_pred, vola_pred
    
class LiMT_V2_RV12(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(LiMT_V2_RV12, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.volume1_head = nn.Linear(d_model, 1)  # 辅助任务：Volume1预测
        self.volume2_head = nn.Linear(d_model, 1)  # 辅助任务：Volume1预测

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)[:, -1, :]  # [N, D]

        # Task-Specific Predictions
        return_pred = self.return_head(features).squeeze(-1)  # [N]
        volume1_pred = self.volume1_head(features).squeeze(-1)  # [N]
        volume2_pred = self.volume2_head(features).squeeze(-1)  # [N]

        return return_pred, volume1_pred, volume2_pred

class LiMT_V2_RC(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, num_classes=10):
        super(LiMT_RC, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.class_head = nn.Linear(d_model, num_classes) # 辅助任务：crossReturn-classification

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)[:, -1, :]  # [N, D]

        # Task-Specific Predictions
        return_pred = self.return_head(features).squeeze(-1)  # [N]
        class_logits = self.class_head(features)  # [N, num_classes]

        return return_pred, class_logits
    
class LiMT_RVC(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, num_classes=10):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.volume_head = nn.Linear(d_model, 1) # 辅助任务：Volume预测
        self.class_head = nn.Linear(d_model, num_classes) # 辅助任务：crossReturn-classification

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)[:, -1, :]  # [N, D]

        # Task-Specific Predictions
        return_pred = self.return_head(features).squeeze(-1)  # [N]
        volume_pred = self.volume_head(features).squeeze(-1) # [N]
        class_logits = self.class_head(features)  # [N, num_classes]

        return return_pred, volume_pred, class_logits

class LiMT_RV12C(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, num_classes=10):
        super().__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        )

        # Task-Specific Heads (直接对Backbone输出做预测，减少冗余层)
        self.return_head = nn.Linear(d_model, 1)  # 主任务：Return预测
        self.volume1_head = nn.Linear(d_model, 1) # 辅助任务：Volume预测
        self.volume2_head = nn.Linear(d_model, 1) # 辅助任务：Volume预测
        self.class_head = nn.Linear(d_model, num_classes) # 辅助任务：crossReturn-classification

    def forward(self, x):
        # Shared Feature Extraction
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D]
        features = self.layers(src)[:, -1, :]  # [N, D]

        # Task-Specific Predictions
        return_pred = self.return_head(features).squeeze(-1)  # [N]
        volume1_pred = self.volume1_head(features).squeeze(-1) # [N]
        volume2_pred = self.volume2_head(features).squeeze(-1) # [N]
        class_logits = self.class_head(features)  # [N, num_classes]

        return return_pred, volume1_pred, volume2_pred, class_logits


class GateFormer(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, seq_len=21,
                       patch_len=4, stride=2, e_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding = stride
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'

        # 1. Patch Embedding:
        # Input: [B, D, L] -> Output: [B*D, patch_num, d_model]
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, self.padding, T_dropout_rate)
        # 2. Project Embedding: 
        # Input: [B, L, D] -> Output: [B, D, d_model]
        self.project_embedding = DataEmbedding_inverted(seq_len, d_model, T_dropout_rate)
        # 3. Head layer
        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        # Input: [B, D, d_model, patch_num] -> Output: [B, D, d_model]
        self.head = FlattenHead(d_feat, self.head_nf, d_model, head_dropout=T_dropout_rate)

        # 4. Gate层
        self.gate_w1 = nn.Linear(d_model, d_model)
        self.gate_w2 = nn.Linear(d_model, d_model)
        self.gate_w3 = nn.Linear(d_model, d_model)
        self.gate_w4 = nn.Linear(d_model, d_model)
        self.gate_sigmoid = nn.Sigmoid()

        # 5. Temporal & Feature Attention
        enc_layers_temporal = nn.TransformerEncoderLayer(
                                    d_model=d_model,
                                    nhead=t_nhead,
                                    dim_feedforward=d_model,
                                    dropout=T_dropout_rate,
                                    layer_norm_eps=1e-5,
                                    batch_first=True, 
                                    norm_first=True 
                                )

        # temporal-wise attention
        enc_layers_feature = nn.TransformerEncoderLayer(
                                    d_model=d_model,
                                    nhead=t_nhead,
                                    dim_feedforward=d_model,
                                    dropout=T_dropout_rate,
                                    layer_norm_eps=1e-5,
                                    batch_first=True,
                                    norm_first=True
                                )
        
        # temporal-wise attention
        self.enc_temporal = nn.TransformerEncoder(enc_layers_temporal, num_layers=e_layers)

        # variate-wise attention
        self.enc_variate = nn.TransformerEncoder(enc_layers_feature, num_layers=e_layers)

        # 修改projection层为两步预测
        self.feature_projection = nn.Linear(d_model, 1)  # 将每个特征的d_model维表示映射为标量
        self.final_projection = nn.Linear(d_feat, 1)     # 将所有特征的预测整合为最终预测



    def forward(self, x):
        # x: [B, T, D]  batch_size, seq_len, feat_dim
        src = x[:, :, :self.gate_input_start_index]
        
        # 1. Global Embedding
        global_embedding = self.project_embedding(src, None)  # [B, D, d_model]
        
        # 2. Local Pattern Embedding
        x_enc = src.permute(0, 2, 1)  # [B, D, T]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [B*D, patch_num, d_model]
        
        # print("enc_out shape:", enc_out.shape, "n_vars:", n_vars)
        # 3. Temporal Attention
        enc_out = self.enc_temporal(enc_out)  # [B*D, patch_num, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, D, d_model, patch_num]
        temporal_dependency_embedding = self.head(enc_out)  # [B, D, d_model]
        
        # 4. First Gate Fusion
        gate = self.gate_sigmoid(self.gate_w1(global_embedding) + self.gate_w2(temporal_dependency_embedding))
        enc_out = gate * global_embedding + (1 - gate) * temporal_dependency_embedding  # [B, D, d_model]
        
        # 5. Cross Attention
        enc_out_cross = self.enc_variate(enc_out)  # [B, D, d_model]
        
        # 6. Second Gate Fusion
        gate = self.gate_sigmoid(self.gate_w3(enc_out) + self.gate_w4(enc_out_cross))
        enc_out = gate * enc_out + (1 - gate) * enc_out_cross  # [B, D, d_model]
        
        # 7. Final Projection
        pred = self.feature_projection(enc_out).squeeze(-1)  # [B, D]
        
        output = self.final_projection(pred).squeeze(-1) # [B]
        
        return output

class GateFormer_RV1(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, seq_len=21,
                       patch_len=4, stride=2, e_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding = stride
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'

        # 1. Patch Embedding:
        # Input: [B, D, L] -> Output: [B*D, patch_num, d_model]
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, self.padding, T_dropout_rate)
        # 2. Project Embedding: 
        # Input: [B, L, D] -> Output: [B, D, d_model]
        self.project_embedding = DataEmbedding_inverted(seq_len, d_model, T_dropout_rate)
        # 3. Head layer
        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        # Input: [B, D, d_model, patch_num] -> Output: [B, D, d_model]
        self.head = FlattenHead(d_feat, self.head_nf, d_model, head_dropout=T_dropout_rate)

        # 4. Gate层
        self.gate_w1 = nn.Linear(d_model, d_model)
        self.gate_w2 = nn.Linear(d_model, d_model)
        self.gate_w3 = nn.Linear(d_model, d_model)
        self.gate_w4 = nn.Linear(d_model, d_model)
        self.gate_sigmoid = nn.Sigmoid()

        # 5. Temporal & Feature Attention
        enc_layers_temporal = nn.TransformerEncoderLayer(
                                    d_model=d_model,
                                    nhead=t_nhead,
                                    dim_feedforward=d_model,
                                    dropout=T_dropout_rate,
                                    layer_norm_eps=1e-5,
                                    batch_first=True, 
                                    norm_first=True 
                                )

        # temporal-wise attention
        enc_layers_feature = nn.TransformerEncoderLayer(
                                    d_model=d_model,
                                    nhead=t_nhead,
                                    dim_feedforward=d_model,
                                    dropout=T_dropout_rate,
                                    layer_norm_eps=1e-5,
                                    batch_first=True,
                                    norm_first=True
                                )
        
        # temporal-wise attention
        self.enc_temporal = nn.TransformerEncoder(enc_layers_temporal, num_layers=e_layers)

        # variate-wise attention
        self.enc_variate = nn.TransformerEncoder(enc_layers_feature, num_layers=e_layers)

        # 修改projection层为两步预测
        self.ret_pred = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Squeeze(-1),
            nn.Linear(d_feat, 1),
            nn.Squeeze(-1)
        )

        self.vol_pred = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Squeeze(-1),
            nn.Linear(d_feat, 1),
            nn.Squeeze(-1)
        )


    def forward(self, x):
        # x: [B, T, D]  batch_size, seq_len, feat_dim
        src = x[:, :, :self.gate_input_start_index]
        
        # 1. Global Embedding
        global_embedding = self.project_embedding(src, None)  # [B, D, d_model]
        
        # 2. Local Pattern Embedding
        x_enc = src.permute(0, 2, 1)  # [B, D, T]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [B*D, patch_num, d_model]
        
        # print("enc_out shape:", enc_out.shape, "n_vars:", n_vars)
        # 3. Temporal Attention
        enc_out = self.enc_temporal(enc_out)  # [B*D, patch_num, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, D, d_model, patch_num]
        temporal_dependency_embedding = self.head(enc_out)  # [B, D, d_model]
        
        # 4. First Gate Fusion
        gate = self.gate_sigmoid(self.gate_w1(global_embedding) + self.gate_w2(temporal_dependency_embedding))
        enc_out = gate * global_embedding + (1 - gate) * temporal_dependency_embedding  # [B, D, d_model]
        
        # 5. Cross Attention
        enc_out_cross = self.enc_variate(enc_out)  # [B, D, d_model]
        
        # 6. Second Gate Fusion
        gate = self.gate_sigmoid(self.gate_w3(enc_out) + self.gate_w4(enc_out_cross))
        enc_out = gate * enc_out + (1 - gate) * enc_out_cross  # [B, D, d_model]
        
        # 7. Final Projection
        ret_pred = self.ret_pred(enc_out) # [B]
        vol_pred = self.vol_pred(enc_out) # [B]
        
        return ret_pred, vol_pred
    

class GMASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, seq_len=21,
                       patch_len=4, stride=2, e_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding = stride
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'

        # 1. Patch Embedding:
        # Input: [B, D, L] -> Output: [B*D, patch_num, d_model]
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, self.padding, T_dropout_rate)
        # 2. Project Embedding: 
        # Input: [B, L, D] -> Output: [B, D, d_model]
        self.project_embedding = DataEmbedding_inverted(seq_len, d_model, T_dropout_rate)
        # 3. Head layer
        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        # Input: [B, D, d_model, patch_num] -> Output: [B, D, d_model]
        self.head = FlattenHead(d_feat, self.head_nf, d_model, head_dropout=T_dropout_rate)

        # 4. Gate层
        self.gate_w1 = nn.Linear(d_model, d_model)
        self.gate_w2 = nn.Linear(d_model, d_model)
        self.gate_w3 = nn.Linear(d_model, d_model)
        self.gate_w4 = nn.Linear(d_model, d_model)
        self.gate_sigmoid = nn.Sigmoid()

        # 5. Temporal & Feature Attention
        enc_layers_temporal = nn.TransformerEncoderLayer(
                                    d_model=d_model,
                                    nhead=t_nhead,
                                    dim_feedforward=d_model,
                                    dropout=T_dropout_rate,
                                    layer_norm_eps=1e-5,
                                    batch_first=True, 
                                    norm_first=True 
                                )

        # temporal-wise attention
        enc_layers_feature = nn.TransformerEncoderLayer(
                                    d_model=d_model,
                                    nhead=t_nhead,
                                    dim_feedforward=d_model,
                                    dropout=T_dropout_rate,
                                    layer_norm_eps=1e-5,
                                    batch_first=True,
                                    norm_first=True
                                )
        
        # temporal-wise attention
        self.enc_temporal = nn.TransformerEncoder(enc_layers_temporal, num_layers=e_layers)

        # variate-wise attention
        self.enc_variate = nn.TransformerEncoder(enc_layers_feature, num_layers=e_layers)

        # stock-wise attention
        self.stock_attention = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)

        # 修改projection层为两步预测
        self.ret_pred = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Squeeze(-1),
            nn.Linear(d_feat, 1),
            nn.Squeeze(-1)
        )

        self.vol_pred = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Squeeze(-1),
            nn.Linear(d_feat, 1),
            nn.Squeeze(-1)
        )


    def forward(self, x):
        # x: [B, T, D]  batch_size, seq_len, feat_dim
        src = x[:, :, :self.gate_input_start_index]
        
        # 1. Global Embedding
        global_embedding = self.project_embedding(src, None)  # [B, D, d_model]
        
        # 2. Local Pattern Embedding
        x_enc = src.permute(0, 2, 1)  # [B, D, T]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [B*D, patch_num, d_model]
        
        # 3. Temporal Attention
        enc_out = self.enc_temporal(enc_out)  # [B*D, patch_num, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, D, d_model, patch_num]
        temporal_dependency_embedding = self.head(enc_out)  # [B, D, d_model]
        
        # 4. First Gate Fusion
        gate = self.gate_sigmoid(self.gate_w1(global_embedding) + self.gate_w2(temporal_dependency_embedding))
        enc_out = gate * global_embedding + (1 - gate) * temporal_dependency_embedding  # [B, D, d_model]
        
        # 5. Cross Attention
        enc_out_cross = self.enc_variate(enc_out)  # [B, D, d_model]
        
        # 6. Second Gate Fusion
        gate = self.gate_sigmoid(self.gate_w3(enc_out) + self.gate_w4(enc_out_cross))
        enc_out = gate * enc_out + (1 - gate) * enc_out_cross  # [B, D, d_model]

        # 7. Stock-wise Attention
        enc_out = self.stock_attention(enc_out) # [B, D, d_model]
        
        # 8. Final Projection
        pred = self.feature_projection(enc_out).squeeze(-1)  # [B, D]
        
        output = self.final_projection(pred).squeeze(-1) # [B]
        
        return output
    
class GMASTER_RV1(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta, seq_len=21,
                       patch_len=4, stride=2, e_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding = stride
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'

        # 1. Patch Embedding:
        # Input: [B, D, L] -> Output: [B*D, patch_num, d_model]
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, self.padding, T_dropout_rate)
        # 2. Project Embedding: 
        # Input: [B, L, D] -> Output: [B, D, d_model]
        self.project_embedding = DataEmbedding_inverted(seq_len, d_model, T_dropout_rate)
        # 3. Head layer
        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        # Input: [B, D, d_model, patch_num] -> Output: [B, D, d_model]
        self.head = FlattenHead(d_feat, self.head_nf, d_model, head_dropout=T_dropout_rate)

        # 4. Gate层
        self.gate_w1 = nn.Linear(d_model, d_model)
        self.gate_w2 = nn.Linear(d_model, d_model)
        self.gate_w3 = nn.Linear(d_model, d_model)
        self.gate_w4 = nn.Linear(d_model, d_model)
        self.gate_sigmoid = nn.Sigmoid()

        # 5. Temporal & Feature Attention
        enc_layers_temporal = nn.TransformerEncoderLayer(
                                    d_model=d_model,
                                    nhead=t_nhead,
                                    dim_feedforward=d_model,
                                    dropout=T_dropout_rate,
                                    layer_norm_eps=1e-5,
                                    batch_first=True, 
                                    norm_first=True 
                                )

        # temporal-wise attention
        enc_layers_feature = nn.TransformerEncoderLayer(
                                    d_model=d_model,
                                    nhead=t_nhead,
                                    dim_feedforward=d_model,
                                    dropout=T_dropout_rate,
                                    layer_norm_eps=1e-5,
                                    batch_first=True,
                                    norm_first=True
                                )
        
        # temporal-wise attention
        self.enc_temporal = nn.TransformerEncoder(enc_layers_temporal, num_layers=e_layers)

        # variate-wise attention
        self.enc_variate = nn.TransformerEncoder(enc_layers_feature, num_layers=e_layers)

        # stock-wise attention
        self.stock_attention = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)

        # 修改projection层为两步预测
        self.feature_projection = nn.Linear(d_model, 1)  # 将每个特征的d_model维表示映射为标量
        self.final_projection = nn.Linear(d_feat, 1)     # 将所有特征的预测整合为最终预测


    def forward(self, x):
        # x: [B, T, D]  batch_size, seq_len, feat_dim
        src = x[:, :, :self.gate_input_start_index]
        
        # 1. Global Embedding
        global_embedding = self.project_embedding(src, None)  # [B, D, d_model]
        
        # 2. Local Pattern Embedding
        x_enc = src.permute(0, 2, 1)  # [B, D, T]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [B*D, patch_num, d_model]
        
        # 3. Temporal Attention
        enc_out = self.enc_temporal(enc_out)  # [B*D, patch_num, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, D, d_model, patch_num]
        temporal_dependency_embedding = self.head(enc_out)  # [B, D, d_model]
        
        # 4. First Gate Fusion
        gate = self.gate_sigmoid(self.gate_w1(global_embedding) + self.gate_w2(temporal_dependency_embedding))
        enc_out = gate * global_embedding + (1 - gate) * temporal_dependency_embedding  # [B, D, d_model]
        
        # 5. Cross Attention
        enc_out_cross = self.enc_variate(enc_out)  # [B, D, d_model]
        
        # 6. Second Gate Fusion
        gate = self.gate_sigmoid(self.gate_w3(enc_out) + self.gate_w4(enc_out_cross))
        enc_out = gate * enc_out + (1 - gate) * enc_out_cross  # [B, D, d_model]

        # 7. Stock-wise Attention
        enc_out = self.stock_attention(enc_out) # [B, D, d_model]
        
        # 8. Final Projection
        ret_pred = self.ret_pred(enc_out) # [B]
        vol_pred = self.vol_pred(enc_out) # [B]
        
        return ret_pred, vol_pred

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class UnifiedModel(SequenceModel):
    def __init__(
            self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, gate_input_end_index,
            T_dropout_rate, S_dropout_rate, beta, model_name, n_experts, return_gate_weight, k_s, seed=0, **kwargs,
    ):
        super(UnifiedModel, self).__init__(seed=seed, **kwargs)

        seed_everything(seed)

        self.d_model = d_model
        self.d_feat = d_feat

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta
        self.model_name = model_name
        self.n_experts = n_experts
        self.k_s = k_s
        self.return_gate_weight = return_gate_weight

        self.init_model(model_name)

    def init_model(self, model_name='master'):
        if model_name == 'master':            # 单任务学习，只预测return
            self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
            
        elif model_name == 'transformer':     # 单任务学习，只预测return
            self.model = Transformer(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
            
        elif model_name == 'transformer_rv1':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['rv1', 'rv2', 'r_vola1', 'r_vola2', 'r_New1_vola1', 'r_New1_vola2', 'r_New2_vola1', 'r_New2_vola2', 'r_New3_vola1', 'r_New3_vola2', 'r_vol2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = Transformer_RV1(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
            
        elif model_name == 'transformer_rv12': # 多任务学习，预测return+volume1+volume2
            assert self.mode in ['rv12', 'r_New1_vola12', 'r_New2_vola12', 'r_New3_vola12'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = Transformer_RV12(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'transformer_rc': # 多任务学习，预测return+classification
            assert self.mode=='rc', f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = Transformer_RC(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'transformer_rvc': # 多任务学习，预测return+classification
            assert self.mode in ['rv1c', 'rv2c'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = Transformer_RVC(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'transformer_rv12c': # 多任务学习，预测return+classification
            assert self.mode=='rv12c', f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = Transformer_RV12C(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
            
        elif model_name == 'master_rv1':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['rv1', 'rv2', 'r_vola1', 'r_vola2', 'r_New1_vola1', 'r_New1_vola2', 'r_New2_vola1', 'r_New2_vola2', 'r_New3_vola1', 'r_New3_vola2', 'r_vol2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = MASTER_RV1(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
            
        elif model_name == 'master_rv12': # 多任务学习，预测return+volume1+volume2
            assert self.mode in ['rv12', 'r_New1_vola12', 'r_New2_vola12', 'r_New3_vola12'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = MASTER_RV12(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
            
        elif model_name == 'master_rc': # 多任务学习，预测return+classification
            assert self.mode=='rc', f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = MASTER_RC(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'master_rvc': # 多任务学习，预测return+volume1+volume2+classification
            assert self.mode in ['rv1c', 'rv2c'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = MASTER_RVC(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'master_rv12c': # 多任务学习，预测return+volume1+volume2+classification
            assert self.mode=='rv12c', f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = MASTER_RV12C(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'master_hrc':
            assert self.mode in ['r_v1_co', 'r_v2_co'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = MASTER_Hierarchical(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
            
        elif model_name == 'master_hrc_v2':
            assert self.mode in ['r_v1_co', 'r_v2_co'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = MASTER_Hierarchical_v2(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)

        elif model_name == 'limt':     # 单任务学习，只预测return 
            self.model = LiMT(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'limt_woMoE':     # 单任务学习，只预测return 
            self.model = LiMT_RVol2Vola2_woMoE(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'limt_mmoe_rv1':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['rv1', 'rv2', 'r_vola1', 'r_vola2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_MMoE_RV1(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts)
        
        elif model_name == 'limt_mmoe_rVol2Vola2':  # 多任务学习，预测return+volume1+volatility2
            assert self.mode in ['r_vol2_vola2', 'r_vol2_vola2_t1', 'r_vol2_vola2_t2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_MMoE_RVol2Vola2(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts, return_gate_weight=self.return_gate_weight)
        
        elif model_name == 'limt_mmoe_rVol2Vola2_rc':  # 多任务学习，预测return+volume1+volatility2，残差连接的MoE
            assert self.mode in ['r_vol2_vola2', 'r_vol2_vola2_t1', 'r_vol2_vola2_t2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_MMoE_RVol2Vola2_RC(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts, return_gate_weight=self.return_gate_weight)
        
        
        elif model_name == 'limt_mmoe_rVol2Vola2_woSA':  # 多任务学习，预测return+volume1+volatility2
            assert self.mode in ['r_vol2_vola2', 'r_vol2_vola2_t1', 'r_vol2_vola2_t2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_MMoE_RVol2Vola2_woSA(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts, return_gate_weight=self.return_gate_weight)
        
        elif model_name in ['limt_mmoe_RV']:  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['r_vol2_woVola2', 'r_woVol2_vola2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_MMoE_RV(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts, return_gate_weight=self.return_gate_weight)
        
        elif model_name in ['limt_mmoe_R']:  # 单学习
            assert self.mode in ['r_woVol2_woVola2', 'ret'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_MMoE_R(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts, return_gate_weight=self.return_gate_weight)

        elif model_name == 'limt_dsmoe_rVol2Vola2':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['r_vol2_vola2', 'r_vol2_vola2_t1', 'r_vol2_vola2_t2', 'r_vol2_vola2_DataEnhance', 'r_vol2_vola2_t1_DataEnhance'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_DSMOE(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts)
        
        elif model_name == 'limt_dsmoe2_rVol2Vola2':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['r_vol2_vola2', 'r_vol2_vola2_t1', 'r_vol2_vola2_t2', 'r_vol2_vola2_DataEnhance', 'r_vol2_vola2_t1_DataEnhance'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_DSMOE2(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts, k_s=self.k_s)

        elif model_name == 'limt_mmoe1_rv1':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['rv1', 'rv2', 'r_vola1', 'r_vola2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_MMoE1_RV1(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts)
        
        elif model_name == 'limt_mmoe2_rv1':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['rv1', 'rv2', 'r_vola1', 'r_vola2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_MMoE2_RV1(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts)
        
        elif model_name == 'limt_mmoe_rv12':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['rv12'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_MMoE_RV12(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts)
        
        elif model_name == 'limt_mmoe1_rv12':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['rv12'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_MMoE1_RV12(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts)

        elif model_name == 'limt_mmoe2_rv12':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['rv12'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_MMoE2_RV12(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts)
        
        elif model_name == 'limt_mmoe3_rv12':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['rv12'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_MMoE3_RV12(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta, n_experts=self.n_experts)
        
        elif model_name == 'limt_rv1':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['rv1', 'rv2', 'r_vola1', 'r_vola2', 'r_New1_vola1', 'r_New1_vola2', 'r_New2_vola1', 'r_New2_vola2', 'r_New3_vola1', 'r_New3_vola2', 'r_vol2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_RV1(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'limt_rVol2Vola2':  # 多任务学习，预测return+volume1 或 return+volume2
            assert self.mode in ['r_vol2_vola2', 'r_vol2_vola2_t1', 'r_vol2_vola2_t2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_RVol2Vola2(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'limt_rv12': # 多任务学习，预测return+volume1+volume2
            assert self.mode in ['rv12', 'r_New1_vola12', 'r_New2_vola12', 'r_New3_vola12'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_RV12(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)

        elif model_name == 'limt_rc': # 多任务学习，预测return+classification
            assert self.mode=='rc', f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_RC(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
            
        elif model_name == 'limt_rvc': # 多任务学习，预测return+volume++classification
            assert self.mode in ['rv1c', 'rv2c'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_RVC(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'limt_rv12c': # 多任务学习，预测return+volume1+volume2+classification
            assert self.mode=='rv12c', f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = LiMT_RV12C(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)

        elif model_name == 'GateFormer':     # 单任务学习，只预测return
            self.model = GateFormer(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'GateFormer_RV1':     # 单任务学习，只预测return
            self.model = GateFormer_RV1(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'GMASTER':        # 单任务学习，预测return + volume1
            assert self.mode in ['rv1', 'rv2', 'r_vola1', 'r_vola2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = GMASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        
        elif model_name == 'GMASTER_RV1':        # 单任务学习，只预测return + volume1
            assert self.mode in ['rv1', 'rv2', 'r_vola1', 'r_vola2'], f"mode and model_name not compatible! mode:{self.mode}, model_name:{model_name}"
            self.model = GMASTER_RV1(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)

        else:
            raise Exception(f"ERROR: Model_name: {model_name} has not been implemented!")  
        super(UnifiedModel, self).init_model()
        


class MASTERModel(SequenceModel):
    def __init__(
            self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, gate_input_end_index,
            T_dropout_rate, S_dropout_rate, beta, **kwargs,
    ):
        super(MASTERModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_feat = d_feat

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta

        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        super(MASTERModel, self).init_model()
