import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch import nn

import os
import logging

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x):
    return (x - x.mean()).div(x.std())

def cosine_dissimilarity_loss(gates_dict, tasks):
    """
    最大化余弦距离 = 最小化余弦相似度
    返回：所有任务对的平均余弦相似度（最小化该损失使任务间差异变大）
    """
    num_tasks = len(tasks)
    total_similarity = 0.0
    pair_count = 0
    
    for i in range(num_tasks):
        for j in range(i+1, num_tasks):
            gate_i = gates_dict[tasks[i]]  # [batch_size, n_experts]
            gate_j = gates_dict[tasks[j]]
            
            # 计算余弦相似度 (范围[-1,1])
            sim = F.cosine_similarity(gate_i, gate_j, dim=1)  # [batch_size]
            
            # 我们只关心正相似度（对齐方向）
            # 负值表示方向相反，是好的，不予惩罚
            positive_sim = torch.clamp(sim, min=0)  # 只考虑正相似部分
            
            total_similarity += torch.mean(positive_sim)
            pair_count += 1
    
    return total_similarity / pair_count if pair_count > 0 else torch.tensor(0.0)

def js_divergence_loss(gates_dict, tasks):
    """
    通过最大化分布差异来使门控向量不同
    返回：所有任务对间JS散度的平均值（最小化该损失使分布差异变大）
    """
    total_js = 0.0
    pair_count = 0
    
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            P = gates_dict[tasks[i]]
            Q = gates_dict[tasks[j]]
            
            # 计算M = (P + Q)/2
            M = 0.5 * (P + Q)
            
            # 计算KL散度
            kl_pm = F.kl_div(torch.log(M+1e-8), P, reduction='batchmean')
            kl_qm = F.kl_div(torch.log(M+1e-8), Q, reduction='batchmean')
            
            # JS散度 = 0.5*(KL(P||M) + KL(Q||M))
            js_div = 0.5 * (kl_pm + kl_qm)
            
            total_js += js_div
            pair_count += 1
    
    return -total_js / pair_count if pair_count > 0 else torch.tensor(0.0)

def expert_specialization_loss(gates_dict, tasks):
    """
    鼓励每个任务专注于不同的专家组合
    """
    # 第一步：确定每个任务的首选专家
    task_expert_counts = []
    for task in tasks:
        gate = gates_dict[task]  # [batch_size, n_experts]
        
        # 确定每行（样本）中最大的门控值
        _, expert_idx = torch.max(gate, dim=1)  # [batch_size]
        
        # 计算每个专家被选为首选的次数
        one_hot = F.one_hot(expert_idx, num_classes=gate.size(1))
        counts = torch.mean(one_hot.float(), dim=0)  # [n_experts]
        task_expert_counts.append(counts)
    
    # 第二步：计算任务间的专家重叠度
    loss = 0.0
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            # 点积计算重叠度（完全不相交时为0）
            overlap = torch.dot(task_expert_counts[i], task_expert_counts[j])
            loss += overlap
    
    # 除以任务对数量归一化
    return loss / (len(tasks)*(len(tasks)-1)/2)

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)  
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def drop_na(x):
    N = x.shape[0]
    mask = ~x.isnan()
    return mask, x[mask]

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False, seed=0):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            np.random.seed(self.seed) # 设置random_seed确保每次迭代时和enhance数据的打乱顺序是一致的
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)

# 有序分类损失
class OrdinalCrossEntropy(nn.Module):
    def __init__(self, num_bins=10, temperature=0.1):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature
        
    def forward(self, pred, target):
        """ 有序交叉熵损失
        Args:
            pred: [batch_size, num_bins] 分类logits
            target: [batch_size] 类别标签(0到num_bins-1)
        """
        # 确保输入张量是 torch.float32 类型
        pred = pred.float()   # (N, 10)
        target = target.long()  # (N, 1) # 目标标签需要是整数类型 
        # 基础交叉熵
        ce_loss = F.cross_entropy(pred, target)
        
        # 有序性惩罚项
        if self.temperature > 0:
            probs = F.softmax(pred, dim=1) # (N, 10)
            rank_diff = torch.abs(
                torch.arange(self.num_bins, device=pred.device).unsqueeze(0) - # （1， 10） (0, 1,2,3,4,5,6,7,8,9)
                target.float().unsqueeze(1)                                    # （N， 1）  
            ) # (N, 10)
            smooth_loss = torch.mean(probs * rank_diff)
            return ce_loss + self.temperature * smooth_loss
        return ce_loss

class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= '', log_save_path= '',
                 mode='ret', drop_extreme=False, loss_type='mse', logger=None, eps=1e-6, weight1=0.8, weight2=0.1, weight3=0.1, stop_standard="IC"):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix
        self.log_save_path = log_save_path
        self.mode = mode
        self.drop_extreme = drop_extreme
        self.loss_type = loss_type
        self.logger = logger if logger else logging.getLogger(__name__)
        self.eps = torch.tensor(eps, device=self.device)
        self.stop_standard = stop_standard
        # 初始化时设置可学习权重，给return较高的初始权重以指引模型
        # 注意：我们只需要两个权重参数，因为任何情况下权重和都是1
        # self.return_weight = nn.Parameter(torch.tensor(0.7))  # return的初始权重设为0.7
        self.return_weight = torch.tensor(weight1, device=self.device)  # return的初始权重设为0.8
        self.vh_weight = torch.tensor(weight2, device=self.device)  # vol_hit的初始权重设为0.1
        self.vola_weight = torch.tensor(weight3, device=self.device)  # vol_hit的初始权重设为0.1

        # Ordinary Classification loss fuction
        self.ordinal_loss = OrdinalCrossEntropy(num_bins=10, temperature=0.1)  # 默认参数可调整
        
        # vol2和volatility动态加权给样本
        self.ddw_loss1 = LearnableConfidenceWeightedLoss().to(self.device)

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)
    
    def loss_fn(self, pred_ret, label_ret, label_ret_ori=None,
                      pred_vol1=None, label_vol1=None, label_vol1_ori=None,
                      pred_vol2=None, label_vol2=None, label_vol2_ori=None,
                      pred_class=None, label_class=None, label_class_ori=None,
                      pred_vola=None, label_vola=None, label_vola_ori=None,
                      loss_type='mse_rv1', gate_weight=None): # 使用return与volume的label，预留3个label计算空间
        # if isinstance(self.return_weight, nn.Parameter):
        #     # 如果return_weight是可学习参数，需要确保其在(0, 1)值域范围内
        #     return_weight = torch.sigmoid(self.return_weight)
        # else:
        return_weight = self.return_weight
        vh_weight = self.vh_weight
        vola_weight = self.vola_weight

        if loss_type == 'mse':
            mask = ~torch.isnan(label_ret)
            loss = (pred_ret[mask]-label_ret[mask])**2
            return torch.mean(loss)
    
        elif loss_type == 'mse_rv1':   # 使用return + volume1
            mask = ((~torch.isnan(label_ret)) & (~torch.isnan(label_vol1)))
            ret_loss = ((pred_ret[mask]-label_ret[mask])**2).mean()
            vol1_loss = ((pred_vol1[mask]-label_vol1[mask])**2).mean()
            loss = return_weight * ret_loss + (1-return_weight) * vol1_loss
            return loss
        
        elif loss_type == 'mse_rv2': # 使用return + volume2
            mask = ((~torch.isnan(label_ret)) & (~torch.isnan(label_vol2)))
            ret_loss = ((pred_ret[mask]-label_ret[mask])**2).mean()
            vol2_loss = ((pred_vol2[mask]-label_vol2[mask])**2).mean()
            loss = return_weight * ret_loss + (1-return_weight) * vol2_loss
            return loss
            
        elif loss_type == 'mse_rv12': # 使用return + volume1 + volume2
            mask = ((~torch.isnan(label_ret)) & (~torch.isnan(label_vol1)) & (~torch.isnan(label_vol2)))
            ret_loss = ((pred_ret[mask]-label_ret[mask])**2).mean()
            vol1_loss = ((pred_vol1[mask]-label_vol1[mask])**2).mean()
            vol2_loss = ((pred_vol2[mask]-label_vol2[mask])**2).mean()
            # 对于三个任务，让return保持return_weight的权重
            # 将剩余权重(1-return_weight)平均分配给volume1和volume2
            vol_weight = (1 - return_weight) / 2
            loss = return_weight * ret_loss + vol_weight * vol1_loss + vol_weight * vol2_loss
            return loss
        
        elif loss_type == 'mse_r_vola1':   # 使用return + volatility, 二者都是使用zscore之后的mse
            mask = ((~torch.isnan(label_ret) & (~torch.isnan(label_vola))))
            ret_loss = ((pred_ret[mask]-label_ret[mask])**2).mean()
            vola_loss = ((pred_vola[mask]-label_vola[mask])**2).mean()
            loss = return_weight * ret_loss + (1-return_weight) * vola_loss
            return loss
        
        elif loss_type == 'mse_r_vola2':   # 使用return + volatility, 二者都是使用zscore之后的mse; exp(volatility)还作为return的置信度
            mask = ((~torch.isnan(label_ret) & (~torch.isnan(label_vola))))
            ret_loss = ((pred_ret[mask]-label_ret[mask])**2 / (torch.exp(pred_vola[mask]) + self.eps)).mean()
            vola_loss = ((pred_vola[mask]-label_vola[mask])**2).mean()
            loss = return_weight * ret_loss + (1-return_weight) * vola_loss
            return loss

        elif loss_type == 'mse_r_vol2_vola2':   # 使用return + volume2 + volatility2
            mask = ((~torch.isnan(label_ret)) & (~torch.isnan(label_vol2)) & (~torch.isnan(label_vola)))
            ret_loss = ((pred_ret[mask]-label_ret[mask])**2).mean()
            vol2_loss = ((pred_vol2[mask]-label_vol2[mask])**2).mean()
            vola_loss = ((pred_vola[mask]-label_vola[mask])**2).mean()
            loss = return_weight * ret_loss + vh_weight * vol2_loss + vola_weight * vola_loss
            return loss

    def train_epoch(self, data_loader, data_loader_enhance, mode='ret', drop_extreme=False, loss_type='mse'):
        self.model.train()
        losses = []
        preds = []
        ic = []
        ric = []

        for batch_idx, data in enumerate(data_loader):
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            if mode in ['ret', 'ret_DataEnhance']:            # label只有1个，就是收益率
                feature = data[:, :, 0:-1].to(self.device)
                if "DataEnhance" in mode and data_loader_enhance is not None:
                    # 2. 增强数据分支（仅在 data_loader_enhance 存在时执行）
                    data_enhance = data_loader_enhance[batch_idx]  # 按索引获取增强数据
                    data_enhance = torch.squeeze(data_enhance, dim=0)  # (N', T, F)
                    feature_enhance = data_enhance[:, :, 0:-1].to(self.device)
                    # 拼接增强数据
                    origin_stocks_len = feature.shape[0] # 获取目标股票的数量，后续方便索引
                    feature = torch.cat((feature, feature_enhance), dim=0) # 将增强数据与原始数据拼接
                label = data[:, -1, -1].to(self.device)
                # Additional process on labels
                # If you use original data to train, you won't need the following lines because we already drop extreme when we dumped the data.
                # If you use the opensource data to train, use the following lines to drop extreme labels.
                #########################
                if drop_extreme:
                    mask, label = drop_extreme(label)
                    feature = feature[mask, :, :]
                label = zscore(label) # CSZscoreNorm
                #########################
                pred = self.model(feature.float())
                if "DataEnhance" in mode and data_loader_enhance is not None:
                    # 恢复到原始数据的形状
                    pred = pred[:origin_stocks_len] 
                loss = self.loss_fn(pred, label, loss_type=loss_type) # return mse
            elif mode in ['rv1', 'r_New1_vola1', 'r_New2_vola1', 'r_New3_vola1']:     # label有3个，ret+vol1+vol2，但是我们用ret+vol1
                feature = data[:, :, 0:-3].to(self.device)
                label = data[:, -1, -3].to(self.device)
                label_vol1 = data[:, -1, -2].to(self.device)
                if drop_extreme:
                    mask, label = drop_extreme(label)
                    feature = feature[mask, :, :]
                label = zscore(label) # CSZscoreNorm
                label_vol1 = zscore(label_vol1) # CSZscoreNorm
                pred, pred_vol1 = self.model(feature.float())
                assert loss_type=='mse_rv1', f"loss_type and mode not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, pred_vol1=pred_vol1, label_vol1=label_vol1, loss_type=loss_type)
            elif mode in ['rv2', 'r_New1_vola2', 'r_New2_vola2', 'r_New3_vola2']:     # label有3个，ret+vol1+vol2，但是我们用ret+vol1
                feature = data[:, :, 0:-3].to(self.device)
                label = data[:, -1, -3].to(self.device)
                label_vol2 = data[:, -1, -1].to(self.device)
                if drop_extreme:
                    mask, label = drop_extreme(label)
                    feature = feature[mask, :, :]
                label = zscore(label) # CSZscoreNorm
                label_vol2 = zscore(label_vol2) # CSZscoreNorm
                pred, pred_vol2 = self.model(feature.float())
                assert loss_type=='mse_rv2', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, pred_vol2=pred_vol2, label_vol2=label_vol2, loss_type=loss_type)
            elif mode in ['rv12', 'r_New1_vola12', 'r_New2_vola12', 'r_New3_vola12']:    # label有3个，ret+vol1+vol2
                feature = data[:, :, 0:-3].to(self.device)
                label = data[:, -1, -3].to(self.device)
                label_vol1 = data[:, -1, -2].to(self.device)
                label_vol2 = data[:, -1, -1].to(self.device)
                if drop_extreme:
                    mask, label = drop_extreme(label)
                    feature = feature[mask, :, :]
                label = zscore(label) # CSZscoreNorm
                label_vol1 = zscore(label_vol1) # CSZscoreNorm
                label_vol2 = zscore(label_vol2) # CSZscoreNorm
                pred, pred_vol1, pred_vol2 = self.model(feature.float())
                assert loss_type=='mse_rv12', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, pred_vol1=pred_vol1, label_vol1=label_vol1, pred_vol2=pred_vol2, label_vol2=label_vol2, loss_type=loss_type)
            elif mode=='rc': # label有4个，ret+vol1+vol2+cross_classification, 我们使用ret+cross_classification
                feature = data[:, :, 0:-4].to(self.device)
                label = data[:, -1, -4].to(self.device)
                label_class = data[:, -1, -1].to(self.device)
                if drop_extreme:
                    mask, label = drop_extreme(label)
                    feature = feature[mask, :, :]
                label = zscore(label) # CSZscoreNorm
                pred, pred_class = self.model(feature.float())
                assert loss_type=='mse_rc', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, pred_class=pred_class, label_class=label_class, loss_type=loss_type)
            
            elif mode in ['r_v1_co', 'rv1c']: # label有4个，ret+vol1+vol2+cross_classification, 我们使用ret+vol1+cross_classification
                feature = data[:, :, 0:-4].to(self.device)
                label = data[:, -1, -4].to(self.device)
                label_vol1 = data[:, -1, -3].to(self.device)
                label_class = data[:, -1, -1].to(self.device)
                if drop_extreme:
                    mask, label = drop_extreme(label)
                    feature = feature[mask, :, :]
                label = zscore(label) # CSZscoreNorm
                pred, pred_vol1, pred_class = self.model(feature.float())
                assert loss_type in ['mse_r_v1_co', 'mse_rv1c'], f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, pred_vol1=pred_vol1, label_vol1=label_vol1, pred_class=pred_class, label_class=label_class, loss_type=loss_type)

            elif mode in ['r_v2_co', 'rv2c']: # label有4个，ret+vol1+vol2+cross_classification, 我们使用ret+vol2+cross_classification
                feature = data[:, :, 0:-4].to(self.device)
                label = data[:, -1, -4].to(self.device)
                label_vol2 = data[:, -1, -2].to(self.device)
                label_class = data[:, -1, -1].to(self.device)
                if drop_extreme:
                    mask, label = drop_extreme(label)
                    feature = feature[mask, :, :]
                label = zscore(label) # CSZscoreNorm
                pred, pred_vol2, pred_class = self.model(feature.float())
                assert loss_type in ['mse_r_v2_co', 'mse_rv2c'], f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, pred_vol2=pred_vol2, label_vol2=label_vol2, pred_class=pred_class, label_class=label_class, loss_type=loss_type)
            
            elif mode in ['r_v12_co', 'rv12c']: # label有4个，ret+vol1+vol2+cross_classification, 我们使用ret+vol1+vol2+cross_classification
                feature = data[:, :, 0:-4].to(self.device)
                label = data[:, -1, -4].to(self.device)
                label_vol1 = data[:, -1, -3].to(self.device)
                label_vol2 = data[:, -1, -2].to(self.device)
                label_class = data[:, -1, -1].to(self.device)
                if drop_extreme:
                    mask, label = drop_extreme(label)
                    feature = feature[mask, :, :]
                label = zscore(label) # CSZscoreNorm
                pred, pred_vol1, pred_vol2, pred_class = self.model(feature.float())
                assert loss_type in ['mse_r_v12_co', 'mse_rv12c'], f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                    pred_vol1=pred_vol1, label_vol1=label_vol1, 
                                    pred_vol2=pred_vol2, label_vol2=label_vol2,
                                    pred_class=pred_class, label_class=label_class, loss_type=loss_type)

            elif mode in ['r_vola1', 'r_vola2']: # label有5个，ret+vol1+vol2+label_class+volatility, 我们使用ret+volatility
                feature = data[:, :, 0:-5].to(self.device)
                label = data[:, -1, -5].to(self.device)
                label_vola = data[:, -1, -1].to(self.device)
                if drop_extreme:
                    mask, label = drop_extreme(label)
                    feature = feature[mask, :, :]
                label = zscore(label) # CSZscoreNorm
                label_vola = zscore(label_vola) # CSZscoreNorm 确保在损失函数里二者的量级比较接近
                pred, pred_vola = self.model(feature.float())
                assert loss_type in ['mse_r_vola1', 'mse_r_vola2'], f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                    pred_vola=pred_vola, label_vola=label_vola,
                                    loss_type=loss_type)

            elif mode in ['r_vol2_vola2', 'r_vol2_vola2_t1', 'r_vol2_vola2_t2', 'r_vol2_vola2_t1_DataEnhance', 'r_vol2_vola2_DataEnhance']:    # label有3个，ret+vol1+vol2
                feature = data[:, :, 0:-3].to(self.device)
                if "DataEnhance" in mode and data_loader_enhance is not None:
                    data_enhance = data_loader_enhance[batch_idx]  # 按索引获取增强数据
                    data_enhance = torch.squeeze(data_enhance, dim=0)  # (N', T, F)
                    origin_stocks_len = feature.shape[0] # 获取目标股票的数量，后续方便索引
                    feature_enhance = data_enhance[:, :, 0:-3].to(self.device)
                    feature = torch.cat((feature, feature_enhance), dim=0) # 将增强数据与原始数据拼接
                
                label = data[:, -1, -3].to(self.device)
                label_vol2 = data[:, -1, -2].to(self.device)
                label_vola = data[:, -1, -1].to(self.device)
                if drop_extreme:
                    mask, label = drop_extreme(label)
                    feature = feature[mask, :, :]
                
                label = zscore(label) # CSZscoreNorm
                if mode in ['r_vol2_vola2', 'r_vol2_vola2_DataEnhance']: 
                    label_vol2 = zscore(label_vol2) # CSZscoreNorm
                    label_vola = zscore(label_vola) # CSZscoreNorm
                
                assert loss_type in ['mse_r_vol2_vola2', 'mse_r_vol2_vola2_t1', 'mse_r_vol2_vola2_t2', 'mse_r_vol2_vola2_ddw1', \
                                     'mse_r_vol2_vola2_t1_DW11', 'mse_r_vol2_vola2_t1_DW12', \
                                     'mse_r_vol2_vola2_t1_DW21', 'mse_r_vol2_vola2_t1_DW22', \
                                     'mse_r_vol2_vola2_t1_DW31', 'mse_r_vol2_vola2_t1_DW32', \
                                     'mse_r_vol2_vola2_t1_GW1', 'mse_r_vol2_vola2_t1_GW2', 'mse_r_vol2_vola2_t1_GW3'], f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                
                if "GW" in loss_type:
                    pred, pred_vol2, pred_vola, gate_dict = self.model(feature.float()) 
                    if "DataEnhance" in mode and data_loader_enhance is not None: 
                        # 恢复到原始数据的形状
                        pred = pred[:origin_stocks_len] 
                        pred_vol2 = pred_vol2[:origin_stocks_len]
                        pred_vola = pred_vola[:origin_stocks_len]
                    loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                    pred_vol2=pred_vol2, label_vol2=label_vol2, 
                                    pred_vola=pred_vola, label_vola=label_vola, loss_type=loss_type, gate_weight=gate_dict)  
                else:
                    pred, pred_vol2, pred_vola = self.model(feature.float())
                    if "DataEnhance" in mode and data_loader_enhance is not None:
                        # 恢复到原始数据的形状
                        pred = pred[:origin_stocks_len] 
                        pred_vol2 = pred_vol2[:origin_stocks_len]
                        pred_vola = pred_vola[:origin_stocks_len]
                    loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                    pred_vol2=pred_vol2, label_vol2=label_vol2, 
                                    pred_vola=pred_vola, label_vola=label_vola, loss_type=loss_type)
            
            
            elif mode in ['r_vol2_woVola2', 'r_woVol2_vola2', 'r_woVol2_woVola2']:    # label有3个，ret+vol2+vola2, 专注于消融实验
                feature = data[:, :, 0:-3].to(self.device)
                label = data[:, -1, -3].to(self.device)
                label_vol2 = data[:, -1, -2].to(self.device)
                label_vola = data[:, -1, -1].to(self.device)

                label = zscore(label) # CSZscoreNorm
                label_vol2 = zscore(label_vol2) # CSZscoreNorm
                label_vola = zscore(label_vola) # CSZscoreNorm
                if mode == 'r_vol2_woVola2':
                    pred, pred_vol2 = self.model(feature.float())
                    assert loss_type=='mse_r_vol2', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                    loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                        pred_vol2=pred_vol2, label_vol2=label_vol2, 
                                        loss_type=loss_type)
                elif mode == 'r_woVol2_vola2':
                    pred, pred_vola = self.model(feature.float())
                    assert loss_type=='mse_r_vola1', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                    loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                        pred_vola=pred_vola, label_vola=label_vola, 
                                        loss_type=loss_type)
                elif mode == 'r_woVol2_woVola2':
                    pred = self.model(feature.float())
                    assert loss_type=='mse', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                    loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                        loss_type=loss_type)
            
            elif mode in ['r_vol2']:    # label有3个，ret+vol1+vola2
                feature = data[:, :, 0:-3].to(self.device)
                label = data[:, -1, -3].to(self.device)
                label_vol2 = data[:, -1, -2].to(self.device)
                # label_vola = data[:, -1, -1].to(self.device)
                if drop_extreme:
                    mask, label = drop_extreme(label)
                    feature = feature[mask, :, :]
                label = zscore(label) # CSZscoreNorm
                label_vol2 = zscore(label_vol2) # CSZscoreNorm
                # label_vola = zscore(label_vola) # CSZscoreNorm
                pred, pred_vol2 = self.model(feature.float())
                assert loss_type=='mse_r_vol2', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                    pred_vol2=pred_vol2, label_vol2=label_vol2, 
                                    loss_type=loss_type)

            else: 
                raise Exception(f"ERROR: Mode: {mode} has not been implemented!")


            losses.append(loss.item())
            preds.append(pred.ravel())

            # print(pred.shape, label.shape)
            daily_ic, daily_ric = calc_ic(pred.detach().cpu().numpy(), label.detach().cpu().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
          
        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)
        }

        return float(np.mean(losses)), metrics

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # You cannot drop extreme labels for test. 
            label = zscore(label)
                        
            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

        # # 打印多任务学习的权重
        # if isinstance(self.return_weight, nn.Parameter):
        #     self.logger.info("return_weight: %.4f, vola_weight: %.4f" % (torch.sigmoid(self.return_weight), (1-torch.sigmoid(self.return_weight))))
        # else:
        self.logger.info("return_weight: %.4f, vh_weight: %.4f, vola_weight: %.4f" % (self.return_weight, self.vh_weight, self.vola_weight))
        
        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True, seed=0):
        sampler = DailyBatchSamplerRandom(data, shuffle, seed=seed)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None, dl_test=None, dl_train_enhance=None, dl_valid_enhance=None, dl_test_enhance=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True, seed=self.seed)
        if dl_train_enhance is not None: # 是用来在StockAttention那一步增强数据的
            assert (dl_train.start == dl_train_enhance.start) & (dl_train.end == dl_train_enhance.end), "dl_train and dl_train_enhance should have the same start and end date."
            train_loader_enhance = self._init_data_loader(dl_train_enhance, shuffle=True, drop_last=True, seed=self.seed)
        else:
            train_loader_enhance = None
        best_param = None
        valid_ic_best = -np.inf
        patience = 0
        for step in range(self.n_epochs):
            train_loss, metric_train = self.train_epoch(train_loader, train_loader_enhance, mode=self.mode, drop_extreme=self.drop_extreme, loss_type=self.loss_type)
            self.fitted = step
            if dl_valid:
                valid_loss, metric_valid = self.predict(dl_valid, dl_valid_enhance, mode=self.mode, loss_type=self.loss_type)
                if dl_test:
                    test_loss, metric_test = self.predict(dl_test, dl_test_enhance, mode=self.mode, loss_type=self.loss_type)
                    self.logger.info("Epoch %d, train_loss %.6f, valid_loss: %.6f, test_loss: %.6f, train_ic %.4f, valid_ic %.4f, valid_ric %.4f, test_ic %.4f, test_ric %.4f, test_icir %.4f, test_ricir %.4f" % (step, train_loss, valid_loss, test_loss, metric_train['IC'], metric_valid['IC'], metric_valid['RIC'], metric_test['IC'], metric_test['RIC'], metric_test['ICIR'], metric_test['RICIR']))
                else:
                    self.logger.info("Epoch %d, train_loss %.6f, valid_loss: %.6f, train_ic %.4f, valid_ic %.4f, valid_ric %.4f" % (step, train_loss, valid_loss, metric_train['IC'], metric_valid['IC'], metric_valid['RIC']))
            else: self.logger.info("Epoch %d, train_loss %.6f, train_ic %.4f" % (step, train_loss, metric_train['IC']))

            if metric_valid[self.stop_standard] > valid_ic_best:
                valid_ic_best = metric_valid[self.stop_standard]
                patience = 0
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                patience += 1

            if patience > 20:
                break
            # if train_loss <= self.train_stop_loss_thred:
            #     best_param = copy.deepcopy(self.model.state_dict())
            #     torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
            #     break
        
        # 保存最优模型
        torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
        # 将self.model加载保存的最优参数
        self.model.load_state_dict(best_param)

    def predict(self, dl_test, dl_test_enhance, mode='ret', loss_type='mse', return_type="loss_metric"):
        # if self.fitted<0:
        #     raise ValueError("model is not fitted yet!")
        # else:
        #     print('Epoch:', self.fitted)   
        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False, seed=self.seed)
        if dl_test_enhance is not None:
            assert (dl_test.start == dl_test_enhance.start) & (dl_test.end == dl_test_enhance.end), "dl_test and dl_test_enhance should have the same start and end date."
            test_loader_enhance = self._init_data_loader(dl_test_enhance, shuffle=False, drop_last=False, seed=self.seed)
        else:
            test_loader_enhance = None

        losses = []
        preds = []
        preds_vol = []
        preds_vola = []
        ic = []
        ric = []

        self.model.eval()
        for batch_idx, data in enumerate(test_loader):
            data = torch.squeeze(data, dim=0)
            if mode in ['ret', 'ret_DataEnhance']:
                feature = data[:, :, 0:-1].to(self.device)
                if "DataEnhance" in mode and test_loader_enhance is not None:
                    data_enhance = test_loader_enhance[batch_idx]  # 按索引获取增强数据
                    data_enhance = torch.squeeze(data_enhance, dim=0)
                    origin_stocks_len = feature.shape[0] # 获取目标股票的数量，后续方便索引
                    feature_enhance = data_enhance[:, :, 0:-1].to(self.device)
                    feature = torch.cat((feature, feature_enhance), dim=0) # 将增强数据与原始数据拼接
                label = data[:, -1, -1].to(self.device)
            
                # nan label will be automatically ignored when compute metrics.
                # zscorenorm will not affect the results of ranking-based metrics.

                with torch.no_grad():
                    pred = self.model(feature.float())
                if "DataEnhance" in mode and test_loader_enhance is not None:
                    # 恢复到原来的shape
                    pred = pred[:origin_stocks_len]
                loss = self.loss_fn(pred, label, loss_type=loss_type)

            elif mode in ['rv1', 'r_New1_vola1', 'r_New2_vola1', 'r_New3_vola1']:
                feature = data[:, :, 0:-3].to(self.device)
                label = data[:, -1, -3].to(self.device)
                label_vol1 = data[:, -1, -2].to(self.device)
                with torch.no_grad():
                    pred, pred_vol1 = self.model(feature.float())
                assert loss_type=='mse_rv1', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred, label, pred_vol1, label_vol1, pred_vol1, label_vol1, loss_type=loss_type)

            elif mode in ['rv2', 'r_New1_vola2', 'r_New2_vola2', 'r_New3_vola2']:
                feature = data[:, :, 0:-3].to(self.device)
                label = data[:, -1, -3].to(self.device)
                label_vol2 = data[:, -1, -1].to(self.device)
                with torch.no_grad():
                    pred, pred_vol2 = self.model(feature.float())
                assert loss_type=='mse_rv2', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred, label, pred_vol2, label_vol2, pred_vol2, label_vol2, loss_type=loss_type)
            
            elif mode in ['rv12', 'r_New1_vola12', 'r_New2_vola12', 'r_New3_vola12']:
                feature = data[:, :, 0:-3].to(self.device)
                label = data[:, -1, -3].to(self.device)
                label_vol1 = data[:, -1, -2].to(self.device)
                label_vol2 = data[:, -1, -1].to(self.device)
                with torch.no_grad():
                    pred, pred_vol1, pred_vol2 = self.model(feature.float())
                assert loss_type=='mse_rv12', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred, label, pred_vol1, label_vol1, pred_vol2, label_vol2, loss_type=loss_type)
            
            elif mode=='rc': # label有4个，ret+vol1+vol2+cross_classification, 我们使用ret+cross_classification
                feature = data[:, :, 0:-4].to(self.device)
                label = data[:, -1, -4].to(self.device)
                label_class = data[:, -1, -1].to(self.device)
                with torch.no_grad():
                    pred, pred_class = self.model(feature.float())
                assert loss_type=='mse_rc', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, pred_class=pred_class, label_class=label_class, loss_type=loss_type)

            elif mode in ['r_v1_co', 'rv1c']: # label有4个，ret+vol1+vol2+cross_classification, 我们使用ret+vol2+cross_classification(ordinal)
                feature = data[:, :, 0:-4].to(self.device)
                label = data[:, -1, -4].to(self.device)
                label_vol1 = data[:, -1, -3].to(self.device)
                label_class = data[:, -1, -1].to(self.device)
                with torch.no_grad():
                    pred, pred_vol1, pred_class = self.model(feature.float())
                assert loss_type in ['mse_r_v1_co', 'mse_rv1c'], f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                    pred_vol1=pred_vol1, label_vol1=label_vol1,
                                    pred_class=pred_class, label_class=label_class, loss_type=loss_type)

            elif mode in ['r_v2_co', 'rv2c']: # label有4个，ret+vol1+vol2+cross_classification, 我们使用ret+vol2+cross_classification(ordinal)
                feature = data[:, :, 0:-4].to(self.device)
                label = data[:, -1, -4].to(self.device)
                label_vol2 = data[:, -1, -2].to(self.device)
                label_class = data[:, -1, -1].to(self.device)
                with torch.no_grad():
                    pred, pred_vol2, pred_class = self.model(feature.float())
                assert loss_type in ['mse_r_v2_co', 'mse_rv2c'], f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                    pred_vol2=pred_vol2, label_vol2=label_vol2,
                                    pred_class=pred_class, label_class=label_class, loss_type=loss_type)

            elif mode in ['r_v12_co', 'rv12c']: # label有4个，ret+vol1+vol2+cross_classification, 我们使用ret+vol1+vol2+cross_classification(ordinal)
                feature = data[:, :, 0:-4].to(self.device)
                label = data[:, -1, -4].to(self.device)
                label_vol1 = data[:, -1, -3].to(self.device)
                label_vol2 = data[:, -1, -2].to(self.device)
                label_class = data[:, -1, -1].to(self.device)
                with torch.no_grad():
                    pred, pred_vol1, pred_vol2, pred_class = self.model(feature.float())
                assert loss_type in ['mse_r_v12_co', 'mse_rv12c'], f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                    pred_vol1=pred_vol1, label_vol1=label_vol1,
                                    pred_vol2=pred_vol2, label_vol2=label_vol2,
                                    pred_class=pred_class, label_class=label_class, loss_type=loss_type)

            elif mode in ['r_vola1', 'r_vola2']: # label有5个，ret+vol1+vol2+label_class+volatility, 我们使用ret+volatility
                feature = data[:, :, 0:-5].to(self.device)
                label = data[:, -1, -5].to(self.device)
                label_vola = data[:, -1, -1].to(self.device)
                with torch.no_grad():
                    pred, pred_vola = self.model(feature.float())
                assert loss_type in ['mse_r_vola1', 'mse_r_vola2'], f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                    pred_vola=pred_vola, label_vola=label_vola,
                                    loss_type=loss_type)
            
            elif mode in ['r_vol2_vola2', 'r_vol2_vola2_t1', 'r_vol2_vola2_t2', 'r_vol2_vola2_t1_DataEnhance', 'r_vol2_vola2_DataEnhance']:    # label有3个，ret+vol2+vola2
                feature = data[:, :, 0:-3].to(self.device)
                if "DataEnhance" in mode and test_loader_enhance is not None:
                    data_enhance = test_loader_enhance[batch_idx]  # 按索引获取增强数据
                    data_enhance = torch.squeeze(data_enhance, dim=0) # (N, T, F)
                    origin_stocks_len = feature.shape[0] # 获取目标股票的数量，后续方便索引
                    feature_enhance = data_enhance[:, :, 0:-3].to(self.device)
                    feature = torch.cat((feature, feature_enhance), dim=0) # 将增强数据与原始数据拼接

                label = data[:, -1, -3].to(self.device)
                label_vol2 = data[:, -1, -2].to(self.device)
                label_vola = data[:, -1, -1].to(self.device)
                
                assert loss_type in ['mse_r_vol2_vola2', 'mse_r_vol2_vola2_t1', 'mse_r_vol2_vola2_t2', 'mse_r_vol2_vola2_ddw1', \
                                     'mse_r_vol2_vola2_t1_DW11', 'mse_r_vol2_vola2_t1_DW12', \
                                     'mse_r_vol2_vola2_t1_DW21', 'mse_r_vol2_vola2_t1_DW22', \
                                     'mse_r_vol2_vola2_t1_DW31', 'mse_r_vol2_vola2_t1_DW32', \
                                     'mse_r_vol2_vola2_t1_GW1', 'mse_r_vol2_vola2_t1_GW2', 'mse_r_vol2_vola2_t1_GW3'], f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                with torch.no_grad():
                    if "GW" in loss_type:
                        pred, pred_vol2, pred_vola, gate_dict = self.model(feature.float())  
                        if "DataEnhance" in mode and test_loader_enhance is not None:
                            # 恢复到原始数据的形状
                            pred = pred[:origin_stocks_len] 
                            pred_vol2 = pred_vol2[:origin_stocks_len]
                            pred_vola = pred_vola[:origin_stocks_len]
                        loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                    pred_vol2=pred_vol2, label_vol2=label_vol2, 
                                    pred_vola=pred_vola, label_vola=label_vola, loss_type=loss_type, gate_weight=gate_dict)
                    else:
                        pred, pred_vol2, pred_vola = self.model(feature.float())
                        if "DataEnhance" in mode and test_loader_enhance is not None:
                            # 恢复到原始数据的形状
                            pred = pred[:origin_stocks_len] 
                            pred_vol2 = pred_vol2[:origin_stocks_len]
                            pred_vola = pred_vola[:origin_stocks_len]
                        loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                    pred_vol2=pred_vol2, label_vol2=label_vol2, 
                                    pred_vola=pred_vola, label_vola=label_vola, loss_type=loss_type)
            

            elif mode in ['r_vol2_woVola2', 'r_woVol2_vola2', 'r_woVol2_woVola2']:    # label有3个，ret+vol2+vola2, 专注于消融实验
                feature = data[:, :, 0:-3].to(self.device)
                label = data[:, -1, -3].to(self.device)
                label_vol2 = data[:, -1, -2].to(self.device)
                label_vola = data[:, -1, -1].to(self.device)
                if mode == 'r_vol2_woVola2':
                    with torch.no_grad():
                        pred, pred_vol2 = self.model(feature.float())
                    assert loss_type=='mse_r_vol2', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                    loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                        pred_vol2=pred_vol2, label_vol2=label_vol2, 
                                        loss_type=loss_type)
                elif mode == 'r_woVol2_vola2':
                    with torch.no_grad():
                        pred, pred_vola = self.model(feature.float())
                    assert loss_type=='mse_r_vola1', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                    loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                        pred_vola=pred_vola, label_vola=label_vola, 
                                        loss_type=loss_type)
                elif mode == 'r_woVol2_woVola2':
                    with torch.no_grad():
                        pred = self.model(feature.float())
                    assert loss_type=='mse', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                    loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                        loss_type=loss_type)



            elif mode in ['r_vol2']:    # label有3个，ret+vol2+vola2
                feature = data[:, :, 0:-3].to(self.device)
                label = data[:, -1, -3].to(self.device)
                label_vol2 = data[:, -1, -2].to(self.device)
                # label_vola = data[:, -1, -1].to(self.device)
                with torch.no_grad():
                    pred, pred_vol2 = self.model(feature.float())
                assert loss_type=='mse_r_vol2', f"Not compatible! mode:{mode}, loss_type:{loss_type}"
                loss = self.loss_fn(pred_ret=pred, label_ret=label, 
                                    pred_vol2=pred_vol2, label_vol2=label_vol2, 
                                    loss_type=loss_type)

            
            if 'vol2' in mode:
                preds_vol.append(pred_vol2)
            if 'vola' in mode:
                preds_vola.append(pred_vola)
            preds.append(pred.ravel())
            losses.append(loss.item())
            daily_ic, daily_ric = calc_ic(pred.cpu().numpy(), label.cpu().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        # predictions = pd.Series(np.concatenate(preds.cpu().numpy()), index=dl_test.get_index())

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)
        }

        if return_type == "loss_metric":
            return float(np.mean(losses)), metrics
        else:
            return {"return":preds, "vol_hit": preds_vol, "volatility": preds_vola}, metrics


# 对return、vol_hit2、volatility进行联合权重加权
class LearnableConfidenceWeightedLoss(nn.Module):
    def __init__(self, init_beta=10, init_gamma=5, init_lambda1=0.2, init_lambda2=0.2):
        super().__init__()
        # 定义可学习的参数
        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.lambda1 = nn.Parameter(torch.tensor(init_lambda1, dtype=torch.float32))
        self.lambda2 = nn.Parameter(torch.tensor(init_lambda2, dtype=torch.float32))
        
        # 应用约束确保权重在合理范围 (0.0 - 1.0)
        self.constraint_lambda = lambda: torch.clamp(self.lambda1, 0.0, 1.0) + torch.clamp(self.lambda2, 0.0, 1.0)
        
    def forward(self, 
                pred_return, true_return, 
                pred_vol_hit, true_vol_hit,
                pred_volatility, true_volatility):
        
        # 计算基础MSE损失
        loss_return = (pred_return - true_return) ** 2
        loss_vol_hit = (pred_vol_hit - true_vol_hit) ** 2
        loss_volatility = (pred_volatility - true_volatility) ** 2
        
        # 动态计算return置信度权重
        weight_volhit = torch.sigmoid(self.beta * true_vol_hit)
        weight_volatility = 1 / (1 + torch.abs(self.gamma * true_volatility))
        w_return = weight_volhit * weight_volatility
        
        # 应用参数约束 (确保lambda1 + lambda2 <= 1.0)
        constrained_lambda_sum = self.constraint_lambda()
        lambda1 = self.lambda1 / constrained_lambda_sum
        lambda2 = self.lambda2 / constrained_lambda_sum
        
        # 计算加权损失
        weighted_loss = (
            w_return * loss_return + 
            lambda1 * loss_vol_hit + 
            lambda2 * loss_volatility
        )
        
        return weighted_loss.mean()

    def extra_repr(self):
        # 显示当前参数值
        return (f'beta={self.beta.item():.2f}, gamma={self.gamma.item():.2f}, '
                f'lambda1={self.lambda1.item():.2f}, lambda2={self.lambda2.item():.2f}')