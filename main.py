from himt import *
import pickle
import numpy as np
import time
import pandas as pd
from utils import *
import argparse
import traceback
import logging
# Please install qlib first before load the data.
# Parse command line arguments
parser = argparse.ArgumentParser(description='Model training parameters')
parser.add_argument('--universe', type=str, default='csi500', choices=['csi300', 'csi500', 'csi800', 'csi1000'], help='Universe to use')
parser.add_argument('--d_feat', type=int, default=158, help='Feature dimension')
parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
parser.add_argument('--t_nhead', type=int, default=4, help='Number of heads in temporal attention')
parser.add_argument('--s_nhead', type=int, default=2, help='Number of heads in spatial attention')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--gate_input_start_index', type=int, default=158, help='Gate input start index')
parser.add_argument('--gate_input_end_index', type=int, default=221, help='Gate input end index')
parser.add_argument('--model_name', type=str, default='master', help='Model name')
parser.add_argument('--mode', type=str, default='ret', help='Mode')
parser.add_argument('--drop_extreme', action='store_true', default=False, help='Whether to drop extreme values')
parser.add_argument('--loss_type', type=str, default='mse', help='Loss type')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--GPU', type=int, default=0, help='GPU ID')
parser.add_argument('--train_stop_loss_thred', type=float, default=0.95, help='Training stop loss threshold')
parser.add_argument('--delay', type=int, default=1, help='prediction delay in days')
parser.add_argument('--n_experts', type=int, default=8, help='mmoe experts number')
parser.add_argument('--k_s', type=int, default=2, help='number of shared experts')
parser.add_argument('--w1', type=float, default=0.8, help='multi-task w1 for return loss')
parser.add_argument('--w2', type=float, default=0.1, help='multi-task w2 for vol_hit loss')
parser.add_argument('--w3', type=float, default=0.1, help='multi-task w3 for volatility loss')
parser.add_argument('--return_gate_weight', type=int, default=0, help='whether to return gate weight of moe model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dim')
parser.add_argument('--stop_standard', type=str, default='IC', help='stop training standard')


args = parser.parse_args()

# Assign arguments to variables
universe = args.universe
d_feat = args.d_feat
d_model = args.d_model
t_nhead = args.t_nhead
s_nhead = args.s_nhead
dropout = args.dropout
gate_input_start_index = args.gate_input_start_index
gate_input_end_index = args.gate_input_end_index
model_name = args.model_name
mode = args.mode
n_epoch = args.n_epoch
drop_extreme = args.drop_extreme
loss_type = args.loss_type
delay = args.delay
lr = args.lr
GPU = args.GPU
n_experts = args.n_experts
w1 = args.w1
w2 = args.w2
w3 = args.w3
train_stop_loss_thred = 0.95
return_gate_weight = args.return_gate_weight
seed = args.seed
k_s = args.k_s
hidden_dim = args.hidden_dim
stop_standard = args.stop_standard

enhance_universe = "csi300" if universe == "csi500" else "csi500"

seed_everything(seed)

if 'moe' in model_name:
    file_name = f"seed{seed}+{stop_standard}+{universe}+{model_name}+mode_{mode}+dropEx{int(drop_extreme)}+{loss_type}+delay{delay}+hidden{hidden_dim}+n_experts{n_experts}+t_nheads{t_nhead}+s_nheads{s_nhead}+w1_{w1}+w2_{w2}+w3_{w3}"
else:
    file_name = f"seed{seed}+{stop_standard}+{universe}+{model_name}+mode_{mode}+dropEx{int(drop_extreme)}+{loss_type}+delay{delay}+hidden{hidden_dim}+t_nheads{t_nhead}+s_nheads{s_nhead}+rvc{w1}+w1_{w1}+w2_{w2}+w3_{w3}"
# 将当前目录下的所有.py文件备份
save_path = log_dir(file_name=file_name, log_directory="./csi500_exp_log")
copy_py_files('./', save_path)

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(f'{save_path}/main.log'), logging.StreamHandler()])

# 获取日志记录器
logger = logging.getLogger(__name__)

try:
    logger.info("Starting the main script...")
    logger.info(args)

    if mode in ["ret", "ret_DataEnhance"]:                            # 有一个label，即return
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/{universe}_Alpha158/dl_train.pkl', 'rb') as f:
            dl_train = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/{universe}_Alpha158/dl_valid.pkl', 'rb') as f:
            dl_valid = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/{universe}_Alpha158/dl_test.pkl', 'rb') as f:
            dl_test = pickle.load(f)

        if "DataEnhance" in mode: # 需要增强数据
            # 加载增强数据源
            with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/{enhance_universe}_Alpha158/dl_train.pkl', 'rb') as f:
                dl_train_enhance = pickle.load(f)
            with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/{enhance_universe}_Alpha158/dl_valid.pkl', 'rb') as f:
                dl_valid_enhance = pickle.load(f)
            with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/{enhance_universe}_Alpha158/dl_test.pkl', 'rb') as f:
                dl_test_enhance = pickle.load(f)

    elif mode == 'rv1' or mode == 'rv2' or mode == 'rv12': # 有3个label，ret+vol1+vol2
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/RV2_{universe}_Alpha158/dl_train.pkl', 'rb') as f:
            dl_train = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/RV2_{universe}_Alpha158/dl_valid.pkl', 'rb') as f:
            dl_valid = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/RV2_{universe}_Alpha158/dl_test.pkl', 'rb') as f:
            dl_test = pickle.load(f)
    elif mode in ['rc', 'rv1c', 'rv2c', 'rv12c', 'r_v1_co', 'r_v2_co', 'r_v12_co']: # 有classification的label
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/RV2C_{universe}_Alpha158/dl_train.pkl', 'rb') as f:
            dl_train = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/RV2C_{universe}_Alpha158/dl_valid.pkl', 'rb') as f:
            dl_valid = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/RV2C_{universe}_Alpha158/dl_test.pkl', 'rb') as f:
            dl_test = pickle.load(f)
    # 之前的volatility：delay=1，则是日内30min的return std，delay=5则是日间5天的return std
    elif mode in ['r_vola1', 'r_vola2']:
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/MulLab5_{universe}_Alpha158/dl_train.pkl', 'rb') as f:
            dl_train = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/MulLab5_{universe}_Alpha158/dl_valid.pkl', 'rb') as f:
            dl_valid = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/MulLab5_{universe}_Alpha158/dl_test.pkl', 'rb') as f:
            dl_test = pickle.load(f)
    # 新的volatility：label1=1-abs(close-open) / (high-low)
    elif mode in ['r_New1_vola1', 'r_New1_vola2', 'r_New1_vola12']:
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/RVola1_{universe}_Alpha158/dl_train.pkl', 'rb') as f:
            dl_train = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/RVola1_{universe}_Alpha158/dl_valid.pkl', 'rb') as f:
            dl_valid = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/RVola1_{universe}_Alpha158/dl_test.pkl', 'rb') as f:
            dl_test = pickle.load(f)
    # 新的volatility：label1=(high - low) / close
    elif mode in ['r_New2_vola1', 'r_New2_vola2', 'r_New2_vola12']:
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/RVola2_{universe}_Alpha158/dl_train.pkl', 'rb') as f:
            dl_train = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/RVola2_{universe}_Alpha158/dl_valid.pkl', 'rb') as f:
            dl_valid = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/RVola2_{universe}_Alpha158/dl_test.pkl', 'rb') as f:
            dl_test = pickle.load(f)
    # 新的volatility：label1=(high - low) / vwap
    elif mode in ['r_New3_vola1', 'r_New3_vola2', 'r_New3_vola12']:
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/R_Vol2_Vola2_{universe}_Alpha158/dl_train.pkl', 'rb') as f:
            dl_train = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/R_Vol2_Vola2_{universe}_Alpha158/dl_valid.pkl', 'rb') as f:
            dl_valid = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/R_Vol2_Vola2_{universe}_Alpha158/dl_test.pkl', 'rb') as f:
            dl_test = pickle.load(f)
    elif mode in ['r_vol2_vola2', 'r_vol2', 'r_vol2_vola2_t1', 'r_vol2_vola2_t2', 'r_vol2_vola2_DataEnhance', 'r_vol2_vola2_t1_DataEnhance',
                  'r_woVol2_vola2', 'r_vol2_woVola2', 'r_woVol2_woVola2', ]:
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/R_Vol2_Vola2_{universe}_Alpha158/dl_train.pkl', 'rb') as f:
            dl_train = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/R_Vol2_Vola2_{universe}_Alpha158/dl_valid.pkl', 'rb') as f:
            dl_valid = pickle.load(f)
        with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/R_Vol2_Vola2_{universe}_Alpha158/dl_test.pkl', 'rb') as f:
            dl_test = pickle.load(f)
        
        if "DataEnhance" in mode: # 需要增强数据
            # 加载增强数据源
            with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/R_Vol2_Vola2_{enhance_universe}_Alpha158/dl_train.pkl', 'rb') as f:
                dl_train_enhance = pickle.load(f)
            with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/R_Vol2_Vola2_{enhance_universe}_Alpha158/dl_valid.pkl', 'rb') as f:
                dl_valid_enhance = pickle.load(f)
            with open(f'/home/yanghengyi/workspace/CSPO/data/freq_day/delay{delay}/R_Vol2_Vola2_{enhance_universe}_Alpha158/dl_test.pkl', 'rb') as f:
                dl_test_enhance = pickle.load(f)
            

    if "DataEnhance" not in mode: # 统一处理不需要增强数据的情况
        dl_train_enhance = None
        dl_valid_enhance = None
        dl_test_enhance = None

    logger.info(f"{universe} Data Loaded.")



    if universe == 'csi300' or universe == 'csi500':
        beta = 5
    elif universe == 'csi800':
        beta = 2

    # validation
    ic_val = []
    icir_val = []
    ric_val = []
    ricir_val = []


    ic = []
    icir = []
    ric = []
    ricir = []

    # Training
    ######################################################################################
    # for seed in [0, 1, 2, 3, 4]:
    

    model = UnifiedModel(
                d_feat = d_feat, d_model = hidden_dim, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
                beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
                n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
                save_path=save_path, save_prefix='model',
                log_save_path=save_path, mode=mode, drop_extreme=drop_extreme, loss_type=loss_type, logger=logger, model_name=model_name, n_experts=n_experts,
                weight1=w1, weight2=w2, weight3=w3, return_gate_weight=return_gate_weight, k_s = k_s, stop_standard = stop_standard
                )
    
    start = time.time()
    # Train
    model.fit(dl_train, dl_valid, dl_test, dl_train_enhance, dl_valid_enhance, dl_test_enhance)

    logger.info("Model Trained.")

    # Validation 
    predicions_valid, metrics_valid = model.predict(dl_valid, dl_valid_enhance, mode, loss_type)

    # Test
    predictions, metrics = model.predict(dl_test, dl_test_enhance, mode, loss_type)
    
    running_time = time.time()-start
    
    logger.info('Seed: {:d} time cost : {:.2f} sec'.format(seed, running_time))
    logger.info(metrics)

    # validation
    ic_val.append(metrics_valid['IC'])
    icir_val.append(metrics_valid['ICIR'])
    ric_val.append(metrics_valid['RIC'])
    ricir_val.append(metrics_valid['RICIR'])
        
    # test
    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])

    logger.info("Seed: {:d} IC: {:.4f}".format(seed, metrics['IC']))
    logger.info("Seed: {:d} ICIR: {:.4f}".format(seed, metrics['ICIR']))
    logger.info("Seed: {:d} RIC: {:.4f}".format(seed, metrics['RIC']))
    logger.info("Seed: {:d} RICIR: {:.4f}".format(seed, metrics['RICIR']))
    ######################################################################################

    # Load and Test
    ######################################################################################
    # for seed in [0]:
    #     param_path = f'model\{universe}_{prefix}_{seed}.pkl'

    #     logger.info(f'Model Loaded from {param_path}')
    #     model = MASTERModel(
    #             d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
    #             beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
    #             n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
    #             save_path='model/', save_prefix=universe
    #         )
    #     model.load_param(param_path)
    #     predictions, metrics = model.predict(dl_test)
    #     logger.info(metrics)

    #     ic.append(metrics['IC'])
    #     icir.append(metrics['ICIR'])
    #     ric.append(metrics['RIC'])
    #     ricir.append(metrics['RICIR'])
        
    ######################################################################################

    logger.info("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
    logger.info("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
    logger.info("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
    logger.info("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))

    # 创建一个字典，包含需要的列名和对应的值
    data = {
        "log_name": [file_name],
        "universe": [universe],
        "model_name": [model_name],
        "mode": [mode],
        "delay": [delay],
        "seed": [seed],
        "n_experts": [n_experts],
        "w1": [w1],
        "w2": [w2],
        "w3": [w3],
        "t_head": [t_nhead],
        "s_head": [s_nhead],
        "stop_standard": [stop_standard],
        "hidden_dim": [hidden_dim],
        "drop_extreme": [drop_extreme],
        "loss_type": [loss_type],
        "IC": [round(np.mean(ic), 4)],
        "IC_PM": [round(np.std(ic), 4)],
        "ICIR": [round(np.mean(icir), 4)],
        "ICIR_PM": [round(np.std(icir), 4)],
        "RIC": [round(np.mean(ric), 4)],
        "RIC_PM": [round(np.std(ric), 4)],
        "RICIR": [round(np.mean(ricir), 4)],
        "RICIR_PM": [round(np.std(ricir), 4)],
    }
    # 将字典转换为DataFrame
    df = pd.DataFrame(data)
    # 保存DataFrame到指定目录
    df.to_csv(f"{save_path}/results.csv", index=False)

except Exception as e:
    logger.error("An error occurred: %s", e)
    logger.error(traceback.format_exc())