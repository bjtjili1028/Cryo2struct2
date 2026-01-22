"""
Created on 04 Sep 2023 06:16 AM
@author: nabin

Cryo2Struct-V2 主脚本：
1. 解析命令行和 YAML 配置
2. 划分 3D 网格子体
3. 深度学习推断（氨基酸与原子类型）
4. 从原子／氨基酸概率提取 CA 坐标
5. 聚类、准备 HMM 发射/转移矩阵
6. 调用 Viterbi 进行序列比对

"""

import time  # 用於計算執行時間
from datetime import date  # 用於處理日期
import argparse # 用於處理命令行參數 
import yaml # 用於讀取yml格式的配置文件 
import os # 用於處理檔案和目錄 
import shutil # 用於刪除目錄和文件 
import threading # 用於創建和管理線程
import mrcfile

import torch
from pprint import pprint


# 从 utils 包中导入各子模块
from utils import get_probs_cords_from_atom_amino, clustering_centroid, grid_division, get_ca_from_pred_probs, clustering_centroid_for_c_n, atom_pick
from viterbi import alignment # 导入 Viterbi 算法模块
import subprocess # 用于调用外部推断脚本

import warnings
warnings.filterwarnings("ignore") # 忽略各种警告信息

# 获取当前脚本所在目录，用于后续定位其他文件
script_dir = os.path.dirname(os.path.abspath(__file__))

# 配置文件路径（相对于脚本目录）
config_file_path = f"{script_dir}/config/arguments.yml"
COMMENT_MARKER = '#'

def parse_arguments():
    """
    使用 argparse 从命令行解析 --config 和 --density_map_name 参数
    --config: YAML 配置文件（可选，默认 arguments.yml）
    --density_map_name: 要处理的密度图文件夹名称
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'),
                        default=config_file_path)
    parser.add_argument('--density_map_name', type=str)
    
    return parser.parse_args()


def process_arguments(args):
    """
    将 argparse 读取的参数与 YAML 配置文件合并，去除以 '#' 开头的注释项
    返回一个字典 config_dict，包含所有配置项
    """
    if args.config is not None:
        # 从文件读取 YAML，并过滤注释项
        config_dict = yaml.safe_load(args.config)
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith(COMMENT_MARKER)}
        args.config = args.config.name
    else:
        config_dict = dict()
        
    # 如果命令行指定了 density_map_name，就覆盖 YAML 中的同名字段
    if args.density_map_name is not None:
        config_dict['density_map_name'] = args.density_map_name
    return config_dict


def delete_directory(directory_path):
    """
    递归删除目录（注意：调用前应检查目录是否存在）
    用于在后台清理临时文件夹
    """
    shutil.rmtree(directory_path)

def make_predictions(config_dict):
    """
    核心流程：
    1. 调用 grid_division 将原始密度图分成小块并保存
    2. 调用外部推断脚本对每块数据执行氨基酸和原子类型预测
    3. 在后台启动线程删除 split 目录和 lightning_logs 目录
    """
     # 1. 划分子网格
    start_time = time.time()
    grid_division.create_subgrids(input_data_dir=config_dict['input_data_dir'], density_map_name=config_dict['density_map_name'])
    end_time = time.time()
    print(f"\nCryo2Struct DL: Grid Division Complete! \n[Time] : {end_time - start_time:.2f} seconds")
    
    # 构造路径和脚本列表
    density_map_dir = os.path.join(config_dict['input_data_dir'],config_dict['density_map_name'])
    density_map_split_dir = os.path.join(density_map_dir, f"{config_dict['density_map_name']}_splits")
    
    # 原始脚本路径（硬编码，应改为相对路径）
    # script_name = ['/bml/nabin/charlieCryo/src/cryo2struct_v2/Cryo2Struct_V2_final/infer/atom_amino_joint_inference.py']
    script_name = ['/media/ray-suen/TRANSCEND1/huei/Cryo2Struct2/infer/atom_amino_joint_inference.py']
     # 对应 checkpoint 字典键
    checkpoint_name = ['amino_checkpoint', 'atom_checkpoint']

    # 2. 依次执行脚本（氨基酸 & 原子预测）
    for s in range(len(script_name)):
        start_time = time.time()
        cmd = ['python3', script_name[s], density_map_split_dir, str(config_dict['input_data_dir']),
               str(config_dict['density_map_name']),  str(config_dict[checkpoint_name[s]]), str(config_dict[checkpoint_name[s+1]]), config_dict['infer_run_on'], str(config_dict['infer_on_gpu'] )]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            stdout = result.stdout
            stderr = result.stderr
            return_code = result.returncode
            if return_code == 0:
                print(f"Cryo2Struct DL: Prediction {s + 1} / {len(script_name)} Complete!")
                # print(stdout)
            else:
                print(f"Cryo2Struct Deep Learning Block failed with exit code {return_code}.")
                print("Standard Error:")
                print(stderr)
        except subprocess.CalledProcessError as e:
            print(f"Cryo2Struct Deep Learning Block failed with exit code {e.returncode}.")
            print("Standard Error:")
            print(e.stderr)
        except Exception as e:
            print(f"An error occurred in Cryo2Struct Deep Learning Block: {str(e)}")

        end_time = time.time()
        print(f"[Time] : {end_time - start_time:.2f} seconds")

    # 启动后台线程删除 split 和日志目录
    delete_thread = threading.Thread(target=delete_directory, args=(density_map_split_dir,))
    delete_thread.start() # runs in background to delete the grid division directory
    delete_thread1 = threading.Thread(target=delete_directory, args=(f"{density_map_dir}/lightning_logs",))
    delete_thread1.start()

def extract_probs_cords_from_atom_amino(config_dict):
    """
    从 atom/amino 预测结果提取联合概率和坐标，生成 HMM 的发射和转移矩阵文件
    """
    # 讀取模型輸出檔案
    probability_file_atom = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_atom.txt" # comes from atom_inference.py
    probability_file_atom_spilt_ca = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_spilt_ca_prob.txt" #修改成僅讀取切割後的ca原子
    probability_file_amino = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino.txt" # comes from amino_inference.py
    
    # 輸出共同原子的檔案(胺基酸和原子進行比對後的數據)
    probability_file_amino_atom_common_emi = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino_atom_common_emi.txt" # save common amino and atom
    probability_file_amino_common_emi = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino_emi.txt" # save amino probability as emission
    probability_file_amino_atom_common_ca_prob = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino_atom_common_ca_prob.txt" # save common amino and atom (atom prob)
    save_cords = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_coordinates_ca.txt" # save cords as transition matrix

    # 輸出三種骨幹原子
    split_output_ca  = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_spilt_ca_prob.txt" # save ca atom prob
    split_output_n = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_spilt_n_prob.txt" # save n atom prob
    split_output_c = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_spilt_c_prob.txt" # save c atom prob

    # 删除旧文件
    if os.path.exists(save_cords):
        os.remove(save_cords)
    
    if os.path.exists(probability_file_amino_atom_common_emi):
        os.remove(probability_file_amino_atom_common_emi)

    if os.path.exists(probability_file_amino_common_emi):
        os.remove(probability_file_amino_common_emi)
    ##################################################################################
    ############# 分割三種骨幹原子 #####################
    get_probs_cords_from_atom_amino.split_atom_file(
                    probability_file_atom=probability_file_atom, 
                    split_output_ca = split_output_ca, 
                    split_output_n = split_output_n, 
                    split_output_c = split_output_c,
                    ca_threshold = config_dict['threshold'],
                    mode=config_dict['split_mod'] # mod=1是取最大機率值  mod=2是取0.4當作標準 
                    ) 
    ############# new_atom_pick ########################################################
    # # cal base_target
    # fasta_file = [f for f in os.listdir(f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}") if f.endswith(".fasta")]
    # fasta_file.sort()
    # base_target = atom_pick.determine_target_count(fasta_file[0])
    # target_coverage = int(base_target * config_dict['coverage_factor'])
    # print(f"\n[GRID] cf={config_dict['coverage_factor']:.2f} → target_atoms={target_coverage}")

    # # 基於自適應閾值，乘上倍率
    # ca_t, n_t, c_t = atom_pick.adaptive_threshold_analysis(probability_file_atom, target_coverage)
    # def clamp01(x): return max(0.0, min(1.0, float(x)))
    
    # ca_t = clamp01(ca_t * config_dict['ca_mult'])
    # n_t  = clamp01(n_t  * config_dict['n_mult'])
    # c_t  = clamp01(c_t  * config_dict['c_mult'])

    # print(f"[GRID] thresholds  CA={ca_t:.4f} (x{config_dict['ca_mult']}), N={n_t:.4f} (x{config_dict['n_mult']}), C={c_t:.4f} (x{config_dict['c_mult']})")
    
    # # 3) 解析概率點
    # ca_pts, n_pts, c_pts = atom_pick.parse_probabilities(probability_file_atom, ca_t, n_t, c_t)
    # print(f"After Thresholding：CA: len({ca_pts}), N: len({n_pts}), C: len({c_pts})\n")

    # # 4) NMS
    # ca_nms,ca_final_r = atom_pick.nms_kdtree_adaptive(ca_pts,config_dict['nms_radius'], max_points=target_coverage)
    # n_nms,n_final_r  = atom_pick.nms_kdtree_adaptive(n_pts, config_dict['nms_radius'], max_points=int(target_coverage))
    # c_nms,c_final_r  = atom_pick.nms_kdtree_adaptive(c_pts, config_dict['nms_radius'], max_points=target_coverage)
    # print(f"After NMS：CA: len({ca_nms}), N: len({n_nms}), C: len({c_nms})\n")
    # print(f"CA_final_nms_r:{ca_final_r},N_final_nms_r:{n_final_r},C_final_nms_r:{c_final_r}\n")

    # # output
    # atom_pick.write_centroid_file(ca_nms, split_output_ca)
    # atom_pick.write_centroid_file(n_nms, split_output_n)
    # atom_pick.write_centroid_file(c_nms, split_output_c)

    ##################################################################################
    
    # 调用工具函数生成联合概率和坐标
    get_probs_cords_from_atom_amino.get_joint_probabity_common_threshold(
        probability_file_atom=probability_file_atom_spilt_ca,   # 原本使用 probability_file_atom
        probability_file_amino_atom_common=probability_file_amino_atom_common_emi, 
        probability_file_amino=probability_file_amino, 
        s_c=save_cords, threshold = config_dict['threshold'], 
        # s_c=save_cords, threshold = ca_t,
        probability_file_amino_atom_common_ca_prob=probability_file_amino_atom_common_ca_prob)

##################################################################################
# 執行聚類來準備發射和轉換矩陣
def cluster_emission_transition(config_dict):
    save_cords, save_probs_aa, save_ca_probs= clustering_centroid.main(config_dict)
    combined_output_file_c, coords_output_file_c, save_cords_c,combined_output_file_n, coords_output_file_n, save_cords_n = clustering_centroid_for_c_n.main(config_dict) 
    return save_cords, save_probs_aa, save_ca_probs,combined_output_file_c, coords_output_file_c, save_cords_c,combined_output_file_n, coords_output_file_n, save_cords_n


################# v2 新增
"""
因為其將訓練整合為一個模型，所以無法輸出單獨一個ca的pdb和mrc檔案，所以這塊僅是彌補輸出。
"""
def extract_ca_from_prediction_probabilities(config_dict):
    # 設定參數
    original_map = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/emd_normalized_map.mrc"
    original_map_mrc = mrcfile.open(original_map, mode='r')
    original_map_shape = original_map_mrc.data.shape
    original_map_origin = original_map_mrc.header.origin
    x_origin = original_map_mrc.header.origin['x']
    y_origin = original_map_mrc.header.origin['y']
    z_origin = original_map_mrc.header.origin['z']
    x_voxel = original_map_mrc.voxel_size['x']
    y_voxel = original_map_mrc.voxel_size['y']
    z_voxel = original_map_mrc.voxel_size['z']

    # extract ca from atoms prediction : for visualization and evaluation
    pred_atom_prob = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_atom.txt"
    only_ca_atom_mrc = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_prob_atom_ca_predicted.mrc"
    only_ca_atom_pdb = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_prob_atom_ca_predicted.pdb"
    
    # 刪除舊文件
    if os.path.exists(only_ca_atom_pdb):
        os.remove(only_ca_atom_pdb)
    
    # 得到原始ca的pdb和mac檔案
    get_ca_from_pred_probs.extract_ca_from_atom(pred_atom=pred_atom_prob, outfilename=only_ca_atom_mrc, outfilename_pdb=only_ca_atom_pdb, density_shape=original_map_shape, density_voxel=(x_voxel,y_voxel,z_voxel), density_origin=(x_origin,y_origin,z_origin), origin=original_map_origin)
    
    print("Extracting carbon alpha from ATOMS prediction complete!")

#################

# 主函數，執行整個流程
def main():
    args = parse_arguments() # 解析命令行參數
    config_dict = process_arguments(args) # 處理命令行參數
    print("\n##############- Cryo2Struct-V2 -##############")
    print("\nRunning with below configuration: ")
    
    # 打印配置信息
    for key,value in config_dict.items():
        print("%s : %s"%(key, value))
    print("Date:",date.today())
    print("\n- This might take a bit. Time for a coffee break, maybe! -")
    
    # 進行預測
    make_predictions(config_dict)
    
    ############ v2 新增
    print("\nExtracting CA") # 得到原始ca的pdb和mac檔案
    extract_ca_from_prediction_probabilities(config_dict)
    ############
    
    # preparing for HMM model (提取概率和坐標)
    cluster_start_time = time.time()
    extract_probs_cords_from_atom_amino(config_dict)
    
    # clustering and preparing emission and transition matrix
    coordinate_file, emission_file, save_ca_probs,combined_output_file_c, coords_output_file_c, save_cords_c,combined_output_file_n, coords_output_file_n, save_cords_n = cluster_emission_transition(config_dict)
    # coordinate_file, emission_file, save_ca_probs = cluster_emission_transition(config_dict)
    cluster_end_time = time.time()
    runtime_sec = cluster_end_time - cluster_start_time
    print(f"\nCryo2Struct Clustering Finished {runtime_sec:.2f} seconds.")

    # run viterbi algorithm (執行Viterbi算法)
    alignment.main(coordinate_file, emission_file, config_dict, save_ca_probs)

if __name__ == "__main__":
    main()
