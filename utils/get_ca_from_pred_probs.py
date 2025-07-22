"""
Created on 6 March 2023 01:15 PM
@author: nabin


This script takes in predicted probability file, process it and extracts only ca from it, then finally saves to mrc file.
用途：
  - 從預測概率文件中提取 CA 原子 (Alpha Carbon) 的坐標與概率，
    並根據氨基酸預測、二級結構預測結果，計算三者共同的 CA 分數。
  - 最終將 CA 座標保存為 PDB 格式，同時輸出 MRC 體積文件以便可視化。
"""
import mrcfile
from tqdm import tqdm
import numpy as np
import ast
import math
import os


"""
將實際空間坐標 cord 映射到體素網格索引:
    - cord: 實際坐標值
    - origin: 密度體素的原點坐標
    - voxel: 單位體素大小
返回對應的整數索引。
"""
# 計算座標轉換為索引
def get_index(cord, origin, voxel):
    return math.ceil(math.floor(cord - abs(origin)) / voxel)


# 寫入 PDB 文件的一行 ATOM 記錄
def save(x, y, z, count, save_path):
    atom = 'CA'
    residue_name = "GLY"
    with open(save_path, 'a') as fi:
        fi.write('ATOM')
        fi.write('  ')
        fi.write(str(count).rjust(5)) # 序號
        fi.write('  ')
        fi.write(atom.ljust(4)) # 原子類型
        fi.write(residue_name.rjust(3)) # 殘基名
        fi.write(' ')
        fi.write('A')  # todo: need to change chain id accordingly # 鏈ID (TODO: 根據需要修改)
        fi.write(str(count).rjust(4))
        fi.write('    ')
        fi.write(str(x).rjust(8)) # X 坐標
        fi.write(str(y).rjust(8)) # Y 坐標
        fi.write(str(z).rjust(8)) # Z 坐標
        fi.write(str(1.00).rjust(5)) # 占有率
        fi.write(str(0.00).rjust(5)) # B因子
        fi.write('           ')
        fi.write(atom[0:1].rjust(1)) # 元素符號
        fi.write('  ')
        fi.write('\n')


# 原子 + 二級結構 + 氨基酸三者共同預測的 CA 提取
def extract_ca_from_atom_amino_sec_common_only(pred_atom, pred_amino, pred_sec ,outfilename, outfilename_pdb, density_shape, density_voxel, density_origin, origin):
    
    # get common from atom, sec and amino pred
    # 初始化空間體素數據和索引映射字典
    data = np.zeros(density_shape, dtype=np.float32)
    atom_idx = dict() # 存儲原子概率
    atom_sec_idx = dict() # 存儲原子+二級結構概率
    count = 0
    key_err = 0
    idx_err = 0
    idx_no_err = 0
    
    x_origin = density_origin[0]
    y_origin = density_origin[1]
    z_origin = density_origin[2]
    
    x_voxel = density_voxel[0]
    y_voxel = density_voxel[1]
    z_voxel = density_voxel[2]

    # 讀取原子概率，記錄到字典
    with open(pred_atom, 'r') as atom_prob:
        for line in atom_prob:
            line_a = ast.literal_eval(line)
            ca_prob = line_a[2] # CA 原子概率
            ca_cords = line_a[0]  # 從檔案中提取座標
            x, y, z = ca_cords[0], ca_cords[1], ca_cords[2] # 實際坐標
            # 體素索引
            iz = int(get_index(z, z_origin, z_voxel))
            jy = int(get_index(y, y_origin, y_voxel))
            kx = int(get_index(x, x_origin, x_voxel))
            atom_idx[iz,jy,kx] = ca_prob

     # 讀取二級結構概率，與原子概率做乘積融合
    with open(pred_sec, 'r') as sec_prob:
        for line in sec_prob:
            line_a = ast.literal_eval(line)
            ca_prob = 1 - line_a[1] # 取非 helix 段作 CA 概率
            ca_cords = line_a[0]
            x, y, z = ca_cords[0], ca_cords[1], ca_cords[2]
            iz = int(get_index(z, z_origin, z_voxel))
            jy = int(get_index(y, y_origin, y_voxel))
            kx = int(get_index(x, x_origin, x_voxel))
            try:
                # 原子概率 与 二級結構概率 做平方根乘積
                atom_idx[iz,jy,kx] = np.sqrt(atom_idx[iz,jy,kx] * ca_prob)
                atom_sec_idx[iz,jy,kx] = atom_idx[iz,jy,kx]
            except KeyError:
                key_err += 1

    # 讀取氨基酸概率，與前面融合結果再做融合並輸出
    with open(pred_amino, 'r') as amino_prob:
        for line in amino_prob:
            line_a = ast.literal_eval(line)
            ca_prob = 1 - line_a[1] # 非第2氨基酸概率
            ca_cords = line_a[0]
            x, y, z = ca_cords[0], ca_cords[1], ca_cords[2]
            iz = int(get_index(z, z_origin, z_voxel))
            jy = int(get_index(y, y_origin, y_voxel))
            kx = int(get_index(x, x_origin, x_voxel))
            try:
                atom_sec_idx[iz,jy,kx] = np.sqrt(atom_sec_idx[iz,jy,kx] * ca_prob)
                data[iz,jy,kx] = atom_sec_idx[iz,jy,kx] 
                # 同時寫入 PDB ATOM 記錄
                save(x=x, y=y, z=z, count=count, save_path=outfilename_pdb)
                count += 1
                idx_no_err += 1
            except KeyError:
                key_err += 1
                idx_err += 1
                
    # 保存為 MRC 文件
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = x_voxel
        mrc.header.origin = origin
        mrc.close()


    print("####################################################")
    print("Atom_Sec_Amino index error", idx_err)
    print("Atom_Sec_Amino NO index error", idx_no_err)
    print("Number of common carbon alphas", idx_no_err)

# 原子 + 氨基酸共同預測的 CA 提取
def extract_ca_from_atom_amino_common_only(pred_atom, pred_amino, outfilename, outfilename_pdb, density_shape, density_voxel, density_origin, origin):
    data = np.zeros(density_shape, dtype=np.float32)
    atom_idx = dict()
    count = 0
    key_err = 0
    idx_err = 0
    idx_no_err = 0
    x_origin = density_origin[0]
    y_origin = density_origin[1]
    z_origin = density_origin[2]
    x_voxel = density_voxel[0]
    y_voxel = density_voxel[1]
    z_voxel = density_voxel[2]

    with open(pred_atom, 'r') as atom_prob:
        for line in atom_prob:
            line_a = ast.literal_eval(line)
            ca_prob = line_a[2]
            ca_cords = line_a[0]
            x, y, z = ca_cords[0], ca_cords[1], ca_cords[2]
            iz = int(get_index(z, z_origin, z_voxel))
            jy = int(get_index(y, y_origin, y_voxel))
            kx = int(get_index(x, x_origin, x_voxel))
            atom_idx[iz,jy,kx] = ca_prob

    with open(pred_amino, 'r') as amino_prob:
        for line in amino_prob:
            line_a = ast.literal_eval(line)
            ca_prob = 1 - line_a[1]
            ca_cords = line_a[0]
            x, y, z = ca_cords[0], ca_cords[1], ca_cords[2]
            iz = int(get_index(z, z_origin, z_voxel))
            jy = int(get_index(y, y_origin, y_voxel))
            kx = int(get_index(x, x_origin, x_voxel))
            try:
                # atom_idx[iz,jy,kx] = np.sqrt(atom_idx[iz,jy,kx] * ca_prob)
                a = np.sqrt(atom_idx[iz,jy,kx] * ca_prob)
                data[iz,jy,kx] = atom_idx[iz,jy,kx] 
                save(x=x, y=y, z=z, count=count, save_path=outfilename_pdb)
                count += 1
                idx_no_err += 1
            except KeyError:
                key_err += 1
                idx_err += 1
            except IndexError:
                pass

    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = x_voxel
        mrc.header.origin = origin
        mrc.close()

    print("####################################################")
    print("Atom_Amino index error", idx_err)
    print("Atom_Amino NO index error", idx_no_err)
    print("Number of common carbon alphas", idx_no_err)

# 氨基酸預測的 CA 提取
def extract_ca_from_amino(pred_amino, outfilename, outfilename_pdb, density_shape, density_voxel, density_origin, origin):
    # use this for data visualization, only ca from amino prediction
    data = np.zeros(density_shape, dtype=np.float32)
    count = 0
    idx_err = 0
    idx_no_err = 0
    x_origin = density_origin[0]
    y_origin = density_origin[1]
    z_origin = density_origin[2]
    x_voxel = density_voxel[0]
    y_voxel = density_voxel[1]
    z_voxel = density_voxel[2]
    with open(pred_amino, 'r') as atom_prob:
        for line in atom_prob:
            line_a = ast.literal_eval(line)
            ca_prob = 1 - line_a[1]
            ca_cords = line_a[0]
            x, y, z = ca_cords[0], ca_cords[1], ca_cords[2]
            save(x=x, y=y, z=z, count=count, save_path=outfilename_pdb)
            count += 1
            iz = int(get_index(z, z_origin, z_voxel))
            jy = int(get_index(y, y_origin, y_voxel))
            kx = int(get_index(x, x_origin, x_voxel))
            try:
                data[iz,jy,kx] = ca_prob  
                idx_no_err += 1  
            except:
                idx_err += 1

    print("####################################################")
    print("Amino index error", idx_err)
    print("Amino NO index error", idx_no_err)


    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = x_voxel
        mrc.header.origin = origin
        mrc.close()

# 原子預測的 CA 提取
# 最後在主程式使用這段
def extract_ca_from_atom(pred_atom, outfilename, outfilename_pdb, density_shape, density_voxel, density_origin, origin):
    # use this for data visualization, only ca from atom prediction
    data = np.zeros(density_shape, dtype=np.float32)
    idx_err = 0
    idx_no_err = 0
    count = 0
    x_origin = density_origin[0]
    y_origin = density_origin[1]
    z_origin = density_origin[2]
    x_voxel = density_voxel[0]
    y_voxel = density_voxel[1]
    z_voxel = density_voxel[2]
    
    with open(pred_atom, 'r') as atom_prob:
        for line in atom_prob:
            line_a = ast.literal_eval(line)
            ca_prob = line_a[2] # CA 原子概率
            ca_cords = line_a[0] # 從檔案中提取座標
            x, y, z = ca_cords[0], ca_cords[1], ca_cords[2] # 實際坐標
            # 同時寫入 PDB ATOM 記錄
            save(x=x, y=y, z=z, count=count, save_path=outfilename_pdb)
            count += 1
            # 體素索引
            iz = int(get_index(z, z_origin, z_voxel))
            jy = int(get_index(y, y_origin, y_voxel))
            kx = int(get_index(x, x_origin, x_voxel))
            try:
                data[iz,jy,kx] = ca_prob  
                idx_no_err += 1  
            except:
                idx_err += 1
    
    print("####################################################")
    print("Atom index error", idx_err)
    print("Atom NO index error", idx_no_err)
    
    # 保存為 MRC 文件
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = x_voxel
        mrc.header.origin = origin
        mrc.close()