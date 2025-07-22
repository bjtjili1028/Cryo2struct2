import numpy as np
import mrcfile
import os
import math
from copy import deepcopy
import torch
import esm
from pathlib import Path

box_size = 32  # Expected Dimensions to pass to Transformer Unet
core_size = 20  # core of the image where we dnt have to worry about boundry issues

# v2 新增的
# 設定 Torch 模型(esm模型)快取目錄
# os.environ["TORCH_HOME"] = "/esm"
os.environ["TORCH_HOME"] = "/media/ray-suen/TRANSCEND1/huei/Cryo2Struct2/esm"

# v2 新增的
def chain_merger_2(density_map, fasta_name):
    """
    合併多條鏈的 FASTA 序列：讀取 atomic.fasta，
    將每條鏈根據管道分割資訊重複並拼接到一個檔案中
    """
    input_file = f'{density_map}/atomic.fasta'

    output_file = f'{density_map}/{fasta_name}_all_chain_combined.fasta'

    # if os.path.exists(output_file):
        # os.remove(output_file)

    with open(input_file, "r") as input_fp, open(output_file, "w") as output_fp:
        merge_lines = [] # 收集序列內容行
        repeat_count = 1 # 重複次數

        for line in input_fp:
            if line.startswith(">"): # 每當遇到序列標題行
                if merge_lines:
                    # 將上一段序列複製多次寫入檔案
                    merged_line = "".join(merge_lines) * repeat_count
                    output_fp.write(merged_line)
                    merge_lines = []
                    repeat_count = 1

                chain_info = line.split("|")
                if len(chain_info) > 1:
                    # 標題中第二段管道分隔表示鏈次數
                    chain_data = chain_info[1]
                    repeat_count = len(chain_data.split(","))
            else:
                # 收集序列行（去除換行符
                merge_lines.append(line.strip())
        
        # 處理最後一段序列
        if merge_lines:
            merged_line = "".join(merge_lines) * repeat_count
            output_fp.write(merged_line)


# v2 新增的
def generate_esm_embeddings(sequence, save_path):
    """
    使用 ESM2 模型計算蛋白質序列的嵌入，並將結果儲存到指定檔案
    """
    # Load the model and save it to the specified directory
    # 載入 ESM2 模型與字母表
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D() # 原始模型
    # model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()  # 內存不夠版
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    # Load ESM-2 model
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results # 關閉 dropout，確保結果可重現

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    # 準備單序列資料
    data = [
        ("protein1", sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # 計算序列長度（排除 padding）
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    # 推論階段：取得第36層的 token representations # 這個部分要根據模型進行修正
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36], return_contacts=False)
    token_representations = results["representations"][36]
    
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    # 對每條序列做平均池化，得到整序列表示
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    # 將每個維度值逐行輸出到檔案
    with open(save_path, 'w') as file:
        for seq_res in sequence_representations[0]:
            file.write(str(float(seq_res)) + '\n')


# 創建清單，將大圖像切割成小塊，每個小塊傳遞給Transformer Unet進行處理
def create_manifest(full_image):
    """
    將整張 3D 密度圖切分成多個小立方塊（維度 box_size），
    回傳一個列表，供 Transformer Unet 逐塊推論
    """
    # creates a list of box_size tensors. Each tensor is passed to Transformer Unet independently
    image_shape = np.shape(full_image)
    
    # 在三維邊緣補 0 填充，以完整包含所有區塊
    padded_image = np.zeros(
        (image_shape[0] + 2 * box_size, image_shape[1] + 2 * box_size, image_shape[2] + 2 * box_size))
    padded_image[box_size:box_size + image_shape[0], box_size:box_size + image_shape[1],
    box_size:box_size + image_shape[2]] = full_image
    manifest = list()

    # 計算起始點，使核心區塊位於 padded 區域中心
    start_point = box_size - int((box_size - core_size) / 2)
    cur_x = start_point
    cur_y = start_point
    cur_z = start_point
    
    # 三層迴圈：依 core_size 步長切割所有小區塊
    while cur_z + (box_size - core_size) / 2 < image_shape[2] + box_size:
        next_chunk = padded_image[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        manifest.append(next_chunk)
        cur_x += core_size
        if cur_x + (box_size - core_size) / 2 >= image_shape[0] + box_size:
            cur_y += core_size
            cur_x = start_point  # Reset
            if cur_y + (box_size - core_size) / 2 >= image_shape[1] + box_size:
                cur_z += core_size
                cur_y = start_point  # Reset
                cur_x = start_point  # Reset
    return manifest

# 從密度圖目錄中讀取處理後的圖像數據
def get_data(density_map_dir):
    """
    從指定目錄讀取標準化過的 MRC 密度圖，
    並生成切塊 manifest
    """
    protein_manifest = None
    amino_manifest = None
    atom_manifest = None
    
    # 列出資料夾內所有檔案
    processed_maps = [m for m in os.listdir(density_map_dir)]
    for maps in range(len(processed_maps)):
        # 找到目標 MRC 檔
        os.chdir(density_map_dir)
        if processed_maps[maps] == "emd_normalized_map.mrc":
            p_map = mrcfile.open(processed_maps[maps], mode='r')
            protein_data = deepcopy(p_map.data)
            protein_manifest = create_manifest(protein_data)

    return protein_manifest

# 根據Transformer Unet的輸出重建完整的蛋白質密度圖
def reconstruct_map(manifest, image_shape):
    """
    根據模型輸出的 manifest 將小區塊重組回原始大小的 3D 密度圖
    """
    # takes the output of Transformer Unet and reconstructs the full dimension of the protein
    extract_start = int((box_size - core_size) / 2)
    extract_end = int((box_size - core_size) / 2) + core_size
    dimensions = get_manifest_dimensions(image_shape)

    reconstruct_image = np.zeros((dimensions[0], dimensions[1], dimensions[2]))
    counter = 0
    for z_steps in range(int(dimensions[2] / core_size)):
        for y_steps in range(int(dimensions[1] / core_size)):
            for x_steps in range(int(dimensions[0] / core_size)):
                reconstruct_image[x_steps * core_size:(x_steps + 1) * core_size,
                y_steps * core_size:(y_steps + 1) * core_size, z_steps * core_size:(z_steps + 1) * core_size] = \
                    manifest[counter][extract_start:extract_end, extract_start:extract_end,
                    extract_start:extract_end]
                counter += 1
    float_reconstruct_image = np.array(reconstruct_image, dtype=np.float32)
    float_reconstruct_image = float_reconstruct_image[:image_shape[0], :image_shape[1], :image_shape[2]]
    return float_reconstruct_image

# 根據原始圖像尺寸計算manifest的維度
def get_manifest_dimensions(image_shape):
    """
    計算按照 core_size 切割後，重建時所需的三個軸方向維度
    """
    dimensions = [0, 0, 0]
    dimensions[0] = math.ceil(image_shape[0] / core_size) * core_size
    dimensions[1] = math.ceil(image_shape[1] / core_size) * core_size
    dimensions[2] = math.ceil(image_shape[2] / core_size) * core_size
    return dimensions

# v2 新增的
def run_pdb2seq(pdb_file, perl_script_dir, atm_sequence):
    """
    呼叫外部 Perl 腳本，將 PDB 轉為序列 FASTA
    """
    os.system("perl " + perl_script_dir + " " + pdb_file + ">>" + atm_sequence)

# 創建子網格，並將每個小塊保存為壓縮的npz文件
def create_subgrids(input_data_dir, density_map_name):
    """
    整合整個流程：
      1. 找出目錄下的 .fasta 檔案
      2. 如果沒有 atomic.fasta，透過 run_pdb2seq 產生
      3. 生成 ESM 嵌入檔
      4. 讀取 MRC 密度圖，切分並儲存為 .npz
    """
    density_map_dir = os.path.join(input_data_dir,density_map_name)
    # 找到所有 .fasta 檔，並取第一個為 pdb_name
    pdb_files = [l for l in os.listdir(density_map_dir) if l.endswith(".fasta")]
    pdb_files.sort()
    pdb_name = pdb_files[0].split(".")[0]
    pdb_name = pdb_name.split("_")[0]
    pdb_name = pdb_name.lower()


    esm_embeddings = f"{density_map_dir}/atomic_esm_t36_3B_embeds.txt"

    # 如果已存在舊嵌入檔就刪除
    if os.path.exists(esm_embeddings):
        os.remove(esm_embeddings)

    if not os.path.isfile(esm_embeddings):
        print("Generating Embeddings Using: ", pdb_name)
        
        atm_sequence = f'{density_map_dir}/atomic.fasta'

        if not os.path.isfile(atm_sequence):
            perl_script= "./preprocess/pdb2seq.pl"
            perl_script_expand = os.path.abspath(perl_script)
            print(perl_script_expand)
            run_pdb2seq(pdb_file=f"{density_map_dir}/{pdb_name}.pdb",perl_script_dir=perl_script_expand, atm_sequence=atm_sequence)
        
        # 合併所有序列
        with open(atm_sequence,'r') as all_fasta:
            combined_sequence = all_fasta.read()

        # 產生並儲存 ESM 嵌入
        generate_esm_embeddings(sequence=combined_sequence, save_path=esm_embeddings)

    # esm_embeddings = f"{density_map_dir}/{pdb_name}_esm_t36_3B_embeds.txt"
    # 讀取嵌入數值
    with open(esm_embeddings, 'r') as esm_emb:
        embeds = [float(line.strip()) for line in esm_emb.readlines()]

    # 取得密度圖分塊 manifest
    protein = get_data(density_map_dir)
    if protein is not None:
        split_map_dir = os.path.join(density_map_dir, f"{density_map_name}_splits")
        os.makedirs(split_map_dir, exist_ok=True)
        for i in range(len(protein)):
            save_file_name = f'{split_map_dir}/{density_map_name}_{i}.npz'
            np.savez_compressed(file=save_file_name, protein_grid=protein[i], embeddings=embeds)
    else:
        print("There is no input map. Please check the input density map's directory")
        exit()



    # print("Done : ", density_map_name)

