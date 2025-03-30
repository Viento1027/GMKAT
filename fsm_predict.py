import torch
import numpy as np
import time
import os
from create_dataset_from_tiff import read_tif, map_cut
from GMKAT import gmkat_base as create_model_gmt

# 开始计时
start_time = time.time()

gis_map = read_tif("G:\\FSM\\data\\normalized_gis_data.tif").transpose(1, 2, 0)
valid = []
for x in range(gis_map.shape[0]):
    for y in range(gis_map.shape[1]):
        # 遍历，在区域内，记录
        if gis_map[x, y, -2] != 0:
            valid.append((x, y))
    if len(valid) > 10000:
        break
valid = np.array(valid)
print("预测样本数量：", valid.shape[0])

model = create_model_gmt()
model.load_state_dict(torch.load("./results/GMTKAN/GMTKAN_weights.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(torch.cuda.is_available())

model.eval()
prediction_dense = np.full((5108, 3755), -1.0)
block_size = 1000
count = 0

for i in range(0, len(valid), block_size):
    block = valid[i:i + block_size]
    areas = []
    for x, y in block:
        area = map_cut(gis_map, x, y, 16)
        areas.append(area)
    areas = np.array(areas).transpose(0, 3, 1, 2)
    areas = torch.from_numpy(areas).float()
    areas = areas.to(device)
    with torch.no_grad():
        probs = model(areas)
    for j, (x, y) in enumerate(block):
        prediction_dense[x, y] = probs[j][0].item()
    count += 1
    print("Batch：", count)
    current_time = time.time()
    elapsed_time = current_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f"程序已运行时间：{hours}小时{minutes}分{seconds:.2f}秒")
print("Finish predict")
save_path = os.path.join(os.getcwd(), 'Flood susceptibility map/GMTKAN_prediction.txt')
np.savetxt(save_path, prediction_dense)
print("Saved!")

# 结束计时
end_time = time.time()
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = elapsed_time % 60
print(f"程序总运行时间：{hours}小时{minutes}分{seconds:.2f}秒")
