from osgeo import gdal, osr
import numpy as np
import random
import os

random.seed(1027)

data_path = {
    'dem': '.\\data\\dem.tif',
    'aspect': '.\\data\\aspect.tif',
    'curvature': '.\\data\\curvature.tif',
    'slope': '.\\data\\slope.tif',
    'twi': '.\\data\\TWI.tif',
    'ndvi': '.\\data\\ndvi.tif',
    'distance_to_rivers': '.\\data\\distance_to_rivers.tif',
    'land_cover': '.\\data\\Land_Cover.tif',
    'rainfall': '.\\data\\rainfall.tif',
    'flood': '.\\data\\flooded.tif'

}


# 读取tif文件数据
def read_tif(filename):
    dataset = gdal.Open(filename)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset
    return im_data


# 读取tif文件信息
def read_tif_info(filename):
    dataset = gdal.Open(filename)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
    im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
    del dataset
    return im_bands, im_width, im_height


# 保存为tif文件
def write_tif(filename, im_data):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_height, im_width, im_bands = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    # dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    # dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[:, :, i])

    del dataset


# 数据读取合并
def data_concat(path_dict):
    complete_data = np.zeros(shape=(5108, 3755, 0))
    for key, value in path_dict.items():
        print(key, '=>', value)
        # if str(key) == 'land_use':
        #     data_land_use = np.where(read_img(value) <= -128, 0, read_img(value))
        #     data_land_use = (data_land_use / 10).astype(int)
        #     print('one-hot finished!')
        # else:
        locals()['data_{0}'.format(key)] = np.expand_dims(np.where(read_tif(value) <= -128, 0, read_tif(value)), axis=2)
        complete_data = np.append(complete_data, locals()['data_{0}'.format(key)], axis=2)
        print(complete_data.shape)
        # print(np.min(locals()['data_{0}'.format(key)]), np.max(locals()['data_{0}'.format(key)]))
    return complete_data


def pondingdata_cut(data, chan, l_size, r_size, save_dir):
    data_num = 0
    height, width = data.shape[0], data.shape[1]

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(height):
        for j in range(width):
            # 判断当前点是否为积水点且有 20% 概率被选中
            if (data[i, j, chan - 1] == 1) and (random.random() < 0.2):
                # 边界检查，确保裁剪区域不超出图像边界
                if (i - l_size >= 0) and (i + r_size < height) and (j - l_size >= 0) and (j + r_size < width):
                    # 裁剪区域
                    ponding_data = data[i - l_size:i + r_size, j - l_size:j + r_size, 0:chan - 1]

                    # 保存裁剪的数据
                    path = os.path.join(save_dir, f'pondingdata_{data_num}.tif')
                    write_tif(path, ponding_data)

                    data_num += 1  # 更新裁剪数据数量

                    # 每裁剪 10000 个积水点时打印进度
                    if (data_num % 10000) == 0:
                        print("已切割：", data_num)

    # 最终结果
    print('共切割积水点数据：', data_num)
    return data_num


def nonpondingdata_cut(data, chan, num, l_size, r_size, save_dir):
    i = 0
    height, width = data.shape[0], data.shape[1]

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    while i < num:
        inside = int(l_size * r_size * 2)  # 初始化内部统计变量
        far = True  # 用于判断是否远离积水点

        # 随机生成裁剪的中心点坐标
        x = random.randint(l_size, height - r_size)
        y = random.randint(l_size, width - r_size)

        # 检查当前点是否为非积水点
        if data[x, y, chan - 1] == 0:
            # 检查附近是否有积水点
            for j in range(x - 4, x + 5):
                for k in range(y - 4, y + 5):
                    if data[j, k, 1] == 0:
                        inside -= 1  # 非积水点数减1
                    if data[j, k, chan - 1] == 1:
                        far = False  # 如果附近有积水点，则标记为不远离

        # 如果该区域内有足够的非积水点且距离积水点足够远
        if inside > 0 and far:
            # 裁剪非积水点数据
            nonponding_data = data[x - l_size:x + r_size, y - l_size:y + r_size, 0:chan - 1]

            # 保存裁剪的数据
            path = os.path.join(save_dir, f'nonpondingdata_{i}.tif')
            write_tif(path, nonponding_data)

            i += 1  # 更新裁剪计数

            # 每裁剪 10000 个非积水点时打印进度
            if i % 10000 == 0:
                print(f"已切割：{i}")

    # 打印最终裁剪数量
    print(f'共切割非积水点数据：{num}')


# 有效数据归一化
def normalize_valid(matrix):
    # 创建一个掩码，标记无效数值所在的位置
    mask = (matrix[:, :, -2] != 0)
    # 分离最后一个通道
    last_channel = matrix[:, :, -1:]
    other_channels = matrix[:, :, :-1]
    # 提取有效数值的部分
    valid_values = matrix[mask]
    valid_values = valid_values[:, :-1]
    # 计算有效数值部分的均值和标准差
    mean = np.mean(valid_values, axis=0, keepdims=True)
    std = np.std(valid_values, axis=0, keepdims=True)
    # 对有效数值进行归一化操作
    normalized_other_channels = (other_channels - mean) / (std + 1e-7)
    # 合并归一化的通道和未归一化的最后一个通道
    normalized_matrix = np.concatenate((normalized_other_channels, last_channel), axis=-1)
    # 将范围外数据重新赋为0
    for x in range(normalized_matrix.shape[0]):
        for y in range(normalized_matrix.shape[1]):
            if ~mask[x, y]:
                normalized_matrix[x, y, :-1] = 0

    return normalized_matrix


if __name__ == '__main__':
    gis_data = data_concat(data_path)
    normalized_gis_data = normalize_valid(gis_data)
    ponding_num = pondingdata_cut(normalized_gis_data, normalized_gis_data.shape[2], 8, 8,
                                  '.\data\\flood_16\\1\\')
    nonpondingdata_cut(normalized_gis_data, normalized_gis_data.shape[2], ponding_num, 8, 8,
                       '.\\data\\flood_16\\0\\')
    write_tif(".\\data\\normalized_gis_data.tif", normalized_gis_data)
    print("Finished!")
