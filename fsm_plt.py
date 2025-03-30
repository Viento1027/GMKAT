from create_dataset_from_tiff import *
import matplotlib.pyplot as plt
import numpy as np


def rewrite_geotiff(origin_path, save_path, dataset):
    # 读取GeoTIFF文件
    origin_file = gdal.Open(origin_path, gdal.GA_ReadOnly)
    if origin_file is None:
        print("文件不存在或无法打开。")
        return False
    # 获取原始数据信息
    cols = origin_file.RasterXSize
    rows = origin_file.RasterYSize
    bands = origin_file.RasterCount
    # 获取地理坐标信息
    geotransform = origin_file.GetGeoTransform()
    x_origin = geotransform[0]
    y_origin = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    # 获取投影信息
    projection = origin_file.GetProjection()
    srs = osr.SpatialReference(wkt=projection)
    # 创建新的GeoTIFF文件
    driver = gdal.GetDriverByName('GTiff')
    new_tif = driver.Create(save_path, cols, rows, bands, gdal.GDT_Float32)
    # 将地理信息写入新文件
    new_tif.SetGeoTransform((x_origin, pixel_width, 0, y_origin, 0, pixel_height))
    new_tif.SetProjection(srs.ExportToWkt())
    # 将新数据写入新文件
    new_tif.GetRasterBand(1).WriteArray(dataset)
    print("新的GeoTIFF文件已保存。")


gis_map = read_tif(".\\data\\normalized_gis_data.tif").transpose(1, 2, 0)

prediction_dense = np.loadtxt('.\\Flood_susceptibility_map\\GMTKAN_prediction.txt')

savetif_path = '.\\fsm_map_resnet.tif'
original = '.\\data\\降雨.tif'

rewrite_geotiff(original, savetif_path, prediction_dense)
# 绘制
x = np.arange(5108)  # x轴坐标
y = np.arange(3755)  # y轴坐标
X, Y = np.meshgrid(x, y)
prediction_dense[prediction_dense == -1] = np.nan
plt.figure(figsize=(6, 6))
plt.pcolormesh(prediction_dense, cmap='RdYlGn_r', shading='auto', vmin=0, vmax=1)
plt.gca().invert_yaxis()
plt.colorbar(label='probability')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Flood Susceptibility Map')
plt.savefig("FSM.png")
