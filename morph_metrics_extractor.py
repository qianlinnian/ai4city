# ==========================================
# 🌟 1. 强行修复 NumPy 别名兼容性问题（保持置顶）
import numpy as np
np.bool = np.bool_
# ==========================================

import os
import cv2
import mxnet as mx
from mxnet import image
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ----------------- 2. 配置与环境初始化 -----------------
IMAGE_DIR = r"./resized_images"  # 存放你 2048*1024 街景图片的路径
OUTPUT_DIR = r"./output_results"  # 处理结果根目录

# 自动创建 3 个指定的输出存储子文件夹
SEG_DIR = os.path.join(OUTPUT_DIR, "segmentation_results")  # 语义分割图
EDGE_DIR = os.path.join(OUTPUT_DIR, "edge_density_maps")    # 计算边缘密度的边缘图
SKYLINE_DIR = os.path.join(OUTPUT_DIR, "skyline_boundary_maps")  # 计算天际线变化率的边界图

for path in [SEG_DIR, EDGE_DIR, SKYLINE_DIR]:
    os.makedirs(path, exist_ok=True)

# 设定使用 CPU
ctx = mx.cpu()

print("正在载入预训练语义分割模型...")
model = gluoncv.model_zoo.get_model('deeplab_resnet101_citys', pretrained=True, ctx=ctx)
print("模型载入成功！")


# ----------------- 3. 各指标计算函数定义 -----------------

def get_segmentation_and_save(img_path, img_name):
    """
    进行语义分割，保存分割结果图，并返回分割分类矩阵(H, W)
    """
    img = image.imread(img_path)
    img_transformed = test_transform(img, ctx=ctx)
    output = model.predict(img_transformed)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy().astype(np.uint8)
    
    # 自动上色并保存语义分割结果图
    pred_palette = get_color_pallete(predict, 'citys')
    # get_color_pallete 返回 Pillow 的调色板模式（P）图像；JPEG 不支持直接
    # 保存 P 模式，因此统一转成 RGB，避免 "cannot write mode P as JPEG"。
    pred_palette.convert("RGB").save(os.path.join(SEG_DIR, f"seg_{img_name}"))
    
    return predict


def calculate_color_richness(img_path):
    """
    指标 5：色彩丰富度 (Color Richness, CR) 
    """
    img_bgr = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    
    # 过滤 S < 0.1 (OpenCV 中 S 范围 0-255，故 0.1 * 255 ≈ 25.5)
    valid_mask = s >= 25
    total_valid_pixels = np.sum(valid_mask)
    
    if total_valid_pixels == 0:
        return 0.0
        
    valid_h = h[valid_mask]
    K = 24
    bin_edges = np.linspace(0, 180, K + 1)
    hist, _ = np.histogram(valid_h, bins=bin_edges)
    p_k = hist / total_valid_pixels
    
    tau = 0.005
    N_color = np.sum(p_k >= tau)
    cr = N_color / K
    return float(cr)


def calculate_edge_density_and_save(img_path, img_name):
    """
    指标 6：边缘密度 (Edge Density, ED)
    """
    img_bgr = cv2.imread(img_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 保存边缘图
    cv2.imwrite(os.path.join(EDGE_DIR, f"edge_{img_name}"), edges)
    
    n_edge = np.sum(edges > 0)
    n_valid = edges.size  
    edge_density = n_edge / n_valid
    return float(edge_density)


def calculate_skyline_and_save(predict, img_path, img_name):
    """
    指标 7：天际线变化率 (Skyline Variation Rate, SVR)
    """
    sky_mask = (predict == 10)
    H, W = predict.shape
    y_coords = np.zeros(W, dtype=np.int32)
    
    for x in range(W):
        col = sky_mask[:, x]
        sky_indices = np.where(col)[0]
        if len(sky_indices) > 0:
            y_coords[x] = sky_indices[-1] # 取该列天空最底部的 y 坐标作为边界
        else:
            y_coords[x] = 0
            
    diff_sum = np.sum(np.abs(np.diff(y_coords)))
    svr = (diff_sum / ((W - 1) * H)) * 100.0
    
    # 在原图上画出天际线并保存
    img_bgr = cv2.imread(img_path)
    for x in range(W - 1):
        pt1 = (x, y_coords[x])
        pt2 = (x + 1, y_coords[x+1])
        cv2.line(img_bgr, pt1, pt2, (0, 0, 255), 3) 
        
    cv2.imwrite(os.path.join(SKYLINE_DIR, f"skyline_{img_name}"), img_bgr)
    return float(svr)


# ----------------- 4. 批处理主循环流程 -----------------

def process_pipeline():
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 错误：配置的图像文件夹 {IMAGE_DIR} 不存在，请检查！")
        return
        
    img_names = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if len(img_names) == 0:
        print(f"⚠️ 警告：没有在 {IMAGE_DIR} 中找到有效图片。")
        return

    # 存储最终指标的记录列表
    records = []

    # 🌟 完整保留的 19 类分割标签字典
    col_map = {
        0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light',
        7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider',
        13: 'car', 14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
    }

    print(f"\n🚀 开始处理全景街景图片，共找到 {len(img_names)} 张图像...")
    
    for img_name in tqdm(img_names, desc="街景图像多指标提取中"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        try:
            # 1. 获取分割预测矩阵
            predict = get_segmentation_and_save(img_path, img_name)
            total_pixels = predict.size
            
            # 2. 🌟 核心保留：通过字典计算原始 19 类别的独立占比
            raw_seg_results = {}
            for class_id, class_name in col_map.items():
                pixel_count = np.sum(predict == class_id)
                raw_seg_results[class_name] = pixel_count / total_pixels
            
            # 3. 依据 19 类占比，精确计算前 4 项形态指标
            # [指标 1] 绿视率 = vegetation + terrain
            green_rate = raw_seg_results['vegetation'] + raw_seg_results['terrain']
            
            # [指标 2] 蓝视率 (Cityscapes默认无水体分类，保留 0.0)
            blue_rate = 0.0 
            
            # [指标 3] 人造物占比 (道路+人行道+建筑+墙+篱笆+杆+红绿灯+指示牌+所有交通工具)
            built_rate = (
                raw_seg_results['building'] + raw_seg_results['road'] + raw_seg_results['sidewalk'] +
                raw_seg_results['wall'] + raw_seg_results['fence'] + raw_seg_results['pole'] +
                raw_seg_results['traffic light'] + raw_seg_results['traffic sign'] +
                raw_seg_results['car'] + raw_seg_results['truck'] + raw_seg_results['bus'] +
                raw_seg_results['train'] + raw_seg_results['motorcycle'] + raw_seg_results['bicycle']
            )
            
            # [指标 4] 天空可视率
            sky_rate = raw_seg_results['sky']
            
            # [指标 5] 色彩丰富度 (自定义 HSV 算法)
            color_richness = calculate_color_richness(img_path)
            
            # [指标 6] 边缘密度 (Canny 算法提取)
            edge_density = calculate_edge_density_and_save(img_path, img_name)
            
            # [指标 7] 天际线变化率
            svr = calculate_skyline_and_save(predict, img_path, img_name)
            
            # 4. 整合 19 类详细占比 + 7 项核心评估指标
            record = {
                "图像名称": img_name,
                # --- 7项综合指标 ---
                "综合-绿视率(GVI)": f"{green_rate:.4%}",
                "综合-蓝视率(BVI)": f"{blue_rate:.4%}",
                "综合-人造物占比": f"{built_rate:.4%}",
                "综合-天空可视率": f"{sky_rate:.4%}",
                "综合-色彩丰富度(CR)": f"{color_richness:.4f}",
                "综合-边缘密度(ED)": f"{edge_density:.4%}",
                "综合-天际线变化率(SVR)": f"{svr:.4f}%"
            }
            
            # --- 19类原始分割细分指标（格式化为百分比存入） ---
            for class_name, ratio in raw_seg_results.items():
                record[f"原始-{class_name}"] = f"{ratio:.4%}"
                
            records.append(record)
            
        except Exception as e:
            print(f"\n❌ 图片 {img_name} 处理出错，跳过。错误原因: {e}")
            continue

    # ----------------- 5. 输出写入 Excel -----------------
    df_results = pd.DataFrame(records)
    excel_output_path = os.path.join(OUTPUT_DIR, "metrics_results.xlsx")
    df_results.to_excel(excel_output_path, index=False)
    
    print("\n" + "="*50)
    print("🎉 处理完成！")
    print(f"📊 包含 19类细分占比 与 7项指标 的 Excel 结果已保存至: {excel_output_path}")
    print(f"📂 分割结果图、边缘密度图、天际线图分别存储在 {OUTPUT_DIR} 下。")
    print("="*50)

if __name__ == "__main__":
    process_pipeline()
