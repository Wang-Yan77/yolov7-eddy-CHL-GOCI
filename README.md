# Submesoscale Eddy Identification from GOCI-I Chlorophyll-a (YOLOv7)

本仓库是论文配套代码：

> Wang, Y., Chen, G., Yang, J., Gui, Z., and Peng, D.: *A submesoscale eddy identification dataset in the northwest Pacific Ocean derived from GOCI I chlorophyll a data based on deep learning*, Earth Syst. Sci. Data, 16, 5737–5752, 2024. https://doi.org/10.5194/essd-16-5737-2024

该项目基于 YOLOv7 对 GOCI-I 叶绿素浓度（chlorophyll-a）影像中的**次中尺度涡旋**进行目标检测，支持：

- 训练（train）
- 单图/视频推理（predict）
- 批量切片推理并输出逐日 JSON（dir_predict）
- mAP 评估（get_map）
- 先验框聚类（kmeans_for_anchors）

---

## 1. 项目结构

```text
.
├── train.py                     # 训练入口
├── predict.py                   # 推理入口（多模式）
├── get_map.py                   # VOC mAP 评估
├── kmeans_for_anchors.py        # anchors 聚类
├── yolo.py                      # 推理模型封装
├── nets/                        # 网络结构与损失
├── utils/                       # 数据加载、训练、后处理等工具
├── Preprocessing_training_set/  # 数据集预处理脚本
├── model_data/                  # 类别、anchors、权重、字体等
└── img/                         # 示例输入
```

---

## 2. 环境要求

建议 Python 3.8+。

安装依赖：

```bash
pip install -r requirements.txt
```

> 说明：
> - 训练/推理默认使用 PyTorch + CUDA；无 GPU 时可在配置中关闭 CUDA。
> - `predict.py` 的目录批处理模式依赖 `torchvision.ops.nms`。

---

## 3. 数据准备（训练）

当前训练流程采用 VOC 风格标注组织方式（脚本中可见 `VOCdevkit` 路径约定）：

1. 准备图像与标注（XML）。
2. 生成训练/验证划分文本（如 `2007_train.txt`, `2007_val.txt`）。
3. 在 `model_data/eddy.txt` 中设置类别（例如 `AE`, `CE`）。

如果需要，可使用 `Preprocessing_training_set/` 中脚本做切片、增强、VOC 列表生成等预处理。

---

## 4. 如何运行

### 4.1 训练

```bash
python train.py
```

训练前请重点检查 `train.py` 内参数：

- `model_path`：预训练权重路径
- `classes_path`：类别文件路径
- `input_shape`：输入分辨率
- `batch_size`、`Freeze_Train`、`UnFreeze_Epoch`
- 数据集根目录（VOC 路径）

---

### 4.2 推理

```bash
python predict.py
```

在 `predict.py` 中通过 `mode` 切换功能：

- `predict`：单图
- `video`：视频/摄像头
- `fps`：测速
- `dir_predict`：目录批量推理（逐日融合切片结果并输出 JSON）
- `heatmap`：热力图
- `export_onnx`：导出 ONNX

> `dir_predict` 模式下，需要根据你的本地数据目录修改按小时组织的输入输出路径。

---

### 4.3 mAP 评估

```bash
python get_map.py
```

根据脚本注释配置：

- GT 与预测结果路径
- 评估阈值（如 IoU）
- 是否绘图/可视化

---

## 5. 代码改进说明（本次整理）

本次对推理脚本与说明文档进行了工程化整理，核心目标是提高可维护性和可复现性：

1. **`predict.py` 结构优化**
   - 将地理网格加载、像素投影、日期遍历、目录推理拆分成独立函数。
   - 修复切片行列索引比较中的类型不一致问题（`int` vs `str`）。
   - 使用 `os.path` 统一路径处理，避免重复拼接或平台依赖问题。
   - 增加函数级注释，明确输入输出含义。

2. **README 重写**
   - 增补了项目背景、功能、目录结构、环境、训练与推理步骤。
   - 明确了 `dir_predict` 的数据组织要求，便于论文代码复现。

---

## 6. 输出结果说明（`dir_predict`）

逐日 JSON 中包含：

- 日期
- AE/CE 数量
- 每个检测框的像素坐标
- 涡旋中心经纬度
- 框角点经纬度
- 内接半径与椭圆面积估计
- 置信度

这与论文中构建涡旋识别数据集的产品化流程保持一致。

---

## 7. 引用

如果你使用了本仓库或其衍生数据/方法，请引用：

```bibtex
@article{wang2024submesoscale,
  author  = {Wang, Y. and Chen, G. and Yang, J. and Gui, Z. and Peng, D.},
  title   = {A submesoscale eddy identification dataset in the northwest Pacific Ocean derived from GOCI I chlorophyll a data based on deep learning},
  journal = {Earth System Science Data},
  volume  = {16},
  pages   = {5737--5752},
  year    = {2024},
  doi     = {10.5194/essd-16-5737-2024}
}
```

---

## 8. License

See `LICENSE` for details.
