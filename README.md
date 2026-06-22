# YOLO 标注、训练与推理一体化工具

本程序是基于 Qt/C++ 开发的桌面工具，用于完成目标检测数据标注、YOLO 模型训练、模型导出/量化、Python 推理和 ONNX Runtime C++ 推理。

## 1. 启动程序

已编译程序位于：

```text
build/Desktop_Qt_6_11_0_MSVC2022_64bit-Debug/YOLOAnnotator.exe
```

使用 Qt Creator 时：

1. 打开项目根目录的 `CMakeLists.txt`。
2. 选择 Qt 6.11.0 MSVC 2022 64-bit Kit。
3. 配置并构建项目。
4. 运行 `YOLOAnnotator`。

程序首次启动时会使用默认类别 `motorbike`。Python、输出目录、类别等设置会通过 `QSettings` 自动保存。

主窗口顶部菜单分为：

- `文件`：打开图片目录、选择标签目录、保存标注和退出。
- `编辑`：撤销、重新标注（清空当前图片全部框）和删除选中标注框。
- `工作区`：在“图片标注”“模型训练”“量化与推理”三个独立页面之间切换。

程序不再使用重复的第二行工具栏。标注类别、上一张、下一张和旋转等高频操作位于图片标注页面内部。

## 2. 标注使用流程

### 2.1 准备图片

支持以下图片格式：

```text
jpg、jpeg、png、bmp、tif、tiff、webp
```

点击“文件 → 打开图片目录”或工具栏“打开目录”，选择图片文件夹。右侧图片列表会显示目录内全部图片。

### 2.2 设置标签保存目录

点击“文件 → 选择标签目录”。

- 如果不选择，YOLO `.txt` 标签保存在图片同目录。
- 如果选择独立目录，全部标签保存到该目录。
- 图片与标签按不含扩展名的文件名对应，例如 `001.jpg` 对应 `001.txt`。

### 2.3 管理类别

- 点击“添加类别”，输入新类别名称。
- 在工具栏下拉框中选择当前标注类别。
- 选中已有框后切换下拉框，可修改该框的类别。
- 删除类别时不会重排其他类别 ID。
- 当前图片仍使用某个类别时，程序会阻止删除；应先把相关框修改为其他类别。

### 2.4 创建和编辑标注框

- 创建：在图片空白处按住鼠标左键拖动。
- 仅单击而没有拖动不会创建标注框；水平和垂直方向都至少拖动 4 个图像像素后才会创建。
- 选中：左键单击已有标注框。
- 移动：选中框后按住左键拖动。
- 缩放：拖动选中框四角或四边的白色控制点。
- 修改类别：选中框后切换类别下拉框。
- 删除：选中框后按 `Delete`，或使用“编辑 → 删除选中框”。
- 撤销：`Ctrl+Z`。
- 重做/重新标注：`Ctrl+Y`，确认后清空当前图片的全部标注框，以便从头标注；清空后可按 `Ctrl+Z` 恢复。

图片标注页面顶部同时提供“撤销”和“重做（清空全部框）”按钮。“重做”仅在当前图片存在标注框时可用，并在执行前显示确认提示。

标注框会被限制在图片范围内，过小的框不会创建。

### 2.5 浏览图片

- 工具栏“上一张”“下一张”。
- `A` 或方向键上：上一张。
- `S` 或方向键下：下一张。
- 鼠标滚轮：以鼠标位置为中心缩放。
- 鼠标右键拖动：平移画布。
- `R`：顺时针旋转视图 90°。

切换图片时会自动保存上一张图片的标注，也可以按 `Ctrl+S` 手动保存。

左侧“标注详情”会显示当前图片的所有框，并与画布选中状态同步。状态区会显示图片总数、已标注数和未标注数。

## 3. YOLO 数据集目录要求

推荐使用标准目录结构：

```text
dataset/
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

训练界面中应分别选择：

```text
训练图片目录：dataset/images/train
验证图片目录：dataset/images/val
```

程序会自动检查对应的 `dataset/labels/train` 和 `dataset/labels/val`，并在输出目录生成本次训练使用的 `data.yaml`。

也支持以下结构：

```text
train/
├─ images/
└─ labels/

val/
├─ images/
└─ labels/
```

## 4. 训练使用流程

### 4.1 配置运行环境

通过“工作区 → 模型训练”进入独立训练页面，然后填写：

- 训练图片目录：训练集图片目录。
- 验证图片目录：验证集图片目录。
- Python：安装了 PyTorch 和 Ultralytics 的 Python 解释器，例如 `python` 或具体的 `python.exe`。
- Ultralytics 源码：可选。如果使用已通过 pip 安装的 Ultralytics，保持为空；如果使用本地源码，选择源码根目录。
- 输出目录：保存训练 YAML、日志和模型结果的目录。

可先在终端验证环境：

```powershell
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

### 4.2 设置训练参数

- 模型：默认使用随程序部署的 `yolo26n.pt`，也可以填写其他 `.pt` 模型的完整路径。
- 设备：空值表示自动选择；CPU 可填 `cpu`；第一张 GPU 可填 `0`。
- Epochs：训练轮数。
- Batch：批量大小。
- Image size：输入图片尺寸，常用值为 `640`。
- Learning rate：初始学习率。
- 优化器：`auto`、`SGD`、`Adam` 或 `AdamW`。

### 4.3 开始训练

1. 确认训练集、验证集和标签目录正确。
2. 点击“开始训练”。
3. 日志区显示原始运行信息。
4. 进度条显示当前 epoch。
5. 训练结果摘要表显示结果目录、`best.pt`、`last.pt`，以及 `results.csv` 最后一轮的 epoch、Precision、Recall、mAP 和 loss。
6. 训练结果图区显示当前训练目录中的曲线图、混淆矩阵、批次样例等图片，可通过“上一张/下一张”切换；点击“选择结果目录”可浏览任意历史训练目录，摘要表会同步更新。程序启动时也会自动加载输出目录中最近的 `train_*` 结果。
7. 需要提前结束时点击“停止训练”。

训练完成后会记录：

- 训练开始与结束时间；
- 使用的数据集 YAML；
- 模型和输出目录；
- 每个 epoch 的训练指标；
- `best.pt` 与 `last.pt` 路径。

这些信息保存在应用数据目录的 `yolo_results.db` 中。

## 5. 模型导出与量化

通过“工作区 → 量化与推理”进入独立量化推理页面。

### 5.1 普通 FP32 导出

1. 选择训练得到的 `.pt` 模型。
2. 选择目标格式，例如 `onnx`。
3. 不勾选 FP16 或 INT8。
4. 点击“开始导出/量化”。

### 5.2 FP16 导出

1. 选择源 `.pt` 模型。
2. 选择支持 FP16 的目标格式和设备。
3. 勾选 `FP16`。
4. 不要同时勾选 INT8。
5. 点击“开始导出/量化”。

### 5.3 INT8 后训练量化

1. 选择源 `.pt` 模型。
2. 选择目标格式。
3. 勾选 `INT8`。
4. 选择用于校准的 `data.yaml`。
5. 不要同时勾选 FP16。
6. 点击“开始导出/量化”。

实际可用格式和精度取决于 Ultralytics、PyTorch、CUDA、TensorRT、OpenVINO 等运行环境。环境不支持时，错误会显示在日志区。

## 6. Python 推理流程

Python 后端支持 Ultralytics 能加载的 `.pt`、`.onnx`、`.engine` 等模型。

1. 在“模型”中选择待推理模型。
2. 点击“选择图片”处理单张图片，或点击“选择目录”批量处理。
3. 后端选择“Python / Ultralytics”。
4. 设置 Confidence 和 IoU。
5. 设置 Python、Ultralytics 源码和输出目录。
6. 点击“开始推理”。

推理结束后：

- 右下区域显示带检测框的结果图片。
- 使用“上一张”“下一张”翻页。
- 点击“检测明细”查看当前任务的类别、置信度和坐标。
- 点击“类别统计”查看当前任务的类别占比图。
- 结果展示区左侧会列出当前图片的检测框记录；点击一条记录，右侧图片会用青色粗框实时高亮对应目标。切换图片后列表和高亮会自动更新，原结果图片文件不会被修改。

## 7. ONNX Runtime C++ 推理流程

### 7.1 配置 ONNX Runtime

当前工作区已配置官方 ONNX Runtime 1.26.0 CPU / Windows x64 SDK：

```text
third_party/onnxruntime-win-x64-1.26.0
```

现有 Qt 6.11 / MSVC 2022 构建目录的 CMake 缓存已设置为：

```text
ONNXRUNTIME_ROOT=E:/QT/project/2/third_party/onnxruntime-win-x64-1.26.0
```

构建后 `onnxruntime.dll` 会自动复制到 `YOLOAnnotator.exe` 所在目录。若新建或清空构建目录，需要在新 Kit 的 CMake 配置中重新设置上述 `ONNXRUNTIME_ROOT`。

下载与 MSVC x64 匹配的 ONNX Runtime C/C++ SDK，目录应包含：

```text
onnxruntime/
├─ include/onnxruntime_cxx_api.h
└─ lib/onnxruntime.lib
```

在 Qt Creator 的 CMake 配置中增加：

```text
ONNXRUNTIME_ROOT=ONNX Runtime SDK 根目录
```

或使用命令行：

```powershell
E:\QT\qt\Tools\CMake_64\bin\cmake.exe -S . -B build/onnx `
  -DONNXRUNTIME_ROOT="D:/Libraries/onnxruntime"
```

重新构建后，CMake 输出中应出现：

```text
ONNX Runtime C++ inference enabled
```

如果未配置 SDK，程序仍可编译，但选择 C++ 推理时会提示“ONNX Runtime 未启用”。

### 7.2 执行 C++ 推理

1. 选择 ONNX 模型。
2. 选择单张图片或图片目录。
3. 后端选择“C++ / ONNX Runtime”。
4. 设置 Confidence 和 IoU。
5. 选择输出目录。
6. 点击“开始推理”。

C++ 后端会执行图片缩放和填充、ONNX Runtime 推理、YOLO 输出解析、NMS、结果绘制和数据库保存。

当前实现支持 Ultralytics 常见的原始 YOLO 检测输出，以及 YOLO26 的 `[1,N,6]` 端到端输出（`x1,y1,x2,y2,confidence,class_id`）。自定义输出节点或 float16 输入张量的 ONNX 模型可能仍需额外适配，建议先使用标准 FP32 ONNX 模型验证流程。

## 8. 数据库说明

数据库文件名为：

```text
yolo_results.db
```

默认存放在 Qt 返回的应用本地数据目录，而不是项目或构建目录。主要数据表包括：

- `training_tasks`：训练任务、状态和模型产物。
- `training_metrics`：每个 epoch 的指标。
- `inference_tasks`：推理任务、模型、输入、后端和结果目录。
- `detections`：检测类别、置信度和边界框坐标。

推理窗口中的明细与统计默认只显示当前推理任务，避免不同批次结果混合。

## 9. 常见问题

### 点击训练后提示找不到桥接脚本

正常构建会把以下文件复制到可执行文件目录：

```text
qt_train_bridge.py
qt_quant_infer_bridge.py
```

如果手动移动了 `.exe`，需要同时复制这两个脚本。

### Python 无法导入 ultralytics

检查界面中的 Python 是否是正确环境，或填写本地 Ultralytics 源码目录。

### 找不到 labels 目录

检查训练集和验证集是否符合第 3 节的目录结构，并确认图片与标签文件同名。

### 切换图片后旧标签仍然存在

删除全部框后仍需保存。当前版本支持写入空 `.txt`，可正确清除已有标签内容。

### C++ 推理不可用

确认已经设置 `ONNXRUNTIME_ROOT` 并重新运行 CMake，而不只是重新编译旧配置。

### `.engine` 模型能否使用 C++ 后端

不能。当前 C++ 后端仅使用 ONNX Runtime 加载 ONNX 模型；TensorRT `.engine` 应使用 Python/Ultralytics 后端。

## 10. 推荐完整工作流程

1. 整理原始图片并打开图片目录。
2. 建立类别并完成矩形框标注。
3. 检查已标注/未标注数量，保存标签。
4. 按标准 YOLO 结构划分 `images/train`、`images/val`、`labels/train`、`labels/val`。
5. 在训练控制台选择训练集和验证集，配置环境及参数。
6. 启动训练，观察日志、指标和曲线。
7. 使用 `best.pt` 执行 Python 推理，先验证模型效果。
8. 将 `best.pt` 导出为 ONNX，按需要启用 FP16 或 INT8。
9. 使用 Python 后端验证导出模型。
10. 配置 ONNX Runtime 后，使用 C++ 后端对同一批图片推理并比较结果。
11. 在检测明细、统计图和数据库中检查并保存结果。
