# yolo-cs2-detect

基于 **YOLO + 自研 C++ 原生叠加层** 的 CS2（Counter-Strike 2）游戏内实时目标检测系统。

覆盖「屏幕捕获 → 模型推理 → 结果渲染」完整链路：用 `dxcam` 抓取屏幕 → YOLO 推理 →
通过 C++（pybind11）编写的 Win32 透明叠加层，把检测框实时绘制到游戏画面上。

---

## 功能特性

- **实时检测闭环**：dxcam GPU 侧屏幕捕获 → YOLO 实时推理 → 叠加层低延迟绘制
- **自研 C++ 透明叠加层**（`overlay/`）：Win32 分层窗口 + `UpdateLayeredWindow` 逐像素
  Alpha 合成，绕过 `WM_PAINT` 消息循环、零 GDI 绘制调用、直写像素缓冲交由 DWM 合成，
  实现**无边框 / 鼠标穿透 / 置顶**
- **交互式模型选择**：启动时自动扫描 `models/` 下的权重并列出，由用户选择（不硬编码）
- **多模型 Benchmark**：自动发现权重，横向对比延迟 / FPS / 显存 / mAP，
  基于**帕累托前沿**给出选型推荐

## 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows（叠加层依赖 Win32 分层窗口 + DWM 合成） |
| Python | 3.12 |
| 编译器 | MSVC（Visual Studio 2022 或 Build Tools，需勾选「使用 C++ 的桌面开发」） |
| GPU | NVIDIA 显卡可选；CPU 也能跑，但帧率很低 |

## 安装

```bash
# 1. 安装依赖（uv 会读取 pyproject.toml + uv.lock）
uv sync

# 2. 编译 C++ 叠加层（生成 overlay/overlay.pyd）
uv run python overlay/build.py
```

> `overlay.pyd` 不入库（编译产物），首次运行 `main.py` 时若缺失会自动尝试编译。

## 使用
使用前你需要下载一些目标检测模型的`.pt`权重文件,将您想要使用的模型权重文件放在models目录下即可

### 实时目标检测(cs2游戏一定开启无边框全屏模式,否则无法绘制检测框)

```bash
uv run python main.py
```

启动后会列出 `models/` 下的所有权重供你选择，随后在游戏画面上叠加检测框。
按 `Ctrl+C` 退出。

### 多模型性能 Benchmark  
项目为视觉模型提供了测试选项，如果您需要根据你的设备选择一个合适的检测模型，请将模型权重文件(.pt文件)放在models目录下  

- 测精度：在 `data_yaml` 中配置好各项参数即可；不想测精度，保持 `data_yaml` 文件中的变量为空即可（程序会自动检测，字段不全就跳过精度测试）
- 排除特定权重（如开集模型、微调模型）：`exclude=("world", "_cs2")`

配置完需要的参数后，运行：
```bash
uv run python benchmark.py
```

自动扫描 `models/` 下的全部 `.pt`，逐个测速度 / 显存 / 精度，最后输出对比表格与
帕累托选型推荐。



## 项目结构

```
.
├── main.py              # 实时检测主程序（交互式选模型 + 主循环）
├── benchmark.py         # 多模型 Benchmark + 帕累托选型
├── models/              # 模型权重统一存放于此（.pt 不入库）
├── overlay/             # 自研 C++ 原生透明叠加层
│   ├── overlay.cpp      #   核心实现（入库）
│   ├── build.py         #   一键编译脚本（入库）
│   └── pyproject.toml   #   包声明，uv workspace 成员（入库）
└── test/                # 数据集目录
    ├── images/          #   图片（不入库）
    ├── labels/          #   YOLO 格式标签（不入库）
    ├── data.yaml        #   数据集配置模板（入库）
    └── data_local.yaml  #   本地真实配置（不入库）
```

## 数据与配置

### 数据集
当您想要使用benchmark模块测试模型性能时，你需要准备一些测试图片以及对应符合ultralytics标准的标注，我们推荐你将图片放在数据目录下的images目录中，标注放在labels目录下(标注为.txt格式)  
在`data_yaml`文件中有五个变量：  
1. path: 您的测试集数据文件存放目录，使用时可以忽略
2. train: 您的训练数据文件存放目录，无训练需求时可以忽略
3. val: 用于测评模型精度的数据文件存放目录， 可以自行指定，也可忽略
4. test: 用于测试的图片文件目录，可以自行制定，若留空则自动从`val`对应目录中取得
5. nc: 需要检测的类别数
6. name: 需要检测的类别名称，其长度应该等于nc

### 模型权重

所有 `.pt` 权重统一放在 `models/`。权重体积大，**不入库**，需自行下载或训练后放入。

## 注意事项

1. **编译和运行请使用项目根 `.venv` 里的 Python。**
   `overlay.pyd` 与 Python 小版本绑定（3.12 编的不能在 3.13 里加载）。
   混用解释器会导致 `import overlay` 报 DLL 错误，此时删掉 `overlay.pyd` 重新编译即可。

2. **独占全屏（exclusive fullscreen）下叠加层无法生效。**
   只有游戏处于无边框全屏 / 窗口化时，DWM 才会把叠加层合成到游戏之上。
   独占全屏直接写显卡扫描输出，绕过 DWM，任何 Win32 叠加层都盖不上去。

3. **mAP 对比只在同类模型之间有意义。**
   `test/data.yaml` 定义的是 3 类（t / ct / c4），而通用 YOLO 模型输出的是 COCO 80 类。
   拿通用模型去测 3 类数据集，mAP 会低得离谱且不可比。
   **速度 / FPS / 显存对比不受影响**，可以放心横比。

4. **`.gitignore` 已忽略**：`models/*.pt`、`test/images/`、`test/labels/`、
   `data_local.yaml`、`.venv`、`.workbuddy-ai/`、`*.part`、`*.cache`。

## 技术栈

- **语言**：Python、C++17
- **深度学习**：PyTorch、Ultralytics YOLO（v5 / v8 / v10 / v11 / v26）
- **系统与底层**：Win32 API（分层窗口 · DWM 合成 · 逐像素 Alpha）、pybind11、MSVC
- **工具链**：dxcam、OpenCV、NumPy、uv

