# 🛣️ 基于 YOLO 的道路缺陷检测系统 (Road Defect Detection System)

![Python](https://img.shields.io/badge/Python-3.11.15-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55.0-red)
![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v11%20%7C%20v26-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

本项目是一个功能完整的道路缺陷（如路面坑洼 Pothole）智能检测系统。基于最新的 Ultralytics YOLO 系列模型构建，并提供了一个美观、易用的 Streamlit Web 交互界面。本项目非常适合作为**本科毕业设计**的参考与展示。

> **⚠️ 关于本项目 (About This Project)**
> 
> 本仓库**开源了系统的前端交互应用（App）源码及训练数据可视化模块**。
> 
> **可视化模块说明**：系统内置了自动化分析逻辑，能够读取训练过程中的日志文件（results.csv），自动绘制 Loss 损失曲线、mAP 精度曲线和混淆矩阵。**这部分逻辑已完全开源**，即使不懂代码，也能直观地看到模型的训练效果，非常适合用于撰写论文时的图表生成。
> 
> **🛒 核心模型与训练源码资源包 (Core Models & Training Codes) 包含：**
> 1.  **核心训练代码**：包含 **train.py (单模型)** 和 **train_batch.py (批量实验)**，支持从数据预处理、断点续训到多模型对比评估的完整流程。
> 2.  **全系列模型权重与训练日志**：包含 **YOLOv8 / YOLO11 / YOLO26** 三个系列的 `best.pt` 模型文件，以及完整的训练日志、Loss 曲线图、mAP 指标图和混淆矩阵等评估图表，直接用于论文插图。
> 3.  **（赠品）数据集**：本项目使用 [Ultralytics Project Explorer](https://platform.ultralytics.com/embed/projectexplorer/datasets/project1) 的开源数据集。
>     *   *说明：数据集本身是免费开源的。但由于国内网络环境下载困难，我在资源包中免费提供了已整理好的数据集压缩包，方便大家直接使用。*
> 
> **📥 立即获取资源包：** [🍞 面包多 (Mianbaoduo)](YOUR_MIANBAODUO_LINK) | [🐟 闲鱼 (Xianyu)](YOUR_XIANYU_LINK)

---

## ✨ 核心功能亮点 (Features)

*   **🚀 多模型兼容**: 系统无缝支持 `YOLOv8`、`YOLO11` 以及最新的 `YOLO26` 模型，用户可在侧边栏一键切换，实时对比不同模型的检测效果。
*   **📷 多模态检测**: 支持上传单张图片进行快速检测，也支持导入 `MP4/MKV/AVI` 等格式的视频文件（内置转码功能，确保在浏览器中流畅播放）。
*   **📦 批量处理引擎**: 针对大量数据，提供批量检测功能。系统会自动统计缺陷数量，并生成包含详细信息的 CSV 表格。
*   **📑 自动报告生成**: 一键生成专业的 PDF 检测报告，包含检测综述、缺陷统计和样张展示，可直接用于作业或汇报。
*   **📊 训练可视化**: 独家内置**模型性能对比面板**。无需手动绘制，系统自动解析训练日志，生成专业的 mAP 和 Loss 对比图，助你轻松完成毕设论文的实验分析章节。
*   **🇨🇳 全中文支持**: 从界面菜单到检测框标签（如“路面坑洼”），全部采用中文显示，符合国内用户习惯。

---

## 📺 系统演示 (Demo)

> **🎥 视频演示 (Video Demo)**
> 
> 点击下方图标，跳转至各大视频平台查看系统的详细运行演示：
>
> | 哔哩哔哩 (Bilibili) | 抖音 (Douyin) | 小红书 (Xiaohongshu) |
> | :---: | :---: | :---: |
> | [![Bilibili](https://img.shields.io/badge/Bilibili-演示视频-pink?style=for-the-badge&logo=bilibili)](https://space.bilibili.com/YOUR_UID) | [![Douyin](https://img.shields.io/badge/抖音-演示视频-black?style=for-the-badge&logo=tiktok)](https://www.douyin.com/user/YOUR_UID) | [![Xiaohongshu](https://img.shields.io/badge/小红书-演示视频-red?style=for-the-badge&logo=xiaohongshu)](https://www.xiaohongshu.com/user/profile/YOUR_UID) |
>

---

## 🛠️ 快速开始 (Quick Start)

### 第一阶段：部署开源版（运行系统界面）

此步骤将教你如何运行本仓库的开源代码，体验系统的 UI 交互功能。

1.  **环境准备 (Environment Setup)**
    *   **安装 Anaconda**: 如果你还没有安装 Anaconda，请参考[此教程](https://tcn196ka4swf.feishu.cn/wiki/AE1lwBz5EilnrukWYDTcmzQknNd?from=from_copylink)进行安装。
    *   **打开终端 (Important)**:
        *   **方法 A (推荐)**: 进入本项目文件夹，在地址栏输入 `cmd` 并回车，即可直接在当前目录下打开终端。
        *   **方法 B**: 打开终端后，使用 `cd` 命令跳转到项目目录（例如：`cd D:\毕设\YOLO_Road_defect_detection`）。
    *   **创建并激活环境**:
        ```bash
        conda create -n task1 python=3.11.15 -y
        conda activate task1
        ```
    *   **安装 PyTorch (关键步骤)**:
        本项目基于 PyTorch 2.5.1 开发。推荐访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取最适合你机器的安装命令。
        
        或者，你可以直接使用以下常用命令（请根据你的显卡驱动版本选择）：
        *   **方案 A (推荐新显卡/CUDA 12.1)**:
            ```bash
            conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
            ```
        *   **方案 B (兼容旧显卡/CUDA 11.8)**:
            ```bash
            conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
            ```
        *   **方案 C (无显卡/仅使用 CPU)**:
            ```bash
            conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
            ```

2.  **安装依赖**
    下载本仓库代码，并安装必要的第三方库：
    ```bash
    pip install -r requirements.txt
    ```

3.  **运行系统**
    在终端中执行以下命令：
    ```bash
    streamlit run app/app.py
    ```
    成功后，浏览器会自动打开系统界面。
    *注意：此时由于没有加载模型文件，点击检测可能会提示“未找到模型”。*

### 第二阶段：加载模型（开启检测功能）

为了让系统能够识别路面缺陷，你需要加载训练好的 YOLO 模型文件（`.pt`）。

#### 🅰️ 如果你购买了资源包 (推荐)
资源包中已包含 **YOLOv8 / YOLO11 / YOLO26** 全系列训练好的高精度模型。
1.  **解压资源包**：找到 `models` 文件夹。
2.  **复制文件**：将整个 `models` 文件夹直接复制到本项目的根目录下。
3.  **完成**：刷新网页，侧边栏会自动加载所有模型。

#### 🅱️ 如果你自己训练模型
如果你自行训练了模型：
1.  **新建文件夹**：在项目根目录下创建一个名为 `models` 的文件夹。
2.  **放置模型**：将你训练得到的 `best.pt` 文件复制到 `models/` 文件夹中（建议重命名为清晰的名称，如 `yolov8n_my_train.pt`）。
3.  **完成**：刷新网页，即可在侧边栏选择你的模型进行检测。

---

## 📂 项目结构 (Project Structure)

```text
YOLO_Road_defect_detection/
├── app/
│   ├── app.py          # Streamlit 系统主入口
│   ├── analysis.py     # 训练日志分析与绘图模块
│   └── report_gen.py   # PDF 报告生成模块
├── data/               # 训练数据集 (Images/Labels)
├── models/             # 训练好的模型文件 (.pt)
├── training/           # 模型训练代码
├── docs/               # 演示图片与文档
├── requirements.txt    # 项目依赖库列表
└── README.md           # 项目说明文档
```
*(注：`data/` (数据集)、`training/` (训练代码) 和 `models/` (模型文件) 为资源包专有内容，未包含在开源仓库中。)*

---

## 💰 训练成本核算 (Training Cost Analysis)

以下是基于国内主流 GPU 云服务商（如 AutoDL）的成本估算逻辑：

1.  **硬件配置**: NVIDIA RTX A4000 (24GB 显存)
2.  **训练规模**:
    *   **模型数量**: 15 个模型 (YOLOv8/11/26 系列 x n/s/m/l/x 五种规格)
    *   **训练轮数**: 100 Epochs / 模型
3.  **实际耗时与成本**:
    *   *总耗时*: 33 小时 40 分 (约 33.7 小时)
    *   *云服务器费率*: ¥0.97 / 小时
    *   **总算力成本**: **33.7 小时 x ¥0.97 ≈ ¥32.69**

> **💡 省钱建议**: 自己租用服务器复现所有实验不仅耗时耗力，且算力成本往往高于直接获取成品。
>
> **🛒 核心模型与训练源码资源包 (Core Models & Training Codes) 包含：**
> 1.  **核心训练代码**：包含 **train.py (单模型)** 和 **train_batch.py (批量实验)**，支持从数据预处理、断点续训到多模型对比评估的完整流程。
> 2.  **全系列模型权重与训练日志**：包含 **YOLOv8 / YOLO11 / YOLO26** 三个系列的 `best.pt` 模型文件，以及完整的训练日志、Loss 曲线图、mAP 指标图和混淆矩阵等评估图表，直接用于论文插图。
> 3.  **（赠品）数据集**：整理好的 Ultralytics 开源数据集压缩包。
>
> **📥 立即获取资源包：** [🍞 面包多 (Mianbaoduo)](YOUR_MIANBAODUO_LINK) | [🐟 闲鱼 (Xianyu)](YOUR_XIANYU_LINK)

---

## 👨‍💻 更多项目 & 联系作者 (More Projects & Contact)

### 💎 付费版毕设 (Premium Thesis Project)

#### 🌟 核心优势 (Core Advantages)
1.  **🛡️ 原创代码，拒绝查重**：代码由本人原创开发，且每年进行更新迭代，确保代码结构的新颖性，有效规避查重风险。
2.  **🔒 一校一份，防止撞车**：严格执行“一所学校只售出一份”的原则，从源头上杜绝同学之间毕设雷同的尴尬情况。
3.  **👨‍🏫 全程售后，无忧毕业**：提供从购买到答辩结束的全程技术支持。遇到任何代码运行、环境配置或逻辑理解问题，均可直接通过微信联系问询。

#### 📚 包含的全套文档 (Documentation Included)
付费版不仅仅是代码，更包含了一整套完整的毕设文档资料，助你轻松应对论文撰写与答辩：
*   📊 **数据集可视化分析报告**
*   🏗️ **系统架构说明文档**
*   🧠 **YOLO 模型架构与训练参数详解**
*   📈 **模型指标说明文档**
*   ▶️ **项目运行教程**
*   ⚙️ **配置修改指南**

👉 **[查看所有毕设项目介绍 (开源/付费)](https://tcn196ka4swf.feishu.cn/wiki/LXE5wumyNir8MBkLz51ckWnfnTc?from=from_copylink)**


### 🤝 联系方式

如果你需要**付费完整版（含源码、模型、数据集）**，欢迎联系我！

*   **更多毕设资源**: [👉 点击查看飞书毕设大全](https://tcn196ka4swf.feishu.cn/wiki/LXE5wumyNir8MBkLz51ckWnfnTc?from=from_copylink)

*   **联系方式**:
    *   **微信**: `DY_DaoYi` (添加请备注：**道路检测**)
    *   **B站 / 抖音 / 小红书**: 搜索 **[你的自媒体账号名]** (关注我的个人主页，获取最新教程)
