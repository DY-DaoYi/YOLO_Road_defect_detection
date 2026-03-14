import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import time
import analysis # 导入自定义分析模块
from moviepy import VideoFileClip # 导入 moviepy 用于视频转换
import report_gen # 导入自定义 PDF 生成模块

# 设置页面配置
st.set_page_config(
    page_title="基于YOLO的道路缺陷检测系统",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 类别名称映射
CLASS_NAME_MAP = {
    "pothole": "路面坑洼"
}

def get_cn_name(name):
    return CLASS_NAME_MAP.get(name, name)

# 侧边栏
st.sidebar.title("🛠️ 系统配置")

# 模型选择逻辑
models_dir = Path("models")
results_dir = Path("all_results")

model_files = []
model_display_names = []

# 1. 尝试在 all_results 中查找训练好的模型
if results_dir.exists():
    for model_folder in results_dir.iterdir():
        if model_folder.is_dir():
            # 检查 weights/best.pt 是否存在
            best_pt = model_folder / "weights" / "best.pt"
            if best_pt.exists():
                model_files.append(best_pt)
                model_display_names.append(f"{model_folder.name} (Trained)")

# 2. 如果未找到训练模型或仅添加基础模型，则回退到 models/ 目录
if models_dir.exists():
    # 2a. 检查 models/ 中的子目录 (例如 models/yolov8l_pothole/weights/best.pt)
    for model_folder in models_dir.iterdir():
        if model_folder.is_dir():
            best_pt = model_folder / "weights" / "best.pt"
            if best_pt.exists():
                model_files.append(best_pt)
                model_display_names.append(f"{model_folder.name} (Trained)")

    # 2b. 检查 models/ 中的独立 .pt 文件 (例如 models/yolov8n.pt)
    for m in models_dir.glob("*.pt"):
        model_files.append(m)
        model_display_names.append(m.name)

if not model_files:
    st.sidebar.warning("未找到模型文件！请确保 'all_results' 或 'models' 目录不为空。")
    # 演示用的回退默认选项 (可能会加载失败)
    model_options = ["yolo26n.pt", "yolo11n.pt", "yolov8n.pt"]
    model_map = {m: m for m in model_options}
else:
    model_options = model_display_names
    model_map = dict(zip(model_display_names, model_files))

selected_model_name = st.sidebar.selectbox(
    "选择模型",
    model_options,
    index=0 if model_options else 0
)

# 加载模型
@st.cache_resource(ttl="1h") # 添加 TTL 以强制偶尔刷新
def load_model(model_path):
    try:
        model = YOLO(model_path)
        # 将模型名称修补为中文
        # 调试：打印原始名称
        print(f"Original names: {model.names}")
        
        # 方法 1：按名称映射
        for id, name in model.names.items():
            if name in CLASS_NAME_MAP:
                model.names[id] = CLASS_NAME_MAP[name]
        
        # 方法 2：强制映射类别 0 (因为我们知道这是坑洼检测)
        if 0 in model.names and model.names[0] == 'pothole':
             model.names[0] = "路面坑洼"
             
        print(f"Patched names: {model.names}")
        return model
    except Exception as e:
        st.error(f"无法加载模型: {e}")
        return None

if selected_model_name in model_map:
    model_path = model_map[selected_model_name]
    st.sidebar.info(f"加载路径: {model_path}")
    model = load_model(str(model_path))
else:
    model = None

# 参数
conf_thres = st.sidebar.slider("置信度阈值 (Confidence)", 0.0, 1.0, 0.25, 0.05)
iou_thres = st.sidebar.slider("IOU 阈值", 0.0, 1.0, 0.45, 0.05)

# 主要内容
st.title("🛣️ 基于YOLO的道路缺陷检测系统")
st.markdown("### 🚀 支持 YOLOv8 / YOLO11 / YOLO26")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 单张图片/视频检测", "WB 批量检测 & 报告", "📊 模型性能对比", "ℹ️ 项目说明"])

with tab1:
    st.header("单任务检测")

    source_type = st.radio("选择输入类型", ["图片", "视频"], horizontal=True)
    
    if source_type == "图片":
        uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption="原始图片")
            
            if st.button("开始检测", type="primary"):
                with st.spinner("检测中..."):
                    start_time = time.time()
                    if model:
                        results = model.predict(tmp_path, conf=conf_thres, iou=iou_thres)
                        end_time = time.time()
                        
                        # 修正绘图标签
                        for r in results:
                            for id, name in r.names.items():
                                if name in CLASS_NAME_MAP:
                                    r.names[id] = CLASS_NAME_MAP[name]
                        
                        # 绘制结果
                        res_plotted = results[0].plot()
                        res_image = Image.fromarray(res_plotted[..., ::-1]) # BGR 转 RGB
                        
                        with col2:
                            st.image(res_image, caption=f"检测结果 ({end_time - start_time:.2f}s)")
                        
                        # 显示指标
                        boxes = results[0].boxes
                        if len(boxes) > 0:
                            st.success(f"检测到 {len(boxes)} 个缺陷")
                            # 创建简单的结果 DataFrame
                            data = []
                            for box in boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                name = model.names[cls]
                                cn_name = get_cn_name(name)
                                data.append({"类别": cn_name, "置信度": f"{conf:.2f}"})
                            
                            st.table(pd.DataFrame(data))
                        else:
                            st.info("未检测到缺陷")
            
            os.unlink(tmp_path)

    elif source_type == "视频":
        uploaded_video = st.file_uploader("上传视频", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            st.video(video_path)
            
            if st.button("开始视频检测"):
                st.info("正在处理视频，请稍候...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                vf = cv2.VideoCapture(video_path)
                
                # Get video properties
                width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(vf.get(cv2.CAP_PROP_FPS))
                total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # 输出视频设置
                # 使用带有 mp4v 编解码器的 .mp4 (广泛兼容)
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                frame_count = 0
                
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret:
                        break
                    
                    if model:
                        results = model(frame, conf=conf_thres, iou=iou_thres)
                        # 修补名称
                        for r in results:
                            for id, name in r.names.items():
                                if name in CLASS_NAME_MAP:
                                    r.names[id] = CLASS_NAME_MAP[name]
                        
                        res_plotted = results[0].plot()
                        out.write(res_plotted)
                    else:
                        out.write(frame)
                        
                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))
                        status_text.text(f"Processing frame {frame_count}/{total_frames}")
                    
                vf.release()
                out.release()
                
                status_text.text("Detection complete. Converting for web playback...")
                
                # 使用 moviepy 转换为 H.264 以兼容浏览器
                # 这确保视频可以在 Chrome/Firefox/Edge 中播放
                final_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                try:
                    clip = VideoFileClip(output_path)
                    clip.write_videofile(final_output_path, codec='libx264', audio=False)
                    clip.close()
                    
                    # 清理中间文件
                    try:
                        os.unlink(output_path)
                    except:
                        pass
                        
                    output_path = final_output_path # 切换到 Web 就绪文件
                except Exception as e:
                    st.error(f"Video conversion failed: {e}. Falling back to original file (might not play in browser).")
                
                status_text.text("Processing complete!")
                progress_bar.progress(1.0)
                st.success("视频检测完成")
                
                # 显示结果视频
                st.video(output_path)
                
                # 下载按钮
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="下载检测视频",
                        data=f,
                        file_name="detected_video.mp4",
                        mime="video/mp4"
                    )
                
                # 清理
                try:
                    os.unlink(video_path)
                    # os.unlink(output_path) # 保留输出以供下载，直到会话结束或用户重新运行
                except:
                    pass

with tab2:
    st.header("批量检测与报告生成")
    uploaded_files = st.file_uploader("上传多张图片", type=["jpg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        # 检查文件是否更改或首次运行
        if 'last_uploaded_files' not in st.session_state or st.session_state['last_uploaded_files'] != uploaded_files:
             st.session_state['last_uploaded_files'] = uploaded_files
             # 清除先前的结果
             if 'results_data' in st.session_state: del st.session_state['results_data']
             if 'pdf_data' in st.session_state: del st.session_state['pdf_data']
             if 'pdf_generated' in st.session_state: del st.session_state['pdf_generated']
    
    if uploaded_files and st.button("批量处理"):
        progress_bar = st.progress(0)
        results_data = []
        pdf_data = [] # PDF 生成数据
        
        # 创建报告图片的临时目录
        # 我们需要一个会话持久的临时目录
        if 'report_temp_dir' not in st.session_state:
             st.session_state['report_temp_dir'] = tempfile.mkdtemp()
        report_temp_dir = st.session_state['report_temp_dir']
        
        for i, file in enumerate(uploaded_files):
            # 保存临时输入
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            if model:
                results = model(tmp_path, conf=conf_thres, iou=iou_thres)[0]
                
                # 修正绘图标签
                for id, name in results.names.items():
                    if name in CLASS_NAME_MAP:
                        results.names[id] = CLASS_NAME_MAP[name]
                
                count = len(results.boxes)
                
                # 按类别计数
                details = []
                confs = []
                for box in results.boxes:
                    c = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[c]
                    details.append(get_cn_name(name))
                    confs.append(conf)
                
                # 保存绘图图像用于报告
                res_plotted = results.plot()
                img_save_path = os.path.join(report_temp_dir, f"detected_{i}.jpg")
                # 将 BGR 转换为 RGB 以进行 PIL 保存 (或使用 cv2.imwrite)
                cv2.imwrite(img_save_path, res_plotted)
                
                results_data.append({
                    "文件名": file.name,
                    "缺陷数量": count,
                    "检测详情": ", ".join(details)
                })
                
                pdf_data.append({
                    "filename": file.name,
                    "count": count,
                    "img_path": img_save_path,
                    "details": ", ".join(details),
                    "confs": confs # 存储置信度用于平均值计算
                })
            
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # 保存到会话状态
        st.session_state['results_data'] = results_data
        st.session_state['pdf_data'] = pdf_data
        st.success("批量处理完成！")
        
    # 如果可用，从会话状态渲染结果
    if 'results_data' in st.session_state and st.session_state['results_data']:
        results_data = st.session_state['results_data']
        pdf_data = st.session_state.get('pdf_data', [])
        
        st.subheader("📊 统计结果")
        df = pd.DataFrame(results_data)
        st.dataframe(df)
        
        # 操作按钮行
        btn_col1, btn_col2 = st.columns([1, 1])
        
        with btn_col1:
            # 导出到 CSV
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "📥 下载 CSV 数据",
                csv,
                "detection_report.csv",
                "text/csv",
                key='download-csv'
            )
            
        with btn_col2:
            if st.button("📑 生成 PDF 报告"):
                if not pdf_data:
                    st.error("没有数据可生成 PDF，请重新运行批量处理。")
                else:
                    with st.spinner("正在生成 PDF..."):
                        # 准备摘要
                        total_defects = sum([item['count'] for item in pdf_data])
                        
                        # 计算平均置信度
                        all_confs = []
                        for item in pdf_data:
                            all_confs.extend(item.get('confs', []))
                        
                        avg_conf_str = "N/A"
                        if all_confs:
                            avg_conf = sum(all_confs) / len(all_confs)
                            avg_conf_str = f"{avg_conf:.2%}"
                        
                        summary = {
                            'model': selected_model_name,
                            'total_files': len(uploaded_files),
                            'total_defects': total_defects,
                            'avg_conf': avg_conf_str
                        }
                        
                        # 使用会话临时目录
                        if 'report_temp_dir' not in st.session_state:
                             st.session_state['report_temp_dir'] = tempfile.mkdtemp()
                        report_temp_dir = st.session_state['report_temp_dir']
                        
                        pdf_output_path = os.path.join(report_temp_dir, "report.pdf")
                        success, msg = report_gen.generate_pdf_report(summary, pdf_data, pdf_output_path)
                        
                        if success:
                            # 我们也需要为 PDF 使用下载按钮，但 st.button 会触发重新运行
                            # 一种常见的模式是在生成后显示下载按钮
                            st.session_state['pdf_generated'] = pdf_output_path
                            st.success("PDF 生成成功！请点击下方按钮下载")
                        else:
                            st.error(f"生成失败: {msg}")
            
            # 如果已生成，显示 PDF 下载按钮
            if 'pdf_generated' in st.session_state and os.path.exists(st.session_state['pdf_generated']):
                with open(st.session_state['pdf_generated'], "rb") as f:
                    st.download_button(
                        "📥 点击下载 PDF 报告",
                        f,
                        "detection_report.pdf",
                        "application/pdf",
                        key='download-pdf'
                    )
        
        # 注意：Streamlit 中的临时目录清理很棘手，因为用户可能不会立即下载。
        # 理想情况下使用 cron 作业或会话状态来管理清理。
        # 目前，文件保留在临时目录中，直到操作系统清理它们或重启。

with tab3:
    st.header("模型性能对比分析")
    
    results_dir = Path("all_results")
    # 支持在 all_results 缺失时检查 models/ 目录 (移动后的结构)
    if not results_dir.exists() and Path("models").exists():
        results_dir = Path("models")
        
    if not results_dir.exists():
        st.warning("未找到训练结果目录 (all_results 或 models)，请先上传训练结果或移动模型文件。")
    else:
        # st.info("加载模型训练结果...") # 移除这个一直显示的提示
        # 加载自定义分析模块
        import importlib
        importlib.reload(analysis)
        
        try:
            results_data = analysis.load_all_results(results_dir)
            if not results_data:
                st.warning("未找到有效的 results.csv 文件。")
            else:
                # 1. 汇总表
                st.subheader("🏆 最佳指标汇总")
                summary_df = analysis.get_best_metrics(results_data)
                # 格式化
                st.dataframe(summary_df.style.highlight_max(axis=0, color='lightgreen', subset=['Best mAP50', 'Best mAP50-95']))
                
                # 2. 图表
                st.subheader("📈 训练曲线对比")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**mAP50-95 (验证集)**")
                    fig_map = analysis.plot_comparison(results_data, metric="metrics/mAP50-95(B)", title="mAP50-95 Comparison")
                    st.pyplot(fig_map)
                    
                with col2:
                    st.markdown("**Box Loss (验证集)**")
                    fig_loss = analysis.plot_comparison(results_data, metric="val/box_loss", title="Validation Box Loss")
                    st.pyplot(fig_loss)
                    
                st.markdown("---")
                
                # 3. 详细视图
                st.subheader("🔍 单模型详情")
                selected_detail_model = st.selectbox("选择模型查看详细图表", list(results_data.keys()))
                
                if selected_detail_model:
                    # 找到实际的文件夹路径 (处理潜在的后缀不匹配)
                    model_path = results_dir / selected_detail_model # 默认回退
                    for p in results_dir.iterdir():
                        if p.is_dir() and p.name == selected_detail_model:
                            model_path = p
                            break
                    
                    # 尝试显示混淆矩阵或结果图
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        conf_mat = model_path / "confusion_matrix.png"
                        if conf_mat.exists():
                            st.image(str(conf_mat), caption="混淆矩阵")
                        else:
                            st.info("未找到混淆矩阵图片")
                            
                    with img_col2:
                        res_png = model_path / "results.png"
                        if res_png.exists():
                            st.image(str(res_png), caption="训练结果概览")
                        else:
                            st.info("未找到结果概览图片")
                    
                    st.markdown("#### 📈 详细指标曲线")
                    c1, c2, c3, c4 = st.columns(4)
                    metrics_map = {
                        "F1 Curve": "BoxF1_curve.png",
                        "Precision Curve": "BoxP_curve.png",
                        "Recall Curve": "BoxR_curve.png",
                        "PR Curve": "BoxPR_curve.png"
                    }
                    
                    for col, (label, fname) in zip([c1, c2, c3, c4], metrics_map.items()):
                        img_path = model_path / fname
                        with col:
                            if img_path.exists():
                                st.image(str(img_path), caption=label)
                    
                    st.markdown("#### 🖼️ 验证集预测样本")
                    v1, v2 = st.columns(2)
                    with v1:
                        val_pred = model_path / "val_batch0_pred.jpg"
                        if val_pred.exists():
                            st.image(str(val_pred), caption="验证集预测 (Batch 0)")
                    with v2:
                        val_lbl = model_path / "val_batch0_labels.jpg"
                        if val_lbl.exists():
                            st.image(str(val_lbl), caption="验证集标签 (Batch 0)")

        except Exception as e:
            st.error(f"分析模块出错: {e}")


with tab4:
    st.markdown("""
    ## 📚 项目说明
    
    本项目是基于 **YOLO (You Only Look Once)** 系列模型的道路缺陷检测系统，专为识别路面坑洼等缺陷而设计。
    
    ### ✨ 主要功能
    - **多模型兼容**: 无缝支持 YOLOv8, YOLO11, 以及最新的 YOLO26 模型。
    - **高精度实时检测**: 针对路面坑洼 (Pothole) 场景进行了深度优化和专门训练，支持图片和视频的快速检测。
    - **批量处理与报告生成**: 支持一次性导入多张图片进行批量检测，并能一键生成详细的 PDF 统计报告，方便工程人员存档和分析。
    - **模型性能可视化对比**: 内置训练结果分析模块，直观对比不同模型的 mAP、Loss 等关键指标，辅助选择最佳模型。
    
    ### 🧠 模型介绍
    - **YOLOv8**: 经典的高效目标检测模型，在速度和精度上取得了极佳的平衡。
    - **YOLO11**: 经过迭代升级的模型架构，进一步提升了特征提取能力。
    - **YOLO26**: Ultralytics 最新推出的端到端实时检测模型，摒弃了传统的 NMS（非极大值抑制）后处理步骤，推理速度更快，精度更高。
    
    ### 💻 技术栈
    - **前端展示**: Streamlit (提供快速、交互式的 Web UI)
    - **核心算法**: Ultralytics YOLO (提供模型推理能力)
    - **图像/视频处理**: OpenCV, Pillow, MoviePy
    - **数据分析与报告**: Pandas, Matplotlib, ReportLab
    """)

# 页脚
st.sidebar.markdown("---")
st.sidebar.info("Developed by YourName for Graduation Project")
