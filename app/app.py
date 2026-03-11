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

# Set page config
st.set_page_config(
    page_title="基于YOLO的道路缺陷检测系统",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# Sidebar
st.sidebar.title("🛠️ 系统配置")

# Model Selection
models_dir = Path("models")
model_files = list(models_dir.glob("*.pt"))
# Add default options if no models found (for demo)
if not model_files:
    # Try to find in root as fallback
    model_files = list(Path(".").glob("*.pt"))
    
model_options = [m.name for m in model_files] if model_files else ["yolo26n.pt", "yolo11n.pt", "yolov8n.pt"]

selected_model_name = st.sidebar.selectbox(
    "选择模型",
    model_options,
    index=0 if model_options else 0
)

# Load Model
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"无法加载模型: {e}")
        return None

model_path = models_dir / selected_model_name if (models_dir / selected_model_name).exists() else selected_model_name
model = load_model(str(model_path))

# Parameters
conf_thres = st.sidebar.slider("置信度阈值 (Confidence)", 0.0, 1.0, 0.25, 0.05)
iou_thres = st.sidebar.slider("IOU 阈值", 0.0, 1.0, 0.45, 0.05)

# Main Content
st.title("🛣️ 基于YOLO的道路缺陷检测系统")
st.markdown("### 🚀 支持 YOLOv8 / YOLO11 / YOLO26")

tab1, tab2, tab3 = st.tabs(["🔍 单张图片/视频检测", "WB 批量检测 & 报告", "ℹ️ 项目说明"])

with tab1:
    st.header("单任务检测")
    source_type = st.radio("选择输入类型", ["图片", "视频"], horizontal=True)
    
    if source_type == "图片":
        uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption="原始图片", use_container_width=True)
            
            if st.button("开始检测", type="primary"):
                with st.spinner("检测中..."):
                    start_time = time.time()
                    if model:
                        results = model.predict(tmp_path, conf=conf_thres, iou=iou_thres)
                        end_time = time.time()
                        
                        # Plot results
                        res_plotted = results[0].plot()
                        res_image = Image.fromarray(res_plotted[..., ::-1]) # BGR to RGB
                        
                        with col2:
                            st.image(res_image, caption=f"检测结果 ({end_time - start_time:.2f}s)", use_container_width=True)
                        
                        # Show Metrics
                        boxes = results[0].boxes
                        if len(boxes) > 0:
                            st.success(f"检测到 {len(boxes)} 个缺陷")
                            # Create a simple DataFrame for results
                            data = []
                            for box in boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                name = model.names[cls]
                                data.append({"类别": name, "置信度": f"{conf:.2f}"})
                            
                            st.table(pd.DataFrame(data))
                        else:
                            st.info("未检测到缺陷")
            
            os.unlink(tmp_path)

    elif source_type == "视频":
        uploaded_video = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            st.video(video_path)
            
            if st.button("开始视频检测"):
                st.warning("视频检测可能需要较长时间，请耐心等待...")
                # Video processing logic would go here
                # For simplicity in this demo, we can process frames or just show a placeholder
                # Implementing full video processing in Streamlit requires frame-by-frame loop
                
                vf = cv2.VideoCapture(video_path)
                stframe = st.empty()
                
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret:
                        break
                    
                    if model:
                        results = model(frame, conf=conf_thres, iou=iou_thres)
                        res_plotted = results[0].plot()
                        stframe.image(res_plotted, channels="BGR")
                    
                vf.release()
                st.success("视频检测完成")

with tab2:
    st.header("批量检测与报告生成")
    uploaded_files = st.file_uploader("上传多张图片", type=["jpg", "png"], accept_multiple_files=True)
    
    if uploaded_files and st.button("批量处理"):
        progress_bar = st.progress(0)
        results_data = []
        
        for i, file in enumerate(uploaded_files):
            # Save temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            if model:
                res = model(tmp_path, conf=conf_thres, iou=iou_thres)[0]
                count = len(res.boxes)
                results_data.append({
                    "文件名": file.name,
                    "缺陷数量": count,
                    "检测详情": [model.names[int(c)] for c in res.boxes.cls]
                })
            
            os.unlink(tmp_path)
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        st.success("批量处理完成！")
        df = pd.DataFrame(results_data)
        st.dataframe(df)
        
        # Export to CSV/Excel
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "下载 CSV 报告",
            csv,
            "detection_report.csv",
            "text/csv",
            key='download-csv'
        )

with tab3:
    st.markdown("""
    ## 📚 项目说明
    
    本项目是基于 **YOLO (You Only Look Once)** 系列模型的道路缺陷检测系统。
    
    ### ✨ 主要功能
    - **多模型支持**: 兼容 YOLOv8, YOLO11, YOLO26。
    - **高精度检测**: 针对路面坑洼 (Pothole) 进行专门训练。
    - **批量处理**: 支持一次性导入多张图片并生成统计报告。
    
    ### 🧠 模型介绍
    - **YOLO26**: Ultralytics 最新推出的端到端实时检测模型，无需 NMS，速度更快，精度更高。
    - **YOLO11**: 经典的上一代 SOTA 模型。
    
    ### 💻 技术栈
    - Python
    - Streamlit
    - Ultralytics
    - OpenCV
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed by YourName for Graduation Project")
