import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_all_results(results_dir):
    """
    从所有模型子目录加载 results.csv。
    返回 DataFrames 字典: {model_name: df}
    """
    data = {}
    results_path = Path(results_dir)
    
    # 遍历每个子目录 (例如 yolov8n_pothole)
    for model_dir in results_path.iterdir():
        if model_dir.is_dir():
            csv_path = model_dir / "results.csv"
            if csv_path.exists():
                try:
                    # 读取 CSV，去除列名的空白字符
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.strip()
                    # 保持与侧边栏选择模型名称一致，保留完整的文件夹名称
                    data[model_dir.name] = df
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
            else:
                print(f"Warning: No results.csv found in {model_dir}")
                
    return data

def plot_comparison(data, metric="metrics/mAP50-95(B)", title="Model Comparison"):
    """
    绘制所有模型的特定指标。
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, df in data.items():
        if metric in df.columns:
            plt.plot(df['epoch'], df[metric], label=model_name)
        else:
            print(f"Metric {metric} not found for {model_name}")
            
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt

def get_best_metrics(data):
    """
    提取每个模型的最佳指标值。
    """
    summary = []
    for model_name, df in data.items():
        # 获取最佳 mAP50 和 mAP50-95
        best_map50 = df['metrics/mAP50(B)'].max()
        best_map50_95 = df['metrics/mAP50-95(B)'].max()
        # 获取最终损失 (最后 5 个 epoch 的平均值)
        final_box_loss = df['val/box_loss'].tail(5).mean()
        
        summary.append({
            "Model": model_name,
            "Best mAP50": best_map50,
            "Best mAP50-95": best_map50_95,
            "Final Val Box Loss": final_box_loss
        })
    return pd.DataFrame(summary)
