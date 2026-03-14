from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from datetime import datetime
import tempfile

# Register a font that supports Chinese (SimHei or similar)
# 我们将尝试使用系统字体或回退
try:
    # 尝试 SimHei 的常用 Windows 路径
    font_path = "C:/Windows/Fonts/simhei.ttf"
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('SimHei', font_path))
        FONT_NAME = 'SimHei'
    else:
        # 如果未找到 SimHei，则回退到标准字体 (不支持中文)
        # 或者你可以将字体文件打包在你的项目中
        FONT_NAME = 'Helvetica' 
except:
    FONT_NAME = 'Helvetica'

def generate_pdf_report(summary_data, detail_data, output_path):
    """
    生成检测结果的 PDF 报告。
    
    参数:
        summary_data (dict): {'model': str, 'total_files': int, 'total_defects': int, 'time': str}
        detail_data (list): 字典列表 {'filename': str, 'count': int, 'img_path': str, 'details': str}
        output_path (str): PDF 保存路径
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    # 自定义样式
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontName=FONT_NAME,
        fontSize=24,
        alignment=1, # 居中
        spaceAfter=30
    )
    
    normal_style = ParagraphStyle(
        'NormalCN',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=12,
        leading=20
    )
    
    header_style = ParagraphStyle(
        'HeaderCN',
        parent=styles['Heading2'],
        fontName=FONT_NAME,
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10
    )

    # 1. 标题页
    elements.append(Paragraph("道路缺陷检测报告", title_style))
    elements.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 20))
    
    # 2. 摘要部分
    elements.append(Paragraph("一、检测概览", header_style))
    
    summary_table_data = [
        ["检测模型", summary_data.get('model', 'N/A')],
        ["处理文件数", str(summary_data.get('total_files', 0))],
        ["发现缺陷总数", str(summary_data.get('total_defects', 0))],
        ["平均置信度", summary_data.get('avg_conf', 'N/A')]
    ]
    
    t = Table(summary_table_data, colWidths=[150, 300])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))
    
    # 3. 详细信息部分
    elements.append(Paragraph("二、详细检测结果", header_style))
    
    for item in detail_data:
        # 文件标题
        elements.append(Paragraph(f"文件名: {item['filename']} (缺陷数: {item['count']})", header_style))
        
        # 图片
        if item.get('img_path') and os.path.exists(item['img_path']):
            try:
                # 调整图像大小以适应页面宽度 (A4 宽度 ~595 pts，边距 ~1 英寸)
                # 可用宽度 ~450
                img = Image(item['img_path'], width=400, height=300, kind='proportional')
                elements.append(img)
            except Exception as e:
                elements.append(Paragraph(f"[图片加载失败: {e}]", normal_style))
        
        # 详细文本
        elements.append(Paragraph(f"检测详情: {item['details']}", normal_style))
        elements.append(Spacer(1, 15))
        # 添加分隔线
        # elements.append(Paragraph("_" * 50, normal_style))
        
    # 构建 PDF
    try:
        doc.build(elements)
        return True, "Success"
    except Exception as e:
        return False, str(e)
