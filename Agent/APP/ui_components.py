# Agent/APP/ui_components.py
import os
import glob
import streamlit as st

def render_download_section(target_directory):
    """
    扫描指定目录（及其子目录）下的 .docx 文件，并显示下载按钮区域。
    
    Args:
        target_directory (str): 需要扫描的文件夹路径
    """
    if not target_directory or not os.path.exists(target_directory):
        return

    # 🔥 [修改点] 启用递归搜索 (recursive=True)
    # 这样既能找到根目录的文件，也能找到 idea1_xxx/ 子目录里的文件
    search_pattern = os.path.join(target_directory, "**", "*.docx")
    docx_files = glob.glob(search_pattern, recursive=True)
    
    # 2. 如果有文件，渲染下载区
    if docx_files:
        st.markdown("---")
        st.markdown("### 📥 Download Research Reports")
        st.caption(f"Found {len(docx_files)} generated academic report(s):")
        
        # 使用列布局显示按钮 (每行 3 个)
        cols = st.columns(3)
        
        for idx, doc_path in enumerate(docx_files):
            file_name = os.path.basename(doc_path)
            
            # 为了美观，如果文件名太长，截取一下
            display_name = file_name
            if len(display_name) > 30:
                display_name = display_name[:25] + "..." + display_name[-5:]

            try:
                with open(doc_path, "rb") as f:
                    file_data = f.read()
                    
                # 在对应的列中放置下载按钮
                cols[idx % 3].download_button(
                    label=f"📄 {display_name}",
                    data=file_data,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    help=f"Click to download: {file_name}"
                )
            except Exception as e:
                st.error(f"Error loading {file_name}: {e}")