import streamlit as st
import pandas as pd
import os
import re
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="空间组学分析面板",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Configuration ---
# Makes the script runnable from any location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, 'results_simple')


# --- Helper Functions ---
@st.cache_data
def get_available_resolutions(path):
    """Scan the results directory and return a sorted map of resolutions."""
    resolutions = {}
    if not os.path.isdir(path):
        st.error(f"错误: 结果目录 '{path}' 不存在。请确认分析脚本已成功运行。")
        return {}

    for f in os.listdir(path):
        match = re.search(r'spatial_clusters_r([\d\.]+)\.png', f)
        if match:
            res_str = match.group(1)
            resolutions[float(res_str)] = res_str

    return dict(sorted(resolutions.items()))


@st.cache_data
def find_file(path, pattern):
    """Find a file in a directory that matches a regex pattern."""
    if not os.path.isdir(path):
        return None

    files = [f for f in os.listdir(path) if re.search(pattern, f)]
    if files:
        # Prioritize files with 'optimal' in their name if multiple matches are found
        optimal_files = [f for f in files if 'optimal' in f]
        return os.path.join(path, optimal_files[0] if optimal_files else files[0])
    return None


# --- Main App ---

# --- Sidebar ---
with st.sidebar:
    st.header('🔬 分析参数选择')

    resolutions_map = get_available_resolutions(RESULTS_PATH)

    if not resolutions_map:
        st.warning("在结果目录中未找到任何有效的聚类结果文件 (例如 'spatial_clusters_r0.8.png')。")
        st.info("请先成功运行 `pca_leiden_clustering.py` 脚本以生成结果。")
        st.stop()

    selected_res_float = st.selectbox(
        '选择聚类分辨率 (Resolution)',
        options=list(resolutions_map.keys()),
        format_func=lambda x: f'r = {x:.2f}'
    )
    res_str = resolutions_map[selected_res_float]
    st.markdown("---")
    st.info("这是一个交互式面板，用于探索不同聚类分辨率下的空间组学分析结果。")

# --- Main Panel ---
st.title('空间转录组聚类分析仪表盘')
st.markdown(f"### 当前查看分辨率: **r = {res_str}**")

# Define more granular tabs for a cleaner interface
tab_spatial, tab_umap, tab_marker, tab_qc_eval, tab_qc_spatial = st.tabs([
    "🗺️ 空间聚类视图",
    "🧬 UMAP视图",
    "📊 标志基因分析",
    "📈 聚类质量评估",
    "📉 空间结构检验"
])

# --- Tab 1: Spatial Clustering View ---
with tab_spatial:
    st.header("空间聚类视图 (Spatial Clustering)")
    st.markdown("在组织切片的原始物理空间上查看聚类结果的分布。")
    spatial_img_path = os.path.join(RESULTS_PATH, f'spatial_clusters_r{res_str}.png')
    if os.path.exists(spatial_img_path):
        st.image(spatial_img_path, use_column_width=True)
    else:
        st.warning(f"找不到文件: {os.path.basename(spatial_img_path)}")

# --- Tab 2: UMAP View ---
with tab_umap:
    st.header("UMAP 聚类视图")
    st.markdown("在基于基因表达相似性构建的低维空间中查看聚类结果。")
    umap_img_path = os.path.join(RESULTS_PATH, f'umap_clusters_r{res_str}.png')
    if os.path.exists(umap_img_path):
        st.image(umap_img_path, use_column_width=True)
    else:
        st.warning(f"找不到文件: {os.path.basename(umap_img_path)}")

# --- Tab 3: Marker Gene Exploration ---
with tab_marker:
    st.header("各聚类标志基因分析")
    st.markdown("标志基因是在特定聚类中表达水平显著高于其他聚类的基因，它们可以帮助我们定义和命名细胞类型或状态。")

    st.subheader('标志基因热力点图 (Dot Plot)')
    dotplot_img_path = os.path.join(RESULTS_PATH, f'marker_genes_dotplot_r{res_str}.png')
    if os.path.exists(dotplot_img_path):
        st.image(dotplot_img_path, use_column_width=True)
    else:
        st.warning(f"找不到文件: {os.path.basename(dotplot_img_path)}")

    st.divider()

    st.subheader('标志基因列表 (Top 20)')
    marker_csv_path = os.path.join(RESULTS_PATH, f'marker_genes_r{res_str}.csv')
    if os.path.exists(marker_csv_path):
        try:
            marker_df = pd.read_csv(marker_csv_path, index_col=0)
            st.dataframe(marker_df)
        except Exception as e:
            st.error(f"读取或展示标志基因CSV文件时出错: {e}")
    else:
        st.warning(f"找不到文件: {os.path.basename(marker_csv_path)}")

# --- Tab 4: Clustering Quality Evaluation ---
with tab_qc_eval:
    st.header("聚类质量与参数选择")
    st.markdown("这部分图表不受侧边栏分辨率选择的影响，展示了本次分析的总体质量和最优参数选择依据。")

    st.subheader('聚类评估指标')
    eval_metrics_path = os.path.join(RESULTS_PATH, 'clustering_evaluation_metrics.png')
    if os.path.exists(eval_metrics_path):
        st.image(eval_metrics_path, use_column_width=True)
        st.info("""
        **图表解读:**
        - **轮廓系数 (Silhouette Score):** 衡量聚类内部紧密度的指标，分数越高代表聚类效果越好。
        - **稳定性 (Stability, ARI):** 衡量相邻分辨率之间聚类结果的相似度。曲线进入“平原”表示聚类结果趋于稳定。
        - **红色虚线:** 代表脚本自动选择的“最优”分辨率，通常是稳定性平台的起点。
        """)
    else:
        st.warning(f"找不到文件: clustering_evaluation_metrics.png")

    st.divider()

    st.subheader('最优分辨率下聚类大小分布')
    optimal_dist_path = find_file(RESULTS_PATH, r'cluster_size_distribution_optimal\.png')
    if optimal_dist_path:
        st.image(optimal_dist_path, use_column_width=True, caption="展示了在最优分辨率下，每个聚类包含的细胞数量。")
    else:
        st.warning("找不到最优分辨率下的聚类大小分布图。")

# --- Tab 5: Spatial Structure Test ---
with tab_qc_spatial:
    st.header("空间结构显著性检验")
    st.markdown("该检验用于判断观察到的空间聚类模式是否具有统计学意义。")

    perm_test_pattern = r'spatial_permutation_test_.*\.png'
    perm_test_path = find_file(RESULTS_PATH, perm_test_pattern)
    if perm_test_path:
        st.image(perm_test_path, use_column_width=True)
        st.info("""
        **图表解读:**
        该检验通过将聚类标签随机打乱，来判断观察到的空间聚集模式是否是偶然的。
        - **真实分数 (红线):** 代表真实聚类的空间内聚性（值越低，空间聚集性越好）。
        - **P值 (P-value):** 如果P值很小 (例如 < 0.05), 则有力证明观察到的空间结构是真实存在的，而非随机形成。
        """)
    else:
        st.warning(f"找不到置换检验图 (e.g., spatial_permutation_test_leiden_r*.png)")