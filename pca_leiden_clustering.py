import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, lil_matrix
import gc
import matplotlib as mpl
import warnings
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform

try:
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi',
                     'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'STXihei',
                     'Arial Unicode MS', 'DejaVu Sans']

    available_fonts = []
    for font in chinese_fonts:
        try:
            if any(font.lower() in f.lower() for f in mpl.font_manager.findSystemFonts()):
                available_fonts.append(font)
        except:
            pass

    if available_fonts:
        plt.rcParams['font.sans-serif'] = available_fonts + ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        mpl.font_manager._rebuild()
        print(f"找到可用的中文字体: {available_fonts}")
    else:
        print("找不到中文字体，将使用英文")
        raise Exception("No Chinese fonts available")
except Exception as e:
    print(f"中文字体设置失败: {str(e)}，使用英文标题")

# 设置随机种子
np.random.seed(42)

# 设置scanpy图表参数
sc.settings.verbosity = 1
sc.settings.set_figure_params(
    dpi=300,
    facecolor='white',
    frameon=False,
)


def get_distinct_colors(n):
    import colorsys
    HSV_tuples = [(x*1.0/n, 0.85, 0.9) for x in range(n)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    return RGB_tuples

def find_optimal_resolution(adata, resolutions, output_path, use_rep='X_pca', sample_max=20000):
    """
    通过寻找稳定性平台的"起点"来确定最优分辨率。
    该策略旨在找到达到稳定状态的最低分辨率，避免过度聚类。
    """
    print("正在寻找最优聚类分辨率...")
    scores = []
    cluster_counts = []
    all_labels = []

    # 抽样数据以加速计算
    adata_sample = adata
    if adata.shape[0] > sample_max:
        print(f"数据集过大，抽样 {sample_max} 个细胞用于轮廓系数计算...")
        indices = np.random.choice(adata.shape[0], sample_max, replace=False)
        adata_sample = adata[indices, :].copy()

    X_pca_sample = adata_sample.obsm[use_rep]

    for res in resolutions:
        sc.tl.leiden(adata_sample, resolution=res, key_added=f'leiden_temp', random_state=42)
        labels = adata_sample.obs['leiden_temp']
        all_labels.append(labels.values)
        n_clusters = len(np.unique(labels))
        cluster_counts.append(n_clusters)

        if n_clusters > 1:
            score = silhouette_score(X_pca_sample, labels)
            scores.append(score)
            print(f"分辨率: {res:.2f}, 聚类数: {n_clusters}, 轮廓系数: {score:.4f}")
        else:
            scores.append(-1)
            print(f"分辨率: {res:.2f}, 聚类数: {n_clusters}, 无法计算轮廓系数。")

    del adata_sample
    gc.collect()

    # 计算聚类稳定性 (Adjusted Rand Index)
    stabilities = [adjusted_rand_score(all_labels[i], all_labels[i+1]) for i in range(len(all_labels) - 1)]
    stabilities.append(0)

    scores_df = pd.DataFrame({
        'resolution': resolutions,
        'silhouette_score': scores,
        'n_clusters': cluster_counts,
        'stability': stabilities
    })

    valid_scores_df = scores_df[scores_df['n_clusters'] > 1].copy()

    if valid_scores_df.empty:
        print("警告: 没有有效聚类。将使用默认分辨率0.8。")
        optimal_resolution = 0.8
    else:
        print("使用'稳定平台起点'策略寻找最优分辨率...")
        max_stability = valid_scores_df['stability'].max()

        stable_platform_df = valid_scores_df[valid_scores_df['stability'] >= 0.95 * max_stability]
        
        if not stable_platform_df.empty:
            best_idx = stable_platform_df.index[0]
            optimal_resolution = valid_scores_df.loc[best_idx, 'resolution']
            best_stability = valid_scores_df.loc[best_idx, 'stability']
            n_clus = valid_scores_df.loc[best_idx, 'n_clusters']
            print(f"找到稳定平台的'起点' (第一个达到最大稳定性95%的点)。")
            print(f"最优分辨率: {optimal_resolution:.2f} (产生 {n_clus} 个聚类, 稳定性ARI: {best_stability:.4f})")
        else:
            best_idx = valid_scores_df['stability'].idxmax()
            optimal_resolution = valid_scores_df.loc[best_idx, 'resolution']
            print(f"回退方案：选择稳定性最高的点，分辨率: {optimal_resolution:.2f}")

    # 绘制评估指标图
    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.lineplot(data=scores_df, x='resolution', y='silhouette_score', marker='o', ax=ax1, color='b', label='Silhouette Score')
    ax1.set_xlabel('Resolution')
    ax1.set_ylabel('Silhouette Score', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    sns.lineplot(data=scores_df, x='resolution', y='stability', marker='s', ax=ax2, color='g', label='Stability (ARI)')
    ax2.set_ylabel('Stability (vs. next resolution)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    plt.axvline(x=optimal_resolution, color='r', linestyle='--', label=f'Optimal Resolution: {optimal_resolution:.2f}')
    fig.suptitle('Clustering Evaluation Metrics vs. Resolution')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.grid(True)
    plt.savefig(f"{output_path}/clustering_evaluation_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

    return optimal_resolution, scores_df

def validate_spatial_structure(adata, cluster_key, output_path, n_permutations=100):
    """
    通过置换检验（熵轮换策略）来验证聚类的空间结构的统计显著性。
    """
    print(f"\n--- 开始通过置换检验验证空间结构 ({cluster_key}) ---")
    
    # 1. 定义计算空间凝聚度分数的函数
    def calculate_spatial_cohesion(coords, labels):
        unique_labels = np.unique(labels)
        total_cohesion = 0
        total_weight = 0
        
        for label in unique_labels:
            mask = (labels == label)
            if np.sum(mask) > 1:
                cluster_coords = coords[mask]
                # 计算簇内所有点两两之间的距离，然后取平均值
                avg_dist = pdist(cluster_coords).mean()
                
                weight = len(cluster_coords)
                total_cohesion += avg_dist * weight
                total_weight += weight
        
        return total_cohesion / total_weight if total_weight > 0 else 0

    coords = adata.obsm['spatial']
    original_labels = adata.obs[cluster_key].values

    # 2. 计算真实聚类的空间凝聚度
    real_cohesion_score = calculate_spatial_cohesion(coords, original_labels)
    print(f"真实聚类的空间凝聚度分数: {real_cohesion_score:.4f} (值越低越好)")

    # 3. 进行N次置换
    print(f"正在进行 {n_permutations} 次置换以构建零分布...")
    permuted_scores = []
    for i in range(n_permutations):
        permuted_labels = np.random.permutation(original_labels)
        score = calculate_spatial_cohesion(coords, permuted_labels)
        permuted_scores.append(score)
        if (i+1) % 20 == 0:
            print(f"  完成 {i+1}/{n_permutations} 次置换...")
            
    # 4. 计算P值
    p_value = (np.sum(np.array(permuted_scores) <= real_cohesion_score) + 1) / (n_permutations + 1)
    print(f"置换检验完成。")
    print(f"经验P值: {p_value:.4f}")
    if p_value < 0.05:
        print("结论: 观察到的空间聚类结构具有统计学显著性 (P < 0.05)。")
    else:
        print("结论: 没有足够的证据表明观察到的空间结构是显著的 (P >= 0.05)。")

    # 5. 绘制结果图
    plt.figure(figsize=(10, 6))
    sns.histplot(permuted_scores, kde=True, label='Permuted Scores (Null Distribution)', color='grey')
    plt.axvline(real_cohesion_score, color='r', linestyle='--', label=f'Real Score: {real_cohesion_score:.4f}')
    plt.title(f'Spatial Clustering Permutation Test\nP-value: {p_value:.4f}')
    plt.xlabel('Spatial Cohesion Score (Lower is Better)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}/spatial_permutation_test_{cluster_key}.png", dpi=300, bbox_inches='tight')
    plt.close()

    return p_value

def run_pipeline(data_path, output_path='./results_simple',
                n_pcs=50, n_neighbors=15, spatial_weight=0.5,
                use_english=False):

    os.makedirs(output_path, exist_ok=True)

    title_lang = "en" if use_english else "cn"
    title_dict = {
        "cn": {
            "umap_title": "UMAP聚类 (r={})",
            "spatial_title": "空间聚类 (r={})",
            "cluster_size_title": "聚类大小分布 (r={})",
            "cluster_id": "聚类ID",
            "cell_count": "细胞数量",
            "gene_title": "基因: {}"
        },
        "en": {
            "umap_title": "UMAP Clustering (r={})",
            "spatial_title": "Spatial Clustering (r={})",
            "cluster_size_title": "Cluster Size Distribution (r={})",
            "cluster_id": "Cluster ID",
            "cell_count": "Cell Count",
            "gene_title": "Gene: {}"
        }
    }

    print("加载数据...")
    adata = sc.read_10x_h5(f"{data_path}/filtered_feature_bc_matrix.h5")

    adata.var_names_make_unique()

    expression_barcodes = list(adata.obs_names)
    print(f"表达矩阵包含 {len(expression_barcodes)} 个条形码")

    print("加载空间坐标...")
    spatial_path = f"{data_path}/spatial"

    try:
        spatial_coords = pd.read_parquet(f"{spatial_path}/tissue_positions.parquet")
        print("成功读取parquet文件")

        if 'array_row' in spatial_coords.columns and 'array_col' in spatial_coords.columns:
            coord_cols = ['array_row', 'array_col']
        elif 'pxl_row_in_fullres' in spatial_coords.columns and 'pxl_col_in_fullres' in spatial_coords.columns:
            coord_cols = ['pxl_row_in_fullres', 'pxl_col_in_fullres']
        else:
            coord_cols = list(spatial_coords.columns[-2:])

        print(f"使用坐标列: {coord_cols}")

        if 'barcode' in spatial_coords.columns:
            spatial_coords.set_index('barcode', inplace=True)

            common_barcodes = list(set(expression_barcodes) & set(spatial_coords.index))
            print(f"表达矩阵和坐标文件共有 {len(common_barcodes)} 个匹配的条形码")

            adata = adata[adata.obs_names.isin(common_barcodes)].copy()

            adata.obsm['spatial'] = spatial_coords.loc[adata.obs_names, coord_cols].values
            print(f"成功加载 {adata.obsm['spatial'].shape[0]} 个空间坐标点")
    except Exception as e:
        print(f"读取空间坐标失败: {str(e)}")
        return None

    print("数据预处理...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"预处理后数据形状: {adata.shape}")

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    print(f"选择了 {adata.shape[1]} 个高变异基因")

    batch_size = 5000
    max_cells = 20000

    if adata.shape[0] > max_cells:
        print(f"数据集过大，从 {adata.shape[0]} 个细胞中抽样 {max_cells} 个...")
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)
        print(f"抽样后数据形状: {adata.shape}")

    # PCA降维
    print("执行PCA降维...")
    sc.pp.pca(adata, n_comps=n_pcs)
    print(f"PCA完成，形状: {adata.obsm['X_pca'].shape}")

    # 创建基于空间和基因表达的组合邻域图
    print("构建组合邻域图...")

    # 1. 基于PCA的基因表达邻接图
    print("构建基于基因表达的邻接图...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca')
    conn_expression = adata.obsp['connectivities'].copy()

    # 2. 构建空间邻接图
    print("构建空间邻接图...")
    spatial_coords = adata.obsm['spatial']

    conn_spatial_dist = kneighbors_graph(spatial_coords, n_neighbors=n_neighbors, mode='distance')

    max_known_dist = conn_spatial_dist.data.max()
    if max_known_dist > 0:
        conn_spatial_dist.data = 1 - (conn_spatial_dist.data / max_known_dist)
    else:
        conn_spatial_dist.data = np.ones_like(conn_spatial_dist.data)

    conn_spatial = conn_spatial_dist.maximum(conn_spatial_dist.transpose())

    # 3. 自适应权重结合表达和空间图
    print(f"结合表达与空间信息，空间权重: {spatial_weight}...")
    adata.obsp['connectivities'] = (1 - spatial_weight) * conn_expression + spatial_weight * conn_spatial

    print("自动选择最优聚类分辨率...")
    resolutions_to_test = np.round(np.arange(0.1, 2.1, 0.1), 2)
    optimal_resolution, _ = find_optimal_resolution(adata, resolutions_to_test, output_path)
    print(f"推荐的最优分辨率为: {optimal_resolution:.2f}")

    resolutions_to_analyze = sorted(list(set([optimal_resolution] + [0.4, 0.6, 0.8, 1.0])))
    print(f"将对以下分辨率进行完整分析: {resolutions_to_analyze}")

    # 为所有待分析的分辨率执行Leiden聚类
    for res in resolutions_to_analyze:
        if not adata.obs.get(f'leiden_r{res}'):
             print(f"聚类分辨率: {res}...")
             sc.tl.leiden(adata, resolution=res, key_added=f'leiden_r{res}', random_state=42)

    # UMAP可视化
    print("计算UMAP降维...")
    sc.tl.umap(adata)

    for res in resolutions_to_analyze:
        print(f"--- 正在为分辨率 {res} 生成结果 ---")
        cluster_key = f'leiden_r{res}'
        n_clusters = len(adata.obs[cluster_key].unique())

        umap_title = title_dict[title_lang]["umap_title"].format(f'r={res}, k={n_clusters}')
        spatial_title = title_dict[title_lang]["spatial_title"].format(f'r={res}, k={n_clusters}')

        # 1. UMAP 可视化
        plt.figure(figsize=(12, 12))
        sc.pl.umap(adata, color=cluster_key, title=umap_title,
                   legend_loc='on data', legend_fontsize=12, size=80, show=False)
        plt.savefig(f"{output_path}/umap_clusters_r{res}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 空间聚类图
        plt.figure(figsize=(14, 14))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sc.pl.spatial(adata, img=None, color=cluster_key, title=spatial_title,
                          spot_size=8.0, show=False)
        plt.savefig(f"{output_path}/spatial_clusters_r{res}.png", dpi=400, bbox_inches='tight')
        plt.close()

        # 3. 识别标志基因
        print(f"识别分辨率 {res} 的簇标志基因...")
        sc.tl.rank_genes_groups(adata, cluster_key, method='wilcoxon')

        # 4. 可视化标志基因点图
        n_clusters_to_plot = min(n_clusters, 20)
        clusters_to_plot = list(np.unique(adata.obs[cluster_key]))[:n_clusters_to_plot]
        try:
            plt.figure(figsize=(15, min(20, n_clusters_to_plot * 0.8)))
            sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, groups=clusters_to_plot, dendrogram=False, standard_scale='var', show=False)
            plt.savefig(f"{output_path}/marker_genes_dotplot_r{res}.png", dpi=200, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"绘制分辨率 {res} 的点图时出错: {str(e)}")

        # 5. 导出标志基因列表
        try:
            result = adata.uns['rank_genes_groups']
            groups = result['names'].dtype.names
            markers_df = pd.DataFrame({group: result['names'][group][:20] for group in groups})
            markers_df.to_csv(f"{output_path}/marker_genes_r{res}.csv")
        except Exception as e:
            print(f"导出分辨率 {res} 的标志基因时出错: {str(e)}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 18))
    axes = axes.flatten()

    for i, res in enumerate(resolutions_to_analyze[:4]):
        ax = axes[i]
        n_clusters = len(adata.obs[f'leiden_r{res}'].unique())
        spatial_title = title_dict[title_lang]["spatial_title"].format(f'r={res}, k={n_clusters}')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sc.pl.spatial(adata, img=None, color=f'leiden_r{res}',
                     title=spatial_title,
                     spot_size=4.0, ax=ax, show=False)  # 增加点的大小

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(f"{output_path}/spatial_clusters_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    optimal_cluster_key = f'leiden_r{optimal_resolution}'
    cluster_counts = adata.obs[optimal_cluster_key].value_counts().sort_index()
    plt.figure(figsize=(12, 7))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
    n_clus_optimal = len(cluster_counts)
    plt.title(title_dict[title_lang]["cluster_size_title"].format(f"r={optimal_resolution:.2f}, k={n_clus_optimal}"), fontsize=16)
    plt.xlabel(title_dict[title_lang]["cluster_id"], fontsize=14)
    plt.ylabel(title_dict[title_lang]["cell_count"], fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}/cluster_size_distribution_optimal.png", dpi=300)
    plt.close()

    validate_spatial_structure(adata, optimal_cluster_key, output_path)

    adata.write(f"{output_path}/adata_results.h5ad")

    print(f"\n聚类分析完成！结果已保存至: {output_path}")
    return adata

if __name__ == "__main__":
    data_path = "../binned_outputs/square_008um"
    output_path = "./results_simple"

    print("直接使用英文标题避免中文字体问题...")
    adata = run_pipeline(data_path, output_path, use_english=True)