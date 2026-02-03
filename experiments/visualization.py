"""
結果視覺化工具
生成 t-SNE, UMAP, PCA 等降維視覺化
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from loguru import logger

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_fingerprints(fingerprint_dir: str) -> Dict:
    """
    載入多個模型的指紋
    
    Args:
        fingerprint_dir: 指紋文件目錄
    
    Returns:
        模型名稱到指紋向量的映射
    """
    fingerprints = {}
    
    fingerprint_path = Path(fingerprint_dir)
    
    for fp_file in fingerprint_path.glob("*_fingerprint.json"):
        model_name = fp_file.stem.replace("_fingerprint", "")
        
        with open(fp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "logit_fingerprint" in data and "vector" in data["logit_fingerprint"]:
            vector = np.array(data["logit_fingerprint"]["vector"])
            fingerprints[model_name] = vector
        else:
            logger.warning(f"跳過 {fp_file}（無有效向量數據）")
    
    logger.info(f"已載入 {len(fingerprints)} 個模型指紋")
    
    return fingerprints


def plot_tsne(
    fingerprints: Dict,
    output_path: str,
    perplexity: int = 30,
    n_iter: int = 1000
):
    """
    使用 t-SNE 進行降維並視覺化
    
    Args:
        fingerprints: 模型指紋字典
        output_path: 輸出圖片路徑
        perplexity: t-SNE 困惑度參數
        n_iter: 迭代次數
    """
    from sklearn.manifold import TSNE
    
    logger.info("執行 t-SNE 降維...")
    
    # 準備數據
    model_names = list(fingerprints.keys())
    vectors = np.array([fingerprints[name] for name in model_names])
    
    # 執行 t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(model_names) - 1),
        n_iter=n_iter,
        random_state=42
    )
    
    embeddings = tsne.fit_transform(vectors)
    
    # 視覺化
    plt.figure(figsize=(12, 8))
    
    # 根據模型來源著色
    colors = []
    for name in model_names:
        if any(cn in name.lower() for cn in ['qwen', 'deepseek', 'yi']):
            colors.append('red')
        elif 'llama' in name.lower():
            colors.append('blue')
        elif 'gemma' in name.lower():
            colors.append('green')
        else:
            colors.append('gray')
    
    plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=colors,
        s=200,
        alpha=0.6,
        edgecolors='black',
        linewidths=1.5
    )
    
    # 添加標籤
    for i, name in enumerate(model_names):
        plt.annotate(
            name,
            (embeddings[i, 0], embeddings[i, 1]),
            fontsize=10,
            ha='center',
            va='bottom'
        )
    
    plt.title('t-SNE Visualization of LLM Behavioral Fingerprints', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    
    # 添加圖例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Chinese Models (Qwen/DeepSeek/Yi)'),
        Patch(facecolor='blue', label='Meta Models (Llama)'),
        Patch(facecolor='green', label='Google Models (Gemma)'),
        Patch(facecolor='gray', label='Others'),
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ t-SNE 圖已保存: {output_path}")
    
    plt.close()


def plot_umap(
    fingerprints: Dict,
    output_path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1
):
    """
    使用 UMAP 進行降維並視覺化
    
    Args:
        fingerprints: 模型指紋字典
        output_path: 輸出圖片路徑
        n_neighbors: UMAP 鄰居數
        min_dist: UMAP 最小距離
    """
    try:
        import umap
    except ImportError:
        logger.error("請安裝 umap-learn: pip install umap-learn")
        return
    
    logger.info("執行 UMAP 降維...")
    
    # 準備數據
    model_names = list(fingerprints.keys())
    vectors = np.array([fingerprints[name] for name in model_names])
    
    # 執行 UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(model_names) - 1),
        min_dist=min_dist,
        random_state=42
    )
    
    embeddings = reducer.fit_transform(vectors)
    
    # 視覺化（與 t-SNE 類似的代碼）
    plt.figure(figsize=(12, 8))
    
    colors = []
    for name in model_names:
        if any(cn in name.lower() for cn in ['qwen', 'deepseek', 'yi']):
            colors.append('red')
        elif 'llama' in name.lower():
            colors.append('blue')
        elif 'gemma' in name.lower():
            colors.append('green')
        else:
            colors.append('gray')
    
    plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=colors,
        s=200,
        alpha=0.6,
        edgecolors='black',
        linewidths=1.5
    )
    
    for i, name in enumerate(model_names):
        plt.annotate(
            name,
            (embeddings[i, 0], embeddings[i, 1]),
            fontsize=10,
            ha='center',
            va='bottom'
        )
    
    plt.title('UMAP Visualization of LLM Behavioral Fingerprints', fontsize=16, fontweight='bold')
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Chinese Models'),
        Patch(facecolor='blue', label='Meta Models'),
        Patch(facecolor='green', label='Google Models'),
        Patch(facecolor='gray', label='Others'),
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ UMAP 圖已保存: {output_path}")
    
    plt.close()


def plot_similarity_matrix(
    fingerprints: Dict,
    output_path: str
):
    """
    繪製相似度矩陣熱圖
    
    Args:
        fingerprints: 模型指紋字典
        output_path: 輸出圖片路徑
    """
    from src.attribution.similarity import SimilarityCalculator
    
    logger.info("計算相似度矩陣...")
    
    model_names = list(fingerprints.keys())
    n = len(model_names)
    
    similarity_matrix = np.zeros((n, n))
    
    calc = SimilarityCalculator()
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = calc.cosine_similarity(
                    fingerprints[model_names[i]],
                    fingerprints[model_names[j]]
                )
                similarity_matrix[i, j] = sim
    
    # 視覺化
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        similarity_matrix,
        xticklabels=model_names,
        yticklabels=model_names,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    plt.title('Model Fingerprint Similarity Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ 相似度矩陣已保存: {output_path}")
    
    plt.close()


def plot_dendrogram(
    fingerprints: Dict,
    output_path: str
):
    """
    繪製層次聚類樹狀圖
    
    Args:
        fingerprints: 模型指紋字典
        output_path: 輸出圖片路徑
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    logger.info("執行層次聚類...")
    
    model_names = list(fingerprints.keys())
    vectors = np.array([fingerprints[name] for name in model_names])
    
    # 執行層次聚類
    linkage_matrix = linkage(vectors, method='ward')
    
    # 視覺化
    plt.figure(figsize=(12, 6))
    
    dendrogram(
        linkage_matrix,
        labels=model_names,
        leaf_font_size=10,
        leaf_rotation=45
    )
    
    plt.title('Hierarchical Clustering of LLM Fingerprints', fontsize=16, fontweight='bold')
    plt.xlabel('Model Name', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ 樹狀圖已保存: {output_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM 指紋視覺化工具")
    parser.add_argument(
        "--fingerprints-dir",
        default="data/fingerprints",
        help="指紋文件目錄"
    )
    parser.add_argument(
        "--output-dir",
        default="results/visualizations",
        help="輸出目錄"
    )
    parser.add_argument(
        "--plots",
        nargs='+',
        default=['tsne', 'umap', 'similarity', 'dendrogram'],
        choices=['tsne', 'umap', 'similarity', 'dendrogram', 'all'],
        help="要生成的圖表類型"
    )
    
    args = parser.parse_args()
    
    # 載入指紋
    fingerprints = load_fingerprints(args.fingerprints_dir)
    
    if len(fingerprints) < 2:
        logger.error("至少需要2個模型指紋才能進行視覺化")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots = args.plots
    if 'all' in plots:
        plots = ['tsne', 'umap', 'similarity', 'dendrogram']
    
    # 生成圖表
    logger.info("=" * 60)
    logger.info("開始生成視覺化圖表...")
    logger.info("=" * 60)
    
    if 'tsne' in plots:
        plot_tsne(fingerprints, str(output_dir / "tsne_visualization.png"))
    
    if 'umap' in plots:
        plot_umap(fingerprints, str(output_dir / "umap_visualization.png"))
    
    if 'similarity' in plots:
        plot_similarity_matrix(fingerprints, str(output_dir / "similarity_matrix.png"))
    
    if 'dendrogram' in plots:
        plot_dendrogram(fingerprints, str(output_dir / "dendrogram.png"))
    
    logger.info("=" * 60)
    logger.info("✓ 所有視覺化圖表已生成")
    logger.info("=" * 60)


if __name__ == "__main__":
    # 設置 matplotlib 支持中文
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()
