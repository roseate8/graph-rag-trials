#!/usr/bin/env python3
"""
Comprehensive Embedding Statistical Analysis

This script provides detailed statistical analysis of embeddings generated 
from processed document chunks, including visualizations and recommendations.

Usage: python embedding_analysis.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_processed_chunks(file_path):
    """Load processed chunks from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {data['total_chunks']} chunks")
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def analyze_chunk_sizes(chunks):
    """Analyze chunk size distribution."""
    print("\n" + "="*60)
    print("📊 1. CHUNK SIZE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Extract word counts
    word_counts = [chunk.get('word_count', 0) for chunk in chunks]
    
    # Create comprehensive statistics
    word_count_stats = {
        'count': len(word_counts),
        'mean': np.mean(word_counts),
        'median': np.median(word_counts),
        'std': np.std(word_counts),
        'min': np.min(word_counts),
        'max': np.max(word_counts),
        'q25': np.percentile(word_counts, 25),
        'q75': np.percentile(word_counts, 75)
    }
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📊 Chunk Size Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Main histogram
    ax1.hist(word_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(word_count_stats['mean'], color='red', linestyle='--', linewidth=2, 
               label=f"Mean: {word_count_stats['mean']:.1f}")
    ax1.axvline(word_count_stats['median'], color='green', linestyle='--', linewidth=2, 
               label=f"Median: {word_count_stats['median']:.1f}")
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Chunk Sizes (Word Count)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2.boxplot(word_counts, vert=True, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_ylabel('Word Count')
    ax2.set_title('Box Plot of Chunk Sizes')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    sorted_counts = np.sort(word_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    ax3.plot(sorted_counts, cumulative, linewidth=2, color='purple')
    ax3.set_xlabel('Word Count')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution of Chunk Sizes')
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics table
    ax4.axis('off')
    stats_text = f"""
📈 Chunk Size Statistics

Count:      {word_count_stats['count']:,}
Mean:       {word_count_stats['mean']:.2f}
Median:     {word_count_stats['median']:.2f}
Std Dev:    {word_count_stats['std']:.2f}

Min:        {word_count_stats['min']:,}
Max:        {word_count_stats['max']:,}
Q25:        {word_count_stats['q25']:.2f}
Q75:        {word_count_stats['q75']:.2f}

IQR:        {word_count_stats['q75'] - word_count_stats['q25']:.2f}
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print insights
    print(f"\n🔍 Key Insights:")
    print(f"   • Average chunk size: {word_count_stats['mean']:.1f} words")
    print(f"   • Most chunks are between {word_count_stats['q25']:.0f} and {word_count_stats['q75']:.0f} words")
    
    very_short = sum(1 for wc in word_counts if wc < 50)
    very_long = sum(1 for wc in word_counts if wc > 500)
    print(f"   • {very_short} chunks ({very_short/len(word_counts)*100:.1f}%) are very short (< 50 words)")
    print(f"   • {very_long} chunks ({very_long/len(word_counts)*100:.1f}%) are very long (> 500 words)")
    
    return word_count_stats, word_counts

def analyze_embeddings(chunks):
    """Analyze embedding properties."""
    print("\n" + "="*60)
    print("🧮 2. EMBEDDING DIMENSIONALITY AND PROPERTIES ANALYSIS")
    print("="*60)
    
    # Extract embeddings
    embeddings = []
    for chunk in chunks:
        if 'embedding' in chunk and chunk['embedding']:
            embedding = np.array(chunk['embedding'])
            embeddings.append(embedding)
    
    if not embeddings:
        print("❌ No embeddings found in the data")
        return None, None
    
    embeddings_matrix = np.array(embeddings)
    
    print(f"🔢 Embedding Analysis:")
    print(f"   • Total embeddings: {len(embeddings):,}")
    print(f"   • Embedding dimension: {embeddings_matrix.shape[1]:,}")
    print(f"   • Matrix shape: {embeddings_matrix.shape}")
    
    # Calculate statistics
    embedding_stats = {
        'mean_norm': np.mean([np.linalg.norm(emb) for emb in embeddings]),
        'std_norm': np.std([np.linalg.norm(emb) for emb in embeddings]),
        'mean_values': np.mean(embeddings_matrix, axis=0),
        'std_values': np.std(embeddings_matrix, axis=0),
        'min_value': np.min(embeddings_matrix),
        'max_value': np.max(embeddings_matrix),
        'sparsity': np.mean(embeddings_matrix == 0) * 100
    }
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🧮 Embedding Properties Analysis', fontsize=16, fontweight='bold')
    
    # 1. Distribution of embedding norms
    norms = [np.linalg.norm(emb) for emb in embeddings]
    ax1.hist(norms, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax1.axvline(embedding_stats['mean_norm'], color='red', linestyle='--', 
               label=f"Mean: {embedding_stats['mean_norm']:.3f}")
    ax1.set_xlabel('L2 Norm')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Embedding Norms')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of embedding values
    flat_embeddings = embeddings_matrix.flatten()
    ax2.hist(flat_embeddings, bins=100, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Embedding Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of All Embedding Values')
    ax2.grid(True, alpha=0.3)
    
    # 3. Mean and std per dimension (sample first 100 dimensions)
    dims_to_show = min(100, len(embedding_stats['mean_values']))
    x_dims = range(dims_to_show)
    ax3.plot(x_dims, embedding_stats['mean_values'][:dims_to_show], 
            label='Mean', alpha=0.7, linewidth=2)
    ax3.fill_between(x_dims, 
                    embedding_stats['mean_values'][:dims_to_show] - embedding_stats['std_values'][:dims_to_show],
                    embedding_stats['mean_values'][:dims_to_show] + embedding_stats['std_values'][:dims_to_show],
                    alpha=0.3, label='±1 Std Dev')
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Value')
    ax3.set_title(f'Mean ± Std Dev per Dimension (First {dims_to_show})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics summary
    ax4.axis('off')
    stats_text = f"""
🧮 Embedding Statistics

Dimensions:     {embeddings_matrix.shape[1]:,}
Total vectors:  {len(embeddings):,}

Norm Statistics:
Mean norm:      {embedding_stats['mean_norm']:.4f}
Std norm:       {embedding_stats['std_norm']:.4f}

Value Range:
Min value:      {embedding_stats['min_value']:.4f}
Max value:      {embedding_stats['max_value']:.4f}

Sparsity:       {embedding_stats['sparsity']:.2f}%
Memory usage:   {embeddings_matrix.nbytes / 1024 / 1024:.1f} MB
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print insights
    print(f"\n🔍 Embedding Quality Insights:")
    print(f"   • Average embedding norm: {embedding_stats['mean_norm']:.4f}")
    print(f"   • Value range: [{embedding_stats['min_value']:.4f}, {embedding_stats['max_value']:.4f}]")
    print(f"   • Sparsity: {embedding_stats['sparsity']:.2f}% (zeros)")
    print(f"   • Memory usage: {embeddings_matrix.nbytes / 1024 / 1024:.1f} MB")
    
    return embeddings_matrix, embedding_stats

def analyze_content_structure(chunks):
    """Analyze content types and document structure."""
    print("\n" + "="*60)
    print("📄 3. CONTENT TYPE AND DOCUMENT STRUCTURE ANALYSIS")
    print("="*60)
    
    # Extract content metadata
    chunk_types = [chunk.get('chunk_type', 'unknown') for chunk in chunks]
    doc_ids = [chunk.get('doc_id', 'unknown') for chunk in chunks]
    section_paths = [chunk.get('section_path', []) for chunk in chunks]
    
    # Analyze section depths
    section_depths = [len(path) if path else 0 for path in section_paths]
    
    # Count occurrences
    type_counts = Counter(chunk_types)
    doc_counts = Counter(doc_ids)
    depth_counts = Counter(section_depths)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📄 Content Type and Structure Analysis', fontsize=16, fontweight='bold')
    
    # 1. Chunk types distribution
    if type_counts:
        types, type_values = zip(*type_counts.most_common())
        colors1 = plt.cm.Set3(np.linspace(0, 1, len(types)))
        ax1.pie(type_values, labels=types, autopct='%1.1f%%', colors=colors1, startangle=90)
        ax1.set_title('Distribution of Chunk Types')
    
    # 2. Document distribution
    if doc_counts:
        docs, doc_values = zip(*doc_counts.most_common())
        ax2.bar(range(len(docs)), doc_values, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Document')
        ax2.set_ylabel('Number of Chunks')
        ax2.set_title('Chunks per Document')
        ax2.set_xticks(range(len(docs)))
        ax2.set_xticklabels([doc[:20] + '...' if len(doc) > 20 else doc for doc in docs], 
                           rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
    
    # 3. Section depth distribution
    if depth_counts:
        depths, depth_values = zip(*sorted(depth_counts.items()))
        ax3.bar(depths, depth_values, color='lightblue', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Section Depth')
        ax3.set_ylabel('Number of Chunks')
        ax3.set_title('Distribution of Section Depths')
        ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4.axis('off')
    
    # Get top sections
    all_sections = []
    for path in section_paths:
        if path:
            all_sections.extend(path)
    section_counter = Counter(all_sections)
    top_sections = section_counter.most_common(5)
    
    summary_text = f"""
📊 Content Structure Summary

Chunk Types:
{chr(10).join([f'  • {t}: {v:,} ({v/len(chunks)*100:.1f}%)' for t, v in type_counts.most_common()])}

Documents:
{chr(10).join([f'  • {d[:30]}: {v:,}' for d, v in list(doc_counts.most_common())[:3]])}

Section Depths:
• Average depth: {np.mean(section_depths):.1f}
• Max depth: {max(section_depths) if section_depths else 0}
• Chunks with sections: {sum(1 for d in section_depths if d > 0):,}

Top Sections:
{chr(10).join([f'  • {s[:25]}: {c}' for s, c in top_sections[:3]])}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print insights
    print(f"\n🔍 Content Structure Insights:")
    print(f"   • Total unique documents: {len(doc_counts)}")
    if type_counts:
        print(f"   • Most common chunk type: {type_counts.most_common(1)[0][0]} ({type_counts.most_common(1)[0][1]:,} chunks)")
    print(f"   • Average section depth: {np.mean(section_depths):.1f}")
    print(f"   • {sum(1 for d in section_depths if d == 0):,} chunks ({sum(1 for d in section_depths if d == 0)/len(chunks)*100:.1f}%) have no section structure")
    
    return type_counts, doc_counts, section_depths

def analyze_correlations(chunks, embeddings_matrix):
    """Analyze correlations between chunk properties and embedding characteristics."""
    print("\n" + "="*60)
    print("🔗 4. CORRELATION ANALYSIS")
    print("="*60)
    
    if embeddings_matrix is None:
        print("❌ No embeddings available for correlation analysis")
        return
    
    # Create DataFrame for correlation analysis
    analysis_data = []
    embeddings = [embeddings_matrix[i] for i in range(len(embeddings_matrix))]
    
    for i, chunk in enumerate(chunks):
        if i < len(embeddings):  # Ensure we have embedding for this chunk
            embedding = embeddings[i]
            
            # Calculate embedding characteristics
            emb_norm = np.linalg.norm(embedding)
            emb_mean = np.mean(embedding)
            emb_std = np.std(embedding)
            emb_max = np.max(embedding)
            emb_min = np.min(embedding)
            emb_range = emb_max - emb_min
            
            # Get chunk properties
            word_count = chunk.get('word_count', 0)
            section_depth = len(chunk.get('section_path', []))
            content_length = len(str(chunk.get('content', '')))
            
            analysis_data.append({
                'word_count': word_count,
                'content_length': content_length,
                'section_depth': section_depth,
                'embedding_norm': emb_norm,
                'embedding_mean': emb_mean,
                'embedding_std': emb_std,
                'embedding_range': emb_range,
                'embedding_max': emb_max,
                'embedding_min': emb_min
            })
    
    df = pd.DataFrame(analysis_data)
    
    # Calculate correlation matrix
    correlation_matrix = df.corr()
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔗 Correlation Analysis: Chunk Properties vs Embedding Characteristics', 
                 fontsize=16, fontweight='bold')
    
    # 1. Correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax1, fmt='.3f', cbar_kws={'shrink': 0.8})
    ax1.set_title('Correlation Matrix')
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # 2. Word count vs embedding norm scatter
    ax2.scatter(df['word_count'], df['embedding_norm'], alpha=0.6, s=20, color='blue')
    ax2.set_xlabel('Word Count')
    ax2.set_ylabel('Embedding Norm')
    ax2.set_title('Word Count vs Embedding Norm')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr_coef = df['word_count'].corr(df['embedding_norm'])
    ax2.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Content length vs embedding std
    ax3.scatter(df['content_length'], df['embedding_std'], alpha=0.6, s=20, color='green')
    ax3.set_xlabel('Content Length (characters)')
    ax3.set_ylabel('Embedding Std Dev')
    ax3.set_title('Content Length vs Embedding Variability')
    ax3.grid(True, alpha=0.3)
    
    corr_coef2 = df['content_length'].corr(df['embedding_std'])
    ax3.text(0.05, 0.95, f'Correlation: {corr_coef2:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Section depth analysis
    depth_stats = df.groupby('section_depth').agg({
        'embedding_norm': ['mean', 'std'],
        'word_count': 'count'
    }).round(3)
    
    # Flatten column names
    depth_stats.columns = ['_'.join(col).strip() for col in depth_stats.columns]
    depth_stats = depth_stats.reset_index()
    
    if len(depth_stats) > 1:
        ax4.bar(depth_stats['section_depth'], depth_stats['embedding_norm_mean'], 
                yerr=depth_stats['embedding_norm_std'], capsize=5, alpha=0.7, color='orange')
        ax4.set_xlabel('Section Depth')
        ax4.set_ylabel('Mean Embedding Norm')
        ax4.set_title('Embedding Norm by Section Depth')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient section depth\nvariability for analysis', 
                ha='center', va='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.set_title('Section Depth Analysis')
    
    plt.tight_layout()
    plt.show()
    
    # Print key correlations
    print(f"\n🔍 Key Correlations:")
    print(f"   • Word count ↔ Embedding norm: {df['word_count'].corr(df['embedding_norm']):.3f}")
    print(f"   • Content length ↔ Embedding std: {df['content_length'].corr(df['embedding_std']):.3f}")
    print(f"   • Section depth ↔ Embedding norm: {df['section_depth'].corr(df['embedding_norm']):.3f}")
    print(f"   • Word count ↔ Content length: {df['word_count'].corr(df['content_length']):.3f}")
    
    return df

def perform_clustering_analysis(embeddings_matrix, chunks):
    """Perform clustering analysis on embeddings."""
    print("\n" + "="*60)
    print("🎯 5. CLUSTERING ANALYSIS")
    print("="*60)
    
    if embeddings_matrix is None or len(embeddings_matrix) < 10:
        print("❌ Insufficient data for clustering analysis")
        return
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("❌ scikit-learn not installed. Skipping clustering analysis.")
        print("   Install with: pip install scikit-learn")
        return
    
    print("🔬 Performing clustering analysis...")
    
    # Prepare embedding matrix
    X = embeddings_matrix
    
    # Determine optimal number of clusters using elbow method
    max_clusters = min(10, len(embeddings_matrix) // 10)  # Reasonable upper bound
    if max_clusters >= 2:
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        # Choose optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Dimensionality reduction for visualization
        print("📉 Reducing dimensionality for visualization...")
        
        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('🎯 Clustering Analysis: Natural Groupings in Embedding Space', 
                     fontsize=16, fontweight='bold')
        
        # 1. Elbow method plot
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. PCA visualization
        colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
        for i in range(optimal_k):
            mask = cluster_labels == i
            ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=30)
        
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax3.set_title('PCA Visualization of Clusters')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print results
        print(f"\n🎯 Clustering Results:")
        print(f"   • Optimal number of clusters: {optimal_k}")
        print(f"   • Silhouette score: {silhouette_scores[optimal_k-2]:.3f}")
        print(f"   • PCA explained variance: {sum(pca.explained_variance_ratio_):.1%}")
        
        # Cluster size analysis
        cluster_sizes = Counter(cluster_labels)
        print(f"\n📊 Cluster Sizes:")
        for cluster_id in sorted(cluster_sizes.keys()):
            size = cluster_sizes[cluster_id]
            percentage = size / len(cluster_labels) * 100
            print(f"   • Cluster {cluster_id}: {size:,} chunks ({percentage:.1f}%)")
        
        return optimal_k, silhouette_scores[optimal_k-2]
    
    else:
        print("⚠️ Not enough data points for meaningful clustering analysis")
        return None, None

def generate_summary_report(chunks, word_count_stats, embedding_stats, 
                          type_counts, doc_counts, optimal_k=None, silhouette_score=None):
    """Generate comprehensive summary report."""
    print("\n" + "="*60)
    print("📋 EMBEDDING ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    # Overall statistics
    total_chunks = len(chunks)
    total_embeddings = len([c for c in chunks if 'embedding' in c and c['embedding']])
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total chunks processed: {total_chunks:,}")
    print(f"   • Embeddings generated: {total_embeddings:,}")
    print(f"   • Coverage: {total_embeddings/total_chunks*100:.1f}%" if total_chunks > 0 else "   • Coverage: 0%")
    
    if word_count_stats:
        print(f"\n📏 CHUNK SIZE ANALYSIS:")
        print(f"   • Average chunk size: {word_count_stats['mean']:.1f} words")
        print(f"   • Size range: {word_count_stats['min']}-{word_count_stats['max']} words")
        print(f"   • Standard deviation: {word_count_stats['std']:.1f} words")
    
    if embedding_stats:
        print(f"\n🧮 EMBEDDING QUALITY:")
        print(f"   • Average norm: {embedding_stats['mean_norm']:.4f}")
        print(f"   • Value range: [{embedding_stats['min_value']:.4f}, {embedding_stats['max_value']:.4f}]")
        print(f"   • Sparsity: {embedding_stats['sparsity']:.2f}%")
    
    if type_counts:
        print(f"\n📄 CONTENT DISTRIBUTION:")
        for chunk_type, count in type_counts.most_common():
            print(f"   • {chunk_type}: {count:,} chunks ({count/total_chunks*100:.1f}%)")
    
    if doc_counts:
        print(f"\n📚 DOCUMENT DISTRIBUTION:")
        print(f"   • Total documents: {len(doc_counts)}")
        for doc_id, count in list(doc_counts.most_common())[:5]:
            print(f"   • {doc_id[:40]}: {count:,} chunks")
    
    if optimal_k and silhouette_score:
        print(f"\n🎯 CLUSTERING INSIGHTS:")
        print(f"   • Natural clusters found: {optimal_k}")
        print(f"   • Clustering quality (silhouette): {silhouette_score:.3f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if word_count_stats:
        word_counts = [chunk.get('word_count', 0) for chunk in chunks]
        very_short = sum(1 for wc in word_counts if wc < 50)
        very_long = sum(1 for wc in word_counts if wc > 500)
        optimal_range = sum(1 for wc in word_counts if 100 <= wc <= 300)
        
        if word_count_stats['std'] > word_count_stats['mean'] * 0.5:
            print("   ⚠️  High variability in chunk sizes - consider more consistent chunking")
        
        if very_short / total_chunks > 0.1:
            print("   ⚠️  Many very short chunks - may impact embedding quality")
        
        if very_long / total_chunks > 0.05:
            print("   ⚠️  Some very long chunks - consider breaking them down further")
        
        if optimal_range / total_chunks > 0.7:
            print("   ✅ Good chunk size distribution - most chunks in optimal range")
    
    if embedding_stats:
        if embedding_stats['sparsity'] > 10:
            print("   ⚠️  High sparsity in embeddings - check embedding model performance")
        
        if embedding_stats['mean_norm'] < 0.1 or embedding_stats['mean_norm'] > 10:
            print("   ⚠️  Unusual embedding norms - verify embedding model configuration")
        else:
            print("   ✅ Embedding norms appear normal")
    
    if silhouette_score:
        if silhouette_score > 0.5:
            print("   ✅ Strong natural clustering - good for semantic search")
        elif silhouette_score > 0.3:
            print("   ✅ Moderate clustering structure detected")
        else:
            print("   ⚠️  Weak clustering - embeddings may be too similar or noisy")
    
    print(f"\n🎉 Analysis complete! Use these insights to optimize your RAG system.")
    print("=" * 60)

def main():
    """Main analysis function."""
    print("🚀 EMBEDDING STATISTICAL ANALYSIS")
    print("=" * 60)
    print("📚 Libraries imported successfully!")
    print("🎨 Plotting style configured")
    
    # Load the data
    data_path = Path("../output/processed_chunks.json")
    data = load_processed_chunks(data_path)
    
    if not data:
        print("❌ Cannot proceed without data")
        return
    
    chunks = data['chunks']
    total_chunks = data['total_chunks']
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total chunks: {total_chunks:,}")
    
    if chunks:
        sample_chunk = chunks[0]
        print(f"\n🔍 Sample Chunk Structure:")
        for key, value in sample_chunk.items():
            if key == 'embedding':
                print(f"   {key}: vector of length {len(value) if value else 0}")
            elif key == 'content':
                print(f"   {key}: {len(str(value))} characters")
            else:
                print(f"   {key}: {value}")
    
    # Run all analyses
    try:
        # 1. Chunk size analysis
        word_count_stats, word_counts = analyze_chunk_sizes(chunks)
        
        # 2. Embedding analysis
        embeddings_matrix, embedding_stats = analyze_embeddings(chunks)
        
        # 3. Content structure analysis
        type_counts, doc_counts, section_depths = analyze_content_structure(chunks)
        
        # 4. Correlation analysis
        if embeddings_matrix is not None:
            df = analyze_correlations(chunks, embeddings_matrix)
        
        # 5. Clustering analysis
        optimal_k, silhouette_score = perform_clustering_analysis(embeddings_matrix, chunks)
        
        # 6. Summary report
        generate_summary_report(chunks, word_count_stats, embedding_stats, 
                              type_counts, doc_counts, optimal_k, silhouette_score)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
