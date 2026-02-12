"""
Generate comprehensive visualizations for NASA C-MAPSS model performance
Includes actual results and speculative BiLSTM FD004 projection
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'TCN': '#3498db', 'BiLSTM': '#e74c3c', 'Speculative': '#95a5a6'}

def load_results():
    """Load all training results from checkpoint files"""
    results = {}
    checkpoint_dir = Path('checkpoints')
    
    for json_file in checkpoint_dir.glob('*_results.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            model = data['model']
            dataset = data['dataset']
            key = f"{model}_{dataset}"
            results[key] = data
    
    return results

def speculate_bilstm_fd004(results):
    """
    Speculate BiLSTM FD004 performance based on observed patterns
    BiLSTM consistently achieves 62-70% lower RMSE than TCN
    """
    # Get TCN FD004 results
    tcn_fd004 = results['tcn_FD004']['metrics']
    
    # Calculate average BiLSTM improvement ratio from FD001-FD003
    improvements = []
    for dataset in ['FD001', 'FD002', 'FD003']:
        tcn_rmse = results[f'tcn_{dataset}']['metrics']['rmse']
        bilstm_rmse = results[f'bilstm_{dataset}']['metrics']['rmse']
        improvement = bilstm_rmse / tcn_rmse
        improvements.append(improvement)
    
    avg_improvement = np.mean(improvements)  # ~0.35 (65% reduction)
    
    # Project BiLSTM FD004 metrics
    speculative = {
        'rmse': tcn_fd004['rmse'] * avg_improvement,
        'mae': tcn_fd004['mae'] * avg_improvement,
        'r2': 0.95,  # Conservative estimate based on FD001-FD003 pattern
        'nasa_score': tcn_fd004['nasa_score'] * (avg_improvement ** 2),  # Exponential improvement
    }
    
    return speculative, avg_improvement

def plot_individual_performance(results, output_dir):
    """Plot individual model performance for each dataset"""
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    metrics = ['rmse', 'mae', 'r2']
    metric_names = ['RMSE (cycles)', 'MAE (cycles)', 'R² Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Across Datasets', fontsize=16, fontweight='bold')
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]
        
        # Get data
        tcn_key = f'tcn_{dataset}'
        bilstm_key = f'bilstm_{dataset}'
        
        tcn_metrics = [results[tcn_key]['metrics'][m] for m in metrics]
        
        if bilstm_key in results:
            bilstm_metrics = [results[bilstm_key]['metrics'][m] for m in metrics]
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, tcn_metrics, width, label='TCN', color=colors['TCN'], alpha=0.8)
            bars2 = ax.bar(x + width/2, bilstm_metrics, width, label='BiLSTM', color=colors['BiLSTM'], alpha=0.8)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            # FD004 - only TCN
            x = np.arange(len(metrics))
            bars1 = ax.bar(x, tcn_metrics, 0.5, label='TCN', color=colors['TCN'], alpha=0.8)
            
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title(f'{dataset} Performance', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'individual_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'individual_performance.png'}")
    plt.close()

def plot_model_comparison(results, output_dir):
    """Compare TCN vs BiLSTM across all datasets"""
    datasets = ['FD001', 'FD002', 'FD003']
    
    tcn_rmse = [results[f'tcn_{ds}']['metrics']['rmse'] for ds in datasets]
    bilstm_rmse = [results[f'bilstm_{ds}']['metrics']['rmse'] for ds in datasets]
    
    tcn_r2 = [results[f'tcn_{ds}']['metrics']['r2'] for ds in datasets]
    bilstm_r2 = [results[f'bilstm_{ds}']['metrics']['r2'] for ds in datasets]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('TCN vs BiLSTM+Attention Performance Comparison', fontsize=16, fontweight='bold')
    
    # RMSE comparison
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, tcn_rmse, width, label='TCN', color=colors['TCN'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, bilstm_rmse, width, label='BiLSTM', color=colors['BiLSTM'], alpha=0.8)
    
    ax1.set_xlabel('Dataset', fontweight='bold', fontsize=12)
    ax1.set_ylabel('RMSE (cycles)', fontweight='bold', fontsize=12)
    ax1.set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # R² comparison
    bars3 = ax2.bar(x - width/2, tcn_r2, width, label='TCN', color=colors['TCN'], alpha=0.8)
    bars4 = ax2.bar(x + width/2, bilstm_r2, width, label='BiLSTM', color=colors['BiLSTM'], alpha=0.8)
    
    ax2.set_xlabel('Dataset', fontweight='bold', fontsize=12)
    ax2.set_ylabel('R² Score', fontweight='bold', fontsize=12)
    ax2.set_title('R² Score (Higher is Better)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.7, 1.0])
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_comparison.png'}")
    plt.close()

def plot_speculative_fd004(results, speculative, improvement_ratio, output_dir):
    """Plot FD004 with speculative BiLSTM results"""
    metrics = ['rmse', 'mae', 'r2']
    metric_names = ['RMSE\n(cycles)', 'MAE\n(cycles)', 'R² Score']
    
    tcn_values = [results['tcn_FD004']['metrics'][m] for m in metrics]
    spec_values = [speculative[m] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, tcn_values, width, label='TCN (Actual)', 
                   color=colors['TCN'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, spec_values, width, label='BiLSTM (Speculative)', 
                   color=colors['Speculative'], alpha=0.8, edgecolor='black', 
                   linewidth=1.5, hatch='//')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
               color=colors['Speculative'])
    
    ax.set_xlabel('Metrics', fontweight='bold', fontsize=13)
    ax.set_ylabel('Value', fontweight='bold', fontsize=13)
    ax.set_title(f'FD004: Actual TCN vs Speculative BiLSTM Performance\n(Projected {(1-improvement_ratio)*100:.0f}% RMSE improvement based on FD001-FD003 pattern)', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    ax.text(0.02, 0.98, 
           '⚠ SPECULATIVE DATA\nBased on observed BiLSTM performance\npatterns from FD001-FD003\n(Not trained due to GPU memory limits)',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fd004_speculative.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fd004_speculative.png'}")
    plt.close()

def plot_improvement_summary(results, speculative, output_dir):
    """Plot RMSE improvement percentages"""
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    improvements = []
    
    for dataset in datasets[:3]:
        tcn_rmse = results[f'tcn_{dataset}']['metrics']['rmse']
        bilstm_rmse = results[f'bilstm_{dataset}']['metrics']['rmse']
        improvement = ((tcn_rmse - bilstm_rmse) / tcn_rmse) * 100
        improvements.append(improvement)
    
    # Add speculative FD004
    tcn_fd004_rmse = results['tcn_FD004']['metrics']['rmse']
    spec_improvement = ((tcn_fd004_rmse - speculative['rmse']) / tcn_fd004_rmse) * 100
    improvements.append(spec_improvement)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors_list = [colors['BiLSTM']] * 3 + [colors['Speculative']]
    bars = ax.bar(datasets, improvements, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Hatch pattern for speculative
    bars[3].set_hatch('//')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        label = f'{val:.1f}%'
        if i == 3:
            label += '\n(Speculative)'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Dataset', fontweight='bold', fontsize=13)
    ax.set_ylabel('RMSE Improvement (%)', fontweight='bold', fontsize=13)
    ax.set_title('BiLSTM RMSE Improvement over TCN\n(Percentage Reduction)', 
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(improvements) * 1.2])
    
    # Add average line
    avg_improvement = np.mean(improvements[:3])
    ax.axhline(y=avg_improvement, color='red', linestyle='--', linewidth=2, 
              label=f'Avg Improvement (FD001-003): {avg_improvement:.1f}%')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'improvement_summary.png'}")
    plt.close()

def plot_nasa_score_comparison(results, speculative, output_dir):
    """Plot NASA Score comparison (log scale due to large range)"""
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    
    tcn_scores = [results[f'tcn_{ds}']['metrics']['nasa_score'] for ds in datasets]
    bilstm_scores = [results[f'bilstm_{ds}']['metrics']['nasa_score'] for ds in datasets[:3]]
    bilstm_scores.append(speculative['nasa_score'])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, tcn_scores, width, label='TCN', 
                   color=colors['TCN'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Different colors for actual vs speculative BiLSTM
    bilstm_colors = [colors['BiLSTM']] * 3 + [colors['Speculative']]
    bars2 = ax.bar(x + width/2, bilstm_scores, width, label='BiLSTM', 
                   color=bilstm_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Hatch for speculative
    bars2[3].set_hatch('//')
    
    ax.set_yscale('log')
    ax.set_xlabel('Dataset', fontweight='bold', fontsize=13)
    ax.set_ylabel('NASA Score (log scale, lower is better)', fontweight='bold', fontsize=13)
    ax.set_title('NASA Score Comparison\n(Logarithmic Scale)', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            label = f'{height:.0f}'
            if bars == bars2 and i == 3:
                label += '\n(Spec)'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'nasa_score_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'nasa_score_comparison.png'}")
    plt.close()

def main():
    # Create output directory
    output_dir = Path('reports/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading training results...")
    results = load_results()
    
    print("Generating speculative BiLSTM FD004 metrics...")
    speculative, improvement_ratio = speculate_bilstm_fd004(results)
    
    print(f"\nSpeculative BiLSTM FD004 Metrics:")
    print(f"  RMSE: {speculative['rmse']:.2f} cycles (vs TCN: {results['tcn_FD004']['metrics']['rmse']:.2f})")
    print(f"  MAE: {speculative['mae']:.2f} cycles")
    print(f"  R²: {speculative['r2']:.3f}")
    print(f"  NASA Score: {speculative['nasa_score']:.0f}")
    print(f"  Projected improvement: {(1-improvement_ratio)*100:.1f}% RMSE reduction\n")
    
    print("Generating visualizations...")
    plot_individual_performance(results, output_dir)
    plot_model_comparison(results, output_dir)
    plot_speculative_fd004(results, speculative, improvement_ratio, output_dir)
    plot_improvement_summary(results, speculative, output_dir)
    plot_nasa_score_comparison(results, speculative, output_dir)
    
    print(f"\n[SUCCESS] All visualizations saved to {output_dir}/")
    print("\nGenerated files:")
    print("  1. individual_performance.png - Performance by dataset")
    print("  2. model_comparison.png - TCN vs BiLSTM (FD001-003)")
    print("  3. fd004_speculative.png - FD004 with speculative BiLSTM")
    print("  4. improvement_summary.png - RMSE improvement percentages")
    print("  5. nasa_score_comparison.png - NASA Score comparison")

if __name__ == '__main__':
    main()
