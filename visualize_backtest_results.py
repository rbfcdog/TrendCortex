#!/usr/bin/env python3
"""
Backtest Results Visualization
================================

Creates comprehensive plots from backtest results:
- Equity curves
- Win/Loss distribution
- Exit reason analysis
- Model comparison
- Performance metrics heatmap
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_backtest_results():
    """Load all backtest results from data directory"""
    results_dir = Path("data")
    results = []
    
    print("🔍 Searching for backtest result files...")
    
    # Look for JSON result files
    for json_file in results_dir.glob("*backtest*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['source_file'] = json_file.name
                results.append(data)
                print(f"  ✅ Loaded: {json_file.name}")
        except Exception as e:
            print(f"  ⚠️  Failed to load {json_file.name}: {e}")
    
    if not results:
        print("\n⚠️  No JSON result files found!")
        print("Let me create results from the latest backtest run...")
        return None
    
    return results


def create_mock_results():
    """Create mock results from our documented backtests"""
    results = [
        {
            'name': 'Round 1 - 1h 90 days',
            'metrics': {
                'total_trades': 1,
                'win_rate': 0.0,
                'total_return': -0.0002,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': -0.0002,
                'avg_win': 0.0,
                'avg_loss': -2.36,
            },
            'model_metrics': {
                'random_forest': {'auc': 0.5263, 'accuracy': 0.5698}
            },
            'exit_reasons': {'stop_loss': 1}
        },
        {
            'name': 'Round 2 - 1h 180 days',
            'metrics': {
                'total_trades': 18,
                'win_rate': 0.2222,
                'total_return': -0.0018,
                'profit_factor': 0.77,
                'sharpe_ratio': -1.78,
                'max_drawdown': -0.0072,
                'avg_win': 15.29,
                'avg_loss': -5.65,
            },
            'model_metrics': {
                'random_forest': {'auc': 0.6567, 'accuracy': 0.8997},
                'xgboost': {'auc': 0.6265, 'accuracy': 0.9142},
                'logistic_regression': {'auc': 0.6255, 'accuracy': 0.9075}
            },
            'exit_reasons': {'take_profit': 4, 'stop_loss': 14},
            'signals_generated': 20,
        },
        {
            'name': 'Round 3 - 4h 180 days',
            'metrics': {
                'total_trades': 11,
                'win_rate': 0.2727,
                'total_return': -0.0039,
                'profit_factor': 0.40,
                'sharpe_ratio': -7.53,
                'max_drawdown': -0.0065,
                'avg_win': 8.71,
                'avg_loss': -8.13,
            },
            'model_metrics': {
                'random_forest': {'auc': 0.5427, 'accuracy': 0.7880}
            },
            'exit_reasons': {'stop_loss': 8, 'end_of_backtest': 3},
            'signals_generated': 20,
        },
        {
            'name': 'Round 4 - 1h 365 days',
            'metrics': {
                'total_trades': 4,
                'win_rate': 0.25,
                'total_return': -0.0013,
                'profit_factor': 0.44,
                'sharpe_ratio': -6.41,
                'max_drawdown': -0.0023,
                'avg_win': 10.11,
                'avg_loss': -7.65,
            },
            'model_metrics': {
                'random_forest': {'auc': 0.6230, 'accuracy': 0.9863}
            },
            'exit_reasons': {'stop_loss': 3, 'take_profit': 1},
            'signals_generated': 8,
        }
    ]
    
    return results


def plot_performance_comparison(results, output_dir):
    """Create comparison plots for all backtest rounds"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('🎯 Backtest Performance Comparison - All Rounds', fontsize=16, fontweight='bold')
    
    names = [r['name'] for r in results]
    colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6']
    
    # 1. Win Rate
    ax = axes[0, 0]
    win_rates = [r['metrics']['win_rate'] * 100 for r in results]
    bars = ax.bar(range(len(names)), win_rates, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=27, color='green', linestyle='--', label='Breakeven (27%)', linewidth=2)
    ax.axhline(y=35, color='gold', linestyle='--', label='Target (35%)', linewidth=2)
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Win Rate by Round', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"R{i+1}" for i in range(len(names))], fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, win_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Profit Factor
    ax = axes[0, 1]
    pfs = [r['metrics']['profit_factor'] for r in results]
    bars = ax.bar(range(len(names)), pfs, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='green', linestyle='--', label='Breakeven (1.0)', linewidth=2)
    ax.axhline(y=1.3, color='gold', linestyle='--', label='Target (1.3)', linewidth=2)
    ax.set_ylabel('Profit Factor', fontsize=12, fontweight='bold')
    ax.set_title('Profit Factor by Round', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"R{i+1}" for i in range(len(names))], fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, pfs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. AUC (ML Quality)
    ax = axes[0, 2]
    aucs = [r['model_metrics']['random_forest']['auc'] for r in results]
    bars = ax.bar(range(len(names)), aucs, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0.60, color='gold', linestyle='--', label='Good (0.60)', linewidth=2)
    ax.axhline(y=0.50, color='red', linestyle='--', label='Random (0.50)', linewidth=2)
    ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('ML Model Quality (AUC)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"R{i+1}" for i in range(len(names))], fontsize=10)
    ax.set_ylim(0.4, 0.8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, aucs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Total Trades
    ax = axes[1, 0]
    trades = [r['metrics']['total_trades'] for r in results]
    bars = ax.bar(range(len(names)), trades, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=30, color='gold', linestyle='--', label='Min Sample (30)', linewidth=2)
    ax.set_ylabel('Number of Trades', fontsize=12, fontweight='bold')
    ax.set_title('Trade Count by Round', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"R{i+1}" for i in range(len(names))], fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, trades)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{int(val)}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Total Return
    ax = axes[1, 1]
    returns = [r['metrics']['total_return'] * 100 for r in results]
    colors_return = ['red' if r < 0 else 'green' for r in returns]
    bars = ax.bar(range(len(names)), returns, color=colors_return, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Return by Round', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"R{i+1}" for i in range(len(names))], fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, returns)):
        y_pos = bar.get_height() + (0.01 if val > 0 else -0.02)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
                f'{val:.2f}%', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    # 6. Exit Reasons for Round 2 (best one)
    ax = axes[1, 2]
    round2 = results[1]  # Round 2 has the most data
    exit_reasons = round2['exit_reasons']
    labels = list(exit_reasons.keys())
    values = list(exit_reasons.values())
    colors_exit = ['#27ae60' if 'profit' in l else '#e74c3c' for l in labels]
    
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                        colors=colors_exit, startangle=90,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax.set_title('Exit Reasons (Round 2)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'backtest_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()


def plot_model_comparison(results, output_dir):
    """Compare different ML models"""
    
    # Use Round 2 which has all 3 models
    round2 = results[1]
    models = round2['model_metrics']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('🤖 ML Model Comparison (Round 2 - Best Data)', fontsize=16, fontweight='bold')
    
    model_names = list(models.keys())
    
    # 1. AUC Comparison
    ax = axes[0]
    aucs = [models[m]['auc'] for m in model_names]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(model_names, aucs, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0.60, color='gold', linestyle='--', label='Good (0.60)', linewidth=2)
    ax.axhline(y=0.65, color='green', linestyle='--', label='Excellent (0.65)', linewidth=2)
    ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Model AUC Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(0.5, 0.75)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Accuracy Comparison
    ax = axes[1]
    accuracies = [models[m]['accuracy'] * 100 for m in model_names]
    bars = ax.bar(model_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(80, 95)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / 'model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()


def plot_metrics_heatmap(results, output_dir):
    """Create heatmap of all metrics across rounds"""
    
    # Prepare data
    metrics_to_plot = [
        'win_rate', 'profit_factor', 'sharpe_ratio', 
        'total_return', 'max_drawdown', 'total_trades'
    ]
    
    data = []
    for r in results:
        row = []
        for metric in metrics_to_plot:
            value = r['metrics'][metric]
            # Normalize for better visualization
            if metric == 'win_rate':
                value *= 100  # Convert to percentage
            elif metric == 'total_return' or metric == 'max_drawdown':
                value *= 100  # Convert to percentage
            row.append(value)
        data.append(row)
    
    df = pd.DataFrame(data, 
                      index=[r['name'] for r in results],
                      columns=metrics_to_plot)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize each column for color mapping
    df_norm = df.copy()
    for col in df_norm.columns:
        if col not in ['sharpe_ratio', 'total_return', 'max_drawdown']:
            # Higher is better
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        else:
            # For returns, positive is good
            if col == 'max_drawdown':
                # Less negative is better
                df_norm[col] = 1 - abs(df_norm[col] / df_norm[col].min())
            else:
                df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    
    sns.heatmap(df_norm, annot=df.values, fmt='.2f', cmap='RdYlGn', 
                cbar_kws={'label': 'Normalized Score'}, 
                linewidths=0.5, ax=ax)
    
    ax.set_title('📊 Performance Metrics Heatmap - All Rounds', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    
    output_file = output_dir / 'metrics_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()


def plot_improvement_trajectory(results, output_dir):
    """Show how strategy improved over rounds"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    rounds = [f"R{i+1}" for i in range(len(results))]
    
    # Plot multiple metrics
    metrics = {
        'Win Rate (%)': [r['metrics']['win_rate'] * 100 for r in results],
        'AUC Score (×10)': [r['model_metrics']['random_forest']['auc'] * 10 for r in results],
        'Profit Factor (×10)': [r['metrics']['profit_factor'] * 10 for r in results],
    }
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    markers = ['o', 's', '^']
    
    for (label, values), color, marker in zip(metrics.items(), colors, markers):
        ax.plot(rounds, values, marker=marker, linewidth=2.5, markersize=10,
                label=label, color=color)
        
        # Add value labels
        for i, val in enumerate(values):
            ax.text(i, val + 1, f'{val:.1f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
    
    # Add target lines
    ax.axhline(y=35, color='gold', linestyle='--', label='Win Rate Target (35%)', linewidth=2, alpha=0.7)
    ax.axhline(y=27, color='orange', linestyle=':', label='Breakeven WR (27%)', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Backtest Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('🚀 Strategy Evolution Across Backtest Rounds', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate('Best ML Model', xy=(1, results[1]['model_metrics']['random_forest']['auc'] * 10),
                xytext=(1.5, 68), fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.tight_layout()
    
    output_file = output_dir / 'improvement_trajectory.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()


def create_summary_report(results, output_dir):
    """Create a text summary report"""
    
    report = []
    report.append("=" * 80)
    report.append("📊 BACKTEST RESULTS SUMMARY REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Backtests Analyzed: {len(results)}\n")
    
    for i, result in enumerate(results, 1):
        report.append(f"\n{'='*80}")
        report.append(f"Round {i}: {result['name']}")
        report.append(f"{'='*80}")
        
        metrics = result['metrics']
        report.append(f"\n📈 Trading Performance:")
        report.append(f"  • Total Trades: {metrics['total_trades']}")
        report.append(f"  • Win Rate: {metrics['win_rate']:.2%}")
        report.append(f"  • Profit Factor: {metrics['profit_factor']:.2f}")
        report.append(f"  • Total Return: {metrics['total_return']:.2%}")
        report.append(f"  • Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        report.append(f"  • Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        if 'avg_win' in metrics:
            report.append(f"\n💰 Trade Statistics:")
            report.append(f"  • Average Win: ${metrics['avg_win']:.2f}")
            report.append(f"  • Average Loss: ${metrics['avg_loss']:.2f}")
            if metrics['avg_loss'] != 0:
                rr = abs(metrics['avg_win'] / metrics['avg_loss'])
                report.append(f"  • Win/Loss Ratio: {rr:.2f}:1")
        
        report.append(f"\n🤖 ML Model Performance:")
        for model_name, model_metrics in result['model_metrics'].items():
            report.append(f"  • {model_name}:")
            report.append(f"    - AUC: {model_metrics['auc']:.4f}")
            report.append(f"    - Accuracy: {model_metrics['accuracy']:.2%}")
        
        if 'exit_reasons' in result:
            report.append(f"\n🚪 Exit Reasons:")
            for reason, count in result['exit_reasons'].items():
                pct = count / metrics['total_trades'] * 100
                report.append(f"  • {reason}: {count} ({pct:.1f}%)")
        
        if 'signals_generated' in result:
            report.append(f"\n📡 Signal Statistics:")
            report.append(f"  • Signals Generated: {result['signals_generated']}")
            report.append(f"  • Trades Executed: {metrics['total_trades']}")
            execution_rate = metrics['total_trades'] / result['signals_generated'] * 100
            report.append(f"  • Execution Rate: {execution_rate:.1f}%")
    
    # Best round
    report.append(f"\n\n{'='*80}")
    report.append("🏆 BEST PERFORMING ROUND")
    report.append(f"{'='*80}")
    
    best_auc_idx = max(range(len(results)), 
                       key=lambda i: results[i]['model_metrics']['random_forest']['auc'])
    best_wr_idx = max(range(len(results)), 
                      key=lambda i: results[i]['metrics']['win_rate'])
    
    report.append(f"\n• Best ML Model (AUC): {results[best_auc_idx]['name']}")
    report.append(f"  AUC: {results[best_auc_idx]['model_metrics']['random_forest']['auc']:.4f}")
    
    report.append(f"\n• Best Win Rate: {results[best_wr_idx]['name']}")
    report.append(f"  Win Rate: {results[best_wr_idx]['metrics']['win_rate']:.2%}")
    
    # Key insights
    report.append(f"\n\n{'='*80}")
    report.append("💡 KEY INSIGHTS")
    report.append(f"{'='*80}")
    report.append("\n✅ Strengths:")
    report.append(f"  • Best AUC achieved: {max(r['model_metrics']['random_forest']['auc'] for r in results):.4f}")
    report.append("  • ML model shows predictive power (AUC > 0.60)")
    report.append(f"  • Total trades executed: {sum(r['metrics']['total_trades'] for r in results)}")
    
    report.append("\n⚠️  Areas for Improvement:")
    avg_wr = sum(r['metrics']['win_rate'] for r in results) / len(results)
    report.append(f"  • Average win rate: {avg_wr:.2%} (target: >35%)")
    report.append("  • Entry timing needs refinement (add confirmation)")
    report.append("  • All rounds show negative returns (need fixes)")
    
    report.append("\n🚀 Next Steps:")
    report.append("  1. Implement entry confirmation (wait for price action)")
    report.append("  2. Change label strategy (predict trade outcomes)")
    report.append("  3. Add SMOTE balancing for training data")
    report.append("  4. Test with volume and pattern features")
    
    report.append("\n" + "="*80)
    
    # Save report
    output_file = output_dir / 'backtest_summary_report.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Saved: {output_file}")
    
    # Also print to console
    print("\n" + '\n'.join(report))


def main():
    print("="*80)
    print("📊 BACKTEST RESULTS VISUALIZATION")
    print("="*80)
    
    # Create output directory
    output_dir = Path("backtest_visualizations")
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Load results
    results = load_backtest_results()
    
    if results is None:
        print("\n📝 Using documented backtest results...")
        results = create_mock_results()
    
    print(f"\n✅ Loaded {len(results)} backtest results")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    try:
        plot_performance_comparison(results, output_dir)
        plot_model_comparison(results, output_dir)
        plot_metrics_heatmap(results, output_dir)
        plot_improvement_trajectory(results, output_dir)
        create_summary_report(results, output_dir)
        
        print("\n" + "="*80)
        print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Check the '{output_dir}' directory for:")
        print("  • backtest_comparison.png")
        print("  • model_comparison.png")
        print("  • metrics_heatmap.png")
        print("  • improvement_trajectory.png")
        print("  • backtest_summary_report.txt")
        
    except Exception as e:
        print(f"\n❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
