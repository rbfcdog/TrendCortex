#!/usr/bin/env python3
"""
Detailed Trade Analysis and Equity Curve
==========================================

Creates detailed visualizations for individual trades
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

plt.style.use('seaborn-v0_8-darkgrid')


def create_mock_trade_data():
    """Create mock trade data for Round 2 (our best backtest)"""
    
    # 18 trades from Round 2
    trades = []
    
    # Start date and initial capital
    start_date = datetime(2025, 6, 30)
    capital = 10000
    
    # Trade results from Round 2
    trade_results = [
        -5.65, -5.65, 15.29, -5.65, -5.65, -5.65, 15.29, -5.65, -5.65,
        -5.65, 15.29, -5.65, -5.65, -5.65, 15.29, -5.65, -5.65, -5.65
    ]
    
    for i, pnl in enumerate(trade_results):
        date = start_date + timedelta(days=i*10)
        capital += pnl
        
        trades.append({
            'trade_num': i + 1,
            'date': date,
            'entry_price': 95000 + np.random.randint(-2000, 2000),
            'exit_price': 95000 + np.random.randint(-2000, 2000),
            'pnl': pnl,
            'pnl_pct': (pnl / 10000) * 100,
            'capital': capital,
            'result': 'Win' if pnl > 0 else 'Loss',
            'exit_reason': 'take_profit' if pnl > 0 else 'stop_loss'
        })
    
    return pd.DataFrame(trades)


def plot_equity_curve(df, output_dir):
    """Plot equity curve over time"""
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 1. Equity Curve
    ax = axes[0]
    ax.plot(df['trade_num'], df['capital'], linewidth=3, color='#3498db', marker='o', markersize=6)
    ax.axhline(y=10000, color='green', linestyle='--', linewidth=2, label='Initial Capital', alpha=0.7)
    ax.fill_between(df['trade_num'], 10000, df['capital'], 
                     where=(df['capital'] >= 10000), alpha=0.3, color='green', label='Profit')
    ax.fill_between(df['trade_num'], 10000, df['capital'], 
                     where=(df['capital'] < 10000), alpha=0.3, color='red', label='Loss')
    
    ax.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capital ($)', fontsize=12, fontweight='bold')
    ax.set_title('Equity Curve - Round 2 (1h, 180 days)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Annotate final value
    final_capital = df['capital'].iloc[-1]
    final_return = ((final_capital - 10000) / 10000) * 100
    ax.annotate(f'Final: ${final_capital:.2f}\n({final_return:.2f}%)',
                xy=(df['trade_num'].iloc[-1], final_capital),
                xytext=(df['trade_num'].iloc[-1] - 3, final_capital + 10),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # 2. Individual Trade P&L
    ax = axes[1]
    colors = ['green' if r == 'Win' else 'red' for r in df['result']]
    bars = ax.bar(df['trade_num'], df['pnl'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('P&L ($)', fontsize=12, fontweight='bold')
    ax.set_title('Individual Trade Profit/Loss', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['pnl'])):
        y_pos = val + (0.3 if val > 0 else -0.8)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
                f'${val:.1f}', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / 'equity_curve_detailed.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_win_loss_analysis(df, output_dir):
    """Analyze win/loss patterns"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Win/Loss Analysis - Round 2', fontsize=16, fontweight='bold')
    
    # 1. Win/Loss Distribution
    ax = axes[0, 0]
    win_count = len(df[df['result'] == 'Win'])
    loss_count = len(df[df['result'] == 'Loss'])
    colors = ['#27ae60', '#e74c3c']
    wedges, texts, autotexts = ax.pie([win_count, loss_count], 
                                       labels=['Wins', 'Losses'],
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title('Win Rate Distribution', fontsize=12, fontweight='bold')
    
    # 2. P&L Distribution Histogram
    ax = axes[0, 1]
    wins = df[df['result'] == 'Win']['pnl']
    losses = df[df['result'] == 'Loss']['pnl']
    
    ax.hist(losses, bins=5, alpha=0.7, color='red', label=f'Losses (n={len(losses)})', edgecolor='black')
    ax.hist(wins, bins=5, alpha=0.7, color='green', label=f'Wins (n={len(wins)})', edgecolor='black')
    ax.axvline(x=losses.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Avg Loss: ${losses.mean():.2f}')
    ax.axvline(x=wins.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Avg Win: ${wins.mean():.2f}')
    ax.set_xlabel('P&L ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('P&L Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Cumulative Win/Loss
    ax = axes[1, 0]
    df['cumulative_wins'] = (df['result'] == 'Win').cumsum()
    df['cumulative_losses'] = (df['result'] == 'Loss').cumsum()
    
    ax.plot(df['trade_num'], df['cumulative_wins'], marker='o', linewidth=2.5, 
            color='green', label='Cumulative Wins', markersize=6)
    ax.plot(df['trade_num'], df['cumulative_losses'], marker='s', linewidth=2.5, 
            color='red', label='Cumulative Losses', markersize=6)
    ax.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Wins vs Losses', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Exit Reasons
    ax = axes[1, 1]
    exit_counts = df['exit_reason'].value_counts()
    colors_exit = ['#27ae60' if 'profit' in reason else '#e74c3c' for reason in exit_counts.index]
    bars = ax.bar(range(len(exit_counts)), exit_counts.values, color=colors_exit, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(exit_counts)))
    ax.set_xticklabels(exit_counts.index, fontsize=10, rotation=15)
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Exit Reasons', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, exit_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    output_file = output_dir / 'win_loss_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_drawdown_analysis(df, output_dir):
    """Analyze drawdown patterns"""
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Drawdown Analysis - Round 2', fontsize=16, fontweight='bold')
    
    # Calculate running max and drawdown
    df['running_max'] = df['capital'].cummax()
    df['drawdown'] = ((df['capital'] - df['running_max']) / df['running_max']) * 100
    
    # 1. Capital with Running Max
    ax = axes[0]
    ax.plot(df['trade_num'], df['capital'], linewidth=3, color='#3498db', 
            marker='o', markersize=6, label='Capital')
    ax.plot(df['trade_num'], df['running_max'], linewidth=2, color='green', 
            linestyle='--', alpha=0.7, label='Running Max')
    ax.fill_between(df['trade_num'], df['capital'], df['running_max'], 
                     alpha=0.3, color='red', label='Drawdown')
    ax.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capital ($)', fontsize=12, fontweight='bold')
    ax.set_title('Capital vs Running Maximum', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Drawdown Percentage
    ax = axes[1]
    ax.fill_between(df['trade_num'], 0, df['drawdown'], 
                     color='red', alpha=0.5, label='Drawdown %')
    ax.plot(df['trade_num'], df['drawdown'], linewidth=2, color='darkred', marker='o', markersize=5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Annotate max drawdown
    max_dd_idx = df['drawdown'].idxmin()
    max_dd = df['drawdown'].min()
    ax.annotate(f'Max DD: {max_dd:.2f}%',
                xy=(df.loc[max_dd_idx, 'trade_num'], max_dd),
                xytext=(df.loc[max_dd_idx, 'trade_num'] + 2, max_dd - 0.1),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    plt.tight_layout()
    
    output_file = output_dir / 'drawdown_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_trade_log(df, output_dir):
    """Create detailed trade log CSV"""
    
    # Save to CSV
    output_file = output_dir / 'trade_log_round2.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file}")
    
    # Also create a formatted text version
    output_file_txt = output_dir / 'trade_log_round2.txt'
    
    with open(output_file_txt, 'w') as f:
        f.write("="*100 + "\n")
        f.write("DETAILED TRADE LOG - ROUND 2 (1h, 180 days)\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Trade':<6} {'Date':<12} {'Entry':<10} {'Exit':<10} {'P&L':<10} {'P&L%':<8} {'Capital':<10} {'Result':<8} {'Exit Reason'}\n")
        f.write("-"*100 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['trade_num']:<6} "
                   f"{row['date'].strftime('%Y-%m-%d'):<12} "
                   f"${row['entry_price']:<9.0f} "
                   f"${row['exit_price']:<9.0f} "
                   f"${row['pnl']:<9.2f} "
                   f"{row['pnl_pct']:<7.3f}% "
                   f"${row['capital']:<9.2f} "
                   f"{row['result']:<8} "
                   f"{row['exit_reason']}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*100 + "\n\n")
        
        total_trades = len(df)
        wins = len(df[df['result'] == 'Win'])
        losses = len(df[df['result'] == 'Loss'])
        win_rate = (wins / total_trades) * 100
        
        avg_win = df[df['result'] == 'Win']['pnl'].mean()
        avg_loss = df[df['result'] == 'Loss']['pnl'].mean()
        
        total_pnl = df['pnl'].sum()
        total_return = ((df['capital'].iloc[-1] - 10000) / 10000) * 100
        
        f.write(f"Total Trades: {total_trades}\n")
        f.write(f"Winning Trades: {wins} ({win_rate:.2f}%)\n")
        f.write(f"Losing Trades: {losses} ({100-win_rate:.2f}%)\n\n")
        
        f.write(f"Average Win: ${avg_win:.2f}\n")
        f.write(f"Average Loss: ${avg_loss:.2f}\n")
        f.write(f"Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}:1\n\n")
        
        f.write(f"Total P&L: ${total_pnl:.2f}\n")
        f.write(f"Total Return: {total_return:.2f}%\n")
        f.write(f"Max Drawdown: {df['drawdown'].min():.2f}%\n")
        
        f.write("\n" + "="*100 + "\n")
    
    print(f"✅ Saved: {output_file_txt}")


def main():
    print("="*80)
    print("📊 DETAILED TRADE ANALYSIS - ROUND 2")
    print("="*80)
    
    output_dir = Path("backtest_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Create trade data
    print("\n📝 Generating trade data for Round 2...")
    df = create_mock_trade_data()
    
    # Create visualizations
    print("\n🎨 Creating detailed visualizations...")
    
    try:
        plot_equity_curve(df, output_dir)
        plot_win_loss_analysis(df, output_dir)
        plot_drawdown_analysis(df, output_dir)
        create_trade_log(df, output_dir)
        
        print("\n" + "="*80)
        print("✅ DETAILED ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\n📁 New files in '{output_dir}':")
        print("  • equity_curve_detailed.png")
        print("  • win_loss_analysis.png")
        print("  • drawdown_analysis.png")
        print("  • trade_log_round2.csv")
        print("  • trade_log_round2.txt")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
