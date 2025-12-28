#!/bin/bash
# Quick script to view all backtest visualizations

echo "=================================================="
echo "📊 BACKTEST VISUALIZATIONS VIEWER"
echo "=================================================="
echo ""
echo "Opening all visualization files..."
echo ""

cd /home/rodrigodog/TrendCortex/backtest_visualizations

# Check if files exist
if [ ! -f "backtest_comparison.png" ]; then
    echo "❌ Error: Visualization files not found!"
    echo "Run: python visualize_backtest_results.py first"
    exit 1
fi

echo "✅ Found visualization files"
echo ""
echo "📁 Files in this directory:"
ls -lh *.png *.txt *.csv 2>/dev/null
echo ""
echo "=================================================="
echo ""
echo "Opening images with default viewer..."
echo ""

# Try different image viewers (Linux)
if command -v eog &> /dev/null; then
    echo "📊 Opening with Eye of GNOME..."
    eog *.png &
elif command -v feh &> /dev/null; then
    echo "📊 Opening with feh..."
    feh *.png &
elif command -v display &> /dev/null; then
    echo "📊 Opening with ImageMagick..."
    display *.png &
elif command -v xdg-open &> /dev/null; then
    echo "📊 Opening with default application..."
    for img in *.png; do
        xdg-open "$img" &
        sleep 0.5
    done
else
    echo "⚠️  No image viewer found!"
    echo "Images are saved in: $(pwd)"
    echo ""
    echo "You can view them manually or install a viewer:"
    echo "  sudo apt install eog    # Eye of GNOME"
    echo "  sudo apt install feh    # Lightweight viewer"
fi

echo ""
echo "=================================================="
echo "✅ VISUALIZATION FILES READY"
echo "=================================================="
echo ""
echo "📄 Text Reports:"
echo "  • README.md - Complete guide"
echo "  • backtest_summary_report.txt - Analysis summary"
echo "  • trade_log_round2.txt - Trade-by-trade details"
echo ""
echo "📊 Charts:"
echo "  • backtest_comparison.png - All rounds comparison"
echo "  • model_comparison.png - ML models"
echo "  • equity_curve_detailed.png - Capital over time"
echo "  • win_loss_analysis.png - Win/loss breakdown"
echo "  • drawdown_analysis.png - Risk analysis"
echo "  • improvement_trajectory.png - Evolution"
echo "  • metrics_heatmap.png - All metrics"
echo ""
echo "📈 Data Files:"
echo "  • trade_log_round2.csv - Import into Excel/Python"
echo ""
echo "=================================================="
