#!/bin/bash

echo "🔄 Starting V131 build monitoring loop..."
echo "Will check every 2 minutes until build completes"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    echo "$(date '+%H:%M:%S') - Checking build status..."
    python check_build_status.py
    
    # Check if build completed by looking for success/failure indicators
    if python check_build_status.py | grep -q "Build completed successfully"; then
        echo ""
        echo "✅ Build completed successfully! V131 should be available."
        break
    elif python check_build_status.py | grep -q "Build failed"; then
        echo ""
        echo "❌ Build failed. Check GitHub Actions logs."
        break
    fi
    
    echo ""
    echo "⏳ Waiting 2 minutes before next check..."
    sleep 120
done

echo ""
echo "🎯 Monitoring complete!"