#!/bin/bash

echo "🔍 V131 Status Check"
echo "==================="

# Check Docker Hub
echo "📦 Docker Hub Status:"
if curl -s "https://hub.docker.com/v2/repositories/berrylands/multitalk-runpod/tags/v131" | grep -q "last_updated"; then
    echo "✅ V131 is available on Docker Hub!"
    LAST_UPDATED=$(curl -s "https://hub.docker.com/v2/repositories/berrylands/multitalk-runpod/tags/v131" | grep -o '"last_updated":"[^"]*' | cut -d'"' -f4)
    echo "📅 Last updated: $LAST_UPDATED"
    
    echo ""
    echo "🚀 Ready to test! Run:"
    echo "python test_v131_fixed.py"
    
else
    echo "⏳ V131 not yet available on Docker Hub"
fi

echo ""
echo "🔗 Monitor GitHub Actions: https://github.com/berrylands/meigen-multitalk-runpod-serverless/actions"

# Check local build if log exists
if [ -f "runpod-multitalk/v131_build.log" ]; then
    echo ""
    echo "📝 Local build status:"
    if grep -q "Build successful" runpod-multitalk/v131_build.log; then
        echo "✅ Local build completed successfully"
    elif grep -q "Build failed" runpod-multitalk/v131_build.log; then
        echo "❌ Local build failed"
    else
        echo "🔄 Local build still in progress"
        echo "📊 Latest progress:"
        tail -n 3 runpod-multitalk/v131_build.log
    fi
fi