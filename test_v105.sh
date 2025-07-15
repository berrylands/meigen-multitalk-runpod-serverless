#!/bin/bash
# Test V105 deployment

echo "🚀 Testing V105 deployment with xfuser..."
echo ""

# Check if API key is set
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "❌ RUNPOD_API_KEY not set"
    exit 1
fi

ENDPOINT_ID="kkx3cfy484jszl"
BASE_URL="https://api.runpod.ai/v2/$ENDPOINT_ID"

# 1. Check endpoint health
echo "1️⃣ Checking endpoint health..."
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" "$BASE_URL/health" | jq .
echo ""

# 2. Submit model check job
echo "2️⃣ Submitting model check job..."
RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input": {"action": "model_check"}}' \
    "$BASE_URL/run")

JOB_ID=$(echo $RESPONSE | jq -r .id)
echo "Job ID: $JOB_ID"
echo ""

# 3. Wait for job completion
echo "3️⃣ Waiting for job completion..."
for i in {1..30}; do
    STATUS_RESPONSE=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" "$BASE_URL/status/$JOB_ID")
    STATUS=$(echo $STATUS_RESPONSE | jq -r .status)
    echo -ne "Status: $STATUS\r"
    
    if [ "$STATUS" = "COMPLETED" ]; then
        echo ""
        echo ""
        echo "✅ Job completed successfully!"
        echo ""
        echo "📊 Model check results:"
        echo $STATUS_RESPONSE | jq '.output.model_info' || echo $STATUS_RESPONSE | jq .
        
        # Check xfuser status
        XFUSER_AVAILABLE=$(echo $STATUS_RESPONSE | jq -r '.output.model_info.xfuser_available')
        XFUSER_VERSION=$(echo $STATUS_RESPONSE | jq -r '.output.model_info.xfuser_version')
        
        echo ""
        if [ "$XFUSER_AVAILABLE" = "true" ]; then
            echo "🎉 SUCCESS: V105 is running with real xfuser version $XFUSER_VERSION!"
        else
            echo "⚠️  WARNING: xfuser is not available in V105"
        fi
        break
    elif [ "$STATUS" = "FAILED" ]; then
        echo ""
        echo ""
        echo "❌ Job failed:"
        echo $STATUS_RESPONSE | jq .
        break
    fi
    
    sleep 2
done

echo ""
echo "4️⃣ Submitting generation test..."
curl -s -X POST \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "action": "generate",
            "audio_1": "audio_1.wav",
            "condition_image": "image_1.png",
            "prompt": "A person talking naturally",
            "output_format": "base64"
        }
    }' \
    "$BASE_URL/run" | jq .

echo ""
echo "Test complete! Check RunPod dashboard for generation results."