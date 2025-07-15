#!/bin/bash
# Test V111 real WAN model implementation

API_KEY="${RUNPOD_API_KEY}"
ENDPOINT_ID="zu0ik6c8yukyl6"
BASE_URL="https://api.runpod.ai/v2/$ENDPOINT_ID"

if [ -z "$API_KEY" ]; then
    echo "❌ RUNPOD_API_KEY not set"
    exit 1
fi

echo "🚀 Testing V111 Real WAN Model Implementation..."
echo ""

# 1. Check model status and initialization
echo "1️⃣ Checking V111 model status..."
RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input": {"action": "model_check"}}' \
    "$BASE_URL/run")

JOB_ID=$(echo $RESPONSE | jq -r .id)
echo "Job ID: $JOB_ID"

if [ "$JOB_ID" = "null" ]; then
    echo "Failed to submit job. Response:"
    echo $RESPONSE | jq .
    exit 1
fi

# Wait for model check
sleep 5

for i in {1..20}; do
    STATUS_RESPONSE=$(curl -s -H "Authorization: Bearer $API_KEY" "$BASE_URL/status/$JOB_ID")
    STATUS=$(echo $STATUS_RESPONSE | jq -r .status)
    
    if [ "$STATUS" = "COMPLETED" ]; then
        echo ""
        echo "✅ Model check completed!"
        echo ""
        
        # Save full response
        echo $STATUS_RESPONSE | jq . > v111_model_check_result.json
        
        # Show system info
        echo "🖥️  System Info:"
        echo "  - CUDA: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.cuda_available')"
        echo "  - Device: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.device')"
        echo "  - PyTorch: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.pytorch_version')"
        echo "  - Version: $(echo $STATUS_RESPONSE | jq -r '.output.version')"
        echo ""
        
        # Show MultiTalk V111 status
        echo "🤖 MultiTalk V111 Status:"
        echo "  - Available: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.multitalk_v111_available')"
        echo "  - Initialized: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.multitalk_v111_initialized')"
        echo "  - xfuser: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.xfuser_available')"
        echo "  - SafeTensors: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.safetensors_available')"
        echo ""
        
        # Show model loading status
        echo "📦 Model Loading Status:"
        if [ "$(echo $STATUS_RESPONSE | jq -r '.output.model_info.multitalk_v111_info')" != "null" ]; then
            echo "  - wav2vec: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.multitalk_v111_info.models_loaded.wav2vec')"
            echo "  - vae: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.multitalk_v111_info.models_loaded.vae')"
            echo "  - clip: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.multitalk_v111_info.models_loaded.clip')"
            echo "  - diffusion: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.multitalk_v111_info.models_loaded.diffusion')"
            echo "  - multitalk: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.multitalk_v111_info.models_loaded.multitalk')"
            echo "  - text_encoder: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.multitalk_v111_info.models_loaded.text_encoder')"
        else
            echo "  - Model info not available"
        fi
        echo ""
        
        # Show key model files
        echo "📁 Key Model Files:"
        echo $STATUS_RESPONSE | jq -r '.output.model_info.volume_exploration.key_model_files | to_entries[] | "  - " + .key + ": " + (.value.exists | tostring) + " (" + (.value.size_mb | tostring) + " MB)"' 2>/dev/null || echo "  None found"
        echo ""
        
        break
        
    elif [ "$STATUS" = "FAILED" ]; then
        echo ""
        echo "❌ Model check failed:"
        echo $STATUS_RESPONSE | jq .
        break
    else
        echo -ne "Status: $STATUS ($i/20)\\r"
        sleep 3
    fi
done

echo ""
echo "2️⃣ Testing video generation with V111..."

# Test generation with sample files from network volume
GEN_RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "action": "generate",
            "audio_1": "wan2.1-i2v-14b-480p/examples/sample_audio.wav",
            "condition_image": "wan2.1-i2v-14b-480p/examples/sample_image.jpg",
            "prompt": "A person talking with natural facial expressions and lip sync",
            "output_format": "s3",
            "sample_steps": 20,
            "text_guidance_scale": 7.5,
            "audio_guidance_scale": 3.5,
            "seed": 42
        }
    }' \
    "$BASE_URL/run")

GEN_JOB_ID=$(echo $GEN_RESPONSE | jq -r .id)
echo "Generation job ID: $GEN_JOB_ID"

if [ "$GEN_JOB_ID" != "null" ]; then
    echo "✅ V111 generation job submitted successfully"
    echo ""
    
    # Monitor generation progress
    echo "3️⃣ Monitoring generation progress..."
    
    for i in {1..30}; do
        GEN_STATUS_RESPONSE=$(curl -s -H "Authorization: Bearer $API_KEY" "$BASE_URL/status/$GEN_JOB_ID")
        GEN_STATUS=$(echo $GEN_STATUS_RESPONSE | jq -r .status)
        
        if [ "$GEN_STATUS" = "COMPLETED" ]; then
            echo ""
            echo "✅ Video generation completed!"
            echo ""
            
            # Save generation result
            echo $GEN_STATUS_RESPONSE | jq . > v111_generation_result.json
            
            # Show generation results
            echo "📹 Generation Results:"
            echo "  - Status: $(echo $GEN_STATUS_RESPONSE | jq -r '.output.status')"
            echo "  - Message: $(echo $GEN_STATUS_RESPONSE | jq -r '.output.message')"
            
            if [ "$(echo $GEN_STATUS_RESPONSE | jq -r '.output.video_url')" != "null" ]; then
                echo "  - Video URL: $(echo $GEN_STATUS_RESPONSE | jq -r '.output.video_url')"
                echo "  - S3 Key: $(echo $GEN_STATUS_RESPONSE | jq -r '.output.s3_key')"
            fi
            
            # Show generation parameters
            echo ""
            echo "⚙️  Generation Parameters:"
            echo "  - Prompt: $(echo $GEN_STATUS_RESPONSE | jq -r '.output.generation_params.prompt')"
            echo "  - Sample Steps: $(echo $GEN_STATUS_RESPONSE | jq -r '.output.generation_params.sample_steps')"
            echo "  - Text Guidance: $(echo $GEN_STATUS_RESPONSE | jq -r '.output.generation_params.text_guidance_scale')"
            echo "  - Audio Guidance: $(echo $GEN_STATUS_RESPONSE | jq -r '.output.generation_params.audio_guidance_scale')"
            echo "  - Seed: $(echo $GEN_STATUS_RESPONSE | jq -r '.output.generation_params.seed')"
            
            break
            
        elif [ "$GEN_STATUS" = "FAILED" ]; then
            echo ""
            echo "❌ Video generation failed:"
            echo $GEN_STATUS_RESPONSE | jq .
            break
        else
            echo -ne "Generation Status: $GEN_STATUS ($i/30)\\r"
            sleep 5
        fi
    done
else
    echo "❌ Failed to submit V111 generation job"
    echo $GEN_RESPONSE | jq .
fi

echo ""
echo "📊 Test Summary:"
echo "  - Model Check: $([ -f v111_model_check_result.json ] && echo "✅ Complete" || echo "❌ Failed")"
echo "  - Generation Test: $([ -f v111_generation_result.json ] && echo "✅ Complete" || echo "❌ Failed")"
echo ""
echo "📋 Results saved to:"
echo "  - v111_model_check_result.json"
echo "  - v111_generation_result.json"
echo ""
echo "🎉 V111 Real WAN Model Implementation Testing Complete!"