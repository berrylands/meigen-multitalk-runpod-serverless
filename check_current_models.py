#!/usr/bin/env python3
"""
Check what models we currently have available
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

def analyze_current_models():
    """Analyze what models we currently have."""
    
    print("Current Model Inventory Analysis")
    print("=" * 60)
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    try:
        job = endpoint.run({"action": "list_models"})
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            time.sleep(2)
        
        if job.status() == "COMPLETED":
            result = job.output()
            models = result.get('models', [])
            total = result.get('total', 0)
            
            print(f"Total models: {total}")
            print(f"Total storage: {sum(m.get('size_mb', 0) for m in models) / 1024:.1f} GB")
            print()
            
            # Categorize models
            video_models = []
            audio_models = []
            enhancement_models = []
            other_models = []
            
            for model in models:
                name = model['name'].lower()
                size_gb = model.get('size_mb', 0) / 1024
                
                if any(keyword in name for keyword in ['wan', 'multitalk', 'stable', 'diffusion', 'video']):
                    video_models.append((model['name'], size_gb))
                elif any(keyword in name for keyword in ['wav2vec', 'chinese', 'kokoro', 'audio']):
                    audio_models.append((model['name'], size_gb))
                elif any(keyword in name for keyword in ['gfp', 'face', 'enhancement']):
                    enhancement_models.append((model['name'], size_gb))
                else:
                    other_models.append((model['name'], size_gb))
            
            print("📹 VIDEO GENERATION MODELS:")
            for name, size in video_models:
                print(f"  ✅ {name} ({size:.1f} GB)")
            if not video_models:
                print("  ❌ No video generation models found")
            
            print(f"\n🎵 AUDIO PROCESSING MODELS:")
            for name, size in audio_models:
                print(f"  ✅ {name} ({size:.1f} GB)")
            if not audio_models:
                print("  ❌ No audio processing models found")
            
            print(f"\n✨ ENHANCEMENT MODELS:")
            for name, size in enhancement_models:
                print(f"  ✅ {name} ({size:.1f} GB)")
            if not enhancement_models:
                print("  ❌ No enhancement models found")
            
            if other_models:
                print(f"\n🔧 OTHER MODELS:")
                for name, size in other_models:
                    print(f"  ✅ {name} ({size:.1f} GB)")
            
            # Assess readiness for MultiTalk
            print(f"\n" + "=" * 60)
            print("MULTITALK READINESS ASSESSMENT:")
            
            has_video = len(video_models) > 0
            has_audio = len(audio_models) > 0
            has_enhancement = len(enhancement_models) > 0
            
            print(f"  Video Generation: {'✅' if has_video else '❌'}")
            print(f"  Audio Processing: {'✅' if has_audio else '❌'}")
            print(f"  Face Enhancement: {'✅' if has_enhancement else '❌'}")
            
            if has_video and has_audio:
                print(f"\n🎉 READY FOR BASIC MULTITALK IMPLEMENTATION!")
                print("Can proceed with video generation pipeline")
            elif has_audio:
                print(f"\n⚠️  PARTIAL READINESS")
                print("Can implement audio processing, need video models")
            else:
                print(f"\n❌ NOT READY")
                print("Missing critical models for MultiTalk")
            
            # Storage analysis
            storage_used = sum(m.get('size_mb', 0) for m in models) / 1024
            storage_available = 100 - storage_used  # 100GB volume
            
            print(f"\n📊 STORAGE ANALYSIS:")
            print(f"  Used: {storage_used:.1f} GB / 100 GB")
            print(f"  Available: {storage_available:.1f} GB")
            
            if storage_available < 5:
                print(f"  ⚠️  Low storage - may need cleanup or larger volume")
            else:
                print(f"  ✅ Sufficient storage remaining")
            
            return models, has_video and has_audio
            
        else:
            print(f"❌ Failed to get model list: {job.output()}")
            return [], False
            
    except Exception as e:
        print(f"Error: {e}")
        return [], False

if __name__ == "__main__":
    models, ready = analyze_current_models()
    
    if ready:
        print(f"\n🚀 Next step: Implement MultiTalk video generation pipeline")
        print("We have the essential models to proceed!")
    else:
        print(f"\n🔄 Next step: Address missing models or implement with available ones")
        print("Will work with what we have and iterate")