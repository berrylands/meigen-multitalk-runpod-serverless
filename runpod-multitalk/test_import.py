#!/usr/bin/env python3
"""Test imports for V136"""
import sys
sys.path.insert(0, '/app/cog_multitalk_reference')

print("Testing imports...")

try:
    print("1. Importing wan...")
    import wan
    print("✅ wan imported")
except Exception as e:
    print(f"❌ wan import failed: {e}")

try:
    print("\n2. Importing wan.configs...")
    from wan.configs import WAN_CONFIGS
    print("✅ wan.configs imported")
except Exception as e:
    print(f"❌ wan.configs import failed: {e}")

try:
    print("\n3. Importing transformers...")
    from transformers import Wav2Vec2FeatureExtractor
    print("✅ transformers imported")
except Exception as e:
    print(f"❌ transformers import failed: {e}")

try:
    print("\n4. Importing src.audio_analysis.wav2vec2...")
    from src.audio_analysis.wav2vec2 import Wav2Vec2Model
    print("✅ src.audio_analysis.wav2vec2 imported")
except Exception as e:
    print(f"❌ src.audio_analysis.wav2vec2 import failed: {e}")

print("\nImport test complete!")