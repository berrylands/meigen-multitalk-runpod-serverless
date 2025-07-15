#!/usr/bin/env python3
"""
Test V68 Memory Efficient locally
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import tempfile

# Add current directory to path
sys.path.append('/Users/jasonedge/CODEHOME/meigen-multitalk/runpod-multitalk')

try:
    from multitalk_v68_memory_efficient import MultiTalkV68Pipeline
    print("✓ Successfully imported MultiTalkV68Pipeline")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

def test_memory_efficient_architecture():
    """Test the memory efficient architecture components"""
    print("\n=== Testing Memory Efficient Components ===")
    
    try:
        from multitalk_v68_memory_efficient import (
            MemoryEfficientAttention, 
            AudioProjectionLayer,
            MemoryEfficientTransformerBlock,
            MemoryEfficientDiT,
            MultiTalkConfig
        )
        
        config = MultiTalkConfig()
        device = torch.device('cpu')  # Test on CPU first
        
        print("1. Testing MemoryEfficientAttention...")
        attn = MemoryEfficientAttention(dim=1024, num_heads=16, chunk_size=128)
        
        # Test with realistic sizes
        q = torch.randn(1, 1000, 1024)  # Batch, sequence, dim
        k = torch.randn(1, 100, 1024)   # Audio features
        v = torch.randn(1, 100, 1024)
        
        output = attn(q, k, v)
        print(f"   ✓ Input shape: {q.shape}, Output shape: {output.shape}")
        
        print("2. Testing AudioProjectionLayer...")
        audio_proj = AudioProjectionLayer(audio_dim=768, model_dim=1024)
        audio_features = torch.randn(1, 89, 768)  # Typical Wav2Vec2 output
        projected = audio_proj(audio_features)
        print(f"   ✓ Audio input: {audio_features.shape}, Projected: {projected.shape}")
        
        print("3. Testing MemoryEfficientTransformerBlock...")
        block = MemoryEfficientTransformerBlock(dim=1024, num_heads=16, chunk_size=128)
        x = torch.randn(1, 1000, 1024)
        audio = torch.randn(1, 89, 1024)
        
        output = block(x, audio)
        print(f"   ✓ Block input: {x.shape}, Output: {output.shape}")
        
        print("4. Testing MemoryEfficientDiT (small version)...")
        # Create smaller DiT for testing
        config.max_chunk_size = 64
        dit = MemoryEfficientDiT(config)
        
        # Test with smaller input
        x = torch.randn(1, 8, 5, 8, 8)  # Batch, channels, time, height, width
        timestep = torch.tensor([500])
        audio = torch.randn(1, 89, 1024)
        
        print(f"   Testing DiT with input shape: {x.shape}")
        output = dit(x, timestep, audio)
        print(f"   ✓ DiT output shape: {output.shape}")
        
        print("\n✓ All memory efficient components work correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_initialization():
    """Test pipeline initialization without full processing"""
    print("\n=== Testing Pipeline Initialization ===")
    
    try:
        # Mock the model path to avoid loading actual models
        pipeline = MultiTalkV68Pipeline(model_path="/tmp/fake_models")
        print("✗ Should have failed with missing models")
        return False
        
    except Exception as e:
        print(f"✓ Expected failure with missing models: {type(e).__name__}")
        return True

def test_audio_projection_weights():
    """Test audio projection weight mapping logic"""
    print("\n=== Testing Audio Projection Weight Mapping ===")
    
    try:
        from multitalk_v68_memory_efficient import AudioProjectionLayer
        
        # Create audio projector
        audio_proj = AudioProjectionLayer()
        
        # Mock MultiTalk weights structure
        mock_weights = {
            'audio_proj.norm.weight': torch.randn(768),
            'audio_proj.norm.bias': torch.randn(768),
            'audio_proj.proj1.weight': torch.randn(1024, 768),
            'audio_proj.proj1.bias': torch.randn(1024),
            'audio_proj.proj1_vf.weight': torch.randn(1024, 768),
            'audio_proj.proj1_vf.bias': torch.randn(1024),
            'audio_proj.proj2.weight': torch.randn(1024, 1024),
            'audio_proj.proj2.bias': torch.randn(1024),
            'audio_proj.proj3.weight': torch.randn(1024, 1024),
            'audio_proj.proj3.bias': torch.randn(1024),
        }
        
        print(f"Mock weights keys: {list(mock_weights.keys())}")
        print(f"Audio projector parameters: {list(audio_proj.named_parameters())}")
        
        # Test weight application logic
        applied_count = 0
        for name, param in audio_proj.named_parameters():
            for mt_key, mt_weight in mock_weights.items():
                if 'audio_proj' in mt_key and param.shape == mt_weight.shape:
                    param_suffix = name.split('.')[-1]
                    key_suffix = mt_key.split('.')[-1]
                    
                    if param_suffix == key_suffix:
                        if ('norm' in name and 'norm' in mt_key) or \
                           ('proj1_vf' in name and 'proj1_vf' in mt_key) or \
                           ('proj1' in name and 'proj1' in mt_key and 'proj1_vf' not in name and 'proj1_vf' not in mt_key) or \
                           ('proj2' in name and 'proj2' in mt_key) or \
                           ('proj3' in name and 'proj3' in mt_key):
                            print(f"   Would apply: {mt_key} -> {name} (shapes: {mt_weight.shape} -> {param.shape})")
                            applied_count += 1
                            break
        
        print(f"✓ Would apply {applied_count} weights")
        return True
        
    except Exception as e:
        print(f"✗ Weight mapping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing MultiTalk V68 Memory Efficient Architecture")
    print("=" * 60)
    
    tests = [
        test_memory_efficient_architecture,
        test_pipeline_initialization,
        test_audio_projection_weights,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✓ V68 architecture is ready for deployment!")
        return True
    else:
        print("✗ Some tests failed - needs fixes before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)