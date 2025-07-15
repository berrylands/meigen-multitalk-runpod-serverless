#!/usr/bin/env python3
"""
Test the device fix for MultiTalk V58
This test verifies that the L-RoPE speaker tensor is properly moved to GPU
"""

import torch
import torch.nn as nn
import tempfile
import sys
from pathlib import Path

def test_device_fix():
    """Test the critical device fix for L-RoPE speaker binding"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test device fix")
        return False
    
    device = torch.device("cuda")
    print(f"✓ Testing on device: {device}")
    
    # Simulate the problematic code from V57
    print("\n=== Testing V57 (Broken) Pattern ===")
    try:
        label_embeddings_v57 = nn.Embedding(8, 768)  # NOT moved to device
        speaker_id = 0
        
        # This should fail - tensor created on CPU but embedding on CPU too initially
        speaker_tensor_v57 = torch.tensor([speaker_id])  # CPU tensor
        
        # Move embedding to GPU but tensor stays on CPU - this causes the error
        label_embeddings_v57 = label_embeddings_v57.to(device)
        
        # This line should fail with device mismatch
        speaker_embed_v57 = label_embeddings_v57(speaker_tensor_v57)
        print("❌ V57 pattern should have failed but didn't")
        return False
        
    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            print(f"✓ V57 pattern correctly fails with: {e}")
        else:
            print(f"❌ V57 pattern failed with unexpected error: {e}")
            return False
    
    # Test the V58 fix
    print("\n=== Testing V58 (Fixed) Pattern ===")
    try:
        # V58 Fix: Move embedding to device in initialization
        label_embeddings_v58 = nn.Embedding(8, 768).to(device)
        speaker_id = 0
        
        # V58 Fix: Create tensor directly on correct device
        speaker_tensor_v58 = torch.tensor([speaker_id], device=device, dtype=torch.long)
        
        # This should work fine
        speaker_embed_v58 = label_embeddings_v58(speaker_tensor_v58)
        
        print(f"✓ V58 pattern works: speaker_embed shape {speaker_embed_v58.shape}")
        print(f"✓ All tensors on same device: {speaker_embed_v58.device}")
        
        # Test that we can do the addition too
        fake_features = torch.randn(1, 100, 768, device=device)
        result = fake_features + speaker_embed_v58.unsqueeze(1)
        print(f"✓ Audio feature addition works: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ V58 pattern failed: {e}")
        return False

def test_multitalk_initialization():
    """Test that MultiTalkV58Pipeline can be imported and has the fix"""
    try:
        # Import the V58 implementation
        sys.path.append(str(Path(__file__).parent))
        from multitalk_v58_device_fix import MultiTalkAudioProcessor, MultiTalkConfig
        
        print("\n=== Testing MultiTalk V58 Audio Processor ===")
        
        config = MultiTalkConfig(device="cuda" if torch.cuda.is_available() else "cpu")
        processor = MultiTalkAudioProcessor(config)
        
        # Check that label_embeddings is on the correct device
        expected_device = torch.device(config.device)
        actual_device = next(processor.label_embeddings.parameters()).device
        
        if actual_device == expected_device:
            print(f"✓ label_embeddings correctly on {actual_device}")
            return True
        else:
            print(f"❌ label_embeddings on {actual_device}, expected {expected_device}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test MultiTalk initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing MultiTalk V58 Device Fix")
    print("=" * 50)
    
    # Test 1: Core device fix pattern
    test1_passed = test_device_fix()
    
    # Test 2: MultiTalk class initialization
    test2_passed = test_multitalk_initialization()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED - V58 device fix is working correctly!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - V58 device fix needs more work")
        sys.exit(1)