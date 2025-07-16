# V121 RunPod Testing Guide

## 🚀 V121 Overview

**Docker Image**: `berrylands/multitalk-runpod:v121`  
**Key Feature**: Mock xfuser implementation to bypass import errors  
**Goal**: Allow MeiGen-MultiTalk to load and show actual implementation issues

## 🔧 What V121 Fixes

V121 creates a mock xfuser module to satisfy import requirements:

```python
# Mock xfuser structure:
xfuser/
├── __init__.py (version 0.4.0-mock)
└── core/
    ├── __init__.py
    └── distributed.py (with mock functions)
```

### Mock Functions:
- `is_dp_last_group()` → Returns `True`
- `get_world_group()` → Returns `None` 
- `get_data_parallel_rank()` → Returns `0`
- `get_data_parallel_world_size()` → Returns `1`
- `get_runtime_state()` → Returns mock object

## 🧪 Testing V121 on RunPod

### 1. Update RunPod Endpoint
```yaml
Container Image: berrylands/multitalk-runpod:v121
Container Disk: 20 GB
Volume Disk: 100 GB
GPU: A100 40GB or RTX 4090
Volume Mount Path: /runpod-volume
```

### 2. Test with Script
```bash
cd /Users/jasonedge/CODEHOME/meigen-multitalk
export RUNPOD_API_KEY=your_actual_api_key
# Update ENDPOINT_ID in test_v121_runpod.py
python test_v121_runpod.py
```

### 3. Manual Test
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {}}'
```

## 📊 Expected Results

### ✅ SUCCESS INDICATORS in Logs:
1. `xfuser 0.4.0-mock (mock) installed`
2. `xfuser distributed mock OK`
3. Handler reports as `V121`
4. No `ModuleNotFoundError: No module named xfuser`
5. MeiGen-MultiTalk imports succeed
6. Progress to actual implementation issues (not import failures)

### ❌ FAILURE INDICATORS:
1. Still getting xfuser import errors
2. Mock module not found
3. Handler fails to start
4. MeiGen-MultiTalk still can't import

## 🎯 V121 Goals

1. **Bypass Import Barriers**: Get past xfuser dependency issues
2. **Reveal Real Issues**: Show actual MeiGen-MultiTalk implementation problems
3. **Progress Beyond Dependencies**: Move from "can't import" to "implementation needs work"

## 📈 Version Progression Context

| Version | Issue | Fix | Expected Result |
|---------|-------|-----|-----------------|
| V119 | Missing dependencies | Minimal NumPy fix | ❌ Still failing |
| V120 | xfuser conflicts | Install real xfuser | ❌ Space/dependency issues |
| **V121** | Import failures | **Mock xfuser** | ✅ Bypass imports, show real errors |

## 🔍 What to Look For

V121 should get us past the import phase and into actual MeiGen-MultiTalk execution. Look for:

1. **Import Success**: No more xfuser not found errors
2. **New Error Types**: Model loading, inference, or generation errors
3. **Progress Indicators**: Handler actually trying to process requests
4. **MeiGen-MultiTalk Activity**: Signs that the actual implementation is running

## 📝 Next Steps Based on Results

- **If V121 works**: Move to fixing actual MeiGen-MultiTalk implementation issues
- **If still import errors**: May need different mock approach or dependency fixes
- **If new errors**: Address specific MeiGen-MultiTalk implementation problems

## 🚀 Ready for RunPod Testing

V121 is specifically designed to overcome the import barrier that's been blocking progress. Test it on RunPod to see what the real implementation issues are!