# V121 Manual Update Guide

## 🎯 Goal
Update the original endpoint `zu0ik6c8yukyl6` to use V121 template while keeping the network volume attached.

## 📋 Manual Steps

### 1. Access RunPod Console
- Go to https://runpod.ai
- Navigate to Serverless → Endpoints
- Find endpoint: `multitalk-v114-complete-offline -fb` (ID: zu0ik6c8yukyl6)

### 2. Update Template
- Click "Edit" on the endpoint
- Change template from: `multitalk-v120-xfuser` 
- Change template to: `multitalk-v121-mock-xfuser`
- **Important**: Keep network volume `pth5bf7dey` attached
- Keep all environment variables as-is
- Save changes

### 3. Verify Configuration
After update, endpoint should have:
- ✅ Template: `multitalk-v121-mock-xfuser`
- ✅ Network Volume: `pth5bf7dey` (meigen-multitalk)
- ✅ Environment variables preserved
- ✅ GPU types: RTX 3090, RTX 4090
- ✅ Serverless configuration intact

## 🧪 Test V121 After Update

Once manually updated, run:
```bash
python test_v121_runpod.py
```

## 📊 Expected V121 Behavior

### ✅ SUCCESS INDICATORS:
1. **No xfuser import errors** - Mock bypasses dependency issues
2. **Handler starts as V121** - Version updated in logs
3. **Model loading begins** - Network volume access working
4. **Progress beyond imports** - Gets to actual MeiGen-MultiTalk execution

### ⚠️ POTENTIAL ISSUES:
1. **Model loading errors** - Network volume or model file issues
2. **MeiGen-MultiTalk failures** - Implementation problems now visible
3. **New error types** - Past import barriers, hitting real bugs

## 🎯 V121 Success Criteria

V121 success means:
- **Import barriers removed** ✅
- **xfuser mocked successfully** ✅ 
- **Reveals actual implementation issues** ✅
- **Progress to real debugging** ✅

Even if V121 fails during execution, it's successful if it gets past the import phase and shows what the real problems are.

## 🔄 Next Steps Based on Results

**If V121 works completely**: 
- We have a working MultiTalk implementation!

**If V121 fails with new errors**:
- Analyze the specific errors (model loading, inference, etc.)
- Create V122 to address the revealed issues
- Continue iterating on actual implementation problems

**If V121 still has import issues**:
- Check mock xfuser implementation
- May need more comprehensive mocking

## 📝 Template Details

The V121 template uses:
- **Image**: `berrylands/multitalk-runpod:v121`
- **Mock xfuser**: Bypasses import dependency issues
- **Based on V115**: Working foundation
- **Environment**: All variables preserved from V120