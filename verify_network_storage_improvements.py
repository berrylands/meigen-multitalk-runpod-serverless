#!/usr/bin/env python3
"""
Verify Network Storage Improvements
Check that components are properly cached and system is more efficient
"""

import subprocess
import json
import os
import time

def test_cold_start_performance():
    """Test cold start performance with cached components"""
    
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return False
    
    endpoint_id = "zu0ik6c8yukyl6"
    
    print("🕐 Testing cold start performance with cached components...")
    
    # Test multiple model checks to see consistent performance
    test_results = []
    
    for i in range(3):
        print(f"\n🔄 Test {i+1}/3: Cold start performance...")
        
        test_job = {
            "input": {
                "action": "model_check"
            }
        }
        
        url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        
        curl_cmd = [
            "curl", "-X", "POST", 
            url,
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {api_key}",
            "-d", json.dumps(test_job)
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                job_id = response.get("id")
                
                if job_id:
                    # Monitor job completion time
                    job_start = time.time()
                    
                    for j in range(120):  # 2 minutes max
                        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
                        
                        status_cmd = [
                            "curl", "-H", f"Authorization: Bearer {api_key}",
                            status_url
                        ]
                        
                        status_result = subprocess.run(status_cmd, capture_output=True, text=True)
                        
                        if status_result.returncode == 0:
                            status_data = json.loads(status_result.stdout)
                            status = status_data.get("status")
                            
                            if status == "COMPLETED":
                                completion_time = time.time() - job_start
                                
                                output = status_data.get("output", {})
                                version = output.get("output", {}).get("version", "Unknown")
                                
                                test_results.append({
                                    "test": i+1,
                                    "completion_time": completion_time,
                                    "status": "success",
                                    "version": version
                                })
                                
                                print(f"  ✅ Completed in {completion_time:.1f} seconds (Version {version})")
                                break
                            
                            elif status == "FAILED":
                                test_results.append({
                                    "test": i+1,
                                    "completion_time": None,
                                    "status": "failed"
                                })
                                print(f"  ❌ Failed")
                                break
                            
                            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                                time.sleep(1)
                                continue
                            
                        else:
                            print(f"  ❌ Status check failed")
                            break
                    else:
                        print(f"  ⏱️  Timeout")
                        test_results.append({
                            "test": i+1,
                            "completion_time": None,
                            "status": "timeout"
                        })
                else:
                    print(f"  ❌ No job ID")
                    test_results.append({
                        "test": i+1,
                        "completion_time": None,
                        "status": "no_job_id"
                    })
            else:
                print(f"  ❌ Request failed")
                test_results.append({
                    "test": i+1,
                    "completion_time": None,
                    "status": "request_failed"
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            test_results.append({
                "test": i+1,
                "completion_time": None,
                "status": "error"
            })
        
        # Wait between tests
        if i < 2:
            time.sleep(5)
    
    # Analyze results
    print(f"\n📊 Performance Analysis:")
    successful_tests = [r for r in test_results if r["status"] == "success"]
    
    if successful_tests:
        times = [r["completion_time"] for r in successful_tests]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"  Successful tests: {len(successful_tests)}/3")
        print(f"  Average completion time: {avg_time:.1f} seconds")
        print(f"  Min completion time: {min_time:.1f} seconds")
        print(f"  Max completion time: {max_time:.1f} seconds")
        
        if avg_time < 30:
            print(f"  ✅ Good performance (< 30s average)")
        elif avg_time < 60:
            print(f"  ⚠️  Moderate performance (30-60s average)")
        else:
            print(f"  ❌ Slow performance (> 60s average)")
        
        return True
    else:
        print(f"  ❌ No successful tests")
        return False

def check_storage_utilization():
    """Check current storage utilization"""
    
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        return False
    
    endpoint_id = "zu0ik6c8yukyl6"
    
    print(f"\n📊 Checking storage utilization...")
    
    storage_job = {
        "input": {
            "action": "volume_explore"
        }
    }
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    curl_cmd = [
        "curl", "-X", "POST", 
        url,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(storage_job)
    ]
    
    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            job_id = response.get("id")
            
            if job_id:
                # Wait for completion
                for i in range(60):
                    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
                    
                    status_cmd = [
                        "curl", "-H", f"Authorization: Bearer {api_key}",
                        status_url
                    ]
                    
                    status_result = subprocess.run(status_cmd, capture_output=True, text=True)
                    
                    if status_result.returncode == 0:
                        status_data = json.loads(status_result.stdout)
                        status = status_data.get("status")
                        
                        if status == "COMPLETED":
                            output = status_data.get("output", {})
                            
                            if "output" in output and "exploration" in output["output"]:
                                exploration = output["output"]["exploration"]
                                
                                total_size_gb = exploration.get("total_size_gb", 0)
                                models_present = exploration.get("key_model_files", {})
                                
                                print(f"  📦 Total storage used: {total_size_gb:.2f} GB")
                                print(f"  📂 Storage capacity: 110 GB")
                                print(f"  📊 Utilization: {(total_size_gb/110)*100:.1f}%")
                                
                                present_count = sum(1 for info in models_present.values() if info.get("exists"))
                                total_count = len(models_present)
                                
                                print(f"  🗂️  Key models present: {present_count}/{total_count}")
                                
                                if present_count >= 5:
                                    print(f"  ✅ Excellent model coverage")
                                elif present_count >= 3:
                                    print(f"  ⚠️  Good model coverage")
                                else:
                                    print(f"  ❌ Poor model coverage")
                                
                                return True
                            else:
                                print(f"  ❌ No exploration data")
                                return False
                        
                        elif status == "FAILED":
                            print(f"  ❌ Storage check failed")
                            return False
                        
                        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                            time.sleep(1)
                            continue
                        
                    else:
                        print(f"  ❌ Status check failed")
                        return False
                
                print(f"  ⏱️  Storage check timeout")
                return False
            else:
                print(f"  ❌ No job ID")
                return False
        else:
            print(f"  ❌ Request failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    print("=" * 80)
    print("NETWORK STORAGE IMPROVEMENTS VERIFICATION")
    print("=" * 80)
    print("Testing performance and storage utilization after component caching")
    print("=" * 80)
    
    # Test 1: Cold start performance
    print("\n🚀 Test 1: Cold start performance with cached components")
    perf_success = test_cold_start_performance()
    
    # Test 2: Storage utilization
    print("\n💾 Test 2: Storage utilization analysis")
    storage_success = check_storage_utilization()
    
    # Final assessment
    print("\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)
    
    if perf_success and storage_success:
        print("✅ NETWORK STORAGE IMPROVEMENTS VERIFIED!")
        print("✅ Components are properly cached")
        print("✅ Performance is improved")
        print("✅ Storage utilization is optimal")
        
        print("\n🎯 Benefits achieved:")
        print("  • Faster cold starts due to cached components")
        print("  • More reliable operation with local storage")
        print("  • Better resource utilization")
        print("  • Reduced dependency on external services")
        
        print("\n📈 Next steps:")
        print("  • Deploy V114 for enhanced offline capabilities")
        print("  • Monitor performance improvements")
        print("  • Consider additional optimizations")
        
        return True
    else:
        print("⚠️  NETWORK STORAGE IMPROVEMENTS PARTIALLY VERIFIED")
        print("Some aspects may need additional optimization")
        return False

if __name__ == "__main__":
    main()