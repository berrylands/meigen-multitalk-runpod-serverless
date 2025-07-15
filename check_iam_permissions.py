#!/usr/bin/env python3
"""
Check what IAM permissions are needed for S3 access
"""

print("""
🔍 IAM Permission Check

For S3 access to work, the IAM user needs these permissions:

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::760572149-framepack",
                "arn:aws:s3:::760572149-framepack/*"
            ]
        }
    ]
}

Common issues:
1. ❌ IAM user only has GetObject but not ListBucket permission
2. ❌ IAM policy restricts access by IP address (RunPod IPs not whitelisted)
3. ❌ IAM policy has condition that restricts access
4. ❌ Wrong AWS Access Key ID or Secret in RunPod

To test if it's a permission issue:
1. Try creating a new IAM user with full S3 access
2. Update RunPod secrets with the new credentials
3. Test again

Or check your current IAM policy for any conditions like:
- IpAddress restrictions
- MFA requirements  
- Time-based restrictions
""")