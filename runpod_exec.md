# URGENT: Execute V102 Build on RunPod Pod

## Pod Details
- **Pod Name**: xfuser-builder-large
- **Pod ID**: bv73xecvgq13rr
- **IP**: 213.173.110.4
- **SSH Port**: 19633
- **Cost**: $0.69/hour
- **Disk**: 700GB total (500GB volume + 200GB container)

## EXECUTE NOW IN RUNPOD WEB TERMINAL:

```bash
curl -sSL https://raw.githubusercontent.com/berrylands/meigen-multitalk-runpod-serverless/master/fast_build_v102.sh | bash
```

## What this does:
1. Installs Docker
2. Clones the repository
3. Builds V102 with REAL xfuser (not stub)
4. Pushes to DockerHub as `berrylands/multitalk-runpod:v102-real-xfuser`

## Access the terminal:
1. Go to https://runpod.io/console/pods
2. Click on "xfuser-builder-large"
3. Click "Connect" → "Start Web Terminal"
4. Paste the command above

The build will take approximately 10-20 minutes with the 700GB of disk space available.