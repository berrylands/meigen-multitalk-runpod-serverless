# GitHub Actions Docker Build Setup

This repository uses GitHub Actions to automatically build and push Docker images to Docker Hub.

## Setup Instructions

### 1. Add Docker Hub Token to GitHub Secrets

1. Go to Docker Hub: https://hub.docker.com/
2. Log in to your account
3. Go to Account Settings → Security
4. Create a new Access Token:
   - Description: `github-meigen-multitalk`
   - Access permissions: `Read, Write, Delete`
5. Copy the token

### 2. Add Secret to GitHub Repository

1. Go to your GitHub repository
2. Navigate to Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Create a secret named `DOCKERHUB_TOKEN`
5. Paste your Docker Hub access token as the value
6. Click "Add secret"

### 3. Trigger the Build

You can trigger the build in three ways:

#### Option A: Manual Trigger (Recommended for first run)
1. Go to Actions tab in your repository
2. Select "Build and Push MultiTalk" workflow
3. Click "Run workflow"
4. Select branch: `main` or `master`
5. Version: `v130-final` (default)
6. Click "Run workflow"

#### Option B: Push to Repository
```bash
git add .github/workflows/docker-build.yml
git commit -m "Add GitHub Actions workflow for V130 Docker build"
git push origin main
```

#### Option C: Direct Trigger via GitHub CLI
```bash
gh workflow run docker-build.yml --field version=v130-final
```

## Workflow Details

The workflow will:
1. Check out the repository
2. Set up Docker Buildx for multi-platform builds
3. Log in to Docker Hub using your token
4. Build the Docker image from `Dockerfile.v130-final`
5. Push to Docker Hub with tags:
   - `jasonedge/multitalk-runpod:v130`
   - `jasonedge/multitalk-runpod:v130-final`
   - `jasonedge/multitalk-runpod:latest`
6. Display build summary with next steps

## After Build Completes

1. Check the Actions tab for build status
2. Once successful, the image will be available at:
   - `jasonedge/multitalk-runpod:v130`
3. Update RunPod template:
   - Template ID: `5y1gyg4n78kqwz`
   - New image: `jasonedge/multitalk-runpod:v130`
4. Test with S3 files (1.wav and multi1.png)

## Troubleshooting

- If the build fails with authentication error, verify your DOCKERHUB_TOKEN secret
- If the build times out, it may need to be split into smaller stages
- Check the Actions logs for detailed error messages