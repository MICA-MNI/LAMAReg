# LAMAReg Docker Containerization

This directory contains all Docker-related files and scripts for containerizing LAMAReg.

## 📁 Directory Structure

```
docker/
├── Dockerfile              # Main Docker image definition
├── .dockerignore           # Files to exclude from Docker build context
├── deploy_to_server.sh     # Automated deployment script (migration + build)
├── migrate_to_server.sh    # Server migration script  
├── build_docker.sh         # Docker build script
├── test_docker.sh          # Docker image testing script
├── build_singularity.sh    # Convert Docker to Singularity SIF
├── test_singularity.sh     # Test Singularity SIF container
├── build_local.sh          # Alternative local build option
└── README.md              # This file
```

## 🚀 Quick Start

### Automated Deployment (Recommended)
```bash
cd docker
./deploy_to_server.sh
```

This single command will:
1. Migrate files to the server
2. Automatically build the Docker image
3. Test the built image
4. Optionally build Singularity SIF
5. Test the SIF container
6. Provide usage instructions

### Manual Steps (if needed)
```bash
# Step 1: Migrate to server
./migrate_to_server.sh

# Step 2: Build on server (run this on the server)
cd /host/cassio/export03/data/enning/lamareg_build
./build_docker.sh

# Step 3: Test the image
# Test the image
./test_docker.sh

# Build Singularity SIF (for HPC)
./build_singularity.sh

# Test SIF
./test_singularity.sh
```

## 🚀 Singularity Usage (HPC Clusters)

```bash
# Basic help
singularity run /path/to/lamareg_latest.sif --help

## 📦 What Gets Built

- **Docker Image**: `localhost:5001/lamareg:latest` (~3.7GB)
- **Singularity SIF**: `/host/cassio/export03/data/enning/singularity/lamareg_latest.sif` (~1.5GB compressed)

## 🎯 LAMAReg Usage Examples

### Basic Commands
```bash
# Help and version info
singularity exec /host/cassio/export03/data/enning/singularity/lamareg_latest.sif lamareg --help
singularity exec /host/cassio/export03/data/enning/singularity/lamareg_latest.sif lamareg register --help

# Full registration pipeline
singularity exec -B /path/to/data:/data /host/cassio/export03/data/enning/singularity/lamareg_latest.sif lamareg register \
  --moving /data/moving.nii.gz --fixed /data/fixed.nii.gz \
  --output /data/registered.nii.gz \
  --moving-parc /data/moving_parc.nii.gz \
  --fixed-parc /data/fixed_parc.nii.gz \
  --registered-parc /data/reg_parc.nii.gz \
  --affine /data/affine.mat \
  --warpfield /data/warp.nii.gz

# Generate brain parcellation only
singularity exec -B /path/to/data:/data /host/cassio/export03/data/enning/singularity/lamareg_latest.sif lamareg synthseg \
  --i /data/input.nii.gz --o /data/parcellation.nii.gz --parc

# Apply existing warpfield
singularity exec -B /path/to/data:/data /host/cassio/export03/data/enning/singularity/lamareg_latest.sif lamareg apply-warpfield \
  --moving /data/moving.nii.gz --fixed /data/fixed.nii.gz \
  --output /data/warped.nii.gz \
  --warpfield /data/warp.nii.gz --affine /data/affine.mat
```

### Alternative CLI Access
If `lamareg` command doesn't work, use:
```bash
singularity exec /host/cassio/export03/data/enning/singularity/lamareg_latest.sif python -m lamareg.cli --help
```
```

## 📦 Docker Image Details

- **Base Image**: Python 3.11-slim
- **Size**: ~2-3GB (includes TensorFlow, ANTsPy, and neuroimaging libraries)
- **Build Time**: 10-15 minutes (first build), 2-5 minutes (subsequent builds)
- **User**: Non-root user `lamareg` for security

## 🧪 Usage Examples

```bash
# Basic help
docker run --rm lamareg:latest lamareg --help

# Process data (mount your data directory)
docker run --rm -v /path/to/your/data:/data lamareg:latest lamareg --input /data/input.nii.gz --output /data/output.nii.gz

# Interactive mode
docker run --rm -it -v /path/to/your/data:/data lamareg:latest bash
```

## 🔧 Configuration

### Server Paths
The scripts are configured for the following server setup:
- **Server Base**: `/host/cassio/export03/data/enning`
- **Build Directory**: `/host/cassio/export03/data/enning/lamareg_build`
- **Backup Directory**: `/host/cassio/export03/data/enning/lamareg_backup`

### Customization
To use different server paths, modify the variables at the top of `migrate_to_server.sh`:
```bash
SERVER_BASE_DIR="/your/server/path"
BUILD_DIR="$SERVER_BASE_DIR/lamareg_build"
```

## 🔍 Troubleshooting

### Singularity Build Issues

**"nodev" Mount Warning**
```bash
# Error: SINGULARITY: WARNING: 'nodev' mount option set on /host/cassio/export03
# Solution: Script automatically detects this and uses tar method
./diagnose_singularity.sh  # Check system status first
./build_singularity.sh     # Will handle filesystem issues automatically
```

**Invalid Tar Header Error**
```bash
# Error: archive/tar: invalid tar header
# Solution: Improved build script with robust error handling
./build_singularity.sh     # Uses fallback methods automatically
```

**SIF File Not Created**
```bash
# Comprehensive diagnostics
./diagnose_singularity.sh

# Manual checks
docker image ls | grep lamareg    # Ensure Docker image exists
df -h /host/cassio/export03       # Check available disk space (need 10GB+)
singularity --version             # Verify Singularity installation
```

### Docker Build Fails
1. Check disk space on server: `df -h`
2. Verify Docker daemon: `docker info`
3. Check for image conflicts: `docker image ls | grep lamareg`
4. Review build logs for specific errors

### Import Errors
- Ensure all dependencies are in requirements.txt
- Check if system dependencies are needed in Dockerfile
- Verify LAMAReg module structure

### Permission Issues
- Verify server directory is writable: `ls -la /host/cassio/export03/data/enning`
- Check Docker daemon permissions
- Ensure Singularity has proper access

### Quick Diagnostics
Run the diagnostic script for comprehensive system check:
```bash
./diagnose_singularity.sh  # Checks Docker, Singularity, filesystem, permissions
```

## 📊 Performance Notes

- **First Build**: Downloads and compiles all dependencies (~10-15 min)
- **Incremental Builds**: Uses Docker layer caching (~2-5 min)
- **Memory Usage**: ~4-6GB RAM during build, ~2-3GB runtime
- **Storage**: ~3-4GB for final image + build cache