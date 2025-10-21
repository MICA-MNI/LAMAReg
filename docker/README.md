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
4. Provide usage instructions

### Manual Steps (if needed)
```bash
# Step 1: Migrate to server
./migrate_to_server.sh

# Step 2: Build on server (run this on the server)
cd /host/cassio/export03/data/enning/lamareg_build
./build_docker.sh

# Step 3: Test the image
./test_docker.sh
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

### Build Fails
1. Check disk space on server
2. Verify all required files are copied
3. Check Docker daemon is running
4. Review build logs for specific errors

### Import Errors
- Ensure all dependencies are in requirements.txt
- Check if system dependencies are needed in Dockerfile

### Permission Issues
- Verify server directory is writable
- Check Docker daemon permissions

## 📊 Performance Notes

- **First Build**: Downloads and compiles all dependencies (~10-15 min)
- **Incremental Builds**: Uses Docker layer caching (~2-5 min)
- **Memory Usage**: ~4-6GB RAM during build, ~2-3GB runtime
- **Storage**: ~3-4GB for final image + build cache