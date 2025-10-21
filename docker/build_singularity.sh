#!/bin/bash
set -eu

# ============================================================================
# LAMAReg Singularity Build Script (Robust Version)
# ============================================================================
# Handles filesystem issues like 'nodev' mounts and tar header problems
# Configured for systems without fakeroot privileges

BASE_DIR="/host/cassio/export03/data/enning"
OUTPUT_DIR="${BASE_DIR}/singularity"
OUTPUT_PATH="${OUTPUT_DIR}/lamareg_latest.sif"
DOCKER_IMAGE="${1:-}"

# Logging function
echo_log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Auto-detect Docker image if not provided
if [[ -z "$DOCKER_IMAGE" ]]; then
    # Try localhost:5001/lamareg:latest first (registry version)
    if docker image inspect "localhost:5001/lamareg:latest" >/dev/null 2>&1; then
        DOCKER_IMAGE="localhost:5001/lamareg:latest"
    # Fall back to lamareg:latest (local build)
    elif docker image inspect "lamareg:latest" >/dev/null 2>&1; then
        DOCKER_IMAGE="lamareg:latest"
    else
        echo_log "❌ No LAMAReg Docker image found. Tried:"
        echo_log "   - localhost:5001/lamareg:latest"
        echo_log "   - lamareg:latest"
        echo_log "💡 Build Docker image first:"
        echo_log "   ./build_docker.sh"
        exit 1
    fi
fi

echo_log "🚀 Starting LAMAReg Singularity Build"
echo_log "===================================="
echo_log "📍 Output: $OUTPUT_PATH"
echo_log "🐳 Docker image: $DOCKER_IMAGE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check Docker image
if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo_log "❌ Docker image not found: $DOCKER_IMAGE"
    echo_log "💡 Build Docker image first:"
    echo_log "   ./build_docker.sh"
    exit 1
fi

DOCKER_SIZE=$(docker image inspect "$DOCKER_IMAGE" --format='{{.Size}}' | awk '{print $1/1024/1024/1024 " GB"}')
echo_log "✅ Found Docker image: $DOCKER_SIZE"

# Check Singularity
if ! command -v singularity >/dev/null 2>&1; then
    echo_log "❌ Singularity not found"
    echo_log "💡 Install Singularity first"
    exit 1
fi

SING_VERSION=$(singularity --version)
echo_log "✅ Singularity version: $SING_VERSION"

# Check available space
AVAILABLE=$(df "$BASE_DIR" | awk 'NR==2 {print int($4/1024/1024)}')
if [ "$AVAILABLE" -lt 10 ]; then
    echo_log "❌ Insufficient space: ${AVAILABLE}GB available"
    echo_log "💡 Need at least 10GB free space"
    exit 1
fi
echo_log "✅ Space check: ${AVAILABLE}GB available"

# Remove existing output
if [ -f "$OUTPUT_PATH" ]; then
    echo_log "⚠️  Removing existing SIF file"
    rm -f "$OUTPUT_PATH"
fi

START_TIME=$(date +%s)

# Check for filesystem issues
MOUNT_INFO=$(mount | grep "$(dirname "$OUTPUT_PATH")" || echo "")
if echo "$MOUNT_INFO" | grep -q nodev; then
    echo_log "⚠️  WARNING: 'nodev' mount detected - using tar method"
    USE_TAR_METHOD=true
else
    USE_TAR_METHOD=false
fi

# ============================================================================
# Method Selection and Execution
# ============================================================================

BUILD_SUCCESS=false
METHOD=""

if [ "$USE_TAR_METHOD" = "false" ]; then
    # Try streaming method first
    echo_log "⚡ Attempting streaming method..."
    
    if timeout 1800 bash -c "
        set -o pipefail
        docker save '$DOCKER_IMAGE' | singularity build --force '$OUTPUT_PATH' docker-archive:/dev/stdin
    " 2>&1; then
        if [[ -f "$OUTPUT_PATH" ]] && [[ -s "$OUTPUT_PATH" ]]; then
            if singularity inspect "$OUTPUT_PATH" >/dev/null 2>&1; then
                BUILD_SUCCESS=true
                METHOD="streaming"
                echo_log "✅ Streaming method succeeded!"
            else
                echo_log "❌ Invalid SIF file from streaming method"
                rm -f "$OUTPUT_PATH" 2>/dev/null || true
            fi
        else
            echo_log "❌ Streaming method produced no output"
        fi
    else
        echo_log "❌ Streaming method failed"
        rm -f "$OUTPUT_PATH" 2>/dev/null || true
    fi
fi

# Fall back to tar method if streaming failed or if filesystem requires it
if [ "$BUILD_SUCCESS" = "false" ]; then
    echo_log "🔧 Using tar method (more reliable for problematic filesystems)..."
    
    TAR_FILE="${BASE_DIR}/lamareg_docker_$$.tar"
    
    echo_log "📤 Exporting Docker image to tar..."
    if docker save "$DOCKER_IMAGE" -o "$TAR_FILE"; then
        TAR_SIZE=$(du -h "$TAR_FILE" | cut -f1)
        echo_log "✅ Docker export complete: $TAR_SIZE"
        
        echo_log "🔧 Building SIF from tar file..."
        if timeout 1800 singularity build --force "$OUTPUT_PATH" "docker-archive://$TAR_FILE" 2>&1; then
            if [[ -f "$OUTPUT_PATH" ]] && [[ -s "$OUTPUT_PATH" ]]; then
                if singularity inspect "$OUTPUT_PATH" >/dev/null 2>&1; then
                    BUILD_SUCCESS=true
                    METHOD="tar"
                    echo_log "✅ Tar method succeeded!"
                else
                    echo_log "❌ Invalid SIF file from tar method"
                    rm -f "$OUTPUT_PATH" 2>/dev/null || true
                fi
            else
                echo_log "❌ Tar method produced no output"
            fi
        else
            echo_log "❌ Singularity build from tar failed"
        fi
        
        # Cleanup tar file
        echo_log "🧹 Cleaning up tar file..."
        rm -f "$TAR_FILE" 2>/dev/null || true
    else
        echo_log "❌ Docker save to tar failed"
    fi
fi

# ============================================================================
# Final Validation and Results
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

if [ "$BUILD_SUCCESS" = "true" ] && [ -f "$OUTPUT_PATH" ] && [ -s "$OUTPUT_PATH" ]; then
    SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
    
    echo_log "============================================="
    echo_log "✅ LAMAReg SINGULARITY BUILD COMPLETE"
    echo_log "============================================="
    echo_log "📦 File: $OUTPUT_PATH"
    echo_log "📊 Size: $SIZE"
    echo_log "⏱️  Time: ${DURATION_MIN}m ${DURATION_SEC}s"
    echo_log "🎯 Method: $METHOD"
    echo_log ""
    echo_log "🧪 Test Commands:"
    echo_log "   # Test LAMAReg CLI"
    echo_log "   singularity exec $OUTPUT_PATH lamareg --help"
    echo_log ""
    echo_log "   # Test Python import"
    echo_log "   singularity exec $OUTPUT_PATH python -c 'import lamareg; print(\"LAMAReg ready!\")'"
    echo_log ""
    echo_log "   # Full registration example"
    echo_log "   singularity exec -B /path/to/data:/data $OUTPUT_PATH lamareg register \\"
    echo_log "     --moving /data/moving.nii.gz --fixed /data/fixed.nii.gz \\"
    echo_log "     --output /data/registered.nii.gz"
    echo_log ""
    echo_log "🚀 LAMAReg SIF ready for HPC deployment!"
    
    # Quick validation test
    echo_log "🧪 Running quick validation test..."
    if singularity exec "$OUTPUT_PATH" python -c "import lamareg; print('✅ LAMAReg import successful')" 2>&1; then
        echo_log "✅ Validation passed - container is functional"
    else
        echo_log "⚠️  Validation warning - container may have issues"
    fi
    
else
    echo_log "============================================="
    echo_log "❌ SINGULARITY BUILD FAILED"
    echo_log "============================================="
    echo_log "⏱️  Time: ${DURATION_MIN}m ${DURATION_SEC}s"
    echo_log "❌ ERROR: SIF file not created or empty"
    echo_log ""
    echo_log "🔍 Troubleshooting:"
    echo_log "   1. Check Docker image: docker image ls | grep lamareg"
    echo_log "   2. Check disk space: df -h $BASE_DIR"
    echo_log "   3. Check Singularity: singularity --version"
    echo_log "   4. Try building Docker image again: ./build_docker.sh"
    echo_log ""
    exit 1
fi