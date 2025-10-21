#!/bin/bash
set -eu

# ============================================================================
# LAMAReg Singularity Build Script (Robust Version)
# ============================================================================
# Handles filesystem issues like 'nodev' mounts and tar header problems

BASE_DIR="/host/cassio/export03/data/enning"
OUTPUT_DIR="${BASE_DIR}/singularity"
OUTPUT_PATH="${OUTPUT_DIR}/lamareg_latest.sif"
DOCKER_IMAGE="${1:-localhost:5001/lamareg:latest}"

# Logging function
log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

log "🚀 Starting LAMAReg Singularity Build"
log "===================================="
log "📍 Output: $OUTPUT_PATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check Docker image
if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    log "❌ Docker image not found: $DOCKER_IMAGE"
    log "💡 Build Docker image first:"
    log "   ./build_docker.sh"
    exit 1
fi

DOCKER_SIZE=$(docker image inspect "$DOCKER_IMAGE" --format='{{.Size}}' | awk '{print $1/1024/1024/1024 " GB"}')
log "✅ Found Docker image: $DOCKER_SIZE"

# Check Singularity
if ! command -v singularity >/dev/null 2>&1; then
    log "❌ Singularity not found"
    log "💡 Install Singularity first"
    exit 1
fi

SING_VERSION=$(singularity --version)
log "✅ Singularity version: $SING_VERSION"

# Check available space
AVAILABLE=$(df "$BASE_DIR" | awk 'NR==2 {print int($4/1024/1024)}')
if [ "$AVAILABLE" -lt 10 ]; then
    log "❌ Insufficient space: ${AVAILABLE}GB available"
    log "💡 Need at least 10GB free space"
    exit 1
fi
log "✅ Space check: ${AVAILABLE}GB available"

# Remove existing output
if [ -f "$OUTPUT_PATH" ]; then
    log "⚠️  Removing existing SIF file"
    rm -f "$OUTPUT_PATH"
fi

START_TIME=$(date +%s)

# Check for filesystem issues
MOUNT_INFO=$(mount | grep "$(dirname "$OUTPUT_PATH")" || echo "")
if echo "$MOUNT_INFO" | grep -q nodev; then
    log "⚠️  WARNING: 'nodev' mount detected - using tar method"
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
    log "⚡ Attempting streaming method..."
    
    if timeout 1800 bash -c "
        set -o pipefail
        docker save '$DOCKER_IMAGE' | singularity build --force --fakeroot '$OUTPUT_PATH' docker-archive:/dev/stdin
    " 2>&1; then
        if [[ -f "$OUTPUT_PATH" ]] && [[ -s "$OUTPUT_PATH" ]]; then
            if singularity inspect "$OUTPUT_PATH" >/dev/null 2>&1; then
                BUILD_SUCCESS=true
                METHOD="streaming"
                log "✅ Streaming method succeeded!"
            else
                log "❌ Invalid SIF file from streaming method"
                rm -f "$OUTPUT_PATH" 2>/dev/null || true
            fi
        else
            log "❌ Streaming method produced no output"
        fi
    else
        log "❌ Streaming method failed"
        rm -f "$OUTPUT_PATH" 2>/dev/null || true
    fi
fi

# Fall back to tar method if streaming failed or if filesystem requires it
if [ "$BUILD_SUCCESS" = "false" ]; then
    log "🔧 Using tar method (more reliable for problematic filesystems)..."
    
    TAR_FILE="${BASE_DIR}/lamareg_docker_$$.tar"
    
    log "📤 Exporting Docker image to tar..."
    if docker save "$DOCKER_IMAGE" -o "$TAR_FILE"; then
        TAR_SIZE=$(du -h "$TAR_FILE" | cut -f1)
        log "✅ Docker export complete: $TAR_SIZE"
        
        log "🔧 Building SIF from tar file..."
        if timeout 1800 singularity build --force --fakeroot "$OUTPUT_PATH" "docker-archive://$TAR_FILE" 2>&1; then
            if [[ -f "$OUTPUT_PATH" ]] && [[ -s "$OUTPUT_PATH" ]]; then
                if singularity inspect "$OUTPUT_PATH" >/dev/null 2>&1; then
                    BUILD_SUCCESS=true
                    METHOD="tar"
                    log "✅ Tar method succeeded!"
                else
                    log "❌ Invalid SIF file from tar method"
                    rm -f "$OUTPUT_PATH" 2>/dev/null || true
                fi
            else
                log "❌ Tar method produced no output"
            fi
        else
            log "❌ Singularity build from tar failed"
        fi
        
        # Cleanup tar file
        log "🧹 Cleaning up tar file..."
        rm -f "$TAR_FILE" 2>/dev/null || true
    else
        log "❌ Docker save to tar failed"
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
    
    log "============================================="
    log "✅ LAMAReg SINGULARITY BUILD COMPLETE"
    log "============================================="
    log "📦 File: $OUTPUT_PATH"
    log "📊 Size: $SIZE"
    log "⏱️  Time: ${DURATION_MIN}m ${DURATION_SEC}s"
    log "🎯 Method: $METHOD"
    log ""
    log "🧪 Test Commands:"
    log "   # Test LAMAReg CLI"
    log "   singularity exec $OUTPUT_PATH lamareg --help"
    log ""
    log "   # Test Python import"
    log "   singularity exec $OUTPUT_PATH python -c 'import lamareg; print(\"LAMAReg ready!\")'"
    log ""
    log "   # Full registration example"
    log "   singularity exec -B /path/to/data:/data $OUTPUT_PATH lamareg register \\"
    log "     --moving /data/moving.nii.gz --fixed /data/fixed.nii.gz \\"
    log "     --output /data/registered.nii.gz"
    log ""
    log "🚀 LAMAReg SIF ready for HPC deployment!"
    
    # Quick validation test
    log "🧪 Running quick validation test..."
    if singularity exec "$OUTPUT_PATH" python -c "import lamareg; print('✅ LAMAReg import successful')" 2>&1; then
        log "✅ Validation passed - container is functional"
    else
        log "⚠️  Validation warning - container may have issues"
    fi
    
else
    log "============================================="
    log "❌ SINGULARITY BUILD FAILED"
    log "============================================="
    log "⏱️  Time: ${DURATION_MIN}m ${DURATION_SEC}s"
    log "❌ ERROR: SIF file not created or empty"
    log ""
    log "🔍 Troubleshooting:"
    log "   1. Check Docker image: docker image ls | grep lamareg"
    log "   2. Check disk space: df -h $BASE_DIR"
    log "   3. Check Singularity: singularity --version"
    log "   4. Try building Docker image again: ./build_docker.sh"
    log ""
    exit 1
fi