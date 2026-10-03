#!/bin/bash
#
# LAMAReg Singularity Build - Convert Docker to SIF
# Optimized for local Docker images with server environment
#

set -e

DOCKER_IMAGE="lamareg"
DOCKER_TAG="${1:-latest}"
FULL_DOCKER_IMAGE="${DOCKER_IMAGE}:${DOCKER_TAG}"

BASE_DIR="/host/cassio/export03/data/enning"
OUTPUT_DIR="${BASE_DIR}/singularity"
OUTPUT_PATH="${OUTPUT_DIR}/lamareg_${DOCKER_TAG}.sif"

# Performance settings for server
export SINGULARITY_CACHEDIR="${BASE_DIR}/.singularity_cache"
export SINGULARITY_TMPDIR="${BASE_DIR}/.singularity_tmp"
export SINGULARITY_MEMORY="32G"  # Adjust based on server RAM
export OMP_NUM_THREADS=$(nproc)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Function to check and kill stuck processes
check_stuck_processes() {
    log "🔍 Checking for stuck processes..."
    
    # Check for singularity processes
    SING_PROCS=$(ps aux | grep singularity | grep -v grep | wc -l)
    if [ "$SING_PROCS" -gt 0 ]; then
        log "🔍 Found $SING_PROCS singularity processes:"
        ps aux | grep singularity | grep -v grep
    fi
    
    # Check for docker save processes  
    DOCKER_PROCS=$(ps aux | grep "docker save" | grep -v grep | wc -l)
    if [ "$DOCKER_PROCS" -gt 0 ]; then
        log "🔍 Found $DOCKER_PROCS docker save processes:"
        ps aux | grep "docker save" | grep -v grep
    fi
    
    # Check temp directory usage
    if [ -d "$SINGULARITY_TMPDIR" ]; then
        TEMP_USAGE=$(du -sh "$SINGULARITY_TMPDIR" 2>/dev/null | cut -f1 || echo "0")
        log "💾 Temp directory: $TEMP_USAGE"
    fi
    
    # Check if output file exists and its size
    if [ -f "$OUTPUT_PATH" ]; then
        OUTPUT_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
        log "📁 Output file exists: $OUTPUT_SIZE"
    else
        log "❌ No output file yet"
    fi
}

# Function to kill all related processes
kill_stuck_build() {
    log "🛑 Killing stuck build processes..."
    
    # Kill singularity processes
    pkill -f singularity 2>/dev/null || true
    
    # Kill docker save processes
    pkill -f "docker save" 2>/dev/null || true
    
    # Clean up temp files
    if [ -d "$SINGULARITY_TMPDIR" ]; then
        rm -rf "$SINGULARITY_TMPDIR"/* 2>/dev/null || true
    fi
    
    # Remove partial output
    if [ -f "$OUTPUT_PATH" ]; then
        rm -f "$OUTPUT_PATH"
    fi
    
    log "✅ Cleanup complete"
}

# ============================================================================
# Pre-flight checks
# ============================================================================
log "🚀 LAMAReg SINGULARITY BUILD"
log "📦 Docker Image: $FULL_DOCKER_IMAGE"
log "📍 Output: $OUTPUT_PATH"

# Check local Docker image exists
if ! docker image inspect "$FULL_DOCKER_IMAGE" &>/dev/null; then
    log "❌ Local Docker image not found: $FULL_DOCKER_IMAGE"
    log "   Available images:"
    docker images | grep lamareg || echo "   No lamareg images found"
    log ""
    log "💡 Build Docker image first:"
    log "   cd /host/cassio/export03/data/enning/lamareg_build"
    log "   ./build_docker.sh"
    exit 1
fi

LOCAL_SIZE=$(docker image inspect "$FULL_DOCKER_IMAGE" --format='{{.Size}}' | awk '{printf "%.1f GB", $1/1024/1024/1024}')
log "✅ Found local Docker image: $LOCAL_SIZE"

# Check Singularity is available
if ! command -v singularity >/dev/null 2>&1; then
    log "❌ Singularity not found in PATH"
    log "   Please install Singularity or check PATH"
    exit 1
fi

SING_VERSION=$(singularity --version 2>/dev/null || echo "unknown")
log "✅ Singularity version: $SING_VERSION"

# Create directories
mkdir -p "$OUTPUT_DIR" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

# Check space
AVAILABLE=$(df -BG "$BASE_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE" -lt 50 ]; then
    log "❌ Need 50GB+ space, only ${AVAILABLE}GB available"
    exit 1
fi
log "✅ Space check: ${AVAILABLE}GB available"

# Remove existing output
if [ -f "$OUTPUT_PATH" ]; then
    log "⚠️  Removing existing SIF file"
    rm -f "$OUTPUT_PATH"
fi

START_TIME=$(date +%s)

# ============================================================================
# Method 1: Streaming (fastest - no intermediate files) 
# ============================================================================
log "⚡ Trying streaming method (fastest)..."

# Check for filesystem issues that might cause problems
if mount | grep "$(dirname "$OUTPUT_PATH")" | grep -q nodev; then
    log "⚠️  WARNING: 'nodev' mount detected - this may cause build issues"
    log "🔧 Attempting build anyway with fallback strategy..."
fi

# Run the actual build with better error handling
log "🔄 Starting docker save | singularity build..."
BUILD_SUCCESS=false

if timeout 3600 bash -c "
    set -o pipefail
    docker save '$FULL_DOCKER_IMAGE' | singularity build --force --fakeroot '$OUTPUT_PATH' docker-archive:///dev/stdin
" 2>&1 | while read line; do
    case \"\$line\" in
        *\"FATAL\"*) 
            log \"❌ SINGULARITY FATAL: \$line\"
            ;;
        *\"ERROR\"*) 
            log \"❌ SINGULARITY ERROR: \$line\"
            ;;
        *\"WARNING\"*) 
            log \"⚠️  SINGULARITY WARNING: \$line\"
            ;;
        *\"INFO\"*) 
            log \"ℹ️  SINGULARITY INFO: \$line\"
            ;;
        *) 
            log \"SINGULARITY: \$line\"
            ;;
    esac
done; then
    # Check if output file was actually created and is valid
    if [[ -f "$OUTPUT_PATH" ]] && [[ -s "$OUTPUT_PATH" ]]; then
        # Verify it's a valid SIF file
        if singularity inspect "$OUTPUT_PATH" >/dev/null 2>&1; then
            BUILD_SUCCESS=true
            log "✅ Streaming method succeeded!"
            METHOD="streaming"
        else
            log "❌ Output file created but not a valid SIF"
            rm -f "$OUTPUT_PATH" 2>/dev/null || true
        fi
    else
        log "❌ Streaming method failed - no valid output file"
        rm -f "$OUTPUT_PATH" 2>/dev/null || true
    fi
else
    log "❌ Streaming method command failed"
    rm -f "$OUTPUT_PATH" 2>/dev/null || true
fi

if [[ "$BUILD_SUCCESS" != "true" ]]; then
    log "⚠️  Streaming failed, trying tar method..."
    
    # ============================================================================
    # Method 2: Tar method (more reliable)
    # ============================================================================
    TAR_FILE="${BASE_DIR}/lamareg_docker_$$.tar"
    
    log "📤 Exporting Docker to tar..."
    
    # Monitor tar creation
    (while [ ! -f "$TAR_FILE" ] || [ $(stat -f%z "$TAR_FILE" 2>/dev/null || stat -c%s "$TAR_FILE" 2>/dev/null || echo 0) -eq 0 ]; do
        sleep 5
        log "⏳ Waiting for tar export to start..."
    done
    
    while [ -f "$TAR_FILE" ] && kill -0 $! 2>/dev/null; do
        SIZE=$(du -h "$TAR_FILE" 2>/dev/null | cut -f1 || echo "0")
        log "📈 Tar progress: $SIZE"
        sleep 30
    done) &
    
    TAR_MONITOR_PID=$!
    
    # Create tar file
    docker save "$FULL_DOCKER_IMAGE" -o "$TAR_FILE" &
    DOCKER_PID=$!
    
    # Wait for docker save to complete
    wait $DOCKER_PID
    
    # Stop tar monitor
    kill $TAR_MONITOR_PID 2>/dev/null || true
    wait $TAR_MONITOR_PID 2>/dev/null || true
    
    TAR_SIZE=$(du -h "$TAR_FILE" | cut -f1)
    log "✅ Export complete: $TAR_SIZE"
    
    log "🔧 Building SIF from tar..."
    
    # Monitor SIF creation
    (while [ ! -f "$OUTPUT_PATH" ] || [ $(stat -f%z "$OUTPUT_PATH" 2>/dev/null || stat -c%s "$OUTPUT_PATH" 2>/dev/null || echo 0) -eq 0 ]; do
        sleep 5
        log "⏳ Waiting for SIF build to start..."
    done
    
    while [ -f "$OUTPUT_PATH" ] && kill -0 $! 2>/dev/null; do
        SIZE=$(du -h "$OUTPUT_PATH" 2>/dev/null | cut -f1 || echo "0")
        log "📈 SIF progress: $SIZE"
        sleep 30
    done) &
    
    SIF_MONITOR_PID=$!
    
    # Build SIF
    singularity build --force \
        "$OUTPUT_PATH" \
        "docker-archive://$TAR_FILE" &
    SINGULARITY_PID=$!
    
    # Wait for singularity build to complete
    wait $SINGULARITY_PID
    
    # Stop SIF monitor
    kill $SIF_MONITOR_PID 2>/dev/null || true
    wait $SIF_MONITOR_PID 2>/dev/null || true
    
    log "🧹 Cleaning up tar file..."
    rm -f "$TAR_FILE"
    
    USED_METHOD="tar"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))
SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)

log "============================================="
log "✅ LAMAReg SINGULARITY BUILD COMPLETE"
log "============================================="
log "📦 File: $OUTPUT_PATH"
log "📊 Size: $SIZE"
log "⏱️  Time: ${DURATION_MIN}m ${DURATION_SEC}s"
log "🎯 Method: $USED_METHOD"
log ""
log "🧪 Test Commands:"
log "   # Test help command"
log "   singularity run $OUTPUT_PATH --help"
log ""
log "   # Test Python import"
log "   singularity exec $OUTPUT_PATH python -c 'import lamareg; print(\"LAMAReg ready!\")'"
log ""
log "   # Process data example"
log "   singularity run -B /path/to/data:/data $OUTPUT_PATH python -m lamareg.cli --input /data/input.nii.gz --output /data/output.nii.gz"
log ""
log "🚀 LAMAReg SIF ready for deployment!"

# Quick verification
if [ -f "$OUTPUT_PATH" ] && [ -s "$OUTPUT_PATH" ]; then
    log "✅ SIF file created successfully"
    
    # Test the SIF file
    log "🧪 Quick test..."
    if timeout 60 singularity exec "$OUTPUT_PATH" python -c "import lamareg; print('LAMAReg import successful')" 2>/dev/null; then
        log "✅ SIF test passed - LAMAReg import works"
    else
        log "⚠️  SIF test failed - but file created (might be container issue)"
    fi
else
    log "❌ ERROR: SIF file not created or empty"
    exit 1
fi