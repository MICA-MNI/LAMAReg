#!/bin/bash
set -eu

# ============================================================================
# LAMAReg Singularity Build Script - Alternative for nodev filesystems
# ============================================================================
# This version tries to work around 'nodev' mount restrictions by using
# alternative temporary locations and build strategies

BASE_DIR="/host/cassio/export03/data/enning"
OUTPUT_DIR="${BASE_DIR}/singularity"
OUTPUT_PATH="${OUTPUT_DIR}/lamareg_latest.sif"
DOCKER_IMAGE="${1:-}"

# Try different temp locations to avoid nodev issues
POSSIBLE_TEMP_DIRS=(
    "/tmp"
    "$HOME/tmp"
    "/var/tmp"
    "${BASE_DIR}/tmp"
)

echo_log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Find a suitable temp directory
find_temp_dir() {
    for temp_dir in "${POSSIBLE_TEMP_DIRS[@]}"; do
        if mkdir -p "$temp_dir" 2>/dev/null && [ -w "$temp_dir" ]; then
            # Test if we can create and delete files
            if touch "$temp_dir/test_$$" 2>/dev/null && rm "$temp_dir/test_$$" 2>/dev/null; then
                echo "$temp_dir"
                return 0
            fi
        fi
    done
    echo ""
    return 1
}

echo_log "🚀 LAMAReg Singularity Build (nodev workaround)"
echo_log "==============================================="

# Auto-detect Docker image
if [[ -z "$DOCKER_IMAGE" ]]; then
    if docker image inspect "localhost:5001/lamareg:latest" >/dev/null 2>&1; then
        DOCKER_IMAGE="localhost:5001/lamareg:latest"
    elif docker image inspect "lamareg:latest" >/dev/null 2>&1; then
        DOCKER_IMAGE="lamareg:latest"
    else
        echo_log "❌ No LAMAReg Docker image found"
        exit 1
    fi
fi

echo_log "🐳 Docker image: $DOCKER_IMAGE"

# Find suitable temp directory
TEMP_DIR=$(find_temp_dir)
if [[ -z "$TEMP_DIR" ]]; then
    echo_log "❌ No suitable temp directory found"
    exit 1
fi

echo_log "📁 Using temp directory: $TEMP_DIR"
export SINGULARITY_TMPDIR="$TEMP_DIR/singularity_$$"
export TMPDIR="$TEMP_DIR"

mkdir -p "$SINGULARITY_TMPDIR"
mkdir -p "$OUTPUT_DIR"

# Cleanup function
cleanup() {
    echo_log "🧹 Cleaning up..."
    rm -rf "$SINGULARITY_TMPDIR" 2>/dev/null || true
    rm -f "$TEMP_DIR/lamareg_docker_$$"* 2>/dev/null || true
}
trap cleanup EXIT

START_TIME=$(date +%s)

echo_log "🔧 Method: Direct tar build with custom temp location"

# Export to tar in temp directory (not on nodev mount)
TAR_FILE="$TEMP_DIR/lamareg_docker_$$.tar"
echo_log "📤 Exporting Docker to: $TAR_FILE"

if docker save "$DOCKER_IMAGE" -o "$TAR_FILE"; then
    TAR_SIZE=$(du -h "$TAR_FILE" | cut -f1)
    echo_log "✅ Export complete: $TAR_SIZE"
    
    echo_log "🔧 Building SIF with external temp directory..."
    
    # Build with explicit temp dir and longer timeout
    if timeout 7200 env SINGULARITY_TMPDIR="$SINGULARITY_TMPDIR" \
       singularity build --force "$OUTPUT_PATH" "docker-archive://$TAR_FILE" 2>&1; then
        
        if [[ -f "$OUTPUT_PATH" ]] && [[ -s "$OUTPUT_PATH" ]]; then
            if singularity inspect "$OUTPUT_PATH" >/dev/null 2>&1; then
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))
                SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
                
                echo_log "✅ BUILD SUCCESSFUL!"
                echo_log "📦 SIF: $OUTPUT_PATH ($SIZE)"
                echo_log "⏱️  Time: $((DURATION/60))m $((DURATION%60))s"
                echo_log ""
                echo_log "🧪 Quick test:"
                echo_log "   singularity exec $OUTPUT_PATH python -c 'import lamareg; print(\"Ready!\")'"
                
                exit 0
            else
                echo_log "❌ Invalid SIF file"
            fi
        else
            echo_log "❌ No output file created"
        fi
    else
        echo_log "❌ Singularity build failed"
    fi
else
    echo_log "❌ Docker export failed"
fi

echo_log "❌ BUILD FAILED"
echo_log "💡 Try using Docker directly instead of Singularity:"
echo_log "   docker run --rm -v /path/to/data:/data $DOCKER_IMAGE lamareg --help"
exit 1