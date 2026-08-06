#!/bin/bash
set -eu

# ============================================================================
# LAMAReg Docker-Only Deployment (Singularity Alternative)
# ============================================================================
# For environments where Singularity builds fail due to filesystem restrictions

BASE_DIR="/host/cassio/export03/data/enning"
DOCKER_IMAGE="${1:-}"

echo_log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

echo_log "🐳 LAMAReg Docker-Only Deployment"
echo_log "================================="

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

# Test Docker functionality
echo_log "🧪 Testing Docker container..."
if docker run --rm "$DOCKER_IMAGE" python -c "import lamareg; print('LAMAReg ready!')" 2>&1; then
    echo_log "✅ Docker container is functional"
else
    echo_log "❌ Docker container test failed"
    exit 1
fi

# Create wrapper scripts for easy usage
SCRIPTS_DIR="${BASE_DIR}/docker_scripts"
mkdir -p "$SCRIPTS_DIR"

echo_log "📝 Creating Docker wrapper scripts..."

# Main LAMAReg script
cat > "$SCRIPTS_DIR/lamareg" << EOF
#!/bin/bash
# LAMAReg Docker Wrapper Script
# Usage: ./lamareg [options]

docker run --rm -v "\$(pwd):/data" -w /data "$DOCKER_IMAGE" lamareg "\$@"
EOF

# Interactive shell script
cat > "$SCRIPTS_DIR/lamareg-shell" << EOF
#!/bin/bash
# LAMAReg Interactive Shell
# Usage: ./lamareg-shell

docker run --rm -it -v "\$(pwd):/data" -w /data "$DOCKER_IMAGE" bash
EOF

# Python script
cat > "$SCRIPTS_DIR/lamareg-python" << EOF
#!/bin/bash
# LAMAReg Python Wrapper
# Usage: ./lamareg-python script.py

docker run --rm -v "\$(pwd):/data" -w /data "$DOCKER_IMAGE" python "\$@"
EOF

# Make scripts executable
chmod +x "$SCRIPTS_DIR"/*

echo_log "✅ Docker wrapper scripts created in: $SCRIPTS_DIR"
echo_log ""
echo_log "📋 Usage Examples:"
echo_log ""
echo_log "   # Basic LAMAReg command"
echo_log "   cd /path/to/your/data"
echo_log "   $SCRIPTS_DIR/lamareg --help"
echo_log ""
echo_log "   # Full registration"
echo_log "   $SCRIPTS_DIR/lamareg register \\"
echo_log "     --moving moving.nii.gz --fixed fixed.nii.gz \\"
echo_log "     --output registered.nii.gz"
echo_log ""
echo_log "   # Brain parcellation"
echo_log "   $SCRIPTS_DIR/lamareg synthseg \\"
echo_log "     --i input.nii.gz --o parcellation.nii.gz --parc"
echo_log ""
echo_log "   # Interactive shell"
echo_log "   $SCRIPTS_DIR/lamareg-shell"
echo_log ""
echo_log "   # Run Python scripts"
echo_log "   $SCRIPTS_DIR/lamareg-python your_script.py"
echo_log ""
echo_log "🎯 Docker deployment complete!"
echo_log "💡 Add $SCRIPTS_DIR to your PATH for global access"
echo_log "💡 Alternative: Use Docker directly:"
echo_log "   docker run --rm -v /path/to/data:/data $DOCKER_IMAGE lamareg --help"