#!/bin/bash

# ============================================================================
# LAMAReg Singularity Build Diagnostics
# ============================================================================
# Helps diagnose and fix common Singularity build issues

BASE_DIR="/host/cassio/export03/data/enning"
DOCKER_IMAGE="localhost:5001/lamareg:latest"

echo "🔍 LAMAReg Singularity Build Diagnostics"
echo "========================================"
echo ""

# Check 1: Docker image
echo "1️⃣  Checking Docker image..."

# Check for registry version first
if docker image inspect "localhost:5001/lamareg:latest" >/dev/null 2>&1; then
    SIZE=$(docker image inspect "localhost:5001/lamareg:latest" --format='{{.Size}}' | awk '{print $1/1024/1024/1024 " GB"}')
    echo "   ✅ Docker image found: localhost:5001/lamareg:latest ($SIZE)"
# Check for local version
elif docker image inspect "lamareg:latest" >/dev/null 2>&1; then
    SIZE=$(docker image inspect "lamareg:latest" --format='{{.Size}}' | awk '{print $1/1024/1024/1024 " GB"}')
    echo "   ✅ Docker image found: lamareg:latest ($SIZE)"
else
    echo "   ❌ Docker image missing. Checked:"
    echo "      - localhost:5001/lamareg:latest"
    echo "      - lamareg:latest"
    echo "   💡 Build it with: ./build_docker.sh"
fi

echo ""

# Check 2: Singularity installation
echo "2️⃣  Checking Singularity..."
if command -v singularity >/dev/null 2>&1; then
    VERSION=$(singularity --version)
    echo "   ✅ Singularity installed: $VERSION"
else
    echo "   ❌ Singularity not found"
    echo "   💡 Install with: sudo apt install singularity-container"
fi

echo ""

# Check 3: Filesystem and mount options
echo "3️⃣  Checking filesystem..."
if [ -d "$BASE_DIR" ]; then
    echo "   ✅ Base directory exists: $BASE_DIR"
    
    # Check mount options
    MOUNT_INFO=$(mount | grep "$(dirname "$BASE_DIR")" | head -1)
    if [ -n "$MOUNT_INFO" ]; then
        echo "   📁 Mount info: $MOUNT_INFO"
        
        if echo "$MOUNT_INFO" | grep -q nodev; then
            echo "   ⚠️  WARNING: 'nodev' option detected - may cause build issues"
            echo "   💡 Solution: Script will automatically use tar method"
        else
            echo "   ✅ Mount options look good"
        fi
    else
        echo "   ℹ️  Could not determine mount options"
    fi
else
    echo "   ❌ Base directory missing: $BASE_DIR"
    echo "   💡 Create it with: mkdir -p $BASE_DIR"
fi

echo ""

# Check 4: Disk space
echo "4️⃣  Checking disk space..."
if [ -d "$BASE_DIR" ]; then
    AVAILABLE=$(df "$BASE_DIR" | awk 'NR==2 {print int($4/1024/1024)}')
    USED=$(df "$BASE_DIR" | awk 'NR==2 {print int($3/1024/1024)}')
    TOTAL=$(df "$BASE_DIR" | awk 'NR==2 {print int($2/1024/1024)}')
    
    echo "   📊 Space: ${USED}GB used / ${TOTAL}GB total (${AVAILABLE}GB available)"
    
    if [ "$AVAILABLE" -lt 10 ]; then
        echo "   ⚠️  Low disk space - need at least 10GB for build"
    else
        echo "   ✅ Sufficient disk space"
    fi
else
    echo "   ❌ Cannot check - directory missing"
fi

echo ""

# Check 5: Permissions
echo "5️⃣  Checking permissions..."
if [ -w "$BASE_DIR" ]; then
    echo "   ✅ Write permission to base directory"
else
    echo "   ❌ No write permission to $BASE_DIR"
    echo "   💡 Fix with: chmod 755 $BASE_DIR"
fi

echo ""

# Check 6: Previous build artifacts
echo "6️⃣  Checking for previous builds..."
SIF_DIR="${BASE_DIR}/singularity"
if [ -d "$SIF_DIR" ]; then
    echo "   📁 Singularity directory exists: $SIF_DIR"
    
    SIF_FILES=$(ls -la "$SIF_DIR"/*.sif 2>/dev/null | wc -l)
    if [ "$SIF_FILES" -gt 0 ]; then
        echo "   📦 Found SIF files:"
        ls -lah "$SIF_DIR"/*.sif 2>/dev/null | while read line; do
            echo "      $line"
        done
    else
        echo "   ℹ️  No SIF files found"
    fi
else
    echo "   ℹ️  Singularity directory will be created"
fi

echo ""

# Check 7: Docker daemon
echo "7️⃣  Checking Docker daemon..."
if docker info >/dev/null 2>&1; then
    echo "   ✅ Docker daemon accessible"
else
    echo "   ❌ Docker daemon not accessible"
    echo "   💡 Start with: systemctl start docker"
fi

echo ""

# Summary and recommendations
echo "📋 Summary and Recommendations"
echo "=============================="

# Count issues
ISSUES=0

# Check for either Docker image
if ! docker image inspect "localhost:5001/lamareg:latest" >/dev/null 2>&1 && ! docker image inspect "lamareg:latest" >/dev/null 2>&1; then
    ISSUES=$((ISSUES + 1))
fi

if ! command -v singularity >/dev/null 2>&1; then
    ISSUES=$((ISSUES + 1))
fi

if [ -d "$BASE_DIR" ]; then
    AVAILABLE=$(df "$BASE_DIR" | awk 'NR==2 {print int($4/1024/1024)}')
    if [ "$AVAILABLE" -lt 10 ]; then
        ISSUES=$((ISSUES + 1))
    fi
else
    ISSUES=$((ISSUES + 1))
fi

if [ "$ISSUES" -eq 0 ]; then
    echo "✅ System looks ready for Singularity build!"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Run: ./build_singularity.sh"
    echo "   2. Test: ./test_singularity.sh"
else
    echo "⚠️  Found $ISSUES potential issue(s) that should be fixed first."
    echo ""
    echo "🛠️  After fixing issues, try:"
    echo "   1. Run diagnostics again: ./diagnose_singularity.sh"
    echo "   2. Build SIF: ./build_singularity.sh"
fi

echo ""
echo "💡 If builds still fail, check the improved build script that:"
echo "   - Automatically detects filesystem issues"
echo "   - Falls back to tar method for problematic mounts"
echo "   - Validates SIF files after creation"
echo "   - Provides detailed error messages"