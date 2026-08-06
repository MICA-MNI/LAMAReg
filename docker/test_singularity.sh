#!/bin/bash
set -eu

echo "🧪 Testing LAMAReg Singularity Container"
echo "========================================"
echo ""

# Find the SIF file
BASE_DIR="/host/cassio/export03/data/enning"
SIF_DIR="${BASE_DIR}/singularity"
SIF_FILE="${1:-${SIF_DIR}/lamareg_latest.sif}"

if [[ ! -f "$SIF_FILE" ]]; then
    echo "❌ SIF file not found: $SIF_FILE"
    echo ""
    echo "Available SIF files:"
    ls -la "$SIF_DIR"/*.sif 2>/dev/null || echo "No SIF files found in $SIF_DIR"
    echo ""
    echo "💡 Build SIF first:"
    echo "   ./build_singularity.sh"
    exit 1
fi

SIF_SIZE=$(du -h "$SIF_FILE" | cut -f1)
echo "✅ SIF file found: $SIF_FILE ($SIF_SIZE)"
echo ""

# Test 1: Basic Python import
echo "🔍 Test 1: Python import test"
echo "Command: singularity exec $SIF_FILE python -c 'import lamareg; print(\"LAMAReg imported successfully\")'"
echo ""
if singularity exec "$SIF_FILE" python -c "import lamareg; print('LAMAReg imported successfully')"; then
    echo "✅ Test 1 passed: Python import works"
else
    echo "❌ Test 1 failed: Python import failed"
    exit 1
fi

echo ""

# Test 2: LAMAReg CLI help command  
echo "🔍 Test 2: LAMAReg CLI help command"
echo "Command: singularity exec $SIF_FILE lamareg --help"
echo ""
if singularity exec "$SIF_FILE" lamareg --help >/dev/null 2>&1; then
    echo "✅ Test 2 passed: lamareg --help works"
elif singularity exec "$SIF_FILE" python -m lamareg.cli --help >/dev/null 2>&1; then
    echo "✅ Test 2 passed: python -m lamareg.cli --help works"
    echo "ℹ️  Note: Use 'python -m lamareg.cli' instead of 'lamareg'"
else
    echo "❌ Test 2 failed: Both CLI approaches failed"
    echo "🔍 Debug info:"
    singularity exec "$SIF_FILE" which python
    singularity exec "$SIF_FILE" python --version
    exit 1
fi

echo ""

# Test 3: LAMAReg subcommands
echo "🔍 Test 3: LAMAReg subcommands test"
echo "Command: singularity exec $SIF_FILE lamareg synthseg --help"
echo ""
if singularity exec "$SIF_FILE" lamareg synthseg --help >/dev/null 2>&1; then
    echo "✅ Test 3 passed: lamareg synthseg command works"
elif singularity exec "$SIF_FILE" python -m lamareg.cli synthseg --help >/dev/null 2>&1; then
    echo "✅ Test 3 passed: python -m lamareg.cli synthseg works"
    echo "ℹ️  Note: Use 'python -m lamareg.cli' for subcommands"
else
    echo "❌ Test 3 failed: LAMAReg subcommands not working"
    exit 1
fi

echo ""

# Test 4: Check key dependencies
echo "🔍 Test 4: Key dependencies check"
echo "Command: singularity exec $SIF_FILE python -c 'import tensorflow, nibabel, antspyx'"
echo ""
if singularity exec "$SIF_FILE" python -c "import tensorflow, nibabel, antspyx; print('Key dependencies available')"; then
    echo "✅ Test 4 passed: Key dependencies available"
else
    echo "❌ Test 4 failed: Missing key dependencies"
    echo "🔍 Checking what's available:"
    singularity exec "$SIF_FILE" python -c "
try:
    import tensorflow as tf
    print(f'✅ TensorFlow {tf.__version__}')
except ImportError:
    print('❌ TensorFlow not available')
    
try:
    import nibabel as nib
    print(f'✅ nibabel {nib.__version__}')
except ImportError:
    print('❌ nibabel not available')
    
try:
    import antspyx as ants
    print(f'✅ ANTsPy available')
except ImportError:
    print('❌ ANTsPy not available')
"
    exit 1
fi

echo ""

# Test 5: Example data access (if available)
echo "🔍 Test 5: File system and example data access"
echo "Command: singularity exec $SIF_FILE ls /app"
echo ""
if singularity exec "$SIF_FILE" ls /app >/dev/null 2>&1; then
    echo "✅ Test 5 passed: File system accessible"
    echo "📁 LAMAReg files in container:"
    singularity exec "$SIF_FILE" ls -la /app | head -10
    
    # Check if example data exists
    if singularity exec "$SIF_FILE" test -d /app/example_data 2>/dev/null; then
        echo "📊 Example data found:"
        singularity exec "$SIF_FILE" ls -la /app/example_data 2>/dev/null || true
    else
        echo "ℹ️  No example data in container (expected for optimized build)"
    fi
else
    echo "❌ Test 5 failed: File system access issue"
    exit 1
fi

echo ""
echo "🎉 All tests passed!"
echo "LAMAReg Singularity container is ready to use"
echo ""
echo "📋 Usage Examples:"
echo ""
echo "   # Basic help"
echo "   singularity exec $SIF_FILE lamareg --help"
echo "   singularity exec $SIF_FILE python -m lamareg.cli --help"
echo ""
echo "   # Full registration pipeline"
echo "   singularity exec -B /path/to/data:/data $SIF_FILE lamareg register \\"
echo "     --moving /data/moving.nii.gz --fixed /data/fixed.nii.gz \\"
echo "     --output /data/registered.nii.gz \\"
echo "     --moving-parc /data/moving_parc.nii.gz \\"
echo "     --fixed-parc /data/fixed_parc.nii.gz \\"
echo "     --registered-parc /data/reg_parc.nii.gz \\"
echo "     --affine /data/affine.mat \\"
echo "     --warpfield /data/warp.nii.gz"
echo ""
echo "   # Generate brain parcellation only"
echo "   singularity exec -B /path/to/data:/data $SIF_FILE lamareg synthseg \\"
echo "     --i /data/input.nii.gz --o /data/parcellation.nii.gz --parc"
echo ""
echo "   # Apply existing warpfield"
echo "   singularity exec -B /path/to/data:/data $SIF_FILE lamareg apply-warpfield \\"
echo "     --moving /data/moving.nii.gz --fixed /data/fixed.nii.gz \\"
echo "     --output /data/warped.nii.gz \\"
echo "     --warpfield /data/warp.nii.gz --affine /data/affine.mat"
echo ""
echo "   # Interactive shell"
echo "   singularity shell -B /path/to/data:/data $SIF_FILE"
echo ""
echo "✨ Singularity container ready for HPC deployment!"
echo "📍 SIF location: $SIF_FILE"