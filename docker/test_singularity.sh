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

# Test 2: CLI help command  
echo "🔍 Test 2: CLI help command"
echo "Command: singularity run $SIF_FILE --help (with fallback to python -m)"
echo ""
if singularity run "$SIF_FILE" --help >/dev/null 2>&1; then
    echo "✅ Test 2 passed: singularity run --help works"
elif singularity exec "$SIF_FILE" python -m lamareg.cli --help >/dev/null 2>&1; then
    echo "✅ Test 2 passed: python -m lamareg.cli --help works"
    echo "ℹ️  Note: Use 'singularity exec [sif] python -m lamareg.cli' for commands"
else
    echo "❌ Test 2 failed: Both CLI approaches failed"
    echo "🔍 Debug info:"
    singularity exec "$SIF_FILE" which python
    singularity exec "$SIF_FILE" python --version
    exit 1
fi

echo ""

# Test 3: Check key dependencies
echo "🔍 Test 3: Key dependencies check"
echo "Command: singularity exec $SIF_FILE python -c 'import tensorflow, nibabel, antspyx'"
echo ""
if singularity exec "$SIF_FILE" python -c "import tensorflow, nibabel, antspyx; print('Key dependencies available')"; then
    echo "✅ Test 3 passed: Key dependencies available"
else
    echo "❌ Test 3 failed: Missing key dependencies"
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

# Test 4: File system access
echo "🔍 Test 4: File system access test"
echo "Command: singularity exec $SIF_FILE ls /app"
echo ""
if singularity exec "$SIF_FILE" ls /app >/dev/null 2>&1; then
    echo "✅ Test 4 passed: File system accessible"
    echo "📁 LAMAReg files in container:"
    singularity exec "$SIF_FILE" ls -la /app | head -10
else
    echo "❌ Test 4 failed: File system access issue"
    exit 1
fi

echo ""
echo "🎉 All tests passed!"
echo "LAMAReg Singularity container is ready to use"
echo ""
echo "📋 Usage Examples:"
echo ""
echo "   # Basic help (try both approaches)"
echo "   singularity run $SIF_FILE --help"
echo "   singularity exec $SIF_FILE python -m lamareg.cli --help"
echo ""
echo "   # Process data (mount directories with -B)"
echo "   singularity exec -B /path/to/data:/data $SIF_FILE python -m lamareg.cli --input /data/input.nii.gz --output /data/output.nii.gz"
echo ""
echo "   # Interactive shell"
echo "   singularity shell -B /path/to/data:/data $SIF_FILE"
echo ""
echo "   # Check LAMAReg version"
echo "   singularity exec $SIF_FILE python -c 'import lamareg; print(\"LAMAReg version:\", getattr(lamareg, \"__version__\", \"unknown\"))'"
echo ""
echo "✨ Singularity container ready for HPC deployment!"