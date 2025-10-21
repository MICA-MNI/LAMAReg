#!/bin/bash
set -eu

echo "🧪 Testing LAMAReg Docker Image"
echo "==============================="
echo ""

# Check if Docker image exists
if ! docker images lamareg:latest | grep -q lamareg; then
    echo "❌ Docker image 'lamareg:latest' not found"
    echo "   Build it first with: ./build_docker.sh"
    exit 1
fi

echo "✅ Docker image found: lamareg:latest"
echo ""

# Test 1: Basic help command
echo "🔍 Test 1: Basic help command"
echo "Command: docker run --rm lamareg:latest lamareg --help"
echo ""
if docker run --rm lamareg:latest lamareg --help; then
    echo "✅ Test 1 passed: Help command works"
else
    echo "❌ Test 1 failed: Help command failed"
    exit 1
fi

echo ""
echo "🔍 Test 2: Python import test"
echo "Command: docker run --rm lamareg:latest python -c 'import lamareg; print(\"LAMAReg imported successfully\")'"
echo ""
if docker run --rm lamareg:latest python -c "import lamareg; print('LAMAReg imported successfully')"; then
    echo "✅ Test 2 passed: Python import works"
else
    echo "❌ Test 2 failed: Python import failed"
    exit 1
fi

echo ""
echo "🔍 Test 3: Check dependencies"
echo "Command: docker run --rm lamareg:latest python -c 'import tensorflow, nibabel, antspyx; print(\"Key dependencies available\")'"
echo ""
if docker run --rm lamareg:latest python -c "import tensorflow, nibabel, antspyx; print('Key dependencies available')"; then
    echo "✅ Test 3 passed: Key dependencies available"
else
    echo "❌ Test 3 failed: Missing key dependencies"
    exit 1
fi

echo ""
echo "🎉 All tests passed!"
echo "LAMAReg Docker image is ready to use"
echo ""
echo "📋 Usage examples:"
echo "   # Basic help"
echo "   docker run --rm lamareg:latest lamareg --help"
echo ""
echo "   # Mount data directory and run LAMAReg"
echo "   docker run --rm -v /path/to/your/data:/data lamareg:latest lamareg [options]"
echo ""
echo "   # Interactive mode"
echo "   docker run --rm -it -v /path/to/your/data:/data lamareg:latest bash"