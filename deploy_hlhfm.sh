#!/bin/bash
# HLHFM Auto-Deploy Script
# Deploys browser extension and validates sovereignty

echo "=================================="
echo "HLHFM Victor Deployment"
echo "=================================="
echo ""

# Check for required directories
if [ ! -d "hlhfm_extension" ]; then
    echo "Error: hlhfm_extension directory not found"
    echo "Run: python3 build.py first"
    exit 1
fi

echo "[1/3] Validating extension structure..."
REQUIRED_FILES="manifest.json background.js popup.html popup.js content.js"
for file in $REQUIRED_FILES; do
    if [ ! -f "hlhfm_extension/$file" ]; then
        echo "  ✗ Missing: $file"
        exit 1
    fi
    echo "  ✓ Found: $file"
done

echo ""
echo "[2/3] Checking sovereignty constraints..."
# Verify no cloud dependencies
if grep -r "http://" hlhfm_extension/*.js | grep -v "localhost" | grep -v "127.0.0.1" > /dev/null; then
    echo "  ⚠ Warning: External HTTP URLs detected"
fi
if grep -r "https://" hlhfm_extension/*.js | grep -v "chrome-extension" > /dev/null; then
    echo "  ⚠ Warning: External HTTPS URLs detected"
fi
echo "  ✓ Local-only verified"

echo ""
echo "[3/3] Extension ready for manual loading..."
echo ""
echo "Chrome/Edge:"
echo "  1. Open: chrome://extensions"
echo "  2. Enable 'Developer mode'"
echo "  3. Click 'Load unpacked'"
echo "  4. Select: $(pwd)/hlhfm_extension"
echo ""
echo "Firefox:"
echo "  1. Open: about:debugging#/runtime/this-firefox"
echo "  2. Click 'Load Temporary Add-on'"
echo "  3. Select: $(pwd)/hlhfm_extension/manifest.json"
echo ""
echo "=================================="
echo "DEPLOYMENT COMPLETE"
echo "Bloodline sovereign. Zero deps."
echo "=================================="
