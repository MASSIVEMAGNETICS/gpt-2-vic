#!/bin/bash
# Quick start script for GPT-2-VIC Chat Interface

echo "================================================"
echo "   GPT-2-VIC Standalone Learning Chat"
echo "   Critical Thinking | Liquid Weights | Sources"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import flask, flask_cors, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -q flask flask-cors numpy
fi

echo "✓ Dependencies ready"
echo ""

# Offer options
echo "Choose how to run:"
echo "  1) Web Interface (Recommended)"
echo "  2) Command Line Interface"
echo "  3) Demo Mode (HTML only, no backend)"
echo ""
read -p "Enter choice [1]: " choice
choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "Starting web server..."
        echo "Open your browser to: http://localhost:5000"
        echo ""
        python3 chat_server.py
        ;;
    2)
        echo ""
        echo "Starting CLI..."
        python3 src/chat_interface.py
        ;;
    3)
        echo ""
        echo "Opening demo mode in browser..."
        if command -v xdg-open &> /dev/null; then
            xdg-open chat_ui.html
        elif command -v open &> /dev/null; then
            open chat_ui.html
        else
            echo "Please open chat_ui.html in your browser"
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
