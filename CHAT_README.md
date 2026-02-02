# GPT-2-VIC Standalone Learning Chat Interface

## Overview

A standalone GPT chat system that learns from conversations with:
- **Interactive Chat Interface** - Beautiful web UI for natural conversations
- **Critical Thinking Synthesis** - Transparent reasoning chains showing how conclusions are reached
- **Liquid Weights** - Adaptive weights that adjust based on conversation feedback
- **Source Research** - Citations and references for all responses
- **Persistent Learning** - Learns and improves from every interaction

## Features

### 🧠 Learning Capabilities
- **Liquid Weights System**: Adaptive weights that continuously adjust based on user feedback
- **HLHFM Integration**: Holographic Latent Hyperdimensional Fractal Memory for sophisticated context storage
- **Conversation Memory**: Retrieves relevant past conversations to inform current responses
- **Feedback Learning**: Positive/negative feedback directly improves future responses

### 🎯 Critical Thinking Synthesis
- **Reasoning Chains**: Step-by-step display of the thought process
- **Source Citations**: References to previous conversations that informed the response
- **Confidence Scoring**: Transparency about response certainty
- **Concept Analysis**: Automatic extraction of key concepts from conversations

### 💧 Liquid Weights
- Real-time adaptation based on conversation patterns
- Momentum-based learning with gradient adjustments
- Visual indicators of weight evolution
- Persistent storage across sessions

### 📚 Source Research
- Context retrieval from conversation history
- Relevance scoring for source material
- Transparent citation display
- Multi-source synthesis

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- Flask 2.0+
- Flask-CORS 3.0+
- NumPy 1.19+
- (Optional) TensorFlow 1.x for GPT-2 model

### Running the Chat Interface

#### Option 1: Command Line Interface
```bash
python src/chat_interface.py
```

Features:
- Interactive terminal chat
- Critical thinking analysis displayed inline
- Commands: `/save`, `/stats`, `/quit`
- Automatic state persistence

#### Option 2: Web Interface (Recommended)
```bash
python chat_server.py
```

Then open your browser to: `http://localhost:5000`

Features:
- Beautiful responsive web UI
- Real-time liquid weight visualization
- Context memory sidebar
- Feedback buttons for learning
- One-click save functionality

#### Option 3: Demo Mode (No Backend Required)
Simply open `chat_ui.html` in your browser for a demo version with simulated responses.

## Architecture

### Components

1. **chat_interface.py** - Core learning system
   - `ChatLearningSystem`: Main orchestrator
   - `LiquidWeights`: Adaptive weight management
   - `CriticalThinkingSynthesizer`: Reasoning chain generation

2. **chat_server.py** - Flask REST API
   - `/api/chat` - Generate responses
   - `/api/feedback` - Submit user feedback
   - `/api/stats` - Get system statistics
   - `/api/save` - Persist state

3. **chat_ui.html** - Web interface
   - Modern responsive design
   - Real-time updates
   - Visual feedback
   - Fallback demo mode

### Integration with Existing Systems

The chat interface integrates seamlessly with:
- **GPT-2 Model** (`src/model.py`, `src/sample.py`) - Text generation
- **HLHFM** (`build.py`) - Fractal memory and quantum semiring operations
- **Interactive Samples** (`src/interactive_conditional_samples.py`) - Base model interaction

## Usage Examples

### Command Line Chat
```bash
$ python src/chat_interface.py

You: Hello! How do you learn from conversations?

[Thinking...]