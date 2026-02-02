#!/usr/bin/env python3
"""
Flask server for GPT-2-VIC Chat Interface
Provides REST API for the chat UI
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.chat_interface import ChatLearningSystem

app = Flask(__name__, static_folder='.')
CORS(app)

# Initialize chat system
chat_system = None

def get_chat_system():
    global chat_system
    if chat_system is None:
        print("Initializing chat system...")
        chat_system = ChatLearningSystem()
        chat_system.load_state()
    return chat_system


@app.route('/')
def index():
    """Serve the chat UI"""
    return send_from_directory('.', 'chat_ui.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat message"""
    try:
        data = request.json
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        system = get_chat_system()
        response, analysis = system.generate_response(user_input)
        
        # Get context
        context = system._retrieve_context(user_input, top_k=3)
        
        return jsonify({
            'response': response,
            'analysis': {
                'reasoning_steps': analysis.get('reasoning_steps', []),
                'sources': analysis.get('sources', []),
                'confidence': analysis.get('confidence', 0.0)
            },
            'context': context
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Handle user feedback"""
    try:
        data = request.json
        is_positive = data.get('positive', False)
        user_input = data.get('user_input', '')
        response = data.get('response', '')
        
        system = get_chat_system()
        feedback_value = 0.8 if is_positive else 0.2
        system.learn_from_interaction(user_input, response, feedback=feedback_value)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    try:
        system = get_chat_system()
        
        return jsonify({
            'chat_count': len(system.chat_history),
            'learning_score': float(system.liquid_weights.get_weights().mean()),
            'weights': {
                'mean': float(system.liquid_weights.get_weights().mean()),
                'std': float(system.liquid_weights.get_weights().std())
            },
            'reasoning_chains': len(system.critical_thinking.reasoning_chains),
            'hlhfm_shards': len(system.hlhfm.memory_shards) if system.hlhfm else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/save', methods=['POST'])
def save():
    """Save system state"""
    try:
        system = get_chat_system()
        system.save_state()
        return jsonify({'status': 'success', 'message': 'State saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("GPT-2-VIC Chat Server Starting")
    print("=" * 70)
    print("Open your browser to: http://localhost:5000")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
