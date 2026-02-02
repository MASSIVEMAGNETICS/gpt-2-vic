#!/usr/bin/env python3
"""
Test suite for chat interface functionality
"""

import os
import sys
import numpy as np
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.chat_interface import LiquidWeights, CriticalThinkingSynthesizer, ChatLearningSystem

def test_liquid_weights():
    """Test liquid weights adaptation"""
    print("Testing Liquid Weights...")
    
    lw = LiquidWeights(dimension=512)
    
    # Test initial state
    assert lw.dimension == 512
    assert len(lw.weights) == 512
    assert np.allclose(lw.weights.mean(), 1.0, atol=0.1)
    
    # Test adjustment
    feedback = np.ones(512) * 0.5
    context = np.random.randn(512)
    lw.adjust(feedback, context)
    
    assert len(lw.adaptation_history) == 1
    assert 'timestamp' in lw.adaptation_history[0]
    
    print("  ✓ Initialization")
    print("  ✓ Weight adjustment")
    print("  ✓ History tracking")
    
    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'weights.json')
        lw.save(filepath)
        assert os.path.exists(filepath)
        
        lw2 = LiquidWeights(dimension=512)
        lw2.load(filepath)
        assert np.allclose(lw2.weights, lw.weights)
    
    print("  ✓ Persistence (save/load)")
    print("✓ Liquid Weights - All tests passed\n")


def test_critical_thinking():
    """Test critical thinking synthesizer"""
    print("Testing Critical Thinking Synthesizer...")
    
    cts = CriticalThinkingSynthesizer()
    
    # Test analysis
    text = "How does machine learning work in neural networks?"
    context = [
        "Machine learning is a subset of AI",
        "Neural networks have multiple layers"
    ]
    
    analysis = cts.analyze(text, context)
    
    assert 'reasoning_steps' in analysis
    assert 'sources' in analysis
    assert 'confidence' in analysis
    assert len(analysis['reasoning_steps']) > 0
    assert len(analysis['sources']) == 2
    
    print("  ✓ Analysis generation")
    print("  ✓ Reasoning steps extraction")
    print("  ✓ Source tracking")
    
    # Test formatting
    formatted = cts.format_analysis(analysis)
    assert 'Critical Thinking Analysis' in formatted
    assert 'Reasoning Chain:' in formatted
    assert 'Sources Referenced:' in formatted
    
    print("  ✓ Output formatting")
    print("✓ Critical Thinking Synthesizer - All tests passed\n")


def test_chat_learning_system():
    """Test chat learning system"""
    print("Testing Chat Learning System...")
    
    # Initialize without model (memory-only mode)
    chat = ChatLearningSystem(model_name='124M')
    
    assert chat.liquid_weights is not None
    assert chat.critical_thinking is not None
    assert len(chat.chat_history) == 0
    
    print("  ✓ System initialization")
    
    # Test embedding creation
    embedding = chat._create_embedding("Hello world")
    assert len(embedding) == chat.dimension
    assert np.allclose(np.linalg.norm(embedding), 1.0)
    
    print("  ✓ Embedding generation")
    
    # Test learning
    chat.learn_from_interaction(
        "What is AI?",
        "AI stands for Artificial Intelligence.",
        feedback=0.8
    )
    
    assert len(chat.chat_history) == 1
    assert len(chat.conversation_embeddings) == 1
    
    print("  ✓ Learning from interactions")
    
    # Test context retrieval
    context = chat._retrieve_context("Tell me about AI", top_k=2)
    assert isinstance(context, list)
    
    print("  ✓ Context retrieval")
    
    # Test response generation (will work in memory-only mode)
    response, analysis = chat.generate_response("How do you learn?")
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert 'reasoning_steps' in analysis
    assert 'sources' in analysis
    
    print("  ✓ Response generation")
    print("  ✓ Analysis integration")
    
    # Test state persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        chat.save_state(tmpdir)
        
        assert os.path.exists(os.path.join(tmpdir, 'liquid_weights.json'))
        assert os.path.exists(os.path.join(tmpdir, 'chat_history.json'))
        
        # Load into new instance
        chat2 = ChatLearningSystem()
        chat2.load_state(tmpdir)
        
        assert len(chat2.chat_history) == len(chat.chat_history)
    
    print("  ✓ State persistence")
    print("✓ Chat Learning System - All tests passed\n")


def test_integration():
    """Test integration of all components"""
    print("Testing System Integration...")
    
    chat = ChatLearningSystem()
    
    # Simulate a conversation
    interactions = [
        ("Hello!", "Hello! How can I help you?"),
        ("What can you do?", "I can chat and learn from our conversations."),
        ("That's interesting!", "Thank you! I'm designed to improve over time.")
    ]
    
    for user_input, expected_type in interactions:
        response, analysis = chat.generate_response(user_input)
        
        # Verify response structure
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify analysis structure
        assert 'reasoning_steps' in analysis
        assert 'sources' in analysis
        assert 'confidence' in analysis
        
        # Provide feedback
        chat.learn_from_interaction(user_input, response, feedback=0.7)
    
    # Check that history accumulated
    assert len(chat.chat_history) >= len(interactions)
    
    # Check that weights adapted
    assert len(chat.liquid_weights.adaptation_history) > 0
    
    # Check that reasoning chains accumulated
    assert len(chat.critical_thinking.reasoning_chains) >= len(interactions)
    
    print("  ✓ Multi-turn conversation")
    print("  ✓ Feedback integration")
    print("  ✓ History accumulation")
    print("  ✓ Weight adaptation")
    print("✓ System Integration - All tests passed\n")


def test_api_compatibility():
    """Test Flask server API compatibility"""
    print("Testing API Compatibility...")
    
    try:
        from chat_server import app
        
        # Check routes exist
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        
        assert '/' in rules
        assert '/api/chat' in rules
        assert '/api/feedback' in rules
        assert '/api/stats' in rules
        assert '/api/save' in rules
        
        print("  ✓ All API endpoints defined")
        print("  ✓ Flask app configuration")
        print("✓ API Compatibility - All tests passed\n")
    except ImportError as e:
        print(f"  ! Flask not available (expected in some environments): {e}")
        print("✓ API Compatibility - Skipped (optional)\n")


def main():
    """Run all tests"""
    print("=" * 70)
    print("GPT-2-VIC Chat Interface Test Suite")
    print("=" * 70)
    print()
    
    try:
        test_liquid_weights()
        test_critical_thinking()
        test_chat_learning_system()
        test_integration()
        test_api_compatibility()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("The chat interface is ready to use!")
        print("Run: python src/chat_interface.py (CLI)")
        print("  or: python chat_server.py (Web UI)")
        print()
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
