#!/usr/bin/env python3
"""
Standalone GPT Chat Interface with Learning and Critical Thinking Synthesis
Integrates GPT-2 model with HLHFM for chat learning and liquid weights
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import tensorflow as tf
    from src import model, sample, encoder
except ImportError:
    print("Note: TensorFlow not available. Running in memory-only mode.")
    tf = None

# Import HLHFM system for learning and liquid weights
try:
    from build import HLHFM, HolographicMemory, QuantumSemiring
except ImportError:
    print("Note: HLHFM not available. Advanced features disabled.")
    HLHFM = None


class LiquidWeights:
    """
    Adaptive liquid weights that adjust based on conversation feedback
    Integrates with HLHFM adaptive gates for dynamic weight adjustment
    """
    
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.weights = np.ones(dimension)
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.velocity = np.zeros(dimension)
        self.adaptation_history = []
        
    def adjust(self, feedback_signal: np.ndarray, context_embedding: np.ndarray):
        """Adjust weights based on feedback and context"""
        # Calculate gradient-like adjustment
        adjustment = feedback_signal * context_embedding
        
        # Apply momentum
        self.velocity = self.momentum * self.velocity + self.learning_rate * adjustment
        self.weights += self.velocity
        
        # Normalize to prevent drift
        self.weights = np.clip(self.weights, 0.1, 10.0)
        self.weights = self.weights / (np.linalg.norm(self.weights) + 1e-10) * np.sqrt(self.dimension)
        
        # Track adaptation
        self.adaptation_history.append({
            'timestamp': time.time(),
            'mean_weight': float(np.mean(self.weights)),
            'std_weight': float(np.std(self.weights))
        })
        
    def get_weights(self) -> np.ndarray:
        """Get current weight vector"""
        return self.weights.copy()
    
    def save(self, filepath: str):
        """Save weights to file"""
        data = {
            'weights': self.weights.tolist(),
            'velocity': self.velocity.tolist(),
            'history': self.adaptation_history
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load weights from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.weights = np.array(data['weights'])
            self.velocity = np.array(data['velocity'])
            self.adaptation_history = data.get('history', [])


class CriticalThinkingSynthesizer:
    """
    Synthesizes critical thinking analysis with source tracking
    """
    
    def __init__(self):
        self.reasoning_chains = []
        self.sources = []
        
    def analyze(self, text: str, context: List[str]) -> Dict[str, Any]:
        """
        Analyze text and generate critical thinking synthesis
        Returns reasoning chain with sources
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'reasoning_steps': [],
            'sources': [],
            'confidence': 0.0
        }
        
        # Extract key concepts (simple word-based for now)
        words = text.lower().split()
        key_concepts = [w for w in words if len(w) > 4][:5]
        
        # Generate reasoning steps
        if context:
            analysis['reasoning_steps'].append({
                'step': 1,
                'type': 'context_retrieval',
                'description': f'Retrieved {len(context)} relevant context items',
                'concepts': key_concepts
            })
        
        analysis['reasoning_steps'].append({
            'step': 2,
            'type': 'concept_analysis',
            'description': f'Identified key concepts: {", ".join(key_concepts[:3])}',
            'concepts': key_concepts
        })
        
        # Track sources
        for i, ctx in enumerate(context[:3]):
            analysis['sources'].append({
                'id': f'source_{i+1}',
                'text': ctx[:100] + '...' if len(ctx) > 100 else ctx,
                'relevance': 1.0 / (i + 1)  # Simple relevance score
            })
        
        # Calculate confidence
        analysis['confidence'] = min(1.0, len(context) * 0.2)
        
        self.reasoning_chains.append(analysis)
        return analysis
    
    def format_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format analysis for display"""
        output = []
        output.append("\n=== Critical Thinking Analysis ===")
        
        if analysis['reasoning_steps']:
            output.append("\nReasoning Chain:")
            for step in analysis['reasoning_steps']:
                output.append(f"  {step['step']}. {step['description']}")
        
        if analysis['sources']:
            output.append("\nSources Referenced:")
            for src in analysis['sources']:
                output.append(f"  [{src['id']}] {src['text']} (relevance: {src['relevance']:.2f})")
        
        output.append(f"\nConfidence: {analysis['confidence']:.2%}")
        output.append("=" * 35)
        
        return "\n".join(output)


class ChatLearningSystem:
    """
    Main chat learning system integrating GPT-2, HLHFM, and liquid weights
    """
    
    def __init__(self, model_name: str = '124M', models_dir: str = 'models'):
        self.model_name = model_name
        self.models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        self.dimension = 512
        
        # Initialize components
        self.liquid_weights = LiquidWeights(self.dimension)
        self.critical_thinking = CriticalThinkingSynthesizer()
        
        # Initialize HLHFM if available
        self.hlhfm = None
        if HLHFM is not None:
            try:
                self.hlhfm = HLHFM(dimension=self.dimension, max_depth=8)
                print("✓ HLHFM learning system initialized")
            except Exception as e:
                print(f"! HLHFM initialization failed: {e}")
        
        # Chat history
        self.chat_history = []
        self.conversation_embeddings = []
        
        # TensorFlow session
        self.sess = None
        self.enc = None
        self.output = None
        self.context_placeholder = None
        
        # Initialize model if TensorFlow available
        self._init_model()
        
    def _init_model(self):
        """Initialize GPT-2 model if available"""
        if tf is None:
            print("! TensorFlow not available - running in memory-only mode")
            return
            
        try:
            self.enc = encoder.get_encoder(self.model_name, self.models_dir)
            hparams = model.default_hparams()
            
            hparams_path = os.path.join(self.models_dir, self.model_name, 'hparams.json')
            if os.path.exists(hparams_path):
                with open(hparams_path) as f:
                    hparams.override_from_dict(json.load(f))
            
            # Create TensorFlow graph
            graph = tf.Graph()
            with graph.as_default():
                self.context_placeholder = tf.placeholder(tf.int32, [1, None])
                self.output = sample.sample_sequence(
                    hparams=hparams,
                    length=150,  # Response length
                    context=self.context_placeholder,
                    batch_size=1,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9
                )
                
                # Create session and restore model
                self.sess = tf.Session(graph=graph)
                saver = tf.train.Saver()
                ckpt = tf.train.latest_checkpoint(os.path.join(self.models_dir, self.model_name))
                if ckpt:
                    saver.restore(self.sess, ckpt)
                    print(f"✓ GPT-2 model loaded: {self.model_name}")
                else:
                    print(f"! Model checkpoint not found at {self.models_dir}/{self.model_name}")
                    self.sess = None
        except Exception as e:
            print(f"! Model initialization failed: {e}")
            self.sess = None
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """Create simple embedding from text using hash-based approach"""
        # Use hash-based approach for more uniform distribution
        embedding = np.zeros(self.dimension)
        for i, char in enumerate(text):
            # Hash character with position for better distribution
            hash_val = hash(f"{char}_{i}") % self.dimension
            embedding[hash_val] += 1.0
        
        # Add length normalization
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        return embedding
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant context from chat history (public method)"""
        if not self.chat_history:
            return []
        
        query_embedding = self._create_embedding(query)
        
        # Simple similarity-based retrieval
        similarities = []
        for i, (msg, embedding) in enumerate(zip(self.chat_history, self.conversation_embeddings)):
            if embedding is not None:
                sim = np.dot(query_embedding, embedding)
                similarities.append((sim, i, msg))
        
        # Sort by similarity and return top_k
        similarities.sort(reverse=True)
        return [msg for _, _, msg in similarities[:top_k]]
    
    def _retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Internal context retrieval (deprecated - use retrieve_context)"""
        return self.retrieve_context(query, top_k)
    
    def learn_from_interaction(self, user_input: str, response: str, feedback: float = 0.5):
        """
        Learn from user interaction using liquid weights and HLHFM
        feedback: 0.0 (negative) to 1.0 (positive)
        """
        # Create embeddings
        input_embedding = self._create_embedding(user_input)
        response_embedding = self._create_embedding(response)
        
        # Adjust liquid weights based on feedback
        feedback_signal = np.full(self.dimension, feedback - 0.5)  # Center around 0
        self.liquid_weights.adjust(feedback_signal, input_embedding)
        
        # Store in HLHFM if available
        if self.hlhfm is not None:
            try:
                # Determine emotion and intent from feedback
                if feedback > 0.7:
                    emotion = "joy"
                    intent = "learn"
                elif feedback > 0.5:
                    emotion = "curiosity"
                    intent = "analyze"
                else:
                    emotion = "neutral"
                    intent = "observe"
                
                # Add to HLHFM
                combined_embedding = input_embedding * self.liquid_weights.get_weights()
                self.hlhfm.add_fractal_content(
                    combined_embedding,
                    emotion=emotion,
                    intent=intent
                )
            except Exception as e:
                print(f"! HLHFM learning failed: {e}")
        
        # Store in history
        self.chat_history.append(f"User: {user_input}\nAssistant: {response}")
        self.conversation_embeddings.append(input_embedding)
    
    def generate_response(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response with critical thinking synthesis
        Returns: (response_text, critical_analysis)
        """
        # Retrieve relevant context
        context = self.retrieve_context(user_input, top_k=3)
        
        # Generate critical thinking analysis
        analysis = self.critical_thinking.analyze(user_input, context)
        
        # Generate response using GPT-2 if available
        response = ""
        if self.sess is not None and self.enc is not None:
            try:
                # Build context with chat history
                context_text = "\n".join(context[-2:]) + f"\nUser: {user_input}\nAssistant:"
                context_tokens = self.enc.encode(context_text)
                
                # Generate
                out = self.sess.run(self.output, feed_dict={
                    self.context_placeholder: [context_tokens]
                })
                
                response_tokens = out[0, len(context_tokens):]
                response = self.enc.decode(response_tokens)
                
                # Clean up response
                response = response.split('\n')[0].strip()
                if not response:
                    response = "I understand. Could you tell me more?"
                    
            except Exception as e:
                print(f"! Generation error: {e}")
                response = "I'm processing your request. [Model error occurred]"
        else:
            # Fallback response without model
            response = f"I understand your message about: {user_input[:50]}... [Running in memory-only mode - TensorFlow not available]"
        
        # Learn from this interaction (neutral feedback initially)
        self.learn_from_interaction(user_input, response, feedback=0.5)
        
        return response, analysis
    
    def save_state(self, directory: str = 'chat_state'):
        """Save chat system state"""
        os.makedirs(directory, exist_ok=True)
        
        # Save liquid weights
        self.liquid_weights.save(os.path.join(directory, 'liquid_weights.json'))
        
        # Save chat history
        with open(os.path.join(directory, 'chat_history.json'), 'w') as f:
            json.dump({
                'history': self.chat_history,
                'reasoning_chains': self.critical_thinking.reasoning_chains
            }, f, indent=2)
        
        print(f"✓ State saved to {directory}/")
    
    def load_state(self, directory: str = 'chat_state'):
        """Load chat system state"""
        if not os.path.exists(directory):
            return
        
        # Load liquid weights
        weights_path = os.path.join(directory, 'liquid_weights.json')
        if os.path.exists(weights_path):
            self.liquid_weights.load(weights_path)
        
        # Load chat history
        history_path = os.path.join(directory, 'chat_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                data = json.load(f)
            self.chat_history = data.get('history', [])
            self.critical_thinking.reasoning_chains = data.get('reasoning_chains', [])
            
            # Recreate embeddings
            self.conversation_embeddings = []
            for msg in self.chat_history:
                user_part = msg.split('\n')[0].replace('User: ', '')
                self.conversation_embeddings.append(self._create_embedding(user_part))
        
        print(f"✓ State loaded from {directory}/")


def run_chat_interface():
    """Run interactive chat interface"""
    print("=" * 70)
    print("GPT-2-VIC Standalone Chat Interface with Learning")
    print("Critical Thinking Synthesis | Liquid Weights | Source Research")
    print("=" * 70)
    print()
    
    # Initialize system
    print("Initializing chat learning system...")
    chat_system = ChatLearningSystem()
    
    # Try to load previous state
    chat_system.load_state()
    
    print()
    print("Commands:")
    print("  /save    - Save current state")
    print("  /stats   - Show system statistics")
    print("  /quit    - Exit chat")
    print("=" * 70)
    print()
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == '/quit':
                chat_system.save_state()
                print("\nGoodbye! State saved.")
                break
            
            elif user_input.lower() == '/save':
                chat_system.save_state()
                print("✓ State saved successfully")
                continue
            
            elif user_input.lower() == '/stats':
                print(f"\n=== System Statistics ===")
                print(f"Chat history: {len(chat_system.chat_history)} interactions")
                print(f"Reasoning chains: {len(chat_system.critical_thinking.reasoning_chains)}")
                print(f"Liquid weights mean: {np.mean(chat_system.liquid_weights.get_weights()):.4f}")
                if chat_system.hlhfm:
                    print(f"HLHFM memory shards: {len(chat_system.hlhfm.memory_shards)}")
                continue
            
            # Generate response
            print("\n[Thinking...]")
            response, analysis = chat_system.generate_response(user_input)
            
            # Display response
            print(f"\nAssistant: {response}")
            
            # Display critical thinking analysis
            print(chat_system.critical_thinking.format_analysis(analysis))
            
            # Ask for feedback (optional)
            feedback_input = input("\nFeedback (good/bad/skip): ").strip().lower()
            if feedback_input == 'good':
                chat_system.learn_from_interaction(user_input, response, feedback=0.8)
                print("✓ Positive feedback recorded")
            elif feedback_input == 'bad':
                chat_system.learn_from_interaction(user_input, response, feedback=0.2)
                print("✓ Negative feedback recorded")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving state...")
        chat_system.save_state()
        print("Goodbye!")


if __name__ == '__main__':
    run_chat_interface()
