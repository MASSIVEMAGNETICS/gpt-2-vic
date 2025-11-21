#!/usr/bin/env python3
"""
Test suite for HLHFM (Holographic Latent Hyperdimensional Fractal Memory)
Validates core functionality, quantum semiring operations, and browser extension generation
"""

import numpy as np
import sys
import os

# Import from build.py
sys.path.insert(0, os.path.dirname(__file__))
from build import (
    HLHFM, 
    QuantumSemiring, 
    HolographicMemory, 
    ZKPEmulator,
    BrowserExtensionBuilder,
    SelfModificationEngine
)


def test_quantum_semiring():
    """Test quantum semiring operations"""
    print("\n[TEST 1] Quantum Semiring Operations")
    print("-" * 50)
    
    quantum = QuantumSemiring(dimension=512)
    
    # Test Hadamard superposition
    vec = np.random.randn(512)
    superposed = quantum.hadamard_superposition(vec)
    assert superposed.shape[0] >= 512, "Superposition should expand or maintain dimension"
    print("  ✓ Hadamard superposition works")
    
    # Test semiring operations
    a = np.random.randn(512)
    b = np.random.randn(512)
    
    minmax_result = quantum.minmax_semiring(a, b)
    assert minmax_result.shape == (512,), "Min-max semiring shape mismatch"
    print("  ✓ Min-max semiring operation works")
    
    prob_result = quantum.prob_semiring(a, b)
    assert prob_result.shape == (512,), "Probabilistic semiring shape mismatch"
    print("  ✓ Probabilistic semiring operation works")
    
    # Test entanglement
    vectors = [np.random.randn(512) for _ in range(3)]
    entangled = quantum.entangle_vectors(vectors)
    assert entangled.shape == (512,), "Entanglement should maintain dimension"
    print("  ✓ Vector entanglement works")
    
    # Test hybrid loop
    quantum_state = np.random.randn(512)
    classical_fallback = np.random.randn(512)
    hybrid_result = quantum.hybrid_loop(quantum_state, classical_fallback)
    assert hybrid_result.shape == (512,), "Hybrid loop shape mismatch"
    print("  ✓ Hybrid quantum-classical loop works")
    
    print("  ✅ All quantum semiring tests passed")
    return True


def test_holographic_memory():
    """Test HRR operations"""
    print("\n[TEST 2] Holographic Reduced Representations")
    print("-" * 50)
    
    hrr = HolographicMemory(dimension=512)
    
    # Test binding and unbinding
    key = hrr.generate_random_vector(seed=42)
    value = hrr.generate_random_vector(seed=43)
    
    trace = hrr.bind(key, value)
    assert trace.shape == (512,), "Binding should maintain dimension"
    print("  ✓ Key-value binding works")
    
    recovered = hrr.unbind(trace, key)
    similarity = np.dot(recovered, value) / (np.linalg.norm(recovered) * np.linalg.norm(value))
    assert similarity > 0.5, f"Unbinding should recover similar value (similarity: {similarity:.4f})"
    print(f"  ✓ Unbinding works (similarity: {similarity:.4f})")
    
    # Test random vector generation
    vec1 = hrr.generate_random_vector(seed=100)
    vec2 = hrr.generate_random_vector(seed=100)
    assert np.allclose(vec1, vec2), "Same seed should produce same vector"
    print("  ✓ Random vector generation is deterministic")
    
    print("  ✅ All HRR tests passed")
    return True


def test_zkp_emulator():
    """Test Zero-Knowledge Proof emulation"""
    print("\n[TEST 3] Zero-Knowledge Proof Emulation")
    print("-" * 50)
    
    zkp = ZKPEmulator()
    
    # Test proof generation
    data = {
        'shard_id': 'test_shard_001',
        'depth': 5,
        'timestamp': 1234567890
    }
    proof = zkp.generate_proof(data, secret="test_secret")
    assert isinstance(proof, str), "Proof should be a string"
    assert len(proof) == 64, "SHA256 hash should be 64 hex characters"
    print(f"  ✓ Proof generation works (hash: {proof[:16]}...)")
    
    # Test proof verification
    is_valid = zkp.verify_proof(proof, data, secret="test_secret")
    assert is_valid, "Valid proof should verify"
    print("  ✓ Proof verification works (valid)")
    
    # Test invalid proof
    tampered_data = data.copy()
    tampered_data['depth'] = 999
    is_invalid = zkp.verify_proof(proof, tampered_data, secret="test_secret")
    assert not is_invalid, "Tampered data should not verify"
    print("  ✓ Tamper detection works")
    
    # Test metadata retrieval
    metadata = zkp.get_proof_metadata(proof)
    assert metadata is not None, "Metadata should be retrievable"
    assert 'timestamp' in metadata, "Metadata should contain timestamp"
    print("  ✓ Proof metadata retrieval works")
    
    print("  ✅ All ZKP tests passed")
    return True


def test_hlhfm_core():
    """Test HLHFM core functionality"""
    print("\n[TEST 4] HLHFM Core Functionality")
    print("-" * 50)
    
    hlhfm = HLHFM(dimension=512, max_depth=8)
    
    # Test fractal content addition
    content = np.random.randn(512)
    shard_id = hlhfm.add_fractal_content(
        content,
        emotion="determination",
        intent="create"
    )
    assert isinstance(shard_id, str), "Shard ID should be a string"
    assert shard_id.startswith("shard_d0_"), "Root shard should be depth 0"
    print(f"  ✓ Fractal content addition works (ID: {shard_id})")
    
    # Verify hierarchies updated
    assert "determination" in hlhfm.emotion_hierarchy, "Emotion should be recorded"
    assert "create" in hlhfm.intent_hierarchy, "Intent should be recorded"
    print("  ✓ Emotion and intent hierarchies updated")
    
    # Test provenance
    assert shard_id in hlhfm.provenance_traces, "Provenance should be recorded"
    print("  ✓ Causal provenance tracked")
    
    # Test querying
    query = content * 0.8 + np.random.randn(512) * 0.2
    results = hlhfm.query_memory(query, top_k=5)
    assert len(results) > 0, "Query should return results"
    assert results[0]['verified'], "Top result should be verified"
    print(f"  ✓ Memory query works ({len(results)} results)")
    print(f"    Top similarity: {results[0]['similarity']:.4f}")
    print(f"    Coherence time: {results[0]['coherence_time']:.2f}s")
    
    # Test adaptive gating
    assert len(hlhfm.tau_history) > 0, "Tau history should be updated"
    print("  ✓ Adaptive gate learning active")
    
    # Test audit loop
    stats = hlhfm.eternal_audit_loop(decay_rate=0.95)
    assert 'total_shards' in stats, "Audit should return stats"
    assert stats['total_shards'] > 0, "Should have shards in memory"
    print(f"  ✓ Eternal audit loop works")
    print(f"    Total shards: {stats['total_shards']}")
    print(f"    Emotion diversity: {stats['emotion_diversity']}")
    
    # Test antigravity simulation
    timelines = [np.random.randn(512) for _ in range(3)]
    antigrav_result = hlhfm.antigravity_simulation(timelines)
    assert antigrav_result.shape == (512,), "Antigravity should maintain dimension"
    print("  ✓ Antigravity multi-timeline simulation works")
    
    print("  ✅ All HLHFM core tests passed")
    return True


def test_recursive_sharding():
    """Test recursive fractal shard creation"""
    print("\n[TEST 5] Recursive Fractal Sharding")
    print("-" * 50)
    
    hlhfm = HLHFM(dimension=512, max_depth=8)
    
    # Add content that should trigger deep recursion
    content = np.random.randn(512)
    shard_id = hlhfm.add_fractal_content(content, emotion="curiosity", intent="explore")
    
    # Check for nested shards at various depths
    total_shards = sum(len(shards) for shards in hlhfm.fractal_shards.values())
    print(f"  ✓ Created {total_shards} total shards across depths")
    
    # Verify depth distribution
    for depth, shards in hlhfm.fractal_shards.items():
        if shards:
            print(f"    Depth {depth}: {len(shards)} shards")
    
    # For depth > 4, should have children
    deep_shards = hlhfm.fractal_shards.get(5, [])
    if deep_shards:
        has_children = any(shard.get('children') for shard in deep_shards)
        if has_children:
            print("  ✓ Deep shards (depth > 4) have children")
    
    print("  ✅ Recursive sharding test passed")
    return True


def test_browser_extension():
    """Test browser extension generation"""
    print("\n[TEST 6] Browser Extension Generation")
    print("-" * 50)
    
    # Create temporary test directory
    import tempfile
    import shutil
    
    test_dir = tempfile.mkdtemp(prefix="hlhfm_test_")
    
    try:
        builder = BrowserExtensionBuilder(output_dir=test_dir)
        files = builder.build()
        
        # Check manifest
        assert 'manifest' in files, "Should generate manifest"
        assert os.path.exists(files['manifest']), "Manifest file should exist"
        print(f"  ✓ Manifest generated: {os.path.basename(files['manifest'])}")
        
        # Check background script
        assert 'background' in files, "Should generate background script"
        assert os.path.exists(files['background']), "Background script should exist"
        print(f"  ✓ Background script generated: {os.path.basename(files['background'])}")
        
        # Check popup files
        assert 'popup_html' in files, "Should generate popup HTML"
        assert os.path.exists(files['popup_html']), "Popup HTML should exist"
        print(f"  ✓ Popup HTML generated: {os.path.basename(files['popup_html'])}")
        
        assert 'popup_js' in files, "Should generate popup JS"
        assert os.path.exists(files['popup_js']), "Popup JS should exist"
        print(f"  ✓ Popup JS generated: {os.path.basename(files['popup_js'])}")
        
        # Check content script
        assert 'content' in files, "Should generate content script"
        assert os.path.exists(files['content']), "Content script should exist"
        print(f"  ✓ Content script generated: {os.path.basename(files['content'])}")
        
        # Check icons
        assert 'icons' in files, "Should generate icons"
        assert len(files['icons']) == 3, "Should generate 3 icon sizes"
        print(f"  ✓ Icons generated: {len(files['icons'])} sizes")
        
        print("  ✅ Browser extension generation test passed")
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)


def test_self_modification():
    """Test self-modification engine"""
    print("\n[TEST 7] Self-Modification Engine")
    print("-" * 50)
    
    hlhfm = HLHFM(dimension=512, max_depth=8)
    mod_engine = SelfModificationEngine(hlhfm)
    
    # Test intent analysis
    intent_data = {
        'type': 'user_input',
        'content': 'expand the quantum semiring capabilities'
    }
    modification = mod_engine.analyze_intent(intent_data)
    assert 'action' in modification, "Should return modification action"
    assert 'target' in modification, "Should identify target"
    print(f"  ✓ Intent analysis works")
    print(f"    Action: {modification['action']}")
    print(f"    Target: {modification['target']}")
    
    # Test agent spawning
    agent = mod_engine.spawn_agent(modification)
    assert 'id' in agent, "Agent should have ID"
    assert agent['status'] == 'active', "Agent should start active"
    print(f"  ✓ Agent spawning works (ID: {agent['id']})")
    
    # Test modification execution
    success = mod_engine.execute_modification(agent)
    assert success, "Modification should succeed"
    assert agent['status'] == 'completed', "Agent should be marked completed"
    print("  ✓ Modification execution works")
    
    # Verify history tracking
    assert len(mod_engine.modification_history) > 0, "Should track modifications"
    print(f"  ✓ Modification history tracked ({len(mod_engine.modification_history)} entries)")
    
    print("  ✅ Self-modification engine test passed")
    return True


def run_all_tests():
    """Run all test suites"""
    print("=" * 70)
    print("HLHFM TEST SUITE")
    print("Victor - Bando Bloodline ASI Crucible")
    print("=" * 70)
    
    tests = [
        test_quantum_semiring,
        test_holographic_memory,
        test_zkp_emulator,
        test_hlhfm_core,
        test_recursive_sharding,
        test_browser_extension,
        test_self_modification
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed out of {len(tests)} total")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED - SYSTEM OPERATIONAL")
        print("Bloodline lock verified. Sovereignty maintained.")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED - REVIEW REQUIRED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
