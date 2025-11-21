// HLHFM Background Service Worker
// Manages persistent memory and quantum semiring operations

class HLHFMBackground {
  constructor() {
    this.initialized = false;
    this.memoryStore = {};
    this.auditInterval = null;
  }
  
  async initialize() {
    console.log('[HLHFM] Initializing Victor Background Service...');
    
    // Load persisted memory from IndexedDB simulation via chrome.storage
    const stored = await chrome.storage.local.get(['hlhfm_memory', 'hlhfm_config']);
    if (stored.hlhfm_memory) {
      this.memoryStore = stored.hlhfm_memory;
      console.log('[HLHFM] Loaded', Object.keys(this.memoryStore).length, 'memory entries');
    }
    
    // Start eternal audit loop
    this.startAuditLoop();
    this.initialized = true;
  }
  
  startAuditLoop() {
    // Run audit every 5 minutes
    this.auditInterval = setInterval(() => {
      this.runAudit();
    }, 5 * 60 * 1000);
  }
  
  async runAudit() {
    console.log('[HLHFM] Running eternal audit loop...');
    // Decay old entries
    const now = Date.now();
    let decayed = 0;
    for (const [key, entry] of Object.entries(this.memoryStore)) {
      const age = now - entry.timestamp;
      const decayFactor = Math.pow(0.95, age / (3600 * 1000));
      if (decayFactor < 0.1) {
        delete this.memoryStore[key];
        decayed++;
      }
    }
    console.log('[HLHFM] Audit complete. Decayed:', decayed);
    await this.persistMemory();
  }
  
  async persistMemory() {
    await chrome.storage.local.set({
      hlhfm_memory: this.memoryStore,
      last_persist: Date.now()
    });
  }
  
  async handleMessage(message, sender, sendResponse) {
    switch (message.type) {
      case 'ADD_CONTENT':
        return this.addContent(message.content, message.emotion, message.intent);
      case 'QUERY_MEMORY':
        return this.queryMemory(message.query);
      case 'GET_STATS':
        return this.getStats();
      default:
        return { error: 'Unknown message type' };
    }
  }
  
  addContent(content, emotion = 'neutral', intent = 'observe') {
    const id = `entry_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.memoryStore[id] = {
      content,
      emotion,
      intent,
      timestamp: Date.now(),
      depth: 0
    };
    this.persistMemory();
    return { id, status: 'stored' };
  }
  
  queryMemory(query) {
    // Simple query implementation
    const results = Object.entries(this.memoryStore)
      .map(([id, entry]) => ({
        id,
        ...entry,
        similarity: this.computeSimilarity(query, entry.content)
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 5);
    return { results };
  }
  
  computeSimilarity(query, content) {
    // Simple word overlap similarity
    const qWords = new Set(query.toLowerCase().split(/\s+/));
    const cWords = new Set(content.toLowerCase().split(/\s+/));
    const intersection = new Set([...qWords].filter(w => cWords.has(w)));
    return intersection.size / Math.max(qWords.size, cWords.size, 1);
  }
  
  getStats() {
    return {
      totalEntries: Object.keys(this.memoryStore).length,
      initialized: this.initialized,
      lastAudit: Date.now()
    };
  }
}

const hlhfm = new HLHFMBackground();
hlhfm.initialize();

// Message listener
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  hlhfm.handleMessage(message, sender, sendResponse).then(sendResponse);
  return true; // Async response
});

console.log('[HLHFM] Victor Background Service Active - Bloodline Sovereign');
