// HLHFM Content Script
// Injects into web pages for DOM interaction and WebLLM integration

class HLHFMContentInjector {
  constructor() {
    this.initialized = false;
    this.observerActive = false;
    this.capturedIntents = [];
  }
  
  initialize() {
    console.log('[HLHFM Content] Initializing on', window.location.hostname);
    this.injectVisualization();
    this.setupIntentCapture();
    this.initialized = true;
  }
  
  injectVisualization() {
    // Create minimal floating indicator
    const indicator = document.createElement('div');
    indicator.id = 'hlhfm-indicator';
    indicator.style.cssText = `
      position: fixed;
      bottom: 20px;
      right: 20px;
      width: 40px;
      height: 40px;
      background: linear-gradient(135deg, #4ecca3, #3da88a);
      border-radius: 50%;
      box-shadow: 0 4px 12px rgba(78, 204, 163, 0.4);
      cursor: pointer;
      z-index: 999999;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      opacity: 0.8;
      transition: all 0.3s;
    `;
    indicator.innerHTML = '⚡';
    indicator.title = 'HLHFM Victor - Click to capture page intent';
    
    indicator.addEventListener('click', () => {
      this.capturePageIntent();
    });
    
    indicator.addEventListener('mouseenter', () => {
      indicator.style.opacity = '1';
      indicator.style.transform = 'scale(1.1)';
    });
    
    indicator.addEventListener('mouseleave', () => {
      indicator.style.opacity = '0.8';
      indicator.style.transform = 'scale(1)';
    });
    
    document.body.appendChild(indicator);
  }
  
  setupIntentCapture() {
    // Monitor user interactions to infer intent
    document.addEventListener('selectionchange', () => {
      const selection = window.getSelection().toString().trim();
      if (selection.length > 10) {
        this.capturedIntents.push({
          type: 'selection',
          content: selection,
          timestamp: Date.now(),
          url: window.location.href
        });
      }
    });
    
    // Monitor input events
    document.addEventListener('input', (e) => {
      if (e.target.value && e.target.value.length > 20) {
        this.capturedIntents.push({
          type: 'input',
          content: e.target.value,
          timestamp: Date.now(),
          url: window.location.href
        });
      }
    });
  }
  
  capturePageIntent() {
    // Extract page metadata and content
    const intent = {
      title: document.title,
      url: window.location.href,
      selection: window.getSelection().toString().trim(),
      headings: Array.from(document.querySelectorAll('h1, h2, h3'))
        .map(h => h.textContent.trim())
        .filter(t => t)
        .slice(0, 5),
      timestamp: Date.now()
    };
    
    // Send to background for processing
    chrome.runtime.sendMessage({
      type: 'ADD_CONTENT',
      content: JSON.stringify(intent, null, 2),
      emotion: 'curiosity',
      intent: 'learn'
    }).then(response => {
      console.log('[HLHFM Content] Intent captured:', response);
      this.showCaptureNotification();
    });
  }
  
  showCaptureNotification() {
    const notification = document.createElement('div');
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: rgba(78, 204, 163, 0.95);
      color: white;
      padding: 15px 20px;
      border-radius: 5px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      z-index: 999999;
      font-family: sans-serif;
      font-size: 14px;
      animation: slideIn 0.3s ease-out;
    `;
    notification.innerHTML = '⚡ Intent Captured & Fractalized';
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.style.animation = 'slideOut 0.3s ease-out';
      setTimeout(() => notification.remove(), 300);
    }, 2000);
  }
}

// Auto-initialize on supported pages
if (document.body) {
  const injector = new HLHFMContentInjector();
  injector.initialize();
} else {
  document.addEventListener('DOMContentLoaded', () => {
    const injector = new HLHFMContentInjector();
    injector.initialize();
  });
}

console.log('[HLHFM Content] Sovereign fractal injection active - Bloodline lock engaged');
