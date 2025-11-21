// HLHFM Popup Interface Controller

document.addEventListener('DOMContentLoaded', async () => {
  await updateStats();
  
  document.getElementById('addBtn').addEventListener('click', addContent);
  document.getElementById('queryBtn').addEventListener('click', queryMemory);
});

async function updateStats() {
  const response = await chrome.runtime.sendMessage({ type: 'GET_STATS' });
  document.getElementById('totalShards').textContent = response.totalEntries || 0;
  document.getElementById('status').textContent = response.initialized ? '✓' : '...';
}

async function addContent() {
  const content = document.getElementById('contentInput').value;
  const emotion = document.getElementById('emotionSelect').value;
  const intent = document.getElementById('intentSelect').value;
  
  if (!content.trim()) {
    alert('Please enter content to fractalize');
    return;
  }
  
  const response = await chrome.runtime.sendMessage({
    type: 'ADD_CONTENT',
    content,
    emotion,
    intent
  });
  
  if (response.status === 'stored') {
    document.getElementById('contentInput').value = '';
    await updateStats();
    showNotification('Content fractalized and stored!');
  }
}

async function queryMemory() {
  const query = document.getElementById('queryInput').value;
  
  if (!query.trim()) {
    alert('Please enter a query');
    return;
  }
  
  const response = await chrome.runtime.sendMessage({
    type: 'QUERY_MEMORY',
    query
  });
  
  const resultsDiv = document.getElementById('results');
  if (response.results && response.results.length > 0) {
    resultsDiv.innerHTML = response.results.map(r => `
      <div class="result-item">
        <strong>Similarity: ${(r.similarity * 100).toFixed(1)}%</strong><br>
        ${r.content.substring(0, 100)}${r.content.length > 100 ? '...' : ''}<br>
        <small>Emotion: ${r.emotion} | Intent: ${r.intent}</small>
      </div>
    `).join('');
  } else {
    resultsDiv.innerHTML = '<div class="result-item">No results found</div>';
  }
}

function showNotification(message) {
  // Simple notification - could be enhanced
  const original = document.getElementById('addBtn').textContent;
  document.getElementById('addBtn').textContent = '✓ ' + message;
  setTimeout(() => {
    document.getElementById('addBtn').textContent = original;
  }, 2000);
}
