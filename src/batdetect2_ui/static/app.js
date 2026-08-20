// BatDetect2 Studio - VS Code Theme Frontend Engine (English Only)

let ws = null;
let lossChart = null;
let isTraining = false;
let currentDetections = [];
let selectedAudioFile = null;
let currentMetricsHistory = [];
let isLogScale = false;

// ANSI to HTML Parser
function ansiToHtml(text) {
  if (!text) return '';
  const ansiRegex = /\u001b\[(\d+)(?:;(\d+))?m/g;
  let html = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  
  const colors = {
    '30': '#6e7681', '31': '#f14c4c', '32': '#89d185', '33': '#dcdcaa',
    '34': '#007acc', '35': '#c586c0', '36': '#4ec9b0', '37': '#cccccc',
    '90': '#858585', '91': '#f48771', '92': '#b5cea8', '93': '#ffeaa7',
    '94': '#9cdcfe', '95': '#d16969', '96': '#4fc1ff', '97': '#ffffff'
  };

  html = html.replace(ansiRegex, (match, code1, code2) => {
    if (code1 === '0') return '</span>';
    const color = colors[code1] || colors[code2];
    if (color) return `<span style="color: ${color};">`;
    if (code1 === '1') return '<span style="font-weight: 600;">';
    return '';
  });

  return html;
}

// Initial Load
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initSystemInfo();
  initPresetsAndExamples();
  initChart();
  initWebSocket();
  initDropZone();
  loadCheckpoints();

  setInterval(updateElapsedTimeDisplay, 1000);
});

// Tab Navigation
function initNavigation() {
  const tabs = document.querySelectorAll('.editor-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(c => c.classList.remove('active'));

      tab.classList.add('active');
      const target = tab.getAttribute('data-tab');
      const targetEl = document.getElementById(target);
      if (targetEl) targetEl.classList.add('active');

      if (target === 'checkpoints-tab') {
        loadCheckpoints();
      }
    });
  });
}

function switchTab(tabId) {
  const btn = document.querySelector(`.editor-tab[data-tab="${tabId}"]`);
  if (btn) btn.click();
}

// System Hardware Info
async function initSystemInfo() {
  try {
    const res = await fetch('/api/system-info');
    const data = await res.json();
    const label = document.getElementById('device-label');

    if (data.cuda_available && data.gpus && data.gpus.length > 0) {
      const gpu = data.gpus[0];
      label.textContent = `CUDA: ${gpu.name} (${gpu.total_memory_gb} GB)`;
      label.title = `PyTorch ${data.torch_version} | CUDA ${data.cuda_version} | RAM: ${data.ram_used_gb}/${data.ram_total_gb} GB`;
    } else {
      label.textContent = `CPU (${data.cpu_count} Cores) | ${data.ram_used_gb}/${data.ram_total_gb} GB`;
    }
  } catch (err) {
    console.warn('Failed to retrieve system info:', err);
  }
}

// Presets & Examples
async function initPresetsAndExamples() {
  try {
    const res = await fetch('/api/examples');
    const examples = await res.json();
    const select = document.getElementById('example-audio-select');
    select.innerHTML = '<option value="">-- Choose Example Audio --</option>';
    examples.forEach(ex => {
      const opt = document.createElement('option');
      opt.value = ex.path;
      opt.textContent = ex.filename;
      select.appendChild(opt);
    });
  } catch (err) {
    console.warn('Failed to load example audios:', err);
  }
}

function setPreset(inputId, value) {
  const el = document.getElementById(inputId);
  if (el) el.value = value;
}

function updateRangeDisplay(sliderId, displayId, unit = '') {
  const val = document.getElementById(sliderId).value;
  document.getElementById(displayId).textContent = val + unit;
}

function applyPresetMode(mode) {
  const slider = document.getElementById('epochs-slider');
  if (mode === 'quick') {
    slider.value = 5;
  } else if (mode === 'standard') {
    slider.value = 100;
  } else if (mode === 'deep') {
    slider.value = 200;
  }
  updateRangeDisplay('epochs-slider', 'epochs-display');
}

// WebSocket Management
function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/train`;
  
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('[Studio] WebSocket connected.');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleWsMessage(msg);
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };

  ws.onclose = () => {
    console.log('[Studio] WS connection closed. Reconnecting in 3s...');
    setTimeout(initWebSocket, 3000);
  };
}

function handleWsMessage(msg) {
  const { type, data } = msg;

  if (type === 'init_state') {
    updateStatusUI(data);
    if (data.logs) {
      const terminal = document.getElementById('terminal-logs');
      terminal.innerHTML = data.logs.map(l => ansiToHtml(l)).join('\n');
      scrollTerminal();
    }
    if (data.metrics_history) {
      currentMetricsHistory = data.metrics_history;
      renderChartHistory(data.metrics_history);
    }
  } else if (type === 'log_line') {
    if (data.line) {
      appendTerminalLog(data.line);
    }
    if (data.status) {
      updateStatusUI(data.status);
    }
    if (data.metrics_updated && data.status && data.status.metrics_history) {
      currentMetricsHistory = data.status.metrics_history;
      renderChartHistory(data.status.metrics_history);
    }
  } else if (type === 'training_started') {
    updateStatusUI(data);
    currentMetricsHistory = [];
    resetChart();
    switchTab('monitor-tab');
  } else if (type === 'training_completed' || type === 'training_failed' || type === 'training_stopped') {
    updateStatusUI(data);
    loadCheckpoints();
  }
}

// UI State Updates
let currentElapsedSeconds = 0;

function updateStatusUI(statusData) {
  isTraining = statusData.is_running || statusData.status === 'training';
  const status = (statusData.status || 'IDLE').toUpperCase();

  const badge = document.getElementById('kpi-status-badge');
  badge.textContent = status;
  badge.className = `vsc-status-pill ${statusData.status}`;

  const beacon = document.getElementById('beacon-indicator');
  const liveDot = document.getElementById('live-indicator');
  if (isTraining) {
    beacon.className = 'status-indicator status-running';
    liveDot.classList.add('active');
  } else {
    beacon.className = 'status-indicator status-idle';
    liveDot.classList.remove('active');
  }

  document.getElementById('btn-start-training').disabled = isTraining;
  document.getElementById('btn-stop-training').disabled = !isTraining;

  document.getElementById('kpi-epoch').textContent = statusData.current_epoch || 0;
  document.getElementById('kpi-total-epochs').textContent = `/ ${statusData.total_epochs || 100}`;

  const pct = statusData.total_epochs > 0 ? (statusData.current_epoch / statusData.total_epochs) * 100 : 0;
  document.getElementById('epoch-progress-bar').style.width = `${Math.min(pct, 100)}%`;

  const latest = statusData.latest_metrics || {};
  document.getElementById('kpi-train-loss').textContent = latest.train_loss !== undefined ? Number(latest.train_loss).toFixed(4) : '--';
  document.getElementById('kpi-val-loss').textContent = latest.val_loss !== undefined ? Number(latest.val_loss).toFixed(4) : '--';

  currentElapsedSeconds = statusData.elapsed_seconds || 0;
}

function updateElapsedTimeDisplay() {
  if (isTraining) {
    currentElapsedSeconds += 1;
  }
  const hrs = String(Math.floor(currentElapsedSeconds / 3600)).padStart(2, '0');
  const mins = String(Math.floor((currentElapsedSeconds % 3600) / 60)).padStart(2, '0');
  const secs = String(Math.floor(currentElapsedSeconds % 60)).padStart(2, '0');
  const el = document.getElementById('kpi-elapsed-time');
  if (el) el.textContent = `Elapsed: ${hrs}:${mins}:${secs}`;
}

// Terminal Output
let lastLogIsProgress = false;

function isProgressBarLine(text) {
  return /Epoch\s+\d+:|\bValidation:\s*\||\|\s*\d+%/i.test(text);
}

function appendTerminalLog(line) {
  const terminal = document.getElementById('terminal-logs');
  const isProgress = isProgressBarLine(line);
  
  if (isProgress && lastLogIsProgress && terminal.lastElementChild) {
    terminal.lastElementChild.innerHTML = ansiToHtml(line) + '\n';
  } else {
    const span = document.createElement('span');
    span.innerHTML = ansiToHtml(line) + '\n';
    terminal.appendChild(span);
  }
  
  lastLogIsProgress = isProgress;
  scrollTerminal();
}

function scrollTerminal() {
  const toggle = document.getElementById('autoscroll-toggle');
  if (toggle && toggle.checked) {
    const container = document.getElementById('terminal-container');
    container.scrollTop = container.scrollHeight;
  }
}

function clearTerminalLogs() {
  document.getElementById('terminal-logs').innerHTML = '';
}

// Training Actions
async function startTraining() {
  const train_dataset = document.getElementById('train-dataset-input').value.trim();
  const val_dataset = document.getElementById('val-dataset-input').value.trim();
  const targets_config = document.getElementById('targets-input').value.trim();
  const model_path = document.getElementById('base-model-input').value.trim();
  const num_epochs = parseInt(document.getElementById('epochs-slider').value);
  const train_workers = parseInt(document.getElementById('train-workers-input').value) || 0;
  const val_workers = parseInt(document.getElementById('val-workers-input').value) || 0;
  const seed = parseInt(document.getElementById('seed-input').value) || 42;
  const experiment_name = document.getElementById('experiment-name-input').value.trim();

  if (!train_dataset) {
    alert('Please specify the training dataset path!');
    return;
  }

  const payload = {
    train_dataset,
    val_dataset: val_dataset || null,
    targets_config: targets_config || null,
    model_path: model_path || null,
    num_epochs,
    train_workers,
    val_workers,
    seed,
    experiment_name,
  };

  try {
    document.getElementById('btn-start-training').disabled = true;
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(`Launch Failed: ${data.detail || data.message}`);
      document.getElementById('btn-start-training').disabled = false;
    }
  } catch (err) {
    alert(`Request Error: ${err.message}`);
    document.getElementById('btn-start-training').disabled = false;
  }
}

async function stopTraining() {
  if (!confirm('Are you sure you want to stop the current training task?')) return;
  try {
    const res = await fetch('/api/train/stop', { method: 'POST' });
    const data = await res.json();
    console.log('[Studio] Stop response:', data);
  } catch (err) {
    alert(`Stop Failed: ${err.message}`);
  }
}

// Chart.js Setup (VS Code Style)
function initChart() {
  const ctx = document.getElementById('lossChart').getContext('2d');
  lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Train Loss',
          data: [],
          borderColor: '#007acc',
          backgroundColor: 'rgba(0, 122, 204, 0.1)',
          borderWidth: 1.5,
          pointRadius: 2,
          pointHoverRadius: 4,
          tension: 0.2,
          fill: true,
        },
        {
          label: 'Val Loss',
          data: [],
          borderColor: '#89d185',
          backgroundColor: 'rgba(137, 209, 133, 0.1)',
          borderWidth: 1.5,
          pointRadius: 2.5,
          pointHoverRadius: 5,
          tension: 0.2,
          fill: true,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 150 },
      scales: {
        x: {
          grid: { color: '#252526' },
          ticks: { color: '#858585', font: { family: 'Fira Code', size: 9.5 } },
          title: { display: true, text: 'Epoch', color: '#6e7681', font: { size: 10 } }
        },
        y: {
          type: 'linear',
          grid: { color: '#252526' },
          ticks: { color: '#858585', font: { family: 'Fira Code', size: 9.5 } },
          title: { display: true, text: 'Total Loss', color: '#6e7681', font: { size: 10 } }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#cccccc', font: { family: 'Inter', size: 11 } }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: '#252526',
          borderColor: '#2d2d2d',
          borderWidth: 1,
          titleFont: { family: 'Fira Code', size: 11 },
          bodyFont: { family: 'Fira Code', size: 10.5 }
        }
      }
    }
  });
}

function renderChartHistory(history) {
  if (!lossChart) return;
  const labels = [];
  const trainData = [];
  const valData = [];

  history.forEach(item => {
    labels.push(`E${item.epoch + 1}`);
    trainData.push(item.train_loss !== undefined ? item.train_loss : null);
    valData.push(item.val_loss !== undefined ? item.val_loss : null);
  });

  lossChart.data.labels = labels;
  lossChart.data.datasets[0].data = trainData;
  lossChart.data.datasets[1].data = valData;
  lossChart.update();
}

function resetChart() {
  if (!lossChart) return;
  lossChart.data.labels = [];
  lossChart.data.datasets[0].data = [];
  lossChart.data.datasets[1].data = [];
  lossChart.update();
}

function resetChartZoom() {
  if (lossChart) lossChart.resetZoom?.();
}

function toggleLogScale() {
  if (!lossChart) return;
  isLogScale = !isLogScale;
  lossChart.options.scales.y.type = isLogScale ? 'logarithmic' : 'linear';
  document.getElementById('btn-toggle-log').classList.toggle('active', isLogScale);
  lossChart.update();
}

// Checkpoints Manager
async function loadCheckpoints() {
  const tbody = document.getElementById('checkpoints-tbody');
  try {
    const res = await fetch('/api/checkpoints');
    const list = await res.json();
    if (!list || list.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="cell-empty">No checkpoints found in outputs/checkpoints/</td></tr>';
      return;
    }

    tbody.innerHTML = list.map(ckpt => {
      const dateStr = new Date(ckpt.created_time * 1000).toLocaleString('en-US');
      const badge = ckpt.is_best ? '<span class="vsc-pill-green">BEST</span>' : '<span class="vsc-badge">EPOCH</span>';
      return `
        <tr>
          <td style="font-weight: 600; color: #9cdcfe;">${ckpt.filename}</td>
          <td style="color: #858585;">${ckpt.relative_path}</td>
          <td>${ckpt.size_mb} MB</td>
          <td style="color: #858585;">${dateStr}</td>
          <td>${badge}</td>
          <td>
            <button class="vsc-btn vsc-btn-outline vsc-btn-xs" onclick="useCheckpointForInference('${ckpt.relative_path}')">Load Test</button>
            <button class="vsc-btn vsc-btn-outline vsc-btn-xs" onclick="useCheckpointForFinetune('${ckpt.relative_path}')">Finetune</button>
            <button class="vsc-btn vsc-btn-danger vsc-btn-xs" onclick="deleteCheckpoint('${ckpt.relative_path}')">Delete</button>
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" class="cell-empty">Failed to load: ${err.message}</td></tr>`;
  }
}

async function deleteCheckpoint(path) {
  if (!confirm(`Are you sure you want to delete checkpoint file:\n${path}?\nThis action cannot be undone.`)) {
    return;
  }
  try {
    const res = await fetch(`/api/checkpoints?path=${encodeURIComponent(path)}`, {
      method: 'DELETE',
    });
    const data = await res.json();
    if (res.ok) {
      loadCheckpoints();
    } else {
      alert(`Delete Failed: ${data.detail}`);
    }
  } catch (err) {
    alert(`Delete Error: ${err.message}`);
  }
}

function useCheckpointForFinetune(path) {
  document.getElementById('base-model-input').value = path;
  switchTab('config-tab');
}

function useCheckpointForInference(path) {
  switchTab('inference-tab');
}

// Quick Inference & Spectrogram
function initDropZone() {
  const dropZone = document.getElementById('audio-drop-zone');
  const fileInput = document.getElementById('audio-file-input');

  dropZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });
}

function handleFileSelected(file) {
  selectedAudioFile = file;
  document.getElementById('selected-file-name').textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  document.getElementById('example-audio-select').value = '';
}

function onSelectExampleAudio() {
  const sel = document.getElementById('example-audio-select');
  if (sel.value) {
    selectedAudioFile = null;
    document.getElementById('selected-file-name').textContent = `Using: ${sel.value}`;
  }
}

async function runInference() {
  const exampleVal = document.getElementById('example-audio-select').value;
  if (!selectedAudioFile && !exampleVal) {
    alert('Please upload an audio file or select an example audio first!');
    return;
  }

  const threshold = document.getElementById('infer-threshold-slider').value;
  const maxDuration = document.getElementById('infer-duration-slider').value;

  const btn = document.getElementById('btn-run-inference');
  btn.disabled = true;
  btn.innerHTML = '<span>Processing echolocation calls & spectrogram...</span>';

  const formData = new FormData();
  if (selectedAudioFile) {
    formData.append('file', selectedAudioFile);
  } else {
    formData.append('preset_path', exampleVal);
  }
  formData.append('detection_threshold', threshold);
  formData.append('max_duration', maxDuration);

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      alert(`Inference Failed: ${data.detail}`);
      return;
    }

    document.getElementById('empty-spec-placeholder').style.display = 'none';
    const specImg = document.getElementById('spectrogram-img');
    specImg.src = data.spectrogram_image;
    specImg.style.display = 'block';

    const playerBox = document.getElementById('audio-player-box');
    const player = document.getElementById('audio-player');
    if (exampleVal) {
      player.src = `/api/audio?path=${encodeURIComponent(exampleVal)}`;
      playerBox.style.display = 'block';
    }

    currentDetections = data.detections || [];
    document.getElementById('detection-count-badge').textContent = `${data.total_detections} calls detected`;
    renderDetectionsTable(currentDetections);

  } catch (err) {
    alert(`Inference Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg class="btn-icon" viewBox="0 0 24 24" fill="currentColor">
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      </svg>
      <span>Run Detection & Generate Spectrogram</span>
    `;
  }
}

function renderDetectionsTable(list) {
  const tbody = document.getElementById('detections-tbody');
  if (!list || list.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="cell-empty">No bat echolocation calls detected above threshold</td></tr>';
    return;
  }

  tbody.innerHTML = list.map(d => `
    <tr>
      <td>${d.id}</td>
      <td style="font-weight:600; color:#4ec9b0;">${d.species}</td>
      <td>${d.start_time}s</td>
      <td>${d.end_time}s</td>
      <td>${d.duration_ms} ms</td>
      <td>${d.low_freq} ~ ${d.high_freq} kHz</td>
      <td><span class="vsc-pill-green">${(d.confidence * 100).toFixed(1)}%</span></td>
    </tr>
  `).join('');
}

function exportDetectionsCSV() {
  if (!currentDetections.length) {
    alert('No detection results to export!');
    return;
  }
  const headers = ['id', 'species', 'start_time', 'end_time', 'duration_ms', 'low_freq_khz', 'high_freq_khz', 'confidence'];
  const rows = currentDetections.map(d => [
    d.id, `"${d.species}"`, d.start_time, d.end_time, d.duration_ms, d.low_freq, d.high_freq, d.confidence
  ]);
  const csvContent = 'data:text/csv;charset=utf-8,\uFEFF' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `batdetect2_detections_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function exportDetectionsJSON() {
  if (!currentDetections.length) {
    alert('No detection results to export!');
    return;
  }
  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(currentDetections, null, 2));
  const link = document.createElement('a');
  link.setAttribute('href', dataStr);
  link.setAttribute('download', `batdetect2_detections_${Date.now()}.json`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
