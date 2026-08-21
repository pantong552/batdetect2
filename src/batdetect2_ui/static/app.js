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
  loadExperimentsList();
  updateSTFTCalculations();
  updateFreqDisplay();
  updateResizeFactorDisplay();

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
      } else if (target === 'experiments-tab') {
        loadExperimentsList();
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
  const batchSelect = document.getElementById('train-batch-size');
  const precSelect = document.getElementById('train-precision');
  const trainInput = document.getElementById('train-dataset-input');
  const valInput = document.getElementById('val-dataset-input');
  const targetsInput = document.getElementById('targets-input');

  if (mode === 'test_data_4gb') {
    slider.value = 20;
    if (batchSelect) batchSelect.value = '4';
    if (precSelect) precSelect.value = '16-mixed';
    if (trainInput) trainInput.value = 'Test_data/dataset.yaml';
    if (valInput) valInput.value = 'Test_data/dataset.yaml';
    if (targetsInput) targetsInput.value = 'Test_data/targets.yaml';
  } else if (mode === 'quick') {
    slider.value = 5;
    if (batchSelect) batchSelect.value = '4';
    if (precSelect) precSelect.value = '16-mixed';
  } else if (mode === 'standard') {
    slider.value = 100;
    if (batchSelect) batchSelect.value = '4';
    if (precSelect) precSelect.value = '16-mixed';
  } else if (mode === 'deep') {
    slider.value = 200;
    if (batchSelect) batchSelect.value = '4';
    if (precSelect) precSelect.value = '16-mixed';
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

  const totalEpochs = statusData.total_epochs || 100;
  let displayEpoch = statusData.current_epoch || 0;

  if (statusData.status === 'completed') {
    displayEpoch = totalEpochs;
  } else if (isTraining && displayEpoch < totalEpochs) {
    displayEpoch = displayEpoch + 1;
  }

  document.getElementById('kpi-epoch').textContent = displayEpoch;
  document.getElementById('kpi-total-epochs').textContent = `/ ${totalEpochs}`;

  const pct = totalEpochs > 0 ? (displayEpoch / totalEpochs) * 100 : 0;
  document.getElementById('epoch-progress-bar').style.width = `${Math.min(Math.max(pct, 0), 100)}%`;

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

// Preprocessing Pipeline Helper & Real-time Calculations
function updateSTFTCalculations() {
  const samplerate = parseInt(document.getElementById('spec-samplerate').value) || 256000;
  const winDuration = parseFloat(document.getElementById('spec-win-duration').value) || 0.002;
  const winOverlap = parseFloat(document.getElementById('spec-win-overlap').value) || 0.75;

  const n_fft = Math.floor(samplerate * winDuration);
  const hop_length = Math.floor(n_fft * (1.0 - winOverlap));

  document.getElementById('spec-win-dur-display').textContent = `${winDuration}s (n_fft=${n_fft})`;
  document.getElementById('spec-win-overlap-display').textContent = `${Math.round(winOverlap * 100)}% (hop=${hop_length})`;
}

function updateFreqDisplay() {
  const minF = parseInt(document.getElementById('spec-min-freq').value) || 0;
  const maxF = parseInt(document.getElementById('spec-max-freq').value) || 0;
  document.getElementById('spec-min-freq-display').textContent = `${(minF / 1000).toFixed(0)} kHz`;
  document.getElementById('spec-max-freq-display').textContent = `${(maxF / 1000).toFixed(0)} kHz`;
}

function updateResizeFactorDisplay() {
  const rf = parseFloat(document.getElementById('spec-resize-factor').value) || 0.5;
  document.getElementById('spec-resize-factor-display').textContent = `${rf}x`;
}

function resetPreprocessingDefaults() {
  document.getElementById('spec-samplerate').value = 256000;
  document.getElementById('spec-resample-method').value = 'poly';
  document.getElementById('spec-win-duration').value = 0.002;
  document.getElementById('spec-win-overlap').value = 0.75;
  document.getElementById('spec-win-fn').value = 'hann';
  document.getElementById('spec-min-freq').value = 10000;
  document.getElementById('spec-max-freq').value = 120000;
  document.getElementById('spec-pcen-tc').value = 0.4;
  document.getElementById('spec-pcen-gain').value = 0.98;
  document.getElementById('spec-pcen-bias').value = 2.0;
  document.getElementById('spec-pcen-power').value = 0.5;
  document.getElementById('spec-sms-toggle').checked = true;
  document.getElementById('spec-resize-height').value = 128;
  document.getElementById('spec-resize-factor').value = 0.5;

  updateSTFTCalculations();
  updateFreqDisplay();
  updateResizeFactorDisplay();
}

function getPreprocessingConfigPayload() {
  const samplerate = parseInt(document.getElementById('spec-samplerate').value) || 256000;
  const resampleMethod = document.getElementById('spec-resample-method').value || 'poly';

  const winDuration = parseFloat(document.getElementById('spec-win-duration').value) || 0.002;
  const winOverlap = parseFloat(document.getElementById('spec-win-overlap').value) || 0.75;
  const winFn = document.getElementById('spec-win-fn').value || 'hann';

  const minFreq = parseInt(document.getElementById('spec-min-freq').value) || 10000;
  const maxFreq = parseInt(document.getElementById('spec-max-freq').value) || 120000;

  const pcenTc = parseFloat(document.getElementById('spec-pcen-tc').value) || 0.4;
  const pcenGain = parseFloat(document.getElementById('spec-pcen-gain').value) || 0.98;
  const pcenBias = parseFloat(document.getElementById('spec-pcen-bias').value) || 2.0;
  const pcenPower = parseFloat(document.getElementById('spec-pcen-power').value) || 0.5;
  const smsEnabled = document.getElementById('spec-sms-toggle').checked;

  const resizeHeight = parseInt(document.getElementById('spec-resize-height').value) || 128;
  const resizeFactor = parseFloat(document.getElementById('spec-resize-factor').value) || 0.5;

  const audio_config = {
    samplerate: samplerate,
    resample: {
      enabled: true,
      method: resampleMethod,
    },
  };

  const spectrogram_transforms = [
    {
      name: 'pcen',
      time_constant: pcenTc,
      gain: pcenGain,
      bias: pcenBias,
      power: pcenPower,
    }
  ];

  if (smsEnabled) {
    spectrogram_transforms.push({
      name: 'spectral_mean_subtraction',
    });
  }

  const preprocess_config = {
    audio_transforms: [],
    spectrogram_transforms: spectrogram_transforms,
    stft: {
      window_duration: winDuration,
      window_overlap: winOverlap,
      window_fn: winFn,
    },
    frequencies: {
      min_freq: minFreq,
      max_freq: maxFreq,
    },
    size: {
      name: 'resize_spec',
      height: resizeHeight,
      resize_factor: resizeFactor,
    },
  };

  return { audio_config, preprocess_config };
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

  const batch_size = parseInt(document.getElementById('train-batch-size').value) || 4;
  const precision = document.getElementById('train-precision').value || '16-mixed';
  const lr = parseFloat(document.getElementById('train-lr').value) || 0.001;
  const optimizerName = document.getElementById('train-optimizer').value || 'adam';
  const schedulerName = document.getElementById('train-scheduler').value || 'cosine_annealing';
  const checkValEvery = parseInt(document.getElementById('val-check-interval').value) || 1;

  let optimizerConfig;
  if (optimizerName === 'adamw') {
    optimizerConfig = {
      name: 'import',
      target: 'torch.optim.AdamW',
      arguments: {
        lr: lr,
        weight_decay: 0.01,
      },
    };
  } else if (optimizerName === 'sgd') {
    optimizerConfig = {
      name: 'import',
      target: 'torch.optim.SGD',
      arguments: {
        lr: lr,
        momentum: 0.9,
      },
    };
  } else {
    optimizerConfig = {
      name: 'adam',
      learning_rate: lr,
    };
  }

  const training_config = {
    trainer: {
      precision: precision,
      max_epochs: num_epochs,
      check_val_every_n_epoch: checkValEvery,
    },
    train_loader: {
      batch_size: batch_size,
      shuffle: true,
    },
    optimizer: optimizerConfig,
  };

  if (schedulerName !== 'none') {
    training_config.scheduler = {
      name: schedulerName,
      t_max: num_epochs,
    };
  }

  const { audio_config, preprocess_config } = getPreprocessingConfigPayload();

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
    audio_config,
    preprocess_config,
    training_config,
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

// ----------------------------------------------------
// EXPERIMENT LOGS (metrics.csv & hparams.yaml) VISUALIZER
// ----------------------------------------------------
let expMetricsChart = null;
let currentExpMetricsData = [];
let currentExpHparamsData = {};
let isExpLogScale = false;

function initExpChart() {
  if (expMetricsChart) return;
  const ctx = document.getElementById('expMetricsChart').getContext('2d');
  expMetricsChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: []
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
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
          title: { display: true, text: 'Metric Value', color: '#6e7681', font: { size: 10 } }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#cccccc', font: { family: 'Inter', size: 10.5 } }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: '#252526',
          borderColor: '#2d2d2d',
          borderWidth: 1,
          titleFont: { family: 'Fira Code', size: 11 },
          bodyFont: { family: 'Fira Code', size: 10 }
        }
      }
    }
  });
}

async function loadExperimentsList() {
  initExpChart();
  const select = document.getElementById('exp-run-select');
  try {
    const res = await fetch('/api/experiments');
    const list = await res.json();
    select.innerHTML = '<option value="">-- Choose Logged Experiment Run --</option>';
    if (!list || list.length === 0) {
      select.innerHTML = '<option value="">No experiment logs found in outputs/logs/</option>';
      return;
    }

    list.forEach((exp, idx) => {
      const opt = document.createElement('option');
      opt.value = exp.id;
      const dateStr = new Date(exp.created_time * 1000).toLocaleString('en-US');
      opt.textContent = `${exp.id} (${dateStr})`;
      select.appendChild(opt);
    });

    // Auto-select latest if available
    if (list.length > 0) {
      select.value = list[0].id;
      onSelectExperimentRun();
    }
  } catch (err) {
    console.warn('Failed to load experiments:', err);
  }
}

async function onSelectExperimentRun() {
  const expId = document.getElementById('exp-run-select').value;
  if (!expId) return;

  try {
    const res = await fetch(`/api/experiments/${encodeURIComponent(expId)}`);
    const data = await res.json();

    currentExpMetricsData = data.metrics || [];
    currentExpHparamsData = data.hparams || {};

    // 1. Render Hparams YAML with syntax formatting
    renderHparamsYaml(currentExpHparamsData);

    // 2. Render Chart Metrics
    renderSelectedExpMetric();

    // 3. Render Metrics Table
    renderExpMetricsTable(currentExpMetricsData);

  } catch (err) {
    alert(`Failed to load experiment details: ${err.message}`);
  }
}

function syntaxHighlightYaml(obj) {
  let str = '';
  try {
    str = JSON.stringify(obj, null, 2);
  } catch (e) {
    str = String(obj);
  }

  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
      let cls = 'yaml-num';
      if (/^"/.test(match)) {
        if (/:$/.test(match)) {
          cls = 'yaml-key';
        } else {
          cls = 'yaml-str';
        }
      } else if (/true|false/.test(match)) {
        cls = 'yaml-bool';
      }
      return '<span class="' + cls + '">' + match + '</span>';
    });
}

function renderHparamsYaml(hparams) {
  const container = document.getElementById('hparams-yaml-display');
  if (!hparams || Object.keys(hparams).length === 0) {
    container.innerHTML = '<span style="color:#858585;">No hparams.yaml found for this run.</span>';
    return;
  }
  container.innerHTML = syntaxHighlightYaml(hparams);
}

function renderSelectedExpMetric() {
  initExpChart();
  if (!expMetricsChart || !currentExpMetricsData.length) return;

  const mode = document.getElementById('exp-metric-select').value;
  const labels = currentExpMetricsData.map(r => `E${(r.epoch !== null && r.epoch !== undefined ? r.epoch : 0) + 1}`);

  let datasets = [];

  if (mode === 'total_loss') {
    datasets = [
      {
        label: 'Total Train Loss',
        data: currentExpMetricsData.map(r => r['total_loss/train']),
        borderColor: '#007acc',
        backgroundColor: 'rgba(0, 122, 204, 0.1)',
        borderWidth: 1.5,
        pointRadius: 2,
        tension: 0.2,
      },
      {
        label: 'Total Val Loss',
        data: currentExpMetricsData.map(r => r['total_loss/val']),
        borderColor: '#89d185',
        backgroundColor: 'rgba(137, 209, 133, 0.1)',
        borderWidth: 1.5,
        pointRadius: 2.5,
        tension: 0.2,
      }
    ];
  } else if (mode === 'classification_loss') {
    datasets = [
      {
        label: 'Classification Train Loss',
        data: currentExpMetricsData.map(r => r['classification_loss/train']),
        borderColor: '#9cdcfe',
        borderWidth: 1.5,
        pointRadius: 2,
        tension: 0.2,
      },
      {
        label: 'Classification Val Loss',
        data: currentExpMetricsData.map(r => r['classification_loss/val']),
        borderColor: '#dcdcaa',
        borderWidth: 1.5,
        pointRadius: 2,
        tension: 0.2,
      }
    ];
  } else if (mode === 'detection_loss') {
    datasets = [
      {
        label: 'Detection Train Loss',
        data: currentExpMetricsData.map(r => r['detection_loss/train']),
        borderColor: '#4ec9b0',
        borderWidth: 1.5,
        pointRadius: 2,
        tension: 0.2,
      },
      {
        label: 'Detection Val Loss',
        data: currentExpMetricsData.map(r => r['detection_loss/val']),
        borderColor: '#ce9178',
        borderWidth: 1.5,
        pointRadius: 2,
        tension: 0.2,
      }
    ];
  } else if (mode === 'size_loss') {
    datasets = [
      {
        label: 'BBox Size Train Loss',
        data: currentExpMetricsData.map(r => r['size_loss/train']),
        borderColor: '#c586c0',
        borderWidth: 1.5,
        pointRadius: 2,
        tension: 0.2,
      },
      {
        label: 'BBox Size Val Loss',
        data: currentExpMetricsData.map(r => r['size_loss/val']),
        borderColor: '#f48771',
        borderWidth: 1.5,
        pointRadius: 2,
        tension: 0.2,
      }
    ];
  } else if (mode === 'mAP') {
    datasets = [
      {
        label: 'Mean Average Precision (mAP)',
        data: currentExpMetricsData.map(r => r['classification/mean_average_precision']),
        borderColor: '#4fc1ff',
        backgroundColor: 'rgba(79, 193, 255, 0.15)',
        borderWidth: 2,
        pointRadius: 3,
        tension: 0.2,
        fill: true,
      },
      {
        label: 'Detection Average Precision (AP)',
        data: currentExpMetricsData.map(r => r['detection/average_precision']),
        borderColor: '#89d185',
        borderWidth: 1.5,
        pointRadius: 2.5,
        tension: 0.2,
      }
    ];
  } else if (mode === 'species_ap') {
    // Collect all species AP keys
    const first = currentExpMetricsData[0] || {};
    const speciesKeys = Object.keys(first).filter(k => k.startsWith('classification/average_precision/'));
    const colors = ['#4ec9b0', '#9cdcfe', '#ce9178', '#dcdcaa', '#c586c0', '#89d185', '#f48771'];

    datasets = speciesKeys.map((key, i) => {
      const speciesName = key.replace('classification/average_precision/', '');
      return {
        label: `AP (${speciesName})`,
        data: currentExpMetricsData.map(r => r[key]),
        borderColor: colors[i % colors.length],
        borderWidth: 1.5,
        pointRadius: 2,
        tension: 0.2,
      };
    });
  }

  expMetricsChart.data.labels = labels;
  expMetricsChart.data.datasets = datasets;
  expMetricsChart.update();
}

function toggleExpLogScale() {
  if (!expMetricsChart) return;
  isExpLogScale = !isExpLogScale;
  expMetricsChart.options.scales.y.type = isExpLogScale ? 'logarithmic' : 'linear';
  document.getElementById('btn-exp-log').classList.toggle('active', isExpLogScale);
  expMetricsChart.update();
}

function renderExpMetricsTable(data) {
  const tbody = document.getElementById('exp-metrics-tbody');
  if (!data || data.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="cell-empty">No metric records found in this run.</td></tr>';
    return;
  }

  tbody.innerHTML = data.map(r => {
    const epoch = r.epoch !== null && r.epoch !== undefined ? r.epoch + 1 : '--';
    const tLoss = r['total_loss/train'] !== null && r['total_loss/train'] !== undefined ? Number(r['total_loss/train']).toFixed(4) : '--';
    const vLoss = r['total_loss/val'] !== null && r['total_loss/val'] !== undefined ? Number(r['total_loss/val']).toFixed(4) : '--';
    const detAP = r['detection/average_precision'] !== null && r['detection/average_precision'] !== undefined ? (Number(r['detection/average_precision']) * 100).toFixed(2) + '%' : '--';
    const mAP = r['classification/mean_average_precision'] !== null && r['classification/mean_average_precision'] !== undefined ? (Number(r['classification/mean_average_precision']) * 100).toFixed(2) + '%' : '--';

    return `
      <tr>
        <td style="font-weight:600; color:#9cdcfe;">Epoch ${epoch}</td>
        <td style="color:#89d185;">${tLoss}</td>
        <td style="color:#dcdcaa;">${vLoss}</td>
        <td>${detAP}</td>
        <td><span class="vsc-pill-green">${mAP}</span></td>
      </tr>
    `;
  }).join('');
}

function exportExpMetricsCSV() {
  if (!currentExpMetricsData.length) {
    alert('No experiment metrics data to export!');
    return;
  }
  const keys = Object.keys(currentExpMetricsData[0]);
  const rows = currentExpMetricsData.map(r => keys.map(k => r[k] !== null && r[k] !== undefined ? r[k] : '').join(','));
  const csvContent = 'data:text/csv;charset=utf-8,\uFEFF' + [keys.join(','), ...rows].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  const expId = document.getElementById('exp-run-select').value || 'run';
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `${expId}_metrics_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

