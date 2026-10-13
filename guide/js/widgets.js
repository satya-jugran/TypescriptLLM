/**
 * TypescriptLLM Guide - Interactive Educational Widgets
 * Implements interactive visualizers for Tensors, RoPE, KV-Cache, BF16 Bits, Sampling & Step Simulator.
 */

// --- 1. Tensor Memory Grid Visualizer ---
export function initTensorGrid() {
  const container = document.getElementById('tensor-grid-container');
  if (!container) return;

  const rows = 2;
  const cols = 3;
  const stride0 = 3;
  const stride1 = 1;
  const sampleValues = [12.5, 4.0, -1.2, 8.8, 0.5, 99.1];

  let gridHtml = `<div class="tensor-grid" style="grid-template-columns: repeat(${cols}, 1fr)">`;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const flatIndex = r * stride0 + c * stride1;
      gridHtml += `
        <div class="tensor-cell ${r === 0 && c === 0 ? 'active' : ''}" data-row="${r}" data-col="${c}" data-flat="${flatIndex}">
          <div style="font-size:0.75rem; color:var(--text-muted)">[${r}, ${c}]</div>
          <div style="font-size:1.1rem; font-weight:bold; color:#f8fafc">${sampleValues[flatIndex]}</div>
          <div style="font-size:0.7rem; color:var(--accent-cyan)">offset: ${flatIndex}</div>
        </div>
      `;
    }
  }
  gridHtml += '</div>';

  container.innerHTML = `
    ${gridHtml}
    <div id="tensor-calc-display" style="background:var(--code-bg); padding:0.75rem 1rem; border-radius:0.5rem; font-family:monospace; font-size:0.85rem; border:1px solid var(--border-color)">
      Index: [0, 0] → Formula: <span style="color:var(--accent-cyan)">row (0) &times; stride[0] (3) + col (0) &times; stride[1] (1)</span> = <strong>Offset 0</strong> in contiguous Float32Array.
    </div>
  `;

  const cells = container.querySelectorAll('.tensor-cell');
  const calcDisplay = document.getElementById('tensor-calc-display');

  cells.forEach(cell => {
    cell.addEventListener('mouseenter', () => {
      cells.forEach(c => c.classList.remove('active'));
      cell.classList.add('active');
      const r = cell.getAttribute('data-row');
      const c = cell.getAttribute('data-col');
      const flat = cell.getAttribute('data-flat');
      if (calcDisplay) {
        calcDisplay.innerHTML = `Index: [${r}, ${c}] → Formula: <span style="color:var(--accent-cyan)">row (${r}) &times; stride[0] (3) + col (${c}) &times; stride[1] (1)</span> = <strong>Offset ${flat}</strong> in contiguous Float32Array.`;
      }
    });
  });
}

// --- 2. RoPE Unit Circle Vector Rotation ---
export function initRopeWidget() {
  const canvas = document.getElementById('rope-canvas');
  const slider = document.getElementById('rope-pos-slider');
  const posDisplay = document.getElementById('rope-pos-val');
  const angleDisplay = document.getElementById('rope-angle-val');
  const normDisplay = document.getElementById('rope-norm-val');

  if (!canvas || !slider) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const baseVector = [0.8, 0.6]; // Initial vector at pos=0 (Length = 1.0)
  const theta = 0.35; // Visual theta frequency

  function draw(pos) {
    const angle = pos * theta;
    const cosVal = Math.cos(angle);
    const sinVal = Math.sin(angle);

    // RoPE 2D rotation: [v1 * cos - v2 * sin, v2 * cos + v1 * sin]
    const rotatedX = baseVector[0] * cosVal - baseVector[1] * sinVal;
    const rotatedY = baseVector[1] * cosVal + baseVector[0] * sinVal;
    const norm = Math.hypot(rotatedX, rotatedY);

    if (posDisplay) posDisplay.innerText = pos;
    if (angleDisplay) angleDisplay.innerText = `${(angle * 180 / Math.PI).toFixed(1)}°`;
    if (normDisplay) normDisplay.innerText = norm.toFixed(3);

    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const radius = w * 0.38;

    ctx.clearRect(0, 0, w, h);

    // Draw Axes
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, cy); ctx.lineTo(w, cy);
    ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
    ctx.stroke();

    // Draw Unit Circle
    ctx.strokeStyle = '#475569';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Original Vector (Dimmed)
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + baseVector[0] * radius, cy - baseVector[1] * radius);
    ctx.stroke();

    // Draw Rotated Vector (Accent Cyan)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    const vx = cx + rotatedX * radius;
    const vy = cy - rotatedY * radius;
    ctx.lineTo(vx, vy);
    ctx.stroke();

    // Draw Arrowhead
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(vx, vy, 5, 0, 2 * Math.PI);
    ctx.fill();
  }

  slider.addEventListener('input', (e) => {
    draw(parseInt(e.target.value, 10));
  });

  draw(0);
}

// --- 3. Sampling Sandbox ---
export function initSamplingSandbox() {
  const container = document.getElementById('sampling-bars-container');
  const tempSlider = document.getElementById('temp-slider');
  const topPSlider = document.getElementById('topp-slider');
  const topKSlider = document.getElementById('topk-slider');
  const tempVal = document.getElementById('temp-val');
  const topPVal = document.getElementById('topp-val');
  const topKVal = document.getElementById('topk-val');
  const sampleBtn = document.getElementById('sample-token-btn');
  const sampleResult = document.getElementById('sample-token-result');

  if (!container || !tempSlider || !topPSlider || !topKSlider) return;

  const rawCandidates = [
    { text: "Paris", logit: 6.2 },
    { text: "France", logit: 4.8 },
    { text: "the", logit: 3.9 },
    { text: "a", logit: 3.2 },
    { text: "Europe", logit: 2.7 },
    { text: "city", logit: 2.1 },
    { text: "capital", logit: 1.5 },
  ];

  let currentDistribution = [];

  function update() {
    const temp = Math.max(0.01, parseFloat(tempSlider.value));
    const topP = parseFloat(topPSlider.value);
    const topK = parseInt(topKSlider.value, 10);

    if (tempVal) tempVal.innerText = temp.toFixed(2);
    if (topPVal) topPVal.innerText = topP.toFixed(2);
    if (topKVal) topKVal.innerText = topK;

    // 1. Temperature scaling
    const scaled = rawCandidates.map(c => ({
      text: c.text,
      exp: Math.exp(c.logit / temp)
    }));

    const sumExp = scaled.reduce((acc, c) => acc + c.exp, 0);
    let probs = scaled.map(c => ({ text: c.text, prob: c.exp / sumExp }));

    // 2. Sort descending
    probs.sort((a, b) => b.prob - a.prob);

    // 3. Top-K Filter
    probs = probs.slice(0, topK);

    // 4. Top-P (Nucleus) Filter
    let cumsum = 0;
    let cutoff = probs.length - 1;
    for (let i = 0; i < probs.length; i++) {
      cumsum += probs[i].prob;
      if (cumsum >= topP) {
        cutoff = i;
        break;
      }
    }
    const retained = probs.slice(0, cutoff + 1);

    // Re-normalize retained
    const retainedSum = retained.reduce((acc, c) => acc + c.prob, 0);
    currentDistribution = retained.map(c => ({
      text: c.text,
      prob: c.prob / retainedSum
    }));

    // Render bars
    container.innerHTML = rawCandidates.map(c => {
      const activeItem = currentDistribution.find(item => item.text === c.text);
      const isRetained = activeItem !== undefined;
      const prob = activeItem ? (activeItem.prob * 100) : 0;

      return `
        <div style="margin-bottom:0.6rem">
          <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:0.2rem">
            <span style="font-weight:600; color:${isRetained ? '#f8fafc' : '#64748b'}">
              "${c.text}" ${!isRetained ? '<span style="font-size:0.75rem; color:#ef4444">(Filtered Out)</span>' : ''}
            </span>
            <span style="font-family:monospace; color:${isRetained ? 'var(--accent-cyan)' : '#64748b'}">
              ${prob.toFixed(1)}%
            </span>
          </div>
          <div style="background:var(--border-color); height:8px; border-radius:4px; overflow:hidden">
            <div style="width:${prob}%; height:100%; background:${isRetained ? 'var(--accent-cyan)' : 'transparent'}; transition: width 0.2s"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  tempSlider.addEventListener('input', update);
  topPSlider.addEventListener('input', update);
  topKSlider.addEventListener('input', update);

  if (sampleBtn && sampleResult) {
    sampleBtn.addEventListener('click', () => {
      const rand = Math.random();
      let acc = 0;
      let chosen = currentDistribution[0]?.text ?? "None";
      for (const item of currentDistribution) {
        acc += item.prob;
        if (rand <= acc) {
          chosen = item.text;
          break;
        }
      }
      sampleResult.innerHTML = `Sampled: <strong style="color:var(--accent-green); font-size:1.1rem">"${chosen}"</strong> (Random roll: ${rand.toFixed(3)})`;
    });
  }

  update();
}

// --- 4. KV-Cache Growth Animation ---
export function initKVCacheWidget() {
  const cacheContainer = document.getElementById('kv-cache-slots');
  const appendBtn = document.getElementById('kv-append-btn');
  const resetBtn = document.getElementById('kv-reset-btn');
  const seqDisplay = document.getElementById('kv-seq-len');

  if (!cacheContainer || !appendBtn || !resetBtn) return;

  const maxSlots = 10;
  let currentTokens = ["The", "capital", "of", "France", "is"]; // initial prefill (5 tokens)
  const nextTokens = ["Paris", ".", "It", "is", "a"];
  let nextIdx = 0;

  function render() {
    if (seqDisplay) seqDisplay.innerText = currentTokens.length;

    let html = '';
    for (let i = 0; i < maxSlots; i++) {
      const isOccupied = i < currentTokens.length;
      const token = isOccupied ? currentTokens[i] : 'empty';
      html += `
        <div style="background:${isOccupied ? 'rgba(56, 189, 248, 0.15)' : 'rgba(255, 255, 255, 0.02)'}; 
                    border:1px solid ${isOccupied ? 'var(--accent-cyan)' : 'var(--border-color)'}; 
                    padding:0.5rem; border-radius:0.375rem; text-align:center; font-family:monospace; font-size:0.8rem">
          <div style="color:var(--text-muted); font-size:0.7rem">Slot ${i}</div>
          <div style="font-weight:bold; color:${isOccupied ? '#f8fafc' : '#475569'}">
            ${isOccupied ? `"${token}"` : '&mdash;'}
          </div>
          <div style="font-size:0.65rem; color:${isOccupied ? 'var(--accent-green)' : '#475569'}">
            ${isOccupied ? 'K, V cached' : 'unallocated'}
          </div>
        </div>
      `;
    }
    cacheContainer.innerHTML = html;
  }

  appendBtn.addEventListener('click', () => {
    if (currentTokens.length < maxSlots && nextIdx < nextTokens.length) {
      currentTokens.push(nextTokens[nextIdx++]);
      render();
    }
  });

  resetBtn.addEventListener('click', () => {
    currentTokens = ["The", "capital", "of", "France", "is"];
    nextIdx = 0;
    render();
  });

  render();
}

// --- 5. BF16 Bit Inspector ---
export function initBf16Widget() {
  const container = document.getElementById('bf16-bits-container');
  const valueDisplay = document.getElementById('bf16-value-display');
  const shiftDisplay = document.getElementById('bf16-shift-display');

  if (!container || !valueDisplay || !shiftDisplay) return;

  // Initial BF16 for 1.0 (0x3F80)
  // Sign: 0, Exp: 01111111 (127), Mantissa: 0000000
  let bits = [
    0, // Sign
    0, 1, 1, 1, 1, 1, 1, 1, // Exponent (8 bits)
    0, 0, 0, 0, 0, 0, 0 // Mantissa (7 bits)
  ];

  function render() {
    let uint16 = 0;
    for (let i = 0; i < 16; i++) {
      if (bits[i]) {
        uint16 |= (1 << (15 - i));
      }
    }

    // Convert via Float32 bit shift
    const uint32 = uint16 << 16;
    const f32Buf = new Float32Array(new Uint32Array([uint32]).buffer);
    const decodedF32 = f32Buf[0];

    valueDisplay.innerText = decodedF32.toPrecision(5);
    shiftDisplay.innerText = `0x${uint16.toString(16).toUpperCase().padStart(4, '0')} << 16 = 0x${uint32.toString(16).toUpperCase().padStart(8, '0')}`;

    let html = '<div style="display:flex; gap:0.25rem; flex-wrap:wrap">';
    for (let i = 0; i < 16; i++) {
      let color = '#38bdf8'; // Mantissa
      let label = 'M';
      if (i === 0) { color = '#f43f5e'; label = 'S'; } // Sign
      else if (i < 9) { color = '#a855f7'; label = 'E'; } // Exponent

      html += `
        <button class="bit-btn" data-index="${i}" style="width:34px; height:48px; border-radius:0.25rem; border:1px solid ${color}; background:${bits[i] ? color : 'transparent'}; color:${bits[i] ? '#000' : '#fff'}; cursor:pointer; font-family:monospace; font-weight:bold; font-size:0.85rem">
          <div>${bits[i]}</div>
          <div style="font-size:0.6rem; opacity:0.8">${label}</div>
        </button>
      `;
    }
    html += '</div>';
    container.innerHTML = html;

    container.querySelectorAll('.bit-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.getAttribute('data-index'), 10);
        bits[idx] = bits[idx] ? 0 : 1;
        render();
      });
    });
  }

  render();
}

// --- 6. Step-by-Step Live Inference Trace Simulator ---
let simPlayTimer = null;

export function initSimulatorWidget() {
  const promptSelect = document.getElementById('sim-prompt-select');
  const customWrap = document.getElementById('sim-custom-prompt-wrap');
  const customInput = document.getElementById('sim-custom-input');
  const customApplyBtn = document.getElementById('sim-custom-apply-btn');
  const samplerSelect = document.getElementById('sim-sampler-select');
  const speedSelect = document.getElementById('sim-speed-select');

  const playBtn = document.getElementById('sim-play-btn');
  const stepSublayerBtn = document.getElementById('sim-step-sublayer-btn');
  const stepTokenBtn = document.getElementById('sim-step-token-btn');
  const resetBtn = document.getElementById('sim-reset-btn');

  const phaseBadge = document.getElementById('sim-phase-badge');
  const statusText = document.getElementById('sim-status-text');
  const pipelineTrack = document.getElementById('pipeline-track');

  const seqCounter = document.getElementById('sim-seq-counter');
  const tokenStream = document.getElementById('sim-token-stream');
  const tokenInspector = document.getElementById('sim-token-inspector');

  const layerCounter = document.getElementById('sim-layer-counter');
  const sublayerDetails = document.getElementById('sim-sublayer-details');

  const logitsContainer = document.getElementById('sim-logits-container');
  const telemetryContainer = document.getElementById('sim-telemetry-container');

  if (!promptSelect || !playBtn || !stepSublayerBtn || !pipelineTrack) return;

  // Clear any existing timer from previous visits
  if (simPlayTimer) {
    clearInterval(simPlayTimer);
    simPlayTimer = null;
  }

  // Pre-computed realistic token scenarios (Vocabulary: SmolLM-135M / LLaMA BPE)
  const scenarios = {
    paris: {
      name: "The capital of France is",
      promptTokens: [
        { text: "The", id: 19556, pos: 0, isPrompt: true },
        { text: " capital", id: 905, pos: 1, isPrompt: true },
        { text: " of", id: 17, pos: 2, isPrompt: true },
        { text: " France", id: 15169, pos: 3, isPrompt: true },
        { text: " is", id: 3511, pos: 4, isPrompt: true },
      ],
      genTokens: [
        {
          text: " Paris",
          id: 15169,
          candidates: [
            { text: " Paris", id: 15169, prob: 0.884, logit: 14.82 },
            { text: " Lyon", id: 24810, prob: 0.042, logit: 11.75 },
            { text: " a", id: 264, prob: 0.025, logit: 11.23 },
            { text: " France", id: 15169, prob: 0.018, logit: 10.91 },
            { text: " known", id: 3740, prob: 0.009, logit: 10.20 },
          ],
        },
        {
          text: ".",
          id: 28723,
          candidates: [
            { text: ".", id: 28723, prob: 0.941, logit: 16.24 },
            { text: ",", id: 28725, prob: 0.032, logit: 12.81 },
            { text: " and", id: 322, prob: 0.011, logit: 11.70 },
            { text: " (", id: 357, prob: 0.008, logit: 11.42 },
            { text: " -", id: 546, prob: 0.004, logit: 10.75 },
          ],
        },
        {
          text: " It",
          id: 2119,
          candidates: [
            { text: " It", id: 2119, prob: 0.685, logit: 13.90 },
            { text: " The", id: 450, prob: 0.182, logit: 12.60 },
            { text: " Paris", id: 15169, prob: 0.063, logit: 11.50 },
            { text: " Located", id: 18451, prob: 0.031, logit: 10.80 },
            { text: " With", id: 2261, prob: 0.019, logit: 10.30 },
          ],
        },
        {
          text: " is",
          id: 3511,
          candidates: [
            { text: " is", id: 3511, prob: 0.827, logit: 15.10 },
            { text: " has", id: 704, prob: 0.091, logit: 12.90 },
            { text: " was", id: 435, prob: 0.042, logit: 12.10 },
            { text: " serves", id: 9812, prob: 0.018, logit: 11.30 },
            { text: " also", id: 721, prob: 0.012, logit: 10.90 },
          ],
        },
        {
          text: " the",
          id: 278,
          candidates: [
            { text: " the", id: 278, prob: 0.742, logit: 14.50 },
            { text: " also", id: 721, prob: 0.123, logit: 12.70 },
            { text: " known", id: 3740, prob: 0.054, logit: 11.90 },
            { text: " located", id: 4210, prob: 0.041, logit: 11.60 },
            { text: " one", id: 574, prob: 0.020, logit: 10.90 },
          ],
        },
      ],
    },
    water: {
      name: "The boiling point of water is",
      promptTokens: [
        { text: "The", id: 19556, pos: 0, isPrompt: true },
        { text: " boiling", id: 32904, pos: 1, isPrompt: true },
        { text: " point", id: 1422, pos: 2, isPrompt: true },
        { text: " of", id: 17, pos: 3, isPrompt: true },
        { text: " water", id: 3058, pos: 4, isPrompt: true },
        { text: " is", id: 3511, pos: 5, isPrompt: true },
      ],
      genTokens: [
        {
          text: " 100",
          id: 28740,
          candidates: [
            { text: " 100", id: 28740, prob: 0.912, logit: 15.60 },
            { text: " 212", id: 28784, prob: 0.054, logit: 12.80 },
            { text: " approximately", id: 4310, prob: 0.018, logit: 11.70 },
            { text: " around", id: 1840, prob: 0.009, logit: 11.00 },
            { text: " variable", id: 13200, prob: 0.003, logit: 9.90 },
          ],
        },
        {
          text: " degrees",
          id: 9876,
          candidates: [
            { text: " degrees", id: 9876, prob: 0.783, logit: 14.20 },
            { text: " °C", id: 18940, prob: 0.145, logit: 12.50 },
            { text: " K", id: 412, prob: 0.032, logit: 11.00 },
            { text: " C", id: 345, prob: 0.021, logit: 10.60 },
            { text: " percent", id: 3211, prob: 0.006, logit: 9.30 },
          ],
        },
        {
          text: " Celsius",
          id: 23150,
          candidates: [
            { text: " Celsius", id: 23150, prob: 0.859, logit: 14.90 },
            { text: " Fahrenheit", id: 27120, prob: 0.101, logit: 12.70 },
            { text: " (", id: 357, prob: 0.024, logit: 11.30 },
            { text: ",", id: 28725, prob: 0.008, logit: 10.20 },
            { text: " at", id: 488, prob: 0.005, logit: 9.70 },
          ],
        },
        {
          text: ".",
          id: 28723,
          candidates: [
            { text: ".", id: 28723, prob: 0.954, logit: 16.50 },
            { text: " (", id: 357, prob: 0.025, logit: 12.80 },
            { text: " at", id: 488, prob: 0.011, logit: 11.90 },
            { text: ",", id: 28725, prob: 0.006, logit: 11.30 },
            { text: " under", id: 1205, prob: 0.002, logit: 10.20 },
          ],
        },
      ],
    },
    math: {
      name: "2 + 2 =",
      promptTokens: [
        { text: "2", id: 28750, pos: 0, isPrompt: true },
        { text: " +", id: 714, pos: 1, isPrompt: true },
        { text: " 2", id: 28750, pos: 2, isPrompt: true },
        { text: " =", id: 788, pos: 3, isPrompt: true },
      ],
      genTokens: [
        {
          text: " 4",
          id: 28782,
          candidates: [
            { text: " 4", id: 28782, prob: 0.968, logit: 17.10 },
            { text: " 5", id: 28785, prob: 0.012, logit: 12.70 },
            { text: " 22", id: 18450, prob: 0.008, logit: 12.30 },
            { text: " four", id: 3201, prob: 0.006, logit: 12.00 },
            { text: " 0", id: 28740, prob: 0.003, logit: 11.30 },
          ],
        },
        {
          text: ".",
          id: 28723,
          candidates: [
            { text: ".", id: 28723, prob: 0.891, logit: 15.80 },
            { text: ",", id: 28725, prob: 0.052, logit: 13.00 },
            { text: "\\n", id: 13, prob: 0.034, logit: 12.50 },
            { text: "!", id: 28742, prob: 0.012, logit: 11.50 },
            { text: " It", id: 2119, prob: 0.007, logit: 11.00 },
          ],
        },
        {
          text: " That",
          id: 2191,
          candidates: [
            { text: " That", id: 2191, prob: 0.584, logit: 13.50 },
            { text: " This", id: 1115, prob: 0.245, logit: 12.60 },
            { text: " which", id: 607, prob: 0.082, logit: 11.50 },
            { text: " is", id: 3511, prob: 0.051, logit: 11.00 },
            { text: " math", id: 7812, prob: 0.021, logit: 10.10 },
          ],
        },
        {
          text: " is",
          id: 3511,
          candidates: [
            { text: " is", id: 3511, prob: 0.842, logit: 15.00 },
            { text: " means", id: 2315, prob: 0.078, logit: 12.40 },
            { text: " was", id: 435, prob: 0.042, logit: 11.80 },
            { text: " makes", id: 3410, prob: 0.021, logit: 11.10 },
            { text: " equals", id: 14210, prob: 0.011, logit: 10.40 },
          ],
        },
        {
          text: " correct",
          id: 6398,
          candidates: [
            { text: " correct", id: 6398, prob: 0.795, logit: 14.70 },
            { text: " true", id: 2012, prob: 0.124, logit: 12.90 },
            { text: " right", id: 1340, prob: 0.052, logit: 12.00 },
            { text: " basic", id: 3105, prob: 0.018, logit: 10.90 },
            { text: " math", id: 7812, prob: 0.007, logit: 10.00 },
          ],
        },
      ],
    },
    sun: {
      name: "The sun rises in the",
      promptTokens: [
        { text: "The", id: 19556, pos: 0, isPrompt: true },
        { text: " sun", id: 8567, pos: 1, isPrompt: true },
        { text: " rises", id: 34506, pos: 2, isPrompt: true },
        { text: " in", id: 297, pos: 3, isPrompt: true },
        { text: " the", id: 278, pos: 4, isPrompt: true },
      ],
      genTokens: [
        {
          text: " east",
          id: 6185,
          candidates: [
            { text: " east", id: 6185, prob: 0.974, logit: 17.50 },
            { text: " morning", id: 3340, prob: 0.015, logit: 13.30 },
            { text: " sky", id: 6710, prob: 0.006, logit: 12.40 },
            { text: " East", id: 7210, prob: 0.003, logit: 11.70 },
            { text: " horizon", id: 14205, prob: 0.001, logit: 10.70 },
          ],
        },
        {
          text: " and",
          id: 322,
          candidates: [
            { text: " and", id: 322, prob: 0.723, logit: 14.10 },
            { text: ",", id: 28725, prob: 0.185, logit: 12.70 },
            { text: ".", id: 28723, prob: 0.062, logit: 11.60 },
            { text: " every", id: 1612, prob: 0.018, logit: 10.40 },
            { text: " sets", id: 15891, prob: 0.009, logit: 9.70 },
          ],
        },
        {
          text: " sets",
          id: 15891,
          candidates: [
            { text: " sets", id: 15891, prob: 0.862, logit: 15.00 },
            { text: " goes", id: 2450, prob: 0.073, logit: 12.50 },
            { text: " brings", id: 6712, prob: 0.031, logit: 11.70 },
            { text: " shines", id: 18920, prob: 0.021, logit: 11.30 },
            { text: " disappears", id: 23100, prob: 0.008, logit: 10.30 },
          ],
        },
        {
          text: " in",
          id: 297,
          candidates: [
            { text: " in", id: 297, prob: 0.925, logit: 16.00 },
            { text: " down", id: 1245, prob: 0.043, logit: 12.80 },
            { text: " at", id: 488, prob: 0.018, logit: 11.90 },
            { text: " on", id: 384, prob: 0.009, logit: 11.20 },
            { text: " towards", id: 4510, prob: 0.003, logit: 10.10 },
          ],
        },
        {
          text: " the",
          id: 278,
          candidates: [
            { text: " the", id: 278, prob: 0.948, logit: 16.40 },
            { text: " evening", id: 6240, prob: 0.031, logit: 13.00 },
            { text: " west", id: 4976, prob: 0.012, logit: 12.10 },
            { text: " night", id: 2410, prob: 0.006, logit: 11.40 },
            { text: " horizon", id: 14205, prob: 0.002, logit: 10.30 },
          ],
        },
        {
          text: " west",
          id: 4976,
          candidates: [
            { text: " west", id: 4976, prob: 0.982, logit: 17.80 },
            { text: " West", id: 7510, prob: 0.009, logit: 13.10 },
            { text: " ocean", id: 8940, prob: 0.004, logit: 12.30 },
            { text: " evening", id: 6240, prob: 0.003, logit: 12.00 },
            { text: " horizon", id: 14205, prob: 0.001, logit: 10.90 },
          ],
        },
      ],
    },
  };

  // Build custom scenario generator
  function createCustomScenario(text) {
    const words = text.trim().split(/\s+/).filter(Boolean);
    const promptTokens = words.map((w, idx) => ({
      text: (idx === 0 ? "" : " ") + w,
      id: 1000 + (Math.abs(w.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0) * 31) % 40000),
      pos: idx,
      isPrompt: true,
    }));

    return {
      name: text,
      promptTokens,
      genTokens: [
        {
          text: " a",
          id: 264,
          candidates: [
            { text: " a", id: 264, prob: 0.725, logit: 14.20 },
            { text: " the", id: 278, prob: 0.152, logit: 12.50 },
            { text: " very", id: 1240, prob: 0.063, logit: 11.60 },
            { text: " not", id: 539, prob: 0.038, logit: 11.10 },
            { text: " rapidly", id: 18450, prob: 0.015, logit: 10.20 },
          ],
        },
        {
          text: " powerful",
          id: 4890,
          candidates: [
            { text: " powerful", id: 4890, prob: 0.812, logit: 15.10 },
            { text: " fast", id: 3210, prob: 0.095, logit: 12.90 },
            { text: " new", id: 602, prob: 0.052, logit: 12.30 },
            { text: " useful", id: 6710, prob: 0.024, logit: 11.50 },
            { text: " key", id: 1840, prob: 0.012, logit: 10.80 },
          ],
        },
        {
          text: " tool",
          id: 3450,
          candidates: [
            { text: " tool", id: 3450, prob: 0.865, logit: 15.30 },
            { text: " system", id: 1420, prob: 0.071, logit: 12.60 },
            { text: " technology", id: 3510, prob: 0.038, logit: 12.00 },
            { text: " model", id: 2740, prob: 0.018, logit: 11.20 },
            { text: " concept", id: 7120, prob: 0.006, logit: 10.10 },
          ],
        },
        {
          text: ".",
          id: 28723,
          candidates: [
            { text: ".", id: 28723, prob: 0.945, logit: 16.40 },
            { text: " for", id: 363, prob: 0.031, logit: 12.70 },
            { text: " that", id: 393, prob: 0.014, logit: 11.80 },
            { text: ",", id: 28725, prob: 0.007, logit: 11.10 },
            { text: " in", id: 297, prob: 0.002, logit: 9.80 },
          ],
        },
      ],
    };
  }

  // Simulator Runtime State
  let currentKey = promptSelect.value;
  let activeScenario = currentKey === 'custom' ? createCustomScenario(customInput.value) : scenarios[currentKey];
  let tokenStepIdx = 0; // index into genTokens (0 .. genTokens.length)
  let sublayerIdx = 0;  // sublayer stage index (0 .. 5)
  let isPlaying = false;
  let selectedToken = null;

  // The 6 Sub-layer Pipeline Stages
  const stageDefs = [
    {
      key: 'token',
      name: '01 Tokenize',
      shortName: 'Tokenize',
      title: 'Stage 01: Tokenization / Sequence Dispatch',
      badge: 'Input Phase',
      badgeClass: 'badge-cyan',
      code: 'src/tokenizer/hf-tokenizer.ts',
      func: 'encode(prompt) / decodeToken(id)',
      getShape: (isPrefill, S) => isPrefill ? `[${S} uint32 IDs]` : `[1 uint32 ID]`,
      getFormula: (isPrefill) => isPrefill
        ? `Prompt text → BPE Partition → Array of uint32 Token IDs`
        : `Token ID t<sub>m</sub> fed at sequence position startPos = m`,
      getDesc: (isPrefill, S) => isPrefill
        ? `Partitioning the prompt string into ${S} discrete BPE token IDs. Memory buffers initialized for prefill.`
        : `Autoregressive step: Feeding the single newly generated token into the model. Past ${S} tokens are already preserved in the KV-cache.`,
    },
    {
      key: 'embed',
      name: '02 Embed',
      shortName: 'Embed',
      title: 'Stage 02: Embedding Table Vector Slicing',
      badge: 'Weight Slicing',
      badgeClass: 'badge-cyan',
      code: 'src/model/transformer.ts',
      func: 'embed_tokens.weight[token_ids]',
      getShape: (isPrefill, S) => isPrefill ? `[${S}, 576]` : `[1, 576]`,
      getFormula: () => `X[i] = embed_tokens.weight[token_id[i]] &isin; R<sup>576</sup>`,
      getDesc: (isPrefill, S) => isPrefill
        ? `Looking up contiguous 576-dimensional vectors for all ${S} prompt tokens from the 49,152 × 576 embedding table.`
        : `Slicing a single 576-float hidden activation vector for the current token. Zero allocations; writes to contiguous Float32Array.`,
    },
    {
      key: 'blocks',
      name: '03 30x Blocks',
      shortName: '30x Blocks',
      title: 'Stage 03: 30x Transformer Blocks (Attn + RoPE + SwiGLU)',
      badge: 'Deep Network',
      badgeClass: 'badge-purple',
      code: 'src/layers/transformer-block.ts',
      func: 'TransformerBlock.forward(x, startPos, kvCache)',
      getShape: (isPrefill, S) => isPrefill ? `[${S}, 576]` : `[1, 576]`,
      getFormula: () => `Y = X + Attention(RMS(X)) + SwiGLU(RMS(X))`,
      getDesc: (isPrefill) => isPrefill
        ? `Parallel processing through all 30 layers: Pre-RMSNorm, Rotary Embeddings (RoPE), Grouped-Query Attention (9 Q heads, 3 KV heads), KV-cache insertion, and SwiGLU MLP.`
        : `Single-token decode pass through 30 layers: 8-way unrolled GEMV (Y = X·W<sup>T</sup>). Queries attend to cached Keys & Values from tokens 0..m without recomputing past layers!`,
    },
    {
      key: 'norm',
      name: '04 Final Norm',
      shortName: 'Final Norm',
      title: 'Stage 04: Final Layer RMSNorm',
      badge: 'Normalization',
      badgeClass: 'badge-purple',
      code: 'src/layers/rms-norm.ts',
      func: 'RMSNorm.forward(x)',
      getShape: (isPrefill, S) => isPrefill ? `[${S}, 576]` : `[1, 576]`,
      getFormula: () => `RMS(x) = &radic;(mean(x<sup>2</sup>) + &epsilon;);  y = (x / RMS(x)) &middot; &gamma;`,
      getDesc: () => `Standardizing activations across the 576 hidden dimension with RMS scaling factor &gamma; to stabilize logits prior to vocabulary projection.`,
    },
    {
      key: 'lmhead',
      name: '05 LM Head',
      shortName: 'LM Head',
      title: 'Stage 05: Vocabulary Logits Projection (LM Head)',
      badge: 'GEMV Projection',
      badgeClass: 'badge-amber',
      code: 'src/model/transformer.ts',
      func: 'lm_head.forward(last_hidden_state)',
      getShape: () => `[1, 576] → [1, 49152]`,
      getFormula: () => `Logits = X<sub>last</sub> &middot; W<sub>head</sub><sup>T</sup> &isin; R<sup>49152</sup>`,
      getDesc: () => `Crucial TypescriptLLM optimization: We project only the LAST token hidden state (M = 1) into vocabulary logits, saving 80% matrix multiply time during prefill!`,
    },
    {
      key: 'sample',
      name: '06 Sample',
      shortName: 'Sample',
      title: 'Stage 06: Sampling Strategy & Emission',
      badge: 'Token Decided',
      badgeClass: 'badge-green',
      code: 'src/samplers/sampler-factory.ts',
      func: 'sampler.sample(logits, contextTokens)',
      getShape: () => `[1, 49152] → [1] Winner Token`,
      getFormula: () => `P = Softmax(Logits / Temperature) → Argmax / Top-P Mass`,
      getDesc: () => `Applying temperature scaling and selection filter. The winning token ID is emitted, appended to the output buffer, and written to the KV cache sequence!`,
    },
  ];

  function renderAll() {
    const isPrefill = tokenStepIdx === 0;
    const currentSeqLen = activeScenario.promptTokens.length + (sublayerIdx >= 5 ? tokenStepIdx + 1 : tokenStepIdx);
    const S = isPrefill ? activeScenario.promptTokens.length : 1;
    const isCompleted = tokenStepIdx >= activeScenario.genTokens.length;

    // 1. Phase Badge & Status Text
    if (isCompleted) {
      phaseBadge.className = 'badge badge-green';
      phaseBadge.innerText = 'Generation Complete';
      statusText.innerText = `Finished (${currentSeqLen} tokens generated)`;
    } else if (isPrefill) {
      phaseBadge.className = 'badge badge-cyan';
      phaseBadge.innerText = 'Prefill Phase';
      statusText.innerText = `Prefill: Processing Prompt (Pos 0..${activeScenario.promptTokens.length - 1})`;
    } else {
      phaseBadge.className = 'badge badge-green';
      phaseBadge.innerText = 'Decode Phase';
      statusText.innerText = `Decode: Generating Token ${tokenStepIdx + 1} of ${activeScenario.genTokens.length}`;
    }

    // 2. Pipeline Track
    let trackHtml = '';
    stageDefs.forEach((st, idx) => {
      let nodeClass = 'pipeline-step-node';
      if (isCompleted) {
        nodeClass += ' completed';
      } else if (idx === sublayerIdx) {
        nodeClass += ' active';
      } else if (idx < sublayerIdx) {
        nodeClass += ' completed';
      }

      const shapeStr = st.getShape(isPrefill, S);
      trackHtml += `
        <div class="${nodeClass}" data-stage-idx="${idx}">
          <div class="step-num">${idx + 1 < 10 ? '0' + (idx + 1) : idx + 1}</div>
          <div class="step-name">${st.shortName}</div>
          <div class="step-shape">${shapeStr}</div>
        </div>
      `;
      if (idx < stageDefs.length - 1) {
        trackHtml += `<div class="pipeline-step-connector">→</div>`;
      }
    });
    pipelineTrack.innerHTML = trackHtml;

    // 3. Token Sequence Stream
    seqCounter.innerText = `${currentSeqLen} / 2048 tokens (Pos 0..${currentSeqLen - 1})`;
    let tokensHtml = '';

    // Prompt tokens
    activeScenario.promptTokens.forEach((t) => {
      const isSel = selectedToken && selectedToken.pos === t.pos;
      tokensHtml += `
        <span class="sim-token-pill prompt-token ${isSel ? 'selected' : ''}" data-token-id="${t.id}" data-pos="${t.pos}" data-is-prompt="true">
          ${t.text}
        </span>
      `;
    });

    // Generated tokens
    const numGenToShow = sublayerIdx >= 5 ? tokenStepIdx + 1 : tokenStepIdx;
    for (let i = 0; i < numGenToShow && i < activeScenario.genTokens.length; i++) {
      const gt = activeScenario.genTokens[i];
      const pos = activeScenario.promptTokens.length + i;
      const isSel = selectedToken && selectedToken.pos === pos;
      const isLatest = i === numGenToShow - 1 && !isCompleted;
      tokensHtml += `
        <span class="sim-token-pill gen-token ${isLatest ? 'new-pulse' : ''} ${isSel ? 'selected' : ''}" data-token-id="${gt.id}" data-pos="${pos}" data-is-prompt="false" data-gen-idx="${i}">
          ${gt.text}
        </span>
      `;
    }

    tokenStream.innerHTML = tokensHtml;

    // Bind token clicks for inspection
    tokenStream.querySelectorAll('.sim-token-pill').forEach(pill => {
      pill.addEventListener('click', () => {
        const id = parseInt(pill.getAttribute('data-token-id'), 10);
        const pos = parseInt(pill.getAttribute('data-pos'), 10);
        const isPrompt = pill.getAttribute('data-is-prompt') === 'true';
        const text = pill.innerText;
        selectedToken = { id, pos, isPrompt, text };
        renderAll();
      });
    });

    // Token Inspector
    if (selectedToken) {
      tokenInspector.style.display = 'block';
      tokenInspector.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem">
          <div><strong style="color:var(--accent-cyan)">Token "${selectedToken.text}"</strong> (ID: <span style="color:var(--accent-amber)">${selectedToken.id}</span>)</div>
          <span class="badge ${selectedToken.isPrompt ? 'badge-cyan' : 'badge-green'}">${selectedToken.isPrompt ? 'Prompt Token' : 'Generated Token'}</span>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.4rem; font-size:0.8rem; color:#cbd5e1">
          <div>Sequence Position (<em>m</em>): <strong style="color:var(--accent-cyan)">${selectedToken.pos}</strong></div>
          <div>Embedding Dim: <strong>576 float32</strong></div>
          <div>KV Cache Slot: <strong style="color:var(--accent-green)">Layers 0..29 [Index ${selectedToken.pos}]</strong></div>
          <div>Byte Size: <strong>${new Blob([selectedToken.text]).size} bytes</strong></div>
        </div>
      `;
    } else {
      tokenInspector.style.display = 'none';
    }

    // 4. Sub-Layer Execution Details
    const curStage = isCompleted ? stageDefs[5] : stageDefs[sublayerIdx];
    layerCounter.innerText = isCompleted
      ? 'All 30 Layers Done'
      : (sublayerIdx === 2 ? 'Layers 0..29 Active' : 'Global Transformer');

    sublayerDetails.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem">
        <h4 style="color:var(--accent-cyan); font-size:1.05rem; margin:0">${curStage.title}</h4>
        <span class="badge ${curStage.badgeClass}">${curStage.badge}</span>
      </div>
      <p style="color:#cbd5e1; font-size:0.88rem; margin-bottom:0.75rem">${curStage.getDesc(isPrefill, S)}</p>

      <div style="background:rgba(255,255,255,0.03); border:1px solid var(--border-color); border-radius:0.35rem; padding:0.6rem 0.75rem; margin-bottom:0.75rem; font-family:monospace; font-size:0.82rem">
        <div style="color:var(--text-muted); font-size:0.72rem; text-transform:uppercase">Mathematical Operation:</div>
        <div style="color:#f8fafc; margin-top:0.15rem">${curStage.getFormula(isPrefill)}</div>
      </div>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; font-family:monospace; font-size:0.78rem">
        <div style="background:rgba(255,255,255,0.02); padding:0.5rem; border-radius:0.35rem; border:1px solid var(--border-color)">
          <span style="color:var(--text-muted)">Tensor Shape:</span> <span style="color:var(--accent-amber)">${curStage.getShape(isPrefill, S)}</span>
        </div>
        <div style="background:rgba(255,255,255,0.02); padding:0.5rem; border-radius:0.35rem; border:1px solid var(--border-color)">
          <span style="color:var(--text-muted)">Source:</span> <span style="color:var(--accent-cyan)">${curStage.code}</span>
        </div>
      </div>
    `;

    // 5. Candidate Probabilities Distribution (Top-5 Bars)
    const targetGenToken = activeScenario.genTokens[Math.min(tokenStepIdx, activeScenario.genTokens.length - 1)];
    const showLogits = sublayerIdx >= 4 || isCompleted;

    if (!showLogits || !targetGenToken) {
      logitsContainer.innerHTML = `
        <div style="text-align:center; padding:1.5rem; color:var(--text-muted); font-size:0.88rem">
          <div style="font-size:1.5rem; margin-bottom:0.4rem">⚡</div>
          Hidden activations are propagating through 30 Transformer Blocks.<br>
          Vocabulary logits will be computed at <strong>Stage 05 (LM Head)</strong>.
        </div>
      `;
    } else {
      const samplerType = samplerSelect.value;
      const candidates = targetGenToken.candidates || [];
      const winnerId = targetGenToken.id;

      let rowsHtml = '';
      candidates.forEach((cand, cIdx) => {
        const isWinner = cand.id === winnerId;
        const probPct = (cand.prob * 100).toFixed(1);
        rowsHtml += `
          <div class="sim-candidate-row ${isWinner ? 'winner' : ''}">
            <div class="sim-candidate-token" title="${cand.text}">"${cand.text}"</div>
            <div class="sim-bar-wrapper">
              <div class="sim-bar-fill" style="width:${probPct}%"></div>
            </div>
            <div class="sim-candidate-prob">${probPct}%</div>
            <div class="sim-candidate-logit">logit: ${cand.logit.toFixed(1)}</div>
            ${isWinner ? '<span style="color:var(--accent-green); font-size:0.8rem; font-weight:bold">✓ Winner</span>' : ''}
          </div>
        `;
      });

      const noteText = samplerType === 'greedy'
        ? `Greedy Sampler (Argmax, T=0.0): Deterministically picks token <strong>"${targetGenToken.text}"</strong> (${(candidates[0]?.prob * 100 || 0).toFixed(1)}%) with highest raw logit.`
        : `Nucleus Sampler (T=0.7, Top-P=0.9): Cumulative probability mass concentrated in top candidates. Sampled token: <strong>"${targetGenToken.text}"</strong>.`;

      logitsContainer.innerHTML = `
        <div>
          ${rowsHtml}
          <div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.6rem; border-top:1px solid rgba(255,255,255,0.05); padding-top:0.4rem">
            ${noteText}
          </div>
        </div>
      `;
    }

    // 6. Hardware Execution Telemetry
    const kvSlotBytes = currentSeqLen * 192 * 4 * 2 * 30; // 30 layers * 2(K,V) * 192 floats * 4 bytes
    const kvKb = (kvSlotBytes / 1024).toFixed(1);

    telemetryContainer.innerHTML = `
      <div class="sim-telemetry-grid">
        <div class="sim-telemetry-item">
          <div class="sim-telemetry-label">Execution Phase</div>
          <div class="sim-telemetry-value" style="color:${isPrefill ? 'var(--accent-cyan)' : 'var(--accent-green)'}">
            ${isPrefill ? 'Prefill (Compute-Bound GEMM)' : 'Decode (Bandwidth-Bound GEMV)'}
          </div>
        </div>
        <div class="sim-telemetry-item">
          <div class="sim-telemetry-label">KV-Cache Buffer</div>
          <div class="sim-telemetry-value">${currentSeqLen} / 2048 slots (${kvKb} KB active)</div>
        </div>
        <div class="sim-telemetry-item">
          <div class="sim-telemetry-label">CPU Dispatch Mode</div>
          <div class="sim-telemetry-value">Float32Array Unrolled 8x (Single Thread)</div>
        </div>
        <div class="sim-telemetry-item">
          <div class="sim-telemetry-label">Simulated Throughput</div>
          <div class="sim-telemetry-value" style="color:var(--accent-amber)">~5.8 tokens/sec (~172 ms/tok)</div>
        </div>
      </div>
    `;

    // 7. Update Control Buttons
    playBtn.innerText = isPlaying ? '⏸ Pause' : (isCompleted ? '✓ Done' : '▶ Auto Play');
    playBtn.disabled = isCompleted;
    stepSublayerBtn.disabled = isCompleted;
    stepTokenBtn.disabled = isCompleted;
  }

  function advanceSublayer() {
    if (tokenStepIdx >= activeScenario.genTokens.length) {
      stopAutoPlay();
      renderAll();
      return;
    }

    if (sublayerIdx < 5) {
      sublayerIdx++;
    } else {
      // Completed full cycle of this token
      tokenStepIdx++;
      sublayerIdx = 0;
      if (tokenStepIdx >= activeScenario.genTokens.length) {
        stopAutoPlay();
      }
    }
    renderAll();
  }

  function advanceFullToken() {
    if (tokenStepIdx >= activeScenario.genTokens.length) return;
    tokenStepIdx++;
    sublayerIdx = 0;
    if (tokenStepIdx >= activeScenario.genTokens.length) {
      stopAutoPlay();
    }
    renderAll();
  }

  function startAutoPlay() {
    isPlaying = true;
    const speedMs = parseInt(speedSelect.value, 10) || 600;
    playBtn.innerText = '⏸ Pause';
    if (simPlayTimer) clearInterval(simPlayTimer);
    simPlayTimer = setInterval(() => {
      advanceSublayer();
    }, speedMs);
  }

  function stopAutoPlay() {
    isPlaying = false;
    if (simPlayTimer) {
      clearInterval(simPlayTimer);
      simPlayTimer = null;
    }
    playBtn.innerText = tokenStepIdx >= activeScenario.genTokens.length ? '✓ Done' : '▶ Auto Play';
  }

  function resetAll() {
    stopAutoPlay();
    tokenStepIdx = 0;
    sublayerIdx = 0;
    selectedToken = null;
    renderAll();
  }

  // Bind Events
  playBtn.onclick = () => {
    if (isPlaying) {
      stopAutoPlay();
    } else {
      if (tokenStepIdx >= activeScenario.genTokens.length) {
        tokenStepIdx = 0;
        sublayerIdx = 0;
      }
      startAutoPlay();
    }
    renderAll();
  };

  stepSublayerBtn.onclick = () => {
    stopAutoPlay();
    advanceSublayer();
  };

  stepTokenBtn.onclick = () => {
    stopAutoPlay();
    advanceFullToken();
  };

  resetBtn.onclick = () => {
    resetAll();
  };

  speedSelect.onchange = () => {
    if (isPlaying) {
      startAutoPlay();
    }
  };

  promptSelect.onchange = () => {
    currentKey = promptSelect.value;
    if (currentKey === 'custom') {
      customWrap.style.display = 'block';
      activeScenario = createCustomScenario(customInput.value);
    } else {
      customWrap.style.display = 'none';
      activeScenario = scenarios[currentKey];
    }
    resetAll();
  };

  if (customApplyBtn) {
    customApplyBtn.onclick = () => {
      activeScenario = createCustomScenario(customInput.value);
      resetAll();
    };
  }

  samplerSelect.onchange = () => {
    renderAll();
  };

  // Initial render
  renderAll();
}

// --- 7. Interactive Transformer Block Architecture Diagram Inspector ---
export function initArchitectureDiagram() {
  const container = document.getElementById('transformer-svg') || document.getElementById('transformer-flowchart');
  const inspector = document.getElementById('diagram-inspector-content');
  if (!container || !inspector) return;

  const blockDetails = {
    'input': {
      title: 'Input Activations (X)',
      badge: 'Input State',
      badgeClass: 'badge-cyan',
      description: 'The sequence of token embeddings (from the embedding table in Layer 0) or the output activations from the preceding TransformerBlock (Layer i - 1).',
      shape: '[1, 576] (Decode) or [S, 576] (Prefill)',
      formula: 'X &isin; R<sup>S &times; 576</sup>',
      code: 'src/layers/transformer-block.ts',
      func: 'forward(x, startPos, kvCache)'
    },
    'rmsnorm1': {
      title: 'RMSNorm 1 (Pre-Attention)',
      badge: 'Normalization',
      badgeClass: 'badge-purple',
      description: 'Root Mean Square Layer Normalization. Normalizes activation variance across hidden_size (576) to stabilize activations and prevent vanishing/exploding gradients.',
      shape: '[S, 576] → [S, 576]',
      formula: 'RMS(x) = &radic;( mean(x<sup>2</sup>) + &epsilon; );  y = (x / RMS(x)) &middot; w',
      code: 'src/layers/rms-norm.ts',
      func: 'RMSNorm.forward(x)'
    },
    'attn': {
      title: 'Grouped-Query Attention (GQA)',
      badge: 'Attention Mechanism',
      badgeClass: 'badge-cyan',
      description: 'Mixes contextual information across sequence tokens. Computes Query (W_q), Key (W_k), and Value (W_v) projections. Applies RoPE to Q and K, stores/retrieves from the KV-Cache, evaluates causal scaled dot-product attention across 9 query heads (grouped into 3 KV heads), and projects with W_o.',
      shape: 'Q: [S, 576], K: [S, 192], V: [S, 192] → Output: [S, 576]',
      formula: 'Attn(Q, K, V) = Softmax( (Q &middot; K<sup>T</sup>) / &radic;d<sub>k</sub> + M<sub>causal</sub> ) &middot; V',
      code: 'src/layers/attention.ts',
      func: 'MultiHeadAttention.forward(x, startPos, kvCache, layerIdx)'
    },
    'add1': {
      title: 'Residual Connection 1 (Add)',
      badge: 'Residual Bypass',
      badgeClass: 'badge-green',
      description: 'Adds the original input X directly to the attention output (h = X + AttnOut). Creates an unhindered gradient highway through the 30 layers, preventing degradation during deep propagation.',
      shape: '[S, 576] + [S, 576] = [S, 576]',
      formula: 'h = AttnOut.addInPlace(x)',
      code: 'src/layers/transformer-block.ts',
      func: 'h = attnOut.addInPlace(x)'
    },
    'rmsnorm2': {
      title: 'RMSNorm 2 (Pre-MLP)',
      badge: 'Normalization',
      badgeClass: 'badge-purple',
      description: 'Normalizes the residual representation h before passing it into the non-linear Feed-Forward Network (MLP).',
      shape: '[S, 576] → [S, 576]',
      formula: 'norm2 = this.postAttentionLayernorm.forward(h)',
      code: 'src/layers/rms-norm.ts',
      func: 'postAttentionLayernorm.forward(h)'
    },
    'mlp': {
      title: 'SwiGLU Gated MLP (Feed-Forward)',
      badge: 'Feed-Forward Network',
      badgeClass: 'badge-amber',
      description: 'Applies non-linear token-level feature transformations. Projects hidden_size (576) up to intermediate_size (1536) through gate_proj and up_proj, computes silu(gate) &odot; up, and projects back to 576 via down_proj.',
      shape: '[S, 576] → Intermediate: [S, 1536] → Output: [S, 576]',
      formula: 'Y = down_proj( silu(gate_proj(x)) &odot; up_proj(x) )',
      code: 'src/layers/mlp.ts',
      func: 'SwiGLUMLP.forward(x)'
    },
    'add2': {
      title: 'Residual Connection 2 (Add)',
      badge: 'Residual Bypass',
      badgeClass: 'badge-green',
      description: 'Adds the normalized pre-MLP representation h to the MLP output (out = h + MLPOut). Gives the final output tensor for this Transformer layer.',
      shape: '[S, 576] + [S, 576] = [S, 576]',
      formula: 'out = mlpOut.addInPlace(h)',
      code: 'src/layers/transformer-block.ts',
      func: 'mlpOut.addInPlace(h)'
    },
    'output': {
      title: 'Block Output (Next Layer / Final Norm)',
      badge: 'Output State',
      badgeClass: 'badge-cyan',
      description: 'Transfers refined representations to TransformerBlock i + 1. If at Layer 29 (the final 30th layer), passes directly to final RMSNorm and the LM Head.',
      shape: '[S, 576]',
      formula: 'Output passed to Layer i + 1 or model.norm',
      code: 'src/model/transformer.ts',
      func: 'layers[l].forward(h, ...)'
    }
  };

  function selectNode(key) {
    const data = blockDetails[key];
    if (!data) return;

    // Update active visual node in SVG
    container.querySelectorAll('.svg-node, .svg-node-add, .svg-res-label, .flow-node, .flow-node-add').forEach(el => {
      if (el.getAttribute('data-block') === key) {
        el.classList.add('active');
      } else {
        el.classList.remove('active');
      }
    });

    inspector.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem">
        <h3 style="margin:0; font-size:1.15rem; color:#fff">${data.title}</h3>
        <span class="badge ${data.badgeClass}">${data.badge}</span>
      </div>
      <p style="font-size:0.9rem; color:#cbd5e1; line-height:1.55; margin-bottom:1rem">${data.description}</p>
      
      <div style="background:var(--code-bg); padding:0.75rem 1rem; border-radius:0.5rem; border:1px solid var(--border-color); margin-bottom:0.75rem; font-family:monospace; font-size:0.8rem">
        <div style="color:var(--text-muted); margin-bottom:0.25rem">Tensor Shape Transformation:</div>
        <div style="color:var(--accent-amber); font-weight:600">${data.shape}</div>
      </div>

      <div style="background:var(--code-bg); padding:0.75rem 1rem; border-radius:0.5rem; border:1px solid var(--border-color); margin-bottom:0.75rem; font-family:monospace; font-size:0.8rem">
        <div style="color:var(--text-muted); margin-bottom:0.25rem">Mathematical Operation:</div>
        <div style="color:var(--accent-cyan)">${data.formula}</div>
      </div>

      <div style="background:rgba(255,255,255,0.03); padding:0.75rem 1rem; border-radius:0.5rem; border:1px solid var(--border-color); font-size:0.8rem">
        <div style="color:var(--text-muted); margin-bottom:0.25rem">Implementation File & Function:</div>
        <div style="font-family:monospace; color:var(--accent-green); font-weight:600">${data.code}</div>
        <div style="font-family:monospace; color:#94a3b8; font-size:0.75rem">${data.func}</div>
      </div>
    `;
  }

  container.querySelectorAll('.svg-node, .svg-node-add, .svg-res-label, .flow-node, .flow-node-add').forEach(el => {
    const key = el.getAttribute('data-block');
    el.addEventListener('click', () => selectNode(key));
    el.addEventListener('mouseenter', () => selectNode(key));
  });

  // Pre-select attention node
  selectNode('attn');
}

// --- 8. Autograd Computation Graph Visualizer (Chapter 9) ---
export function initAutogradWidget() {
  const sandbox = document.getElementById('autograd-sandbox');
  if (!sandbox) return;

  const fwdBtn = document.getElementById('ag-fwd-btn');
  const stepBwdBtn = document.getElementById('ag-step-bwd-btn');
  const fullBwdBtn = document.getElementById('ag-full-bwd-btn');
  const numCheckBtn = document.getElementById('ag-num-check-btn');
  const resetBtn = document.getElementById('ag-reset-btn');
  const statusBadge = document.getElementById('autograd-status-badge');
  const auditBox = document.getElementById('ag-audit-box');

  const inW1 = document.getElementById('ag-in-w1');
  const inX1 = document.getElementById('ag-in-x1');
  const inW2 = document.getElementById('ag-in-w2');
  const inX2 = document.getElementById('ag-in-x2');
  const inB = document.getElementById('ag-in-b');
  const inTarget = document.getElementById('ag-in-target');

  // Node elements
  const nodes = {
    w1: document.getElementById('node-w1'),
    x1: document.getElementById('node-x1'),
    w2: document.getElementById('node-w2'),
    x2: document.getElementById('node-x2'),
    b: document.getElementById('node-b'),
    p1: document.getElementById('node-p1'),
    p2: document.getElementById('node-p2'),
    z: document.getElementById('node-z'),
    act: document.getElementById('node-act'),
    loss: document.getElementById('node-loss'),
  };

  const vals = {
    w1: document.getElementById('val-w1'),
    x1: document.getElementById('val-x1'),
    w2: document.getElementById('val-w2'),
    x2: document.getElementById('val-x2'),
    b: document.getElementById('val-b'),
    p1: document.getElementById('val-p1'),
    p2: document.getElementById('val-p2'),
    z: document.getElementById('val-z'),
    act: document.getElementById('val-act'),
    loss: document.getElementById('val-loss'),
  };

  const grads = {
    w1: document.getElementById('grad-w1'),
    x1: document.getElementById('grad-x1'),
    w2: document.getElementById('grad-w2'),
    x2: document.getElementById('grad-x2'),
    b: document.getElementById('grad-b'),
    p1: document.getElementById('grad-p1'),
    p2: document.getElementById('grad-p2'),
    z: document.getElementById('grad-z'),
    act: document.getElementById('grad-act'),
  };

  // State
  let forwardDone = false;
  let backwardStep = 0; // 0 = not started, 1 = dL/dAct, 2 = dL/dZ, 3 = dL/dP & dL/dB, 4 = dL/dW & dL/dX

  let s = {
    w1: 2.0, x1: 1.5, w2: -1.0, x2: 2.0, b: 0.5, y: 3.0,
    p1: 0, p2: 0, z: 0, yHat: 0, loss: 0,
    dL_dyHat: 0, dL_dz: 0, dL_dp1: 0, dL_dp2: 0, dL_db: 0,
    dL_dw1: 0, dL_dx1: 0, dL_dw2: 0, dL_dx2: 0,
  };

  function readInputs() {
    s.w1 = parseFloat(inW1.value) || 0;
    s.x1 = parseFloat(inX1.value) || 0;
    s.w2 = parseFloat(inW2.value) || 0;
    s.x2 = parseFloat(inX2.value) || 0;
    s.b = parseFloat(inB.value) || 0;
    s.y = parseFloat(inTarget.value) || 0;

    vals.w1.textContent = s.w1.toFixed(2);
    vals.x1.textContent = s.x1.toFixed(2);
    vals.w2.textContent = s.w2.toFixed(2);
    vals.x2.textContent = s.x2.toFixed(2);
    vals.b.textContent = s.b.toFixed(2);
  }

  function clearActiveStates() {
    Object.values(nodes).forEach(n => {
      if (n) {
        n.classList.remove('active-forward');
        n.classList.remove('active-backward');
      }
    });
  }

  function forward() {
    readInputs();
    s.p1 = s.w1 * s.x1;
    s.p2 = s.w2 * s.x2;
    s.z = s.p1 + s.p2 + s.b;
    s.yHat = Math.max(0, s.z);
    s.loss = 0.5 * Math.pow(s.yHat - s.y, 2);

    vals.p1.textContent = s.p1.toFixed(2);
    vals.p2.textContent = s.p2.toFixed(2);
    vals.z.textContent = s.z.toFixed(2);
    vals.act.textContent = s.yHat.toFixed(2);
    vals.loss.textContent = s.loss.toFixed(4);

    clearActiveStates();
    Object.values(nodes).forEach(n => n && n.classList.add('active-forward'));

    forwardDone = true;
    backwardStep = 0;

    statusBadge.className = 'badge badge-cyan';
    statusBadge.textContent = 'Forward Done';

    auditBox.innerHTML = `<strong>Forward Pass Evaluated:</strong><br>
p₁ = ${s.w1} × ${s.x1} = <strong>${s.p1.toFixed(2)}</strong> &nbsp;|&nbsp;
p₂ = ${s.w2} × ${s.x2} = <strong>${s.p2.toFixed(2)}</strong><br>
z = p₁ + p₂ + b = ${s.p1.toFixed(2)} + ${s.p2.toFixed(2)} + ${s.b} = <strong>${s.z.toFixed(2)}</strong><br>
ŷ = ReLU(z) = <strong>${s.yHat.toFixed(2)}</strong> &nbsp;|&nbsp;
Loss = ½·(ŷ − y)² = ½·(${s.yHat.toFixed(2)} − ${s.y})² = <strong style="color:#f87171">${s.loss.toFixed(4)}</strong>`;
  }

  function stepBackward() {
    if (!forwardDone) {
      forward();
    }

    backwardStep++;
    clearActiveStates();

    if (backwardStep === 1) {
      // ∂L/∂ŷ = ŷ − y
      s.dL_dyHat = s.yHat - s.y;
      grads.act.textContent = s.dL_dyHat.toFixed(3);
      nodes.act.classList.add('active-backward');
      nodes.loss.classList.add('active-backward');

      statusBadge.className = 'badge badge-amber';
      statusBadge.textContent = 'Backprop: Step 1';

      auditBox.innerHTML = `<strong>Backward Step 1 — Loss Gradient with respect to ŷ:</strong><br>
Loss = ½·(ŷ − y)²<br>
Rule: ∂L/∂ŷ = ŷ − y = ${s.yHat.toFixed(2)} − ${s.y} = <strong style="color:var(--accent-amber)">${s.dL_dyHat.toFixed(3)}</strong>`;
    } else if (backwardStep === 2) {
      // ∂L/∂z = ∂L/∂ŷ · (z > 0 ? 1 : 0)
      const reluGrad = s.z > 0 ? 1 : 0;
      s.dL_dz = s.dL_dyHat * reluGrad;
      grads.z.textContent = s.dL_dz.toFixed(3);
      nodes.z.classList.add('active-backward');
      nodes.act.classList.add('active-backward');

      statusBadge.className = 'badge badge-amber';
      statusBadge.textContent = 'Backprop: Step 2';

      auditBox.innerHTML = `<strong>Backward Step 2 — Chain Rule through ReLU:</strong><br>
ŷ = max(0, z) &nbsp;⟹&nbsp; d(ReLU)/dz = ${s.z > 0 ? '1.0 (active)' : '0.0 (inactive)'}<br>
Rule: ∂L/∂z = (∂L/∂ŷ) · d(ReLU)/dz = ${s.dL_dyHat.toFixed(3)} × ${reluGrad} = <strong style="color:var(--accent-amber)">${s.dL_dz.toFixed(3)}</strong>`;
    } else if (backwardStep === 3) {
      // ∂L/∂p1 = ∂L/∂z · 1, ∂L/∂p2 = ∂L/∂z · 1, ∂L/∂b = ∂L/∂z · 1
      s.dL_dp1 = s.dL_dz * 1;
      s.dL_dp2 = s.dL_dz * 1;
      s.dL_db = s.dL_dz * 1;

      grads.p1.textContent = s.dL_dp1.toFixed(3);
      grads.p2.textContent = s.dL_dp2.toFixed(3);
      grads.b.textContent = s.dL_db.toFixed(3);

      nodes.p1.classList.add('active-backward');
      nodes.p2.classList.add('active-backward');
      nodes.b.classList.add('active-backward');
      nodes.z.classList.add('active-backward');

      statusBadge.className = 'badge badge-amber';
      statusBadge.textContent = 'Backprop: Step 3';

      auditBox.innerHTML = `<strong>Backward Step 3 — Addition Gate distributes gradient equally:</strong><br>
z = p₁ + p₂ + b &nbsp;⟹&nbsp; ∂z/∂p₁ = 1, ∂z/∂p₂ = 1, ∂z/∂b = 1<br>
∂L/∂p₁ = ${s.dL_dp1.toFixed(3)} &nbsp;|&nbsp;
∂L/∂p₂ = ${s.dL_dp2.toFixed(3)} &nbsp;|&nbsp;
<strong>∂L/∂b = ${s.dL_db.toFixed(3)}</strong> (bias parameter gradient!)`;
    } else if (backwardStep >= 4) {
      // ∂L/∂w1 = ∂L/∂p1 · x1, ∂L/∂x1 = ∂L/∂p1 · w1
      // ∂L/∂w2 = ∂L/∂p2 · x2, ∂L/∂x2 = ∂L/∂p2 · w2
      s.dL_dw1 = s.dL_dp1 * s.x1;
      s.dL_dx1 = s.dL_dp1 * s.w1;
      s.dL_dw2 = s.dL_dp2 * s.x2;
      s.dL_dx2 = s.dL_dp2 * s.w2;

      grads.w1.textContent = s.dL_dw1.toFixed(3);
      grads.x1.textContent = s.dL_dx1.toFixed(3);
      grads.w2.textContent = s.dL_dw2.toFixed(3);
      grads.x2.textContent = s.dL_dx2.toFixed(3);

      nodes.w1.classList.add('active-backward');
      nodes.x1.classList.add('active-backward');
      nodes.w2.classList.add('active-backward');
      nodes.x2.classList.add('active-backward');
      nodes.p1.classList.add('active-backward');
      nodes.p2.classList.add('active-backward');

      statusBadge.className = 'badge badge-green';
      statusBadge.textContent = 'Backward Complete ✓';

      auditBox.innerHTML = `<strong>Backward Step 4 — Multiplication Gate chain rule:</strong><br>
p₁ = w₁ · x₁ &nbsp;⟹&nbsp; <strong>∂L/∂w₁ = (∂L/∂p₁) · x₁</strong> = ${s.dL_dp1.toFixed(3)} × ${s.x1} = <strong style="color:var(--accent-green)">${s.dL_dw1.toFixed(3)}</strong><br>
p₂ = w₂ · x₂ &nbsp;⟹&nbsp; <strong>∂L/∂w₂ = (∂L/∂p₂) · x₂</strong> = ${s.dL_dp2.toFixed(3)} × ${s.x2} = <strong style="color:var(--accent-green)">${s.dL_dw2.toFixed(3)}</strong><br>
All leaf gradients accumulated! AdamW can now update w₁, w₂, and b.`;
    }
  }

  function fullBackward() {
    if (!forwardDone) forward();
    for (let i = 0; i < 4; i++) {
      stepBackward();
    }
  }

  function checkNumerical() {
    if (!forwardDone) forward();
    fullBackward();

    // Central difference for w1: (L(w1 + h) - L(w1 - h)) / (2h)
    const h = 1e-6;
    const calcLossForW1 = (w) => {
      const z = w * s.x1 + s.w2 * s.x2 + s.b;
      const yH = Math.max(0, z);
      return 0.5 * Math.pow(yH - s.y, 2);
    };

    const lPlus = calcLossForW1(s.w1 + h);
    const lMinus = calcLossForW1(s.w1 - h);
    const numGrad = (lPlus - lMinus) / (2 * h);
    const diff = Math.abs(numGrad - s.dL_dw1);

    auditBox.innerHTML = `<strong>🔍 Numerical Gradient Verification for w₁:</strong><br>
Finite difference formula: [L(w₁ + 1e−6) − L(w₁ − 1e−6)] / (2 · 1e−6)<br>
• Analytical Autograd Gradient: <strong>${s.dL_dw1.toFixed(6)}</strong><br>
• Finite Difference Gradient: <strong>${numGrad.toFixed(6)}</strong><br>
• Absolute Difference: <strong style="color:var(--accent-green)">${diff.toExponential(3)}</strong> (Exact Match! Proof that reverse-mode DAG backprop produces true derivatives).`;
  }

  function reset() {
    forwardDone = false;
    backwardStep = 0;
    clearActiveStates();
    readInputs();

    ['p1', 'p2', 'z', 'act', 'loss'].forEach(k => vals[k] && (vals[k].textContent = '—'));
    Object.values(grads).forEach(el => el && (el.textContent = '—'));

    statusBadge.className = 'badge badge-purple';
    statusBadge.textContent = 'Ready';
    auditBox.innerHTML = `Ready. Adjust inputs above and click <strong>▶ Forward Pass</strong>.`;
  }

  fwdBtn.onclick = forward;
  stepBwdBtn.onclick = stepBackward;
  fullBwdBtn.onclick = fullBackward;
  numCheckBtn.onclick = checkNumerical;
  resetBtn.onclick = reset;

  [inW1, inX1, inW2, inX2, inB, inTarget].forEach(input => {
    input.addEventListener('input', () => {
      reset();
    });
  });

  reset();
}

// --- 9. Training Loop & Optimizer Race Visualizer (Chapter 10) ---
export function initTrainingLoopWidget() {
  // 1. Cross-entropy & Perplexity Calculator
  const ceSlider = document.getElementById('ce-slider');
  const cePVal = document.getElementById('ce-p-val');
  const ceLossOut = document.getElementById('ce-loss-out');
  const cePplOut = document.getElementById('ce-ppl-out');
  const ceStateOut = document.getElementById('ce-state-out');

  if (ceSlider) {
    function updateCE() {
      const p = parseFloat(ceSlider.value);
      const loss = -Math.log(p);
      const ppl = 1 / p;

      cePVal.textContent = p.toFixed(4) + (Math.abs(p - 1/65) < 0.003 ? ' (Random Uniform Guess ~1/65)' : '');
      ceLossOut.textContent = loss.toFixed(3);
      cePplOut.textContent = ppl.toFixed(1) + ' chars';

      if (p < 0.005) {
        ceStateOut.textContent = 'Severe Hallucination / Surprisal Spike';
        ceStateOut.style.color = '#ef4444';
      } else if (p < 0.03) {
        ceStateOut.textContent = 'Untrained / Random Guessing (Initial Baseline)';
        ceStateOut.style.color = '#f59e0b';
      } else if (p < 0.25) {
        ceStateOut.textContent = 'Early Training (Vowels & Whitespace Learned)';
        ceStateOut.style.color = '#38bdf8';
      } else if (p < 0.70) {
        ceStateOut.textContent = 'Moderate Confidence (Sub-words & Word Roots)';
        ceStateOut.style.color = '#818cf8';
      } else {
        ceStateOut.textContent = 'High Confidence Next Character Prediction';
        ceStateOut.style.color = '#22c55e';
      }
    }
    ceSlider.addEventListener('input', updateCE);
    updateCE();
  }
}


// --- 10. miniGPT Live Training & Generation Studio (Chapter 11) ---
export function initMiniGPTWidget() {
  // Part 1: Training Simulator
  const runBtn = document.getElementById('minigpt-run-btn');
  const stepBtn = document.getElementById('minigpt-step-btn');
  const resetBtn = document.getElementById('minigpt-reset-btn');
  const stepCounter = document.getElementById('minigpt-step-counter');
  const phaseBadge = document.getElementById('minigpt-phase-badge');
  const pplBadge = document.getElementById('minigpt-ppl-badge');
  const lossVal = document.getElementById('minigpt-loss-val');
  const canvas = document.getElementById('minigpt-loss-canvas');
  const samplesDiv = document.getElementById('minigpt-samples');
  const vocabSizeEl = document.getElementById('minigpt-vocab-size');
  const paramCountEl = document.getElementById('minigpt-param-count');

  if (!runBtn || !canvas) return;

  const simulatedLosses = (() => {
    const losses = [];
    let loss = 4.174 + (Math.random() - 0.5) * 0.05;
    for (let i = 0; i < 100; i++) {
      const decay = Math.exp(-i / 22);
      const noise = (Math.random() - 0.5) * 0.08 * decay + (Math.random() - 0.5) * 0.02;
      loss = loss * 0.965 + 0.08 * decay + noise;
      loss = Math.max(2.05, loss);
      losses.push(+loss.toFixed(4));
    }
    return losses;
  })();

  const sampleSnapshots = {
    0:   'The mX7k!qz-Yp\nW2nh, oA;Qr:ti\njG. xBeDF8lsvP',
    10:  'The nde an the  tnd\nth ae tte n  ne tnd',
    25:  'The and the to the  the and to\nthe end the to',
    50:  'The death the to be, to sleep;\nTo sleep, to be',
    75:  'The heart-ache and to sleep—\nTo sleep, to die,',
    100: 'The pangs of despised love, to die,\nTo sleep, to be',
  };

  const vocabSize = 65;
  const paramCount = '~40,000';

  let currentStep = 0;
  let isRunning = false;
  let autoTimer = null;

  if (vocabSizeEl) vocabSizeEl.textContent = `${vocabSize} characters`;
  if (paramCountEl) paramCountEl.textContent = paramCount;

  function drawLossCanvas() {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth || 300;
    canvas.width = W;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let y = 0; y <= 4; y++) {
      const py = H * (y / 4);
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke();
    }

    const minLoss = 2.0, maxLoss = 4.5;
    const toY = (l) => H - ((l - minLoss) / (maxLoss - minLoss)) * (H - 10) - 5;
    const toX = (i) => (i / 99) * (W - 4) + 2;

    const baselineY = toY(Math.log(vocabSize));
    ctx.strokeStyle = 'rgba(245,158,11,0.35)';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, baselineY); ctx.lineTo(W, baselineY); ctx.stroke();
    ctx.setLineDash([]);

    if (currentStep === 0) return;

    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(56,189,248,0.25)');
    grad.addColorStop(1, 'rgba(56,189,248,0.02)');

    ctx.beginPath();
    ctx.moveTo(toX(0), toY(simulatedLosses[0]));
    for (let i = 1; i < currentStep; i++) {
      ctx.lineTo(toX(i), toY(simulatedLosses[i]));
    }
    ctx.lineTo(toX(currentStep - 1), H);
    ctx.lineTo(toX(0), H);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(simulatedLosses[0]));
    for (let i = 1; i < currentStep; i++) {
      ctx.lineTo(toX(i), toY(simulatedLosses[i]));
    }
    ctx.stroke();

    if (currentStep > 0) {
      const cx = toX(currentStep - 1);
      const cy = toY(simulatedLosses[currentStep - 1]);
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
    }
  }

  function getClosestSample() {
    const snapKeys = [0, 10, 25, 50, 75, 100].sort((a, b) => a - b);
    let best = 0;
    for (const k of snapKeys) {
      if (k <= currentStep) best = k;
    }
    return sampleSnapshots[best];
  }

  function renderTraining() {
    if (stepCounter) stepCounter.textContent = String(currentStep);
    if (lossVal && currentStep > 0) {
      const l = simulatedLosses[currentStep - 1];
      lossVal.textContent = `Loss: ${l}`;
      if (pplBadge) {
        pplBadge.textContent = `PPL: ${Math.exp(l).toFixed(1)}`;
      }
    } else if (pplBadge) {
      pplBadge.textContent = `PPL: 65.0`;
    }

    if (phaseBadge) {
      if (currentStep === 0) {
        phaseBadge.className = 'badge badge-purple';
        phaseBadge.textContent = 'Ready';
      } else if (currentStep < 100) {
        phaseBadge.className = 'badge badge-amber';
        phaseBadge.textContent = 'Training…';
      } else {
        phaseBadge.className = 'badge badge-green';
        phaseBadge.textContent = 'Trained ✓';
      }
    }

    if (samplesDiv) {
      const sample = getClosestSample();
      const stepLabel = currentStep === 0 ? 'Step 0 (untrained random guess)' : `Step ${currentStep}`;
      samplesDiv.innerHTML = currentStep === 0
        ? `<em style="color:var(--text-muted)">Click ▶ Run to start training on Shakespeare corpus…</em>`
        : `<div style="color:var(--text-muted); font-size:0.75rem; margin-bottom:0.4rem">${stepLabel}:</div><div style="white-space:pre; color:#e2e8f0">${sample}</div>`;
    }

    drawLossCanvas();

    runBtn.textContent = isRunning ? '⏸ Pause' : (currentStep >= 100 ? '✓ Done' : '▶ Run 100 Steps');
    runBtn.disabled = currentStep >= 100 && !isRunning;
    stepBtn.disabled = currentStep >= 100 || isRunning;
  }

  function advanceSteps(n) {
    currentStep = Math.min(100, currentStep + n);
    renderTraining();
    if (currentStep >= 100) stopAuto();
  }

  function startAuto() {
    isRunning = true;
    if (autoTimer) clearInterval(autoTimer);
    autoTimer = setInterval(() => {
      if (currentStep >= 100) { stopAuto(); return; }
      currentStep++;
      renderTraining();
    }, 60);
    renderTraining();
  }

  function stopAuto() {
    isRunning = false;
    if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
    renderTraining();
  }

  function resetTraining() {
    stopAuto();
    currentStep = 0;
    renderTraining();
  }

  runBtn.onclick = () => {
    if (isRunning) stopAuto();
    else if (currentStep >= 100) resetTraining();
    else startAuto();
  };

  stepBtn.onclick = () => { stopAuto(); advanceSteps(10); };
  resetBtn.onclick = resetTraining;

  window.addEventListener('resize', drawLossCanvas);
  renderTraining();

  // Part 2: Interactive Generation Studio
  const promptPreset = document.getElementById('gen-prompt-preset');
  const promptInput = document.getElementById('gen-prompt-input');
  const tempSlider = document.getElementById('gen-temp-slider');
  const tempVal = document.getElementById('gen-temp-val');
  const tempBadge = document.getElementById('gen-temp-badge');
  const genPlayBtn = document.getElementById('gen-play-btn');
  const genStepBtn = document.getElementById('gen-step-btn');
  const genResetBtn = document.getElementById('gen-reset-btn');
  const outputBox = document.getElementById('gen-output-box');
  const charCountEl = document.getElementById('gen-char-count');
  const candidatesList = document.getElementById('gen-candidates-list');

  if (!promptInput || !outputBox) return;

  let currentText = promptInput.value || "The ";
  let genTimer = null;
  let isGenerating = false;

  // Realistic candidate pools for Shakespearean contexts
  const charCandidatePools = {
    'The ': [
      { char: 'h', logit: 3.8, desc: 'heart' },
      { char: 'k', logit: 3.2, desc: 'king' },
      { char: 'm', logit: 2.7, desc: 'man' },
      { char: 'w', logit: 2.1, desc: 'world' },
      { char: ' ', logit: 1.0, desc: 'space' },
    ],
    'To be, or not ': [
      { char: 't', logit: 4.5, desc: 'to' },
      { char: ' ', logit: 2.1, desc: 'space' },
      { char: 'b', logit: 1.5, desc: 'be' },
      { char: 's', logit: 0.8, desc: 'so' },
      { char: ',', logit: 0.4, desc: 'comma' },
    ],
    'Romeo: ': [
      { char: 'O', logit: 4.1, desc: 'O' },
      { char: 'I', logit: 3.2, desc: 'I' },
      { char: 'W', logit: 2.5, desc: 'What' },
      { char: 'T', logit: 1.8, desc: 'Thou' },
      { char: '\n', logit: 0.9, desc: 'newline' },
    ],
    'default': [
      { char: ' ', logit: 3.2, desc: 'space' },
      { char: 'e', logit: 2.9, desc: 'vowel' },
      { char: 'a', logit: 2.5, desc: 'vowel' },
      { char: 't', logit: 2.1, desc: 'consonant' },
      { char: 's', logit: 1.7, desc: 'consonant' },
    ]
  };

  function getCandidatesForContext(text) {
    for (const [prefix, pool] of Object.entries(charCandidatePools)) {
      if (prefix !== 'default' && text.endsWith(prefix)) {
        return pool;
      }
    }
    // Dynamic based on last char
    const last = text.slice(-1);
    if (last === ' ') {
      return [
        { char: 't', logit: 3.4, desc: 'the/thou' },
        { char: 'a', logit: 2.8, desc: 'and' },
        { char: 's', logit: 2.3, desc: 'shall' },
        { char: 'w', logit: 2.0, desc: 'with' },
        { char: 'b', logit: 1.6, desc: 'be' },
      ];
    } else if (last === 'e' || last === 'o') {
      return [
        { char: ' ', logit: 3.5, desc: 'space' },
        { char: 'r', logit: 2.6, desc: '-er' },
        { char: 'd', logit: 2.1, desc: '-ed' },
        { char: 's', logit: 1.8, desc: '-es' },
        { char: 'n', logit: 1.2, desc: '-en' },
      ];
    }
    return charCandidatePools.default;
  }

  function computeSoftmax(candidates, temp) {
    const scaled = candidates.map(c => Math.exp(c.logit / temp));
    const sum = scaled.reduce((a, b) => a + b, 0);
    return candidates.map((c, i) => ({
      ...c,
      prob: scaled[i] / sum,
    }));
  }

  function renderGenerationStudio() {
    outputBox.innerHTML = `${escapeHtml(currentText)}<span class="typing-cursor"></span>`;
    charCountEl.textContent = String(currentText.length);

    const temp = parseFloat(tempSlider.value);
    tempVal.textContent = temp.toFixed(1);

    if (temp < 0.4) {
      tempBadge.className = 'badge badge-cyan';
      tempBadge.textContent = `Temp: ${temp} (Greedy / Conservative)`;
    } else if (temp <= 0.8) {
      tempBadge.className = 'badge badge-green';
      tempBadge.textContent = `Temp: ${temp} (Balanced Shakespeare)`;
    } else {
      tempBadge.className = 'badge badge-amber';
      tempBadge.textContent = `Temp: ${temp} (Creative / Chaotic)`;
    }

    const rawCandidates = getCandidatesForContext(currentText);
    const withProbs = computeSoftmax(rawCandidates, temp);

    let html = '';
    withProbs.forEach((c, idx) => {
      const isWinner = idx === 0;
      const charDisplay = c.char === ' ' ? '␣ [space]' : (c.char === '\n' ? '↵ [newline]' : `'${c.char}'`);
      html += `
        <div class="sim-candidate-row ${isWinner ? 'winner' : ''}" style="cursor:pointer" data-char="${escapeHtml(c.char)}">
          <div class="sim-candidate-token" title="${c.desc}">${charDisplay}</div>
          <div class="sim-bar-wrapper">
            <div class="sim-bar-fill" style="width: ${(c.prob * 100).toFixed(1)}%"></div>
          </div>
          <div class="sim-candidate-prob">${(c.prob * 100).toFixed(1)}%</div>
          <div class="sim-candidate-logit">z = ${c.logit.toFixed(1)}</div>
        </div>
      `;
    });
    candidatesList.innerHTML = html;

    // Click candidate to sample it directly
    candidatesList.querySelectorAll('.sim-candidate-row').forEach(row => {
      row.addEventListener('click', () => {
        const char = row.getAttribute('data-char');
        appendChar(char);
      });
    });
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function sampleNextChar() {
    const temp = parseFloat(tempSlider.value);
    const rawCandidates = getCandidatesForContext(currentText);
    const withProbs = computeSoftmax(rawCandidates, temp);

    // Sample according to probability distribution
    const r = Math.random();
    let cum = 0;
    let chosen = withProbs[0].char;
    for (const c of withProbs) {
      cum += c.prob;
      if (r <= cum) {
        chosen = c.char;
        break;
      }
    }
    return chosen;
  }

  function appendChar(c) {
    currentText += c;
    renderGenerationStudio();
  }

  function generateStream(count = 25) {
    isGenerating = true;
    genPlayBtn.textContent = '⏸ Stop';
    let remaining = count;

    if (genTimer) clearInterval(genTimer);
    genTimer = setInterval(() => {
      if (remaining <= 0) {
        stopGeneration();
        return;
      }
      const c = sampleNextChar();
      appendChar(c);
      remaining--;
    }, 70);
  }

  function stopGeneration() {
    isGenerating = false;
    genPlayBtn.textContent = '▶ Generate 25 Chars';
    if (genTimer) {
      clearInterval(genTimer);
      genTimer = null;
    }
  }

  promptPreset.addEventListener('change', () => {
    promptInput.value = promptPreset.value;
    currentText = promptPreset.value;
    renderGenerationStudio();
  });

  promptInput.addEventListener('input', () => {
    currentText = promptInput.value;
    renderGenerationStudio();
  });

  tempSlider.addEventListener('input', renderGenerationStudio);

  genPlayBtn.onclick = () => {
    if (isGenerating) stopGeneration();
    else generateStream(25);
  };

  genStepBtn.onclick = () => {
    stopGeneration();
    appendChar(sampleNextChar());
  };

  genResetBtn.onclick = () => {
    stopGeneration();
    currentText = promptInput.value || "The ";
    renderGenerationStudio();
  };

  renderGenerationStudio();
}

