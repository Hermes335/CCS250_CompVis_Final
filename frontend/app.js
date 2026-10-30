(() => {
  const $ = (s) => document.querySelector(s);
  const MODELS = [
    { key: 'restormer', name: 'Restormer' },
    { key: 'dncnn', name: 'DnCNN' },
    { key: 'nafnet', name: 'NAFNet' },
  ];
  let lastImage = null;
  let lastResponse = null;
  let working = false;
  const show = (id) => {
    document.querySelectorAll('.screen').forEach((el) => el.classList.toggle('active', el.id === id));
    window.scrollTo(0, 0);
  };
  $('#begin').addEventListener('click', () => show('upload'));
  $('#home').addEventListener('click', () => show('landing'));
  $('#again').addEventListener('click', () => { if (!working) show('upload'); });
  $('#uploadBtn').addEventListener('click', () => $('#file').click());
  $('#file').addEventListener('change', (ev) => ev.target.files[0] && processFile(ev.target.files[0]));
  $('#sampleBtn').addEventListener('click', async () => {
    const c = document.createElement('canvas'); c.width = 460; c.height = 320;
    const g = c.getContext('2d');
    const grad = g.createLinearGradient(0, 0, c.width, c.height);
    grad.addColorStop(0, '#2e657c'); grad.addColorStop(1, '#e8b871');
    g.fillStyle = grad; g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = '#153f48'; g.fillRect(20, 70, 140, 220);
    g.fillStyle = '#edecdf'; g.fillRect(70, 100, 70, 70);
    g.fillStyle = '#f5b86a'; g.beginPath(); g.arc(280, 105, 55, 0, 7); g.fill();
    const d = g.getImageData(0, 0, c.width, c.height);
    // Visual smoke-test sample; NOT a scientific dataset or fixed Gaussian noise.
    for (let p = 0; p < d.data.length; p += 4) {
      const z = (Math.random() - .5) * 76;
      for (let k = 0; k < 3; k++) d.data[p + k] = Math.max(0, Math.min(255, d.data[p + k] + z));
    }
    g.putImageData(d, 0, 0);
    c.toBlob((blob) => processFile(new File([blob], 'sample.png', {type: 'image/png'})), 'image/png');
  });
  const drop = $('#drop');
  ['dragenter', 'dragover'].forEach((type) => drop.addEventListener(type, (ev) => {ev.preventDefault(); drop.classList.add('over');}));
  ['dragleave', 'drop'].forEach((type) => drop.addEventListener(type, (ev) => {ev.preventDefault(); drop.classList.remove('over');}));
  drop.addEventListener('drop', (ev) => { const f = ev.dataTransfer.files[0]; if (f) processFile(f); });
  window.addEventListener('paste', (ev) => {
    if (!$('#upload').classList.contains('active')) return;
    const item = [...(ev.clipboardData?.items || [])].find((p) => p.type.startsWith('image/'));
    if (item) processFile(item.getAsFile());
  });
  async function refreshStatus() {
    try {
      const r = await fetch('/api/status');
      const data = await r.json();
      const text = data.mode === 'preview'
        ? 'PREVIEW MODE — only demonstration image filters run. These are NOT Restormer, DnCNN, or NAFNet.'
        : 'REAL MODE — actual model inference; a model fails explicitly if its checkpoint is missing.';
      $('#modeNotice').textContent = text;
      $('#modeNotice').style.color = data.mode === 'preview' ? '#f5c07e' : '#a9e3c3';
    } catch (e) {
      $('#modeNotice').textContent = 'Backend not connected. Open this app through FastAPI (not file://).';
    }
  }
  function addImage(parent, source, alt) {
    const im = document.createElement('img');
    im.src = source; im.alt = alt;
    parent.appendChild(im);
  }
  function save(dataURL, name) {
    const a = document.createElement('a'); a.href = dataURL; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
  }
  async function copyImage(uri, button) {
    try {
      const response = await fetch(uri);
      const blob = await response.blob();
      await navigator.clipboard.write([new ClipboardItem({'image/png': blob})]);
      button.textContent = 'Copied';
      setTimeout(() => {button.textContent = 'Copy';}, 1200);
    } catch (_) { alert('Copy requires a secure context and a browser that supports PNG clipboard. Use Save instead.'); }
  }
  async function shareImage(uri, filename) {
    try {
      const response = await fetch(uri);
      const blob = await response.blob();
      const file = new File([blob], filename, {type: 'image/png'});
      if (navigator.canShare?.({files: [file]})) {
        await navigator.share({files: [file], title: 'iMAGE result'});
      } else {
        alert('Sharing is not supported by this browser. Use Save instead.');
      }
    } catch (err) {
      if (err?.name !== 'AbortError') alert('Sharing unavailable. Use Save instead.');
    }
  }
  function action(container, title, callback) {
    const b = document.createElement('button');
    b.className = 'pill'; b.textContent = title;
    b.addEventListener('click', () => callback(b));
    container.appendChild(b);
  }
  function display(data) {
    lastResponse = data;
    const original = $('#origCanvas');
    const im = new Image();
    im.onload = () => {
      original.width = im.naturalWidth; original.height = im.naturalHeight;
      original.getContext('2d').drawImage(im, 0, 0);
    }; im.src = data.input;
    const originalActions = document.querySelector('#results .grid2 .card .actions');
    originalActions.replaceChildren();
    action(originalActions, 'Copy', (b) => copyImage(data.input, b));
    action(originalActions, 'Save', () => save(data.input, 'image_input.png'));
    action(originalActions, 'Share', () => shareImage(data.input, 'image_input.png'));
    const metrics = $('#metrics'); metrics.replaceChildren();
    const models = $('#models'); models.replaceChildren();
    $('#resultNotice').textContent = (data.mode === 'preview' ? 'PREVIEW ONLY: these outputs are basic demonstration filters, NOT model inference. ' : 'REAL INFERENCE: ') +
      `${data.size.width} × ${data.size.height} px · ` + (data.reference_supplied ? 'clean reference supplied' : 'no clean reference supplied');
    data.models.forEach((entry) => {
      const def = MODELS.find((m) => m.key === entry.key);
      const overview = document.createElement('div');
      const top = document.createElement('div'); top.className = 'm-top';
      const nm = document.createElement('span'); nm.textContent = def.name;
      const score = document.createElement('span');
      score.textContent = entry.error ? 'Unavailable' : (entry.metrics && data.mode === 'real' ? `${entry.metrics.psnr === null ? '∞' : entry.metrics.psnr.toFixed(2)} dB` : 'No ground truth score');
      top.append(nm, score); overview.append(top); metrics.append(overview);
      const card = document.createElement('div'); card.className = 'card model';
      const heading = document.createElement('h3'); heading.textContent = def.name;
      card.append(heading);
      const area = document.createElement('div'); area.className = 'imgbox';
      card.append(area);
      const msg = document.createElement('p'); msg.className = 'status';
      if (entry.error) {
        msg.textContent = entry.error; msg.classList.add('error');
      } else {
        addImage(area, entry.image, `${def.name} denoised output`);
        msg.textContent = `${entry.label} · ${entry.runtime_ms} ms` +
          (entry.parameters_m !== null ? ` · ${entry.parameters_m} M parameters` : '');
        const acts = document.createElement('div'); acts.className = 'actions';
        action(acts, 'Save', () => save(entry.image, `${def.name.toLowerCase()}_result.png`));
        action(acts, 'Copy', (b) => copyImage(entry.image, b));
        action(acts, 'Share', () => shareImage(entry.image, `${def.name.toLowerCase()}_result.png`));
        card.append(acts);
        if (entry.metrics && data.mode === 'real') {
          const p = document.createElement('p'); p.className = 'score';
          p.textContent = `PSNR: ${entry.metrics.psnr === null ? '∞' : entry.metrics.psnr.toFixed(2)} dB · SSIM: ${entry.metrics.ssim.toFixed(4)}`;
          card.append(p);
        }
      }
      card.append(msg); models.append(card);
    });
    const rec = $('#recommendation');
    rec.textContent = data.best && data.mode === 'real'
      ? `Recommended model: ${MODELS.find((m) => m.key === data.best).name}. ${data.best_reason}`
      : data.best_reason;
    show('results');
  }
  async function processFile(file) {
    if (working) return;
    if (!file || !file.type.startsWith('image/')) return alert('Please upload an image.');
    if (file.size > 10 * 1024 * 1024) return alert('Image must be 10 MB or smaller.');
    working = true;
    drop.classList.add('busy');
    $('.overlay .label').textContent = 'Running three denoising methods…';
    $('#upPct').textContent = 'This may take longer on CPU';
    try {
      const payload = new FormData(); payload.append('image', file);
      const reference = $('#reference').files[0];
      if (reference) payload.append('reference', reference);
      const response = await fetch('/api/denoise', {method: 'POST', body: payload});
      const data = await response.json();
      if (!response.ok) throw Error(data.detail || 'Server could not process image');
      display(data);
    } catch (error) {
      alert(`Denoising failed: ${error.message}`);
    } finally {
      working = false; drop.classList.remove('busy');
      $('#file').value = '';
    }
  }
  refreshStatus();
})();
