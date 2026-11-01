/**
 * VisionSphere — Clean Modern Studio Application Controller
 */

// Application State
const AppState = {
    currentImage: null,
    currentModel: 'mobilenet_v3_large',
    currentFilter: 'none',
    activeView: 'original', // 'original' | 'gradcam' | 'saliency'
    rawImageB64: null,
    gradcamB64: null,
    saliencyB64: null
};

// DOM References
const DOM = {
    // Stages
    emptyStage: document.getElementById('empty-stage'),
    activeStage: document.getElementById('active-stage'),
    btnNewImage: document.getElementById('btn-new-image'),

    // Dropzone & Inputs
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('file-input'),
    modelSelector: document.getElementById('model-selector'),
    filterSelector: document.getElementById('filter-selector'),

    // Modals
    btnOpenUrl: document.getElementById('btn-open-url'),
    urlModal: document.getElementById('url-modal'),
    urlForm: document.getElementById('url-form'),
    urlInput: document.getElementById('image-url-input'),
    btnCloseUrl: document.getElementById('btn-close-url'),
    btnCancelUrl: document.getElementById('btn-cancel-url'),

    btnOpenCamera: document.getElementById('btn-open-camera'),
    cameraModal: document.getElementById('camera-modal'),
    btnCloseCamera: document.getElementById('btn-close-camera'),

    // Viewport
    mainImage: document.getElementById('main-image'),
    loadingOverlay: document.getElementById('loading-overlay'),
    toggleOriginal: document.getElementById('toggle-original'),
    toggleGradcam: document.getElementById('toggle-gradcam'),
    toggleSaliency: document.getElementById('toggle-saliency'),
    btnCropMode: document.getElementById('btn-crop-mode'),

    // Results
    topLabel: document.getElementById('top-label'),
    topConfidence: document.getElementById('top-confidence'),
    topBadge: document.getElementById('top-badge'),
    topDescription: document.getElementById('top-description'),
    telemetryTime: document.getElementById('telemetry-time'),
    predictionsContainer: document.getElementById('predictions-container'),
    btnVoice: document.getElementById('btn-voice'),

    // Diagnostics
    diagSharpness: document.getElementById('diag-sharpness'),
    diagBlurStatus: document.getElementById('diag-blur-status'),
    diagLighting: document.getElementById('diag-lighting'),
    diagDimensions: document.getElementById('diag-dimensions')
};

// Toast Notification
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    const color = type === 'error' ? 'text-rose-400 border-rose-500/30' :
                  type === 'success' ? 'text-emerald-400 border-emerald-500/30' :
                  'text-cyan-400 border-cyan-500/30';

    toast.className = `clean-toast px-4 py-2.5 text-xs font-mono flex items-center gap-2.5 ${color}`;
    toast.innerHTML = `<i class="fa-solid fa-${type === 'error' ? 'circle-exclamation' : type === 'success' ? 'circle-check' : 'circle-info'} text-sm"></i><span>${message}</span>`;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(8px)';
        toast.style.transition = 'all 0.25s ease';
        setTimeout(() => toast.remove(), 250);
    }, 3200);
}

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    initDropzone();
    initClipboard();
    initSampleChips();
    initModals();
    initControls();
    initSpeech();
});

// ============================================================================
// STAGE SWITCHING
// ============================================================================
function showActiveStage() {
    DOM.emptyStage.classList.add('hidden');
    DOM.activeStage.classList.remove('hidden');
}

function showEmptyStage() {
    DOM.activeStage.classList.add('hidden');
    DOM.emptyStage.classList.remove('hidden');
    AppState.currentImage = null;
    AppState.rawImageB64 = null;
    AppState.gradcamB64 = null;
    AppState.saliencyB64 = null;
    DOM.mainImage.src = '';
}

// ============================================================================
// DROPZONE & UPLOAD
// ============================================================================
function initDropzone() {
    DOM.dropzone.addEventListener('click', () => DOM.fileInput.click());

    ['dragenter', 'dragover'].forEach(n => {
        DOM.dropzone.addEventListener(n, (e) => {
            e.preventDefault();
            DOM.dropzone.classList.add('drag-active');
        });
    });

    ['dragleave', 'drop'].forEach(n => {
        DOM.dropzone.addEventListener(n, (e) => {
            e.preventDefault();
            DOM.dropzone.classList.remove('drag-active');
        });
    });

    DOM.dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) handleFile(files[0]);
    });

    DOM.fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    DOM.btnNewImage.addEventListener('click', () => showEmptyStage());
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please upload a valid image (PNG, JPG, WEBP, BMP)', 'error');
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        AppState.currentImage = e.target.result;
        classifyImage({ image: e.target.result });
    };
    reader.readAsDataURL(file);
}

// ============================================================================
// CLIPBOARD PASTE
// ============================================================================
function initClipboard() {
    window.addEventListener('paste', (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        for (let item of items) {
            if (item.type.indexOf('image') !== -1) {
                const blob = item.getAsFile();
                const reader = new FileReader();
                reader.onload = (ev) => {
                    AppState.currentImage = ev.target.result;
                    showToast('Pasted image from clipboard', 'success');
                    classifyImage({ image: ev.target.result });
                };
                reader.readAsDataURL(blob);
                break;
            }
        }
    });
}

// ============================================================================
// SAMPLE CHIPS
// ============================================================================
function initSampleChips() {
    document.querySelectorAll('.sample-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const sampleName = chip.getAttribute('data-sample');
            classifyImage({ sampleName: sampleName });
        });
    });
}

// ============================================================================
// MODALS (URL & CAMERA)
// ============================================================================
function initModals() {
    // URL Modal
    DOM.btnOpenUrl.addEventListener('click', () => {
        DOM.urlModal.classList.remove('hidden');
        DOM.urlInput.focus();
    });
    DOM.btnCloseUrl.addEventListener('click', () => DOM.urlModal.classList.add('hidden'));
    DOM.btnCancelUrl.addEventListener('click', () => DOM.urlModal.classList.add('hidden'));
    DOM.urlForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const url = DOM.urlInput.value.trim();
        if (!url) return;
        DOM.urlModal.classList.add('hidden');
        classifyImage({ imageUrl: url });
    });

    // Camera Modal
    DOM.btnOpenCamera.addEventListener('click', () => {
        DOM.cameraModal.classList.remove('hidden');
        if (window.WebcamController) window.WebcamController.startCamera();
    });
    DOM.btnCloseCamera.addEventListener('click', () => {
        DOM.cameraModal.classList.add('hidden');
        if (window.WebcamController) window.WebcamController.stopCamera();
    });
}

// ============================================================================
// CONTROLS & FLOATING TOOLBAR
// ============================================================================
function initControls() {
    DOM.modelSelector.addEventListener('change', (e) => {
        AppState.currentModel = e.target.value;
        if (AppState.currentImage) classifyImage({ image: AppState.currentImage });
    });

    DOM.filterSelector.addEventListener('change', (e) => {
        AppState.currentFilter = e.target.value;
        if (AppState.currentImage) classifyImage({ image: AppState.currentImage });
    });

    // Layer Toggles
    DOM.toggleOriginal.addEventListener('click', () => setLayer('original'));
    DOM.toggleGradcam.addEventListener('click', () => setLayer('gradcam'));
    DOM.toggleSaliency.addEventListener('click', () => setLayer('saliency'));
}

function setLayer(layer) {
    AppState.activeView = layer;
    DOM.toggleOriginal.classList.toggle('active', layer === 'original');
    DOM.toggleGradcam.classList.toggle('active', layer === 'gradcam');
    DOM.toggleSaliency.classList.toggle('active', layer === 'saliency');

    if (layer === 'gradcam' && AppState.gradcamB64) {
        DOM.mainImage.src = AppState.gradcamB64;
    } else if (layer === 'saliency' && AppState.saliencyB64) {
        DOM.mainImage.src = AppState.saliencyB64;
    } else if (AppState.rawImageB64) {
        DOM.mainImage.src = AppState.rawImageB64;
    }
}

// ============================================================================
// SPEECH SYNTHESIS
// ============================================================================
function initSpeech() {
    DOM.btnVoice.addEventListener('click', () => {
        const text = DOM.topLabel.textContent;
        if (!text || text === '--') return;
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(`Identified as ${text}`);
            window.speechSynthesis.speak(u);
        }
    });
}

// ============================================================================
// INFERENCE EXECUTION
// ============================================================================
async function classifyImage(payload) {
    showActiveStage();
    DOM.loadingOverlay.classList.remove('hidden');

    try {
        const body = {
            ...payload,
            model: AppState.currentModel,
            filter: AppState.currentFilter,
            gradcam: true,
            saliency: true
        };

        const res = await fetch('/api/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        const data = await res.json();
        if (data.status !== 'success') throw new Error(data.message || 'Classification failed.');

        renderClassificationResults(data);
    } catch (err) {
        console.error(err);
        showToast(err.message, 'error');
    } finally {
        DOM.loadingOverlay.classList.add('hidden');
    }
}

function renderClassificationResults(data) {
    AppState.rawImageB64 = data.image_b64;
    AppState.gradcamB64 = data.gradcam_b64;
    AppState.saliencyB64 = data.saliency_image;
    AppState.currentImage = data.image_b64;

    setLayer('original');

    // Top Verdict
    const top = data.top_prediction;
    DOM.topLabel.textContent = top.label;
    DOM.topConfidence.textContent = `${top.confidence_pct}%`;
    DOM.topBadge.textContent = top.badge.label;
    DOM.topBadge.style.color = top.badge.color;
    DOM.topBadge.style.borderColor = `${top.badge.color}40`;
    DOM.topDescription.textContent = top.description;
    DOM.telemetryTime.innerHTML = `<i class="fa-solid fa-bolt text-amber-400 mr-1"></i> ${data.telemetry.inference_time_ms} ms`;

    // Predictions List
    DOM.predictionsContainer.innerHTML = '';
    data.predictions.forEach((p, idx) => {
        const row = document.createElement('div');
        row.className = 'flex flex-col gap-1';
        row.innerHTML = `
            <div class="flex items-center justify-between text-xs">
                <span class="text-slate-200 font-semibold flex items-center gap-1.5">
                    <span class="text-[10px] font-mono text-cyan-400">#${idx + 1}</span>
                    ${p.label}
                </span>
                <span class="font-mono text-slate-300 font-medium">${p.confidence_pct}%</span>
            </div>
            <div class="progress-track w-full">
                <div class="progress-bar" style="width: 0%"></div>
            </div>
        `;
        DOM.predictionsContainer.appendChild(row);

        setTimeout(() => {
            const bar = row.querySelector('.progress-bar');
            if (bar) bar.style.width = `${p.confidence_pct}%`;
        }, 50);
    });

    // Diagnostics
    const diag = data.diagnostics;
    DOM.diagSharpness.textContent = `${diag.sharpness_score} / 100`;
    DOM.diagBlurStatus.textContent = diag.is_blurry ? 'Blurry' : 'Sharp';
    DOM.diagBlurStatus.className = `text-[11px] font-semibold ${diag.is_blurry ? 'text-rose-400' : 'text-emerald-400'}`;
    DOM.diagLighting.textContent = diag.lighting_quality;
    DOM.diagDimensions.textContent = `${diag.width} × ${diag.height}`;
}
