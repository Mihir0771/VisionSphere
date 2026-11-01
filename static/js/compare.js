/**
 * VisionSphere — Dual Image Comparison Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    setupSlot('a');
    setupSlot('b');
});

function setupSlot(slotId) {
    const dropzone = document.getElementById(`dropzone-${slotId}`);
    const fileInput = document.getElementById(`file-${slotId}`);
    const preview = document.getElementById(`preview-${slotId}`);
    const placeholder = document.getElementById(`placeholder-${slotId}`);
    const resultBox = document.getElementById(`result-${slotId}`);
    const modelSelect = document.getElementById(`model-${slotId}`);

    let currentImageB64 = null;

    dropzone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(n => dropzone.addEventListener(n, (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-active');
    }));

    ['dragleave', 'drop'].forEach(n => dropzone.addEventListener(n, (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-active');
    }));

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    modelSelect.addEventListener('change', () => {
        if (currentImageB64) runInference();
    });

    function handleFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            currentImageB64 = e.target.result;
            preview.src = e.target.result;
            preview.classList.remove('hidden');
            placeholder.classList.add('hidden');
            runInference();
        };
        reader.readAsDataURL(file);
    }

    async function runInference() {
        resultBox.innerHTML = `
            <div class="flex items-center justify-center gap-2 py-4 text-cyan-400 font-mono text-xs">
                <i class="fa-solid fa-spinner animate-spin"></i> Classifying...
            </div>
        `;

        try {
            const res = await fetch('/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: currentImageB64,
                    model: modelSelect.value
                })
            });

            const data = await res.json();
            if (data.status !== 'success') throw new Error(data.message || 'Error');

            const top = data.top_prediction;
            resultBox.innerHTML = `
                <div class="flex items-baseline justify-between">
                    <span class="text-lg font-bold text-white">${top.label}</span>
                    <span class="text-lg font-mono font-bold text-cyan-400">${top.confidence_pct}%</span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="px-2 py-0.5 rounded text-[10px] font-semibold" style="border: 1px solid ${top.badge.color}; color: ${top.badge.color}">
                        ${top.badge.label}
                    </span>
                    <span class="text-[10px] font-mono text-slate-400">
                        <i class="fa-solid fa-bolt text-yellow-400 mr-1"></i>${data.telemetry.inference_time_ms} ms
                    </span>
                </div>
                <div class="mt-2 text-[11px] text-slate-400">
                    Runner-up: <strong class="text-slate-200">${data.predictions[1]?.label || 'None'}</strong> (${data.predictions[1]?.confidence_pct || 0}%)
                </div>
            `;
        } catch (err) {
            resultBox.innerHTML = `<div class="text-rose-400 text-xs py-2 text-center">${err.message}</div>`;
        }
    }
}
