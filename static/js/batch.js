/**
 * VisionSphere — Batch Classification Controller
 */

let batchResults = [];

document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('batch-dropzone');
    const fileInput = document.getElementById('batch-file-input');
    const modelSelect = document.getElementById('batch-model');
    const tbody = document.getElementById('batch-tbody');
    const batchCount = document.getElementById('batch-count');
    const progressContainer = document.getElementById('batch-progress-container');
    const progressBar = document.getElementById('batch-progress-bar');
    const progressText = document.getElementById('batch-progress-text');
    const progressPct = document.getElementById('batch-progress-pct');
    const btnCsv = document.getElementById('btn-export-csv');
    const btnJson = document.getElementById('btn-export-json');

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
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        if (files.length > 0) processBatch(files);
    });

    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) processBatch(files);
    });

    btnCsv.addEventListener('click', () => exportCSV());
    btnJson.addEventListener('click', () => exportJSON());

    async function processBatch(files) {
        progressContainer.classList.remove('hidden');
        progressBar.style.width = '10%';
        progressText.textContent = `Uploading & processing ${files.length} images...`;
        progressPct.textContent = '10%';

        const formData = new FormData();
        formData.append('model', modelSelect.value);
        files.forEach(f => formData.append('images', f));

        try {
            progressBar.style.width = '45%';
            progressPct.textContent = '45%';

            const res = await fetch('/api/batch-classify', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            if (data.status !== 'success') throw new Error(data.message || 'Batch classification failed.');

            progressBar.style.width = '100%';
            progressPct.textContent = '100%';
            progressText.textContent = `Completed ${data.count} images.`;

            batchResults = data.results;
            batchCount.textContent = data.count;
            btnCsv.disabled = false;
            btnJson.disabled = false;

            renderTable(data.results);
            showToast(`Batch completed: ${data.count} items`, 'success');
        } catch (err) {
            console.error(err);
            progressText.textContent = 'Error: ' + err.message;
            showToast(err.message, 'error');
        }
    }

    function renderTable(results) {
        tbody.innerHTML = '';
        results.forEach(r => {
            const tr = document.createElement('tr');
            tr.className = 'hover:bg-slate-900/50 transition-colors';
            tr.innerHTML = `
                <td class="p-3.5">
                    <img src="${r.thumbnail_b64}" class="w-10 h-10 object-cover rounded-lg border border-slate-700/60">
                </td>
                <td class="p-3.5 font-mono text-slate-300">${r.filename}</td>
                <td class="p-3.5 font-bold text-white text-sm">${r.top_label}</td>
                <td class="p-3.5 font-mono text-cyan-400 font-semibold">${r.confidence_pct}%</td>
                <td class="p-3.5">
                    <span class="px-2 py-0.5 rounded text-[10px] font-semibold" style="border: 1px solid ${r.badge.color}; color: ${r.badge.color}">
                        ${r.badge.label}
                    </span>
                </td>
                <td class="p-3.5 font-mono text-slate-400 text-[11px]">${r.inference_time_ms} ms</td>
            `;
            tbody.appendChild(tr);
        });
    }

    function exportCSV() {
        if (!batchResults.length) return;
        let csv = 'Filename,Top_Label,Confidence_Pct,Verdict,Inference_Time_ms\n';
        batchResults.forEach(r => {
            csv += `"${r.filename}","${r.top_label}",${r.confidence_pct},"${r.badge.label}",${r.inference_time_ms}\n`;
        });

        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', `visionsphere_batch_${Date.now()}.csv`);
        link.click();
    }

    function exportJSON() {
        if (!batchResults.length) return;
        const blob = new Blob([JSON.stringify(batchResults, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', `visionsphere_batch_${Date.now()}.json`);
        link.click();
    }
});
