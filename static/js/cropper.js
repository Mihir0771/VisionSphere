/**
 * VisionSphere — Interactive ROI Object Cropper
 */

const CropperController = {
    canvas: document.getElementById('crop-canvas'),
    img: document.getElementById('main-image'),
    cropBar: document.getElementById('crop-bar'),
    btnToggle: document.getElementById('btn-crop-mode'),
    btnCancel: document.getElementById('btn-cancel-crop'),
    btnConfirm: document.getElementById('btn-confirm-crop'),

    active: false,
    isDrawing: false,
    startX: 0,
    startY: 0,
    currentBox: null,

    init() {
        if (!this.canvas || !this.btnToggle) return;

        this.ctx = this.canvas.getContext('2d');

        this.btnToggle.addEventListener('click', () => this.toggleMode());
        this.btnCancel.addEventListener('click', () => this.disableMode());
        this.btnConfirm.addEventListener('click', () => this.submitCrop());

        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', () => this.onMouseUp());
    },

    toggleMode() {
        if (!this.img.src || this.img.classList.contains('hidden')) {
            showToast('Please load an image before cropping', 'info');
            return;
        }
        this.active ? this.disableMode() : this.enableMode();
    },

    enableMode() {
        this.active = true;
        this.syncCanvasSize();
        this.canvas.classList.remove('pointer-events-none');
        this.cropBar.classList.remove('hidden');
        this.btnToggle.classList.add('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500/50');
        showToast('Click and drag on image to select object ROI', 'info');
    },

    disableMode() {
        this.active = false;
        this.canvas.classList.add('pointer-events-none');
        this.cropBar.classList.add('hidden');
        this.btnToggle.classList.remove('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500/50');
        this.clearCanvas();
        this.currentBox = null;
    },

    syncCanvasSize() {
        const rect = this.img.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.canvas.style.width = `${rect.width}px`;
        this.canvas.style.height = `${rect.height}px`;
        this.canvas.style.top = `${this.img.offsetTop}px`;
        this.canvas.style.left = `${this.img.offsetLeft}px`;
    },

    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    },

    onMouseDown(e) {
        if (!this.active) return;
        const rect = this.canvas.getBoundingClientRect();
        this.startX = e.clientX - rect.left;
        this.startY = e.clientY - rect.top;
        this.isDrawing = true;
    },

    onMouseMove(e) {
        if (!this.active || !this.isDrawing) return;
        const rect = this.canvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;

        const x = Math.min(this.startX, currentX);
        const y = Math.min(this.startY, currentY);
        const w = Math.abs(currentX - this.startX);
        const h = Math.abs(currentY - this.startY);

        this.currentBox = { x, y, w, h };

        this.clearCanvas();

        // Draw HUD selection box
        this.ctx.strokeStyle = '#06b6d4';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(x, y, w, h);

        this.ctx.fillStyle = 'rgba(6, 182, 212, 0.15)';
        this.ctx.fillRect(x, y, w, h);
    },

    onMouseUp() {
        this.isDrawing = false;
    },

    async submitCrop() {
        if (!this.currentBox || this.currentBox.w < 10 || this.currentBox.h < 10) {
            showToast('Please drag a larger selection box', 'error');
            return;
        }

        // Map canvas coordinates to natural image coordinates
        const scaleX = this.img.naturalWidth / this.canvas.width;
        const scaleY = this.img.naturalHeight / this.canvas.height;

        const cropPayload = {
            image: AppState.rawImageB64 || this.img.src,
            x: Math.round(this.currentBox.x * scaleX),
            y: Math.round(this.currentBox.y * scaleY),
            width: Math.round(this.currentBox.w * scaleX),
            height: Math.round(this.currentBox.h * scaleY),
            model: AppState.currentModel
        };

        this.disableMode();

        try {
            DOM.loadingOverlay.classList.remove('hidden');
            const res = await fetch('/api/crop-classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cropPayload)
            });
            const data = await res.json();
            if (data.status !== 'success') throw new Error(data.message || 'Crop classification failed');

            renderClassificationResults({
                image_b64: data.cropped_b64,
                diagnostics: data.diagnostics,
                top_prediction: data.top_prediction,
                predictions: data.predictions,
                telemetry: data.telemetry
            });
            showToast(`Cropped Object Classified: ${data.top_prediction.label}`, 'success');
        } catch (err) {
            console.error(err);
            showToast(err.message, 'error');
        } finally {
            DOM.loadingOverlay.classList.add('hidden');
        }
    }
};

document.addEventListener('DOMContentLoaded', () => {
    CropperController.init();
});
