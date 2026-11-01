/**
 * VisionSphere — Clean Webcam Stream Controller
 */

window.WebcamController = {
    stream: null,
    video: document.getElementById('webcam-video'),
    btnSnap: document.getElementById('btn-snap'),
    btnContinuous: document.getElementById('btn-continuous'),
    statusContinuous: document.getElementById('continuous-status'),
    modal: document.getElementById('camera-modal'),

    isContinuous: false,
    intervalId: null,

    init() {
        if (!this.btnSnap) return;
        this.btnSnap.addEventListener('click', () => this.takeSnapshot(true));
        this.btnContinuous.addEventListener('click', () => this.toggleContinuous());
    },

    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
                audio: false
            });
            this.video.srcObject = this.stream;
        } catch (err) {
            console.error('Camera access error:', err);
            showToast('Unable to access camera: ' + err.message, 'error');
            if (this.modal) this.modal.classList.add('hidden');
        }
    },

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.isContinuous) {
            this.toggleContinuous();
        }
        if (this.video) this.video.srcObject = null;
    },

    captureFrame() {
        if (!this.video || !this.stream) return null;
        const canvas = document.createElement('canvas');
        canvas.width = this.video.videoWidth || 640;
        canvas.height = this.video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg', 0.85);
    },

    takeSnapshot(closeModal = true) {
        const frameB64 = this.captureFrame();
        if (frameB64) {
            if (closeModal) {
                this.stopCamera();
                if (this.modal) this.modal.classList.add('hidden');
            }
            classifyImage({ image: frameB64 });
        }
    },

    toggleContinuous() {
        this.isContinuous = !this.isContinuous;
        if (this.isContinuous) {
            this.statusContinuous.textContent = 'ON';
            this.statusContinuous.className = 'text-emerald-400 font-mono animate-pulse';
            this.intervalId = setInterval(() => {
                this.takeSnapshot(false);
            }, 1800);
            showToast('Continuous live stream enabled', 'info');
        } else {
            this.statusContinuous.textContent = 'OFF';
            this.statusContinuous.className = 'text-rose-400 font-mono';
            clearInterval(this.intervalId);
            this.intervalId = null;
            showToast('Continuous stream stopped', 'info');
        }
    }
};

document.addEventListener('DOMContentLoaded', () => {
    window.WebcamController.init();
});
