// /app/static/script.js

document.addEventListener('DOMContentLoaded', () => {
    // Establish a connection to the server's Socket.IO
    const socket = io();

    // Get references to all the HTML elements we'll be interacting with
    const videoFeed = document.getElementById('video-feed');
    const supervisionOverlay = document.getElementById('supervision-overlay');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const timeDisplay = document.getElementById('time-display');
    const fpsStat = document.getElementById('fps-stat');
    const facesStat = document.getElementById('faces-stat');
    const alertsLog = document.getElementById('alerts-log');

    // --- Socket.IO Event Listeners ---

    socket.on('connect', () => {
        console.log('Connected to server!');
        statusDot.classList.remove('disconnected');
        statusDot.classList.add('connected');
        statusText.textContent = 'CONNECTED';
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server.');
        statusDot.classList.remove('connected');
        statusDot.classList.add('disconnected');
        statusText.textContent = 'DISCONNECTED';
        stopSupervisionUI();
    });

    // This event receives the live video frame from the server
    socket.on('video_frame', (data) => {
        videoFeed.src = `data:image/jpeg;base64,${data.image}`;
        fpsStat.textContent = data.fps;
        facesStat.textContent = data.face_count;
    });

    // This event receives new alerts from the server
    socket.on('new_alert', (alert) => {
        const placeholder = alertsLog.querySelector('.alert-placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        const alertItem = document.createElement('div');
        alertItem.classList.add('alert-item', `severity-${alert.severity}`);
        
        alertItem.innerHTML = `
            <div class="timestamp">${new Date(alert.timestamp).toLocaleTimeString()}</div>
            <div class="message">${alert.type}</div>
            <div class="details">${alert.message}</div>
        `;
        
        alertsLog.prepend(alertItem); // Add new alerts to the top
    });

    socket.on('supervision_stopped', () => {
        stopSupervisionUI();
        console.log('Supervision stopped by server.');
    });

    // --- UI Control Functions ---

    function startSupervisionUI() {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        supervisionOverlay.style.display = 'none';
        // Clear any old alerts
        alertsLog.innerHTML = '<div class="alert-placeholder">Monitoring...</div>';
    }

    function stopSupervisionUI() {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        supervisionOverlay.style.display = 'flex';
        videoFeed.src = "static/placeholder.png"; // Reset to placeholder
        fpsStat.textContent = '--';
        facesStat.textContent = '--';
    }

    // --- Event Handlers for Buttons ---

    startBtn.addEventListener('click', () => {
        console.log('Requesting to start supervision...');
        socket.emit('start_supervision');
        startSupervisionUI();
    });

    stopBtn.addEventListener('click', () => {
        console.log('Requesting to stop supervision...');
        socket.emit('stop_supervision');
        stopSupervisionUI();
    });

    // --- Utility: Clock ---
    function updateTime() {
        timeDisplay.textContent = new Date().toLocaleTimeString();
    }
    setInterval(updateTime, 1000);
    updateTime();
});
