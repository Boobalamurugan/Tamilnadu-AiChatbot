const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const micBtn = document.getElementById("mic-btn");
const audioPlayer = document.getElementById("audio-player");

let mediaRecorder; // Global variable to store the MediaRecorder instance

// Send message on button click
sendBtn.addEventListener("click", sendMessage);

// Handle Enter key press
userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
});

// Speech-to-Text functionality
micBtn.addEventListener("click", () => {
    if (!mediaRecorder) {
        startRecording();
    } else {
        stopRecording();
    }
});

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            let audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                const audioUrl = URL.createObjectURL(audioBlob);

                // Send audio to server for transcription
                fetch("/transcribe", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ audio_url: audioUrl })
                })
                .then(response => response.json())
                .then(data => {
                    userInput.value = data.transcript;
                    sendMessage();
                });
            };

            mediaRecorder.start();
            micBtn.textContent = "Stop";
        });
}

function stopRecording() {
    mediaRecorder.stop();
    mediaRecorder = null;
    micBtn.textContent = "🎤 Speak";
}

function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    // Display user message
    chatBox.innerHTML += `<div class="user-message">${message}</div>`;
    userInput.value = "";

    // Get bot response
    fetch("/get_response", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: message })
    })
    .then(response => response.json())
    .then(data => {
        const botMessage = data.response;
        const audioFile = data.audio_file;

        // Display bot message
        chatBox.innerHTML += `<div class="bot-message">${botMessage}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;

        // Play bot's audio response
        audioPlayer.src = audioFile;
        audioPlayer.play();
    });
}