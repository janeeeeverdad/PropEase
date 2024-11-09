// TOGGLE MENU
function toggleMenu() {
    const menu = document.getElementById('menu');
    menu.style.left = menu.style.left === '0px' ? '-300px' : '0px';
}

// Close menu when clicking outside of it
document.addEventListener('click', function(event) {
    const menu = document.getElementById('menu');
    const icon = document.querySelector('.menu-icon');
    if (!menu.contains(event.target) && !icon.contains(event.target)) {
        menu.style.left = '-300px'; // Hide the menu
    }
});

//PROPOSAL PAGE -- UPLOADING A FILES
function displayUploadedFiles() {
    const fileInput = document.getElementById('file-upload');
    const fileList = document.getElementById('file-list');
    fileList.innerHTML = ''; // Clear existing files
    
    Array.from(fileInput.files).forEach((file) => {
        const fileContainer = document.createElement('div');
        
        // Link wrapper for file bubble
        const fileLink = document.createElement('a');
        fileLink.href = 'proposal_view.html'; 
        fileLink.className = 'file-bubble'; 

        // File icon
        const fileIcon = document.createElement('i');
        fileIcon.className = 'fas fa-file-alt file-icon'; 
        fileLink.appendChild(fileIcon);
        
        // File caption
        const fileCaption = document.createElement('div');
        fileCaption.className = 'file-caption';
        fileCaption.textContent = file.name; 
        
        fileContainer.appendChild(fileLink);
        fileContainer.appendChild(fileCaption);
        fileList.appendChild(fileContainer);
    });
}
function openFileView(file) {
    // Redirect to the second interface page with the file name as a parameter
    window.location.href = `proposal_view.html?file=${encodeURIComponent(file.name)}`;
}

// PROPOSAL PAGE -- VIEWING UPLOADED FILES
// Retrieve File Name from URL
document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const fileName = urlParams.get('file');
    if (fileName) {
        document.getElementById('file-view').innerHTML = `<p>Viewing file: ${fileName}</p>`;
    }
    // Ensure modal remains hidden initially
    const modal = document.getElementById("recommendationModal");
    modal.style.display = "none";
});

// PROPOSAL PAGE -- RECOMMENDATION POP UP
function toggleModal() {
    const modal = document.getElementById("recommendationModal");
    modal.style.display = "flex";
}
function closeModal() {
    const modal = document.getElementById("recommendationModal");
    modal.style.display = "none";
}
window.onclick = function(event) {
    const modal = document.getElementById("recommendationModal");
    if (event.target === modal) {
        modal.style.display = "none";
    }
};

// PROPOSAL PAGE -- SPEECH TO TEXT
let recognition;
let isRecognizing = false;

function initializeSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = function() {
            console.log('Speech recognition started...');
            isRecognizing = true; // Set to true when recognition starts
        };

        recognition.onend = function() {
            console.log('Speech recognition ended.');
            isRecognizing = false; // Set to false when recognition ends
        };

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            console.log('Speech recognized:', transcript);
            document.getElementById('yourTextAreaId').value = transcript; // Replace 'yourTextAreaId' with the target element ID
        };

        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            stopSpeechRecognition(); // Stop recognition on error
        };
    } else {
        alert('Your browser does not support speech recognition. Please use Chrome or another compatible browser.');
    }
}

function toggleSpeechToText() {
    if (!recognition) {
        initializeSpeechRecognition();
    }

    if (isRecognizing) {
        stopSpeechRecognition();
    } else {
        startSpeechRecognition();
    }
}

function startSpeechRecognition() {
    if (recognition && !isRecognizing) {
        recognition.start();
    }
}

function stopSpeechRecognition() {
    if (recognition && isRecognizing) {
        recognition.stop();
    }
}

