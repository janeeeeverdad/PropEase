document.addEventListener('DOMContentLoaded', function () {
    const fileUpload = document.getElementById('file-upload');
    const fileList = document.getElementById('file-list');
    const popup = document.getElementById('popup');
    let selectedFileElement;

    fileUpload.addEventListener('change', displayUploadedFiles);

    function displayUploadedFiles() {
        const files = fileUpload.files;
        fileList.innerHTML = ''; // Clear the list before adding new items

        Array.from(files).forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.classList.add('file-item');
            fileItem.dataset.id = index;
            fileItem.innerHTML = `<img src="file-icon.png" alt="file"><span>${file.name}</span>`;
            fileItem.addEventListener('click', (event) => {
                event.stopPropagation();
                selectedFileElement = fileItem;
                const rect = fileItem.getBoundingClientRect();
                popup.style.top = `${rect.bottom + window.scrollY}px`;
                popup.style.left = `${rect.left}px`;
                popup.style.display = 'block';
            });
            fileList.appendChild(fileItem);
        });
    }

    document.addEventListener('click', function (event) {
        if (!popup.contains(event.target) && !Array.from(fileList.children).some(item => item.contains(event.target))) {
            popup.style.display = 'none';
        }
    });

    window.openFile = function () {
        alert('Open file: ' + selectedFileElement.querySelector('span').textContent);
        // Add your logic to open the file here
    };

    window.deleteFile = function () {
        const fileId = selectedFileElement.dataset.id;
        // Simulate moving the file to the archive page
        window.location.href = 'archive.html?fileId=' + fileId;
        selectedFileElement.remove();
        popup.style.display = 'none';
    };
});
