document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const fileInput = document.getElementById('imageFile');
    const noFileMessage = document.getElementById('no-file-message');
    const errorMessage = document.getElementById('error-message');
    const uploadedImage = document.getElementById('uploadedImage');
    const colorizedImage = document.getElementById('colorizedImage');
    const downloadLink = document.getElementById('downloadLink');

    uploadedImage.style.display = 'none';
    colorizedImage.style.display = 'none';
    downloadLink.style.display = 'none';
    noFileMessage.innerText = '';
    errorMessage.innerText = '';

    if (!fileInput.files.length) {
        noFileMessage.innerText = 'Please select an image file.';
        return;
    }

    const file = fileInput.files[0];
    if (!file.type.startsWith('image/')) {
        noFileMessage.innerText = 'Unsupported file type. Please upload an image.';
        return;
    }

    const reader = new FileReader();
    reader.onload = function(event) {
        uploadedImage.src = event.target.result;
        uploadedImage.style.display = 'block';
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            errorMessage.innerText = 'Error: ' + data.error;
        } else {
            colorizedImage.src = `data:image/png;base64,${data.predicted}`;
            colorizedImage.style.display = 'block';
            downloadLink.href = `data:image/png;base64,${data.predicted}`;
            downloadLink.style.display = 'inline-block';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        errorMessage.innerText = 'Error: ' + error.message;
    });
});

document.getElementById('cancelButton').addEventListener('click', function() {
    document.getElementById('uploadForm').reset();
    document.getElementById('uploadedImage').style.display = 'none';
    document.getElementById('colorizedImage').style.display = 'none';
    document.getElementById('downloadLink').style.display = 'none';
    document.getElementById('no-file-message').innerText = '';
    document.getElementById('error-message').innerText = '';
});