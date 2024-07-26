async function processInput() {
  const question = document.getElementById('question').value;
  const file = document.getElementById('fileInput').files[0];
  const image = document.getElementById('imageInput').files[0];
  

  const formData = new FormData();
  formData.append('question', question);

  if (file) { 
    formData.append('file', file);
  }

  if (image) {
    formData.append('image', image);
  }

  // Show loading text
  document.getElementById('response').textContent = 'Loading...';

  try {
    const response = await fetch('/process_file_and_get_answer/', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const responseData = await response.json();
    // Apply bold formatting 
    document.getElementById('response').textContent = responseData.response_text.replace(/\bex\b/g, "<strong>$1</strong>");
    //document.getElementById('response').textContent = responseData.response_text;
  } catch (error) {
    console.error('Error:', error);
    document.getElementById('response').textContent = 'An error occurred while processing the request.';
  }
}

// Image Preview
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');

imageInput.addEventListener('change', () => {
previewFile(imageInput, imagePreview);
});

// File Preview (Modify for your file types)
const fileInput = document.getElementById('fileInput');
const filePreview = document.getElementById('filePreview');

fileInput.addEventListener('change', () => {
previewFile(fileInput, filePreview);
});

function previewFile(input, previewElement) {
const file = input.files[0];
if (file) {
  const reader = new FileReader();

  // Add PDF Check
  if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
    previewElement.classList.add('pdf-file'); 
  } else {
    previewElement.classList.remove('pdf-file');
  }

  reader.onload = function(event) {
    if (file.type.startsWith('image/')) {
      previewElement.innerHTML = '<img src="' + event.target.result + '" width="200" />';
    } else {
       previewElement.textContent = "File: " + file.name; // Simple text preview for now
    }
  };

  reader.readAsDataURL(file);
}
}

// Clear Form Functionality
function clearForm() {
document.getElementById('question').value = '';
document.getElementById('fileInput').value = '';
document.getElementById('imageInput').value = '';
document.getElementById('response').textContent = '';
document.getElementById('filePreview').textContent = '';
document.getElementById('imagePreview').textContent = '';
}
