document.addEventListener('DOMContentLoaded', () => {
  document.getElementById("submit-assignment").addEventListener("click", submitAssignment);
  document.getElementById("stop-run-code").addEventListener("click", stopRunCode);
  document.getElementById("run-code").addEventListener("click", runCode);

  // Initialize Monaco Editor
  require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.47.0/min/vs' } });
  require(['vs/editor/editor.main'], function() {
    window.editor = monaco.editor.create(document.getElementById('monaco-editor'), {
      value: '#!/bin/bash\n# Type your commands here\n# Example: python main.py',
      language: 'shell',
      theme: 'vs',
      minimap: { enabled: false },
      lineNumbers: 'on',
      roundedSelection: false,
      scrollBeyondLastLine: false,
      readOnly: false,
      fontSize: 14,
      automaticLayout: true,
      wordWrap: 'on',
      wrappingStrategy: 'advanced'
    });
  });

  // Initialize Dropzone for file uploads
  Dropzone.autoDiscover = false;
  
  const markingSchemeDropzone = new Dropzone("#marking-scheme-dropzone", {
    url: "/api/upload",
    maxFiles: 1,
    acceptedFiles: ".txt,.pdf,.doc,.docx",
    dictDefaultMessage: "Drop marking scheme here or click to upload"
  });

  const studentAnswerDropzone = new Dropzone("#student-answer-dropzone", {
    url: "/api/upload",
    maxFiles: 1,
    acceptedFiles: ".txt,.py,.java,.c,.cpp",
    dictDefaultMessage: "Drop student's answer here or click to upload"
  });

  const stdinDropzone = new Dropzone("#stdin-file-dropzone", {
    url: "/api/upload",
    maxFiles: 1,
    acceptedFiles: ".txt",
    dictDefaultMessage: "Drop stdin file here or click to upload"
  });

  // Handle file upload success
  markingSchemeDropzone.on("success", function(file, response) {
    document.getElementById('marking-scheme-filename').value = response.filename;
  });

  studentAnswerDropzone.on("success", function(file, response) {
    document.getElementById('student-answer-zip-filename').value = response.filename;
  });

  stdinDropzone.on("success", function(file, response) {
    document.getElementById('stdin-filename').value = response.filename;
  });

  // Handle submit button click
  document.getElementById('submit-assignment').addEventListener('click', async function() {
    const studentId = document.getElementById('student-id').value;
    const markingSchemeFile = document.getElementById('marking-scheme-filename').value;
    const studentAnswerFile = document.getElementById('student-answer-zip-filename').value;
    const stdinFile = document.getElementById('stdin-filename').value;

    if (!studentId || !markingSchemeFile || !studentAnswerFile) {
      alert('Please fill in all required fields and upload necessary files');
      return;
    }

    // Create form data
    const formData = new FormData();
    formData.append('student_id', studentId);
    formData.append('marking_scheme', markingSchemeFile);
    formData.append('student_answer', studentAnswerFile);
    if (stdinFile) {
      formData.append('stdin_file', stdinFile);
    }

    try {
      // Show loading state
      document.getElementById('marks').textContent = 'Processing...';
      document.getElementById('grade').textContent = 'Processing...';
      document.getElementById('feedback').textContent = 'Processing...';

      // Send request to process submission
      const response = await fetch('/api/process-submission', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Failed to process submission');
      }

      const result = await response.json();

      // Update results
      document.getElementById('marks').textContent = result.marks;
      document.getElementById('grade').textContent = result.grade;
      document.getElementById('feedback').textContent = result.feedback;

    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred while processing the submission');
    }
  });
});

document.addEventListener('DOMContentLoaded', () => {
  const languageSelect = document.getElementById('language');
  const versionSelect = document.getElementById('version');

  const pythonVersions = ['3.9', '3.10', '3.11', '3.12', '3.13'];

  languageSelect.addEventListener('change', () => {
    const selectedLanguage = languageSelect.value;
    versionSelect.innerHTML = '';

    if (selectedLanguage === 'Python') {
      versionSelect.disabled = false;
      pythonVersions.forEach(version => {
        const option = document.createElement('option');
        option.value = version;
        option.textContent = version;
        versionSelect.appendChild(option);
      });
    } else {
      versionSelect.disabled = true;
      const option = document.createElement('option');
      option.value = '-';
      option.textContent = '-';
      versionSelect.appendChild(option);
    }
  });

  // Trigger change event to set initial state
  languageSelect.dispatchEvent(new Event('change'));
  versionSelect.value = '-'; // Set default version
});

function submitAssignment() {
  const submitBtn = document.getElementById("submit-assignment");
  const runBtn = document.getElementById("run-code");

  const studentId = document.getElementById("student-id").value;
  const language = document.getElementById("language").value;
  const version = document.getElementById("version").value;
  const studentAnswerZipFilename = document.getElementById("student-answer-zip-filename").value;
  const markingSchemeFilename = document.getElementById("marking-scheme-filename").value;
  const codeRunResultId = document.getElementById("code-run-result-id").value || undefined;

  if (!language || language === '-') {
    alert("Please select a language.");
    return;
  }

  if (language.toLowerCase() == "python" && (!version || version === '-')) {
    alert("Please select a python version.");
    return;
  }

  if (!markingSchemeFilename) {
    alert("Please upload the marking scheme.");
    return;
  }

  if (!studentAnswerZipFilename) {
    alert("Please upload the student's answer.");
    return;
  }

  if (!studentId) {
    alert("Please enter the student ID.");
    return;
  }

  // Set loading state
  setButtonLoading(submitBtn, true);
  runBtn.disabled = true;


  const data = {
    student_id: studentId,
    language,
    version,
    code_zip_filename: studentAnswerZipFilename,
    marking_scheme_filename: markingSchemeFilename,
    code_run_result_id: codeRunResultId
  };

  fetch("/api/grade", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  })
    .then(response => {
      if (!response.ok) {
        return response.text().then(text => {
          throw new Error(`Server error (${response.status}): ${text}`);
        });
      }
      return response.json();
    })
    .then(data => {
      console.log("Received Data:", data); // Debugging: Ensure data is received correctly

      // Get the result card elements
      const marksElement = document.getElementById("marks");
      const gradeElement = document.getElementById("grade");
      const feedbackElement = document.getElementById("feedback");

      if (marksElement && feedbackElement && gradeElement) {
        marksElement.innerText = data.marks;
        gradeElement.innerText = data.grade || calculateGrade(data.marks);
        if (data.feedback) {
          feedbackElement.innerHTML = formatFeedback(data.feedback);
        } else {
          feedbackElement.innerHTML = "<p>No feedback available</p>";
        }
      } else {
        console.error("Error: Result card elements not found!");
        alert("Error displaying the results. Please check the result card elements.");
      }

      alert("Assignment submitted successfully!");
      setButtonLoading(submitBtn, false);
    })
    .catch(error => {
      console.error("Error:", error);
      alert("An error occurred while submitting the assignment: " + error.message);
      setButtonLoading(submitBtn, false);
    });
}

function stopRunCode() {
  // Logic to stop the running code
  alert("Run code stopped.");
}

function setButtonLoading(button, isLoading) {
  if (isLoading) {
    // Store the original text as a data attribute
    button.dataset.originalText = button.textContent;
    button.classList.add('loading');
    button.textContent = '';
  } else {
    button.classList.remove('loading');
    // Restore the original text from the data attribute
    button.textContent = button.dataset.originalText || button.textContent;
  }
  button.disabled = isLoading;
}

async function checkRunCodeImage() {
  const result = await fetch("/api/code-runner-image-healthy");
  const json = await result.json();
  return json.found;
}

async function runCode() {
  // check if the run code docker image is ready
  if (!(await checkRunCodeImage())) {
    alert("The run code docker image not ready yet, it will take around 60 seconds to load, please be patient :)")
    return;
  }

  const runButton = document.getElementById("run-code");
  const submitButton = document.getElementById("submit-assignment");

  // Validate inputs
  const language = document.getElementById("language").value;
  const version = document.getElementById("version").value;
  const studentAnswerZipFilename = document.getElementById("student-answer-zip-filename").value;
  const commands = window.editor.getValue()
    .split('\n')
    .filter(line => line.trim() && !line.startsWith('#') && line !== '#!/bin/bash');
  console.log(commands)

  if (!language || language === '-') {
    alert("Please select a language.");
    return;
  }

  if (!version || version === '-') {
    alert("Please select a python version.");
    return;
  }

  if (!studentAnswerZipFilename) {
    alert("Please upload the student's answer.");
    return;
  }

  if (commands.length === 0) {
    alert("Please enter at least one command to run.");
    return;
  }

  // Set loading state
  setButtonLoading(runButton, true);
  submitButton.disabled = true;

  const stdinFilename = document.getElementById("stdin-filename").value;

  // Create WebSocket connection
  const ws = new WebSocket(`ws://${window.location.host}/ws/run-code`);

  ws.onopen = () => {
    console.log('WebSocket connection established');
    const data = {
      language,
      version,
      code_zip_filename: studentAnswerZipFilename,
      commands: commands,
      stdin_input_filename: stdinFilename || ""
    };
    ws.send(JSON.stringify(data));
  };

  ws.onmessage = (event) => {
    // live log code run output
    const logsContent = document.getElementById('logs-content');
    const contentJson = JSON.parse(event.data);
    if (contentJson.output) {
      logsContent.innerHTML += contentJson.output + '<br>';
      logsContent.scrollTop = logsContent.scrollHeight;
    } else if (contentJson.code_run_id) {
      document.querySelector("#code-run-result-id").value = contentJson.code_run_id;
    }
  };

  ws.onclose = () => {
    console.log('WebSocket connection closed');
    // Reset button states
    setButtonLoading(runButton, false);
    submitButton.disabled = false;
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    alert('An error occurred while running the code. Please try again.');
    // Reset button states
    setButtonLoading(runButton, false);
    submitButton.disabled = false;
  };
}

// Function to determine grade if not provided
function calculateGrade(marks) {
  if (marks >= 90) return "A+";
  if (marks >= 80) return "A";
  if (marks >= 70) return "B";
  if (marks >= 60) return "C";
  if (marks >= 50) return "D";
  return "F";
}

// Function to format feedback
function formatFeedback(feedback) {
  // Convert **bold text** to <strong>bold text</strong>
  feedback = feedback.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Convert lists (* item or - item) to <ul><li>item</li></ul>
  feedback = feedback.replace(/(?:^|\n)([*-])\s*(.*?)(?=\n|$)/g, '<ul><li>$2</li></ul>');

  // Convert numbered lists (1. item) to <ol><li>item</li></ol>
  feedback = feedback.replace(/(?:^|\n)(\d+)\.\s*(.*?)(?=\n|$)/g, '<ol><li>$2</li></ol>');

  // Convert line breaks (\n) to <br>
  feedback = feedback.replace(/\n/g, '<br>');

  return feedback;
}

Dropzone.options.markingSchemeDropzone = {
  paramName: "file",
  maxFilesize: 20, // MB
  success: function(file, response) {
    console.log("File uploaded successfully:", response.info);
    document.getElementById("marking-scheme-filename").value = response.filename;
  },
  error: function(file, response) {
    console.error("File upload error:", response);
    alert("An error occurred while uploading the marking scheme. Please try again.");
  }
};

Dropzone.options.studentAnswerDropzone = {
  paramName: "file",
  maxFilesize: 20, // MB
  success: function(file, response) {
    console.log("File uploaded successfully:", response.info);
    document.getElementById("student-answer-zip-filename").value = response.filename;
  },
  error: function(file, response) {
    console.error("File upload error:", response);
    alert("An error occurred while uploading the student's answer. Please try again.");
  }
};

Dropzone.options.stdinFileDropzone = {
  paramName: "file",
  maxFilesize: 20, // MB
  success: function(file, response) {
    console.log("File uploaded successfully:", response.info);
    document.getElementById("stdin-filename").value = response.filename;
  },
  error: function(file, response) {
    console.error("File upload error:", response);
    alert("An error occurred while uploading the stdin file. Please try again.");
  }
};
