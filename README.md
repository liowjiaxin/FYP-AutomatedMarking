# Automated Marking System

An automated system for grading programming assignments, built with FastAPI backend and modern web frontend.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage Guide](#usage-guide)
- [Local Batch Marking](#local-batch-marking)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)

## Overview

The Automated Marking System is designed to streamline the process of grading programming assignments. It provides:
- Real-time code execution in isolated Docker containers
- Support for multiple programming languages (Python, C, Java)
- Automated grading based on predefined marking schemes
- WebSocket-based live execution feedback
- Secure file handling and result storage
- Batch processing for multiple submissions

## Features

### Frontend Interface
- Upload student submissions in ZIP format
- Upload and select marking schemes
- Choose programming language (Python, C, Java)
- View marks and feedback
- Real-time code execution
- Live logs display
- Optional stdin file upload

### Backend API
- FastAPI-based RESTful API
- WebSocket support for real-time execution
- Automated code grading
- Support for multiple programming languages
- Secure file handling
- Database integration for result storage

### Evaluation Capabilities
- BLEURT-based feedback quality assessment
- Comparison of automated feedback with teacher-written feedback
- Quantitative evaluation of feedback similarity
- Support for batch evaluation of multiple submissions
- Detailed evaluation reports and metrics

## Getting Started

### Quick Setup (Recommended)
1. Install [Docker](https://docs.docker.com/engine/install/)
2. Clone and cd into the repo:
   ```bash
   git clone [repository-url]
   cd FYP-AutomatedMarking
   ```
3. Start the system:
   ```bash
   docker compose up --build -d
   ```
4. Open a web browser and go to [localhost:8080](http://localhost:8080)

### Manual Setup (Development)
#### Prerequisites
- Python 3.8+
- Docker
- Node.js (for frontend development)

#### Backend Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Start the backend server:
   ```bash
   uvicorn app.main:create_application --reload
   ```

#### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Start the development server:
   ```bash
   python -m http.server 8000
   ```

## Usage Guide

1. Access the web interface at `http://localhost:8080`
2. Upload student submissions and marking schemes
3. Select programming language and version
4. Run code and view results
5. Check grading results and feedback

## Local Batch Marking

The system provides a local batch marking feature for processing submissions without web interface. This is the recommended way to grade large batches of student submissions.

### Features
- Process submissions directly from command line
- No web server required
- Faster processing for large batches
- Direct file system access
- Command-line progress monitoring
- Support for custom marking schemes
- Detailed logging and error reporting
- Resume interrupted batch processing
- LLM-based grading using API key

### Quick Start
1. Set up API key:
   - Create a `.env` file in the project root directory
   - Add your API key: `API_KEY=your_api_key_here`

2. Prepare your files:
   - Place marking scheme PDF in `marking_schemes/` directory
   - Place student ZIP files in `grading_data/` directory
   - Each ZIP should contain the student's code files
   - ZIP filename should be the student ID (e.g., `student123.zip`)

3. Run the batch marking:
   ```bash
   cd FYP-AutomatedMarking/local-batch-marking
   python batch_rag_mark.py
   ```

The script will:
- Process all submissions in alphabetical order
- Show progress in the terminal
- Generate results in the `results/` directory
- Create a summary report when complete

### Results
After running the script, check the `results/` directory for:
- `grading_results_assignmentX.json`: Overall statistics and summary
- Individual student result files (e.g., `student123.json`)
- Error logs for failed submissions

## Project Structure

The project is organized into three main components: backend server, frontend interface, and local batch marking system.

```
FYP-AutomatedMarking/
├── backend/                    # FastAPI backend server
│   ├── app/
│   │   ├── api/              # API endpoints and routes
│   │   │   ├── endpoints/    # API endpoint handlers
│   │   │   ├── models/       # Data models and schemas
│   │   │   └── dependencies.py
│   │   ├── core/            # Core application logic
│   │   ├── db/              # Database models and operations
│   │   ├── rag/             # RAG implementation
│   │   │   ├── rag_grading.py    # Main RAG grading logic
│   │   │   ├── rag_llm.py        # LLM integration
│   │   │   ├── retriever.py      # Vector store and retrieval
│   │   │   ├── document_loader.py # Document loading and processing
│   │   │   ├── vector_store/     # FAISS vector store directory
│   │   │   └── datasets/         # RAG training datasets
│   │   └── grader/          # Traditional grading logic
│   └── requirements.txt      # Python dependencies
│
├── frontend/                  # Web interface
│   ├── static/              # Static assets
│   │   ├── css/            # Stylesheets
│   │   └── js/             # JavaScript files
│   └── templates/          # HTML templates
│
├── local-batch-marking/       # Local batch grading system
│   ├── batch_rag_mark.py    # Main marking script with RAG
│   ├── rag_grader.py        # RAG grading implementation
│   ├── marking_schemes/     # PDF marking schemes
│   │   └── assignment1.pdf  # Example marking scheme
│   ├── grading_data/        # Student submissions
│   │   └── assignment1/     # Organized by assignment
│   │       ├── student1.zip
│   │       ├── student2.zip
│   │       └── student3.zip
│   └── results/            # Grading results
│       ├── grading_results_assignment1.csv
│       └── grading_results_assignment1.json
│
├── .env                     # API key configuration
└── evaluation/             # Evaluation tools and metrics
    ├── bleurt_evaluation.ipynb  # BLEURT-based feedback quality assessment
    └── zip_student_codes/  # Legacy student code storage
```

### Component Details

#### Backend Server
- FastAPI-based RESTful API
- Handles code execution and grading
- Manages database operations
- Provides WebSocket support for real-time updates
- RAG-based grading system for contextual feedback

#### Frontend Interface
- Modern web interface
- Real-time code execution feedback
- File upload and management
- Results visualization

#### Local Batch Marking
- Command-line based grading system
- Processes multiple submissions in batch
- Generates detailed reports
- Supports multiple assignment types
- RAG-enhanced grading for consistent feedback

### File Organization

#### Marking Schemes
- Stored in `local-batch-marking/marking_schemes/`
- PDF format for easy viewing
- One file per assignment
- Contains grading criteria and rubrics

#### Student Submissions
- Organized by assignment in `local-batch-marking/grading_data/`
- Each assignment has its own directory
- Submissions in ZIP format
- Filename format: `studentID.zip`

#### Results
- Generated in `local-batch-marking/results/`
- Available in both CSV and JSON formats
- Includes detailed feedback and scores
- Organized by assignment

## RAG Implementation

The system uses Retrieval Augmented Generation (RAG) to provide more contextual and consistent grading feedback. The RAG implementation consists of several key components:

### Core Components

1. **RAG Grader (`rag_grading.py`)**
   - Main grading logic that combines traditional grading with RAG
   - Processes submissions and marking schemes
   - Integrates with LLM for feedback generation
   - Handles metadata extraction and analysis

2. **Retriever (`retriever.py`)**
   - Manages the FAISS vector store for similar submissions
   - Implements semantic search for relevant examples
   - Stores and retrieves past submissions with metadata
   - Handles document embeddings and similarity matching

3. **Document Loader (`document_loader.py`)**
   - Processes student submissions and marking schemes
   - Extracts text from various file formats (PDF, text, code)
   - Prepares documents for vector storage
   - Handles file encoding and formatting

4. **LLM Integration (`rag_llm.py`)**
   - Interfaces with the language model
   - Formats prompts with retrieved context
   - Processes LLM responses
   - Ensures consistent output format

### RAG Pipeline

1. **Document Processing**
   - Load and process student submission
   - Extract code and metadata
   - Process marking scheme

2. **Context Retrieval**
   - Create query from submission and marking scheme
   - Retrieve similar past submissions
   - Select most relevant examples

3. **Grading with Context**
   - Combine submission with retrieved examples
   - Generate contextual feedback
   - Apply consistent grading standards

4. **Result Storage**
   - Store grading results
   - Update vector store with new submission
   - Generate detailed feedback

### Benefits of RAG

- **Consistent Grading**: Uses past examples to maintain grading standards
- **Contextual Feedback**: Provides more relevant and detailed feedback
- **Learning from Past**: Improves over time as more submissions are processed
- **Scalable**: Efficiently handles large numbers of submissions
- **Adaptable**: Can be fine-tuned for different programming languages and assignments

## API Endpoints

### Main Endpoints
- `POST /api/upload` - Upload files
- `POST /api/grade` - Grade code submissions
- `GET /api/result/{student_id}` - Get student results
- `WS /ws/run-code` - WebSocket for real-time code execution

## Troubleshooting

### Common Issues
1. **Docker Compose Fails to Start**
   - Ensure Docker is running
   - Check if ports 8080 and 8000 are available
   - Try `docker compose down` and then `docker compose up --build`

2. **Code Execution Timeout**
   - Increase the timeout duration in the interface
   - Check if the code has infinite loops
   - Verify input/output file formats

3. **File Upload Issues**
   - Ensure ZIP files are properly formatted
   - Check file size limits
   - Verify file permissions

4. **API Key Issues**
   - Verify the API key is correctly set in `.env`
   - Check for any rate limiting
   - Ensure the API key has sufficient permissions

## BLEURT Evaluation

The system includes a BLEURT-based evaluation tool to assess the quality of automated feedback by comparing it with teacher-written feedback.

### Features
- Uses BLEURT-20 model for text similarity assessment
- Quantitative evaluation of feedback quality
- Batch processing of multiple submissions
- Detailed evaluation reports
- Visualization of evaluation results

### Usage
1. Prepare feedback files:
   - Automated feedback from the system
   - Corresponding teacher-written feedback

2. Run the evaluation:
   ```bash
   cd evaluation
   jupyter notebook bleurt_evaluation.ipynb
   ```

3. View results:
   - Similarity scores for each feedback pair
   - Aggregate statistics
   - Visualizations of feedback quality
