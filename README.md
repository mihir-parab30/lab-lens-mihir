# Lab Lens: AI-Powered Healthcare Intelligence Platform

Multi-Modal Healthcare Intelligence Platform for Medical Report Summarization and Diagnostic Image Analysis

---

## 🏥 Project Overview

Lab Lens is an end-to-end MLOps pipeline for healthcare that combines medical report summarization with diagnostic image analysis. The platform processes MIMIC-III clinical data to provide intelligent insights while ensuring fairness and bias mitigation in healthcare AI systems.

### 🎯 Key Features

- **Medical Report Summarization**: Simplifying discharge summaries using MIMIC-III clinical notes
- **RAG-Powered Patient Q&A**: Answer patient questions about discharge summaries using semantic search
- **Risk Prediction**: Patient risk assessment for readmission and complications
- **Automated Bias Detection**: Comprehensive bias analysis and mitigation
- **Data Quality Validation**: Robust data validation and quality assurance
- **Production-Ready Pipeline**: Complete MLOps pipeline with error handling and logging

### 👥 Team Members

- **Asad Ullah Waraich** 
- **Shruthi Kashetty** 
- **Mihir Harishankar Parab** 
- **Sri Lakshmi Swetha Jalluri** 
- **Dhruv Rameshbhai Gajera** 
- **Shahid Kamal** 

## 🌐 Web Interface Quick Start

To run the Lab Lens web interface locally:

```bash
# Clone repository
git clone https://github.com/kamalshahidnu/lab-lens.git
cd lab-lens

# Install dependencies
pip install -r requirements.txt
pip install pdfplumber

# Set API key (create .env file)
echo "GEMINI_API_KEY=your-api-key-here" > .env

# Run the web interface
streamlit run scripts/file_qa_web.py
```

For detailed setup instructions, see **[WEB_INTERFACE_SETUP.md](./WEB_INTERFACE_SETUP.md)**

**Features:**
- Upload PDF/text documents and ask questions
- Automatic document summarization using MedicalSummarizer
- BioBERT embeddings for better medical text understanding
- Medical term simplification for patient-friendly answers

## 📊 Project Components

### 1. Medical Report Summarization
- **Dataset**: MIMIC-III v1.4 clinical notes and discharge summaries
- **Processing**: Text cleaning, section extraction, abbreviation expansion
- **Features**: 26+ engineered features including complexity metrics
- **Quality**: 95/100 validation score with comprehensive bias analysis

### 2. Chest X-ray Classification
- **Dataset**: MedMNIST-ChestMNIST pre-processed 28×28 chest X-rays
- **Optimization**: CPU-optimized for efficient processing
- **Coverage**: Multiple pathology detection capabilities

### 3. Automated Bias Detection & Mitigation
- **Demographic Analysis**: Gender, ethnicity, age, insurance type
- **Statistical Testing**: T-tests, ANOVA for significance detection
- **Automated Mitigation**: Stratified sampling, oversampling, weighting
- **Compliance Monitoring**: Real-time bias score tracking

### 4. Data Quality Assurance
- **Validation Pipeline**: Schema validation, completeness checks
- **Error Handling**: Comprehensive error management and recovery
- **Logging System**: Centralized logging with performance metrics
- **Monitoring**: Continuous data quality monitoring

### 5. Risk Prediction Model
- **Purpose**: Predicts patient risk levels for readmission or complications
- **Methods**: Rule-based scoring and Gemini AI-based analysis (with fallback)
- **Output**: Risk level (LOW/MEDIUM/HIGH) and risk score (0.0-1.0)
- **Factors Considered**: Age, clinical conditions, lab values, diagnoses, complexity, urgency

### 6. RAG-Powered Patient Q&A System
- **Purpose**: Enable patients to ask questions about their discharge summaries
- **Technology**: Retrieval-Augmented Generation with semantic search
- **Features**: Single-patient mode, patient-friendly answers, source citations
- **Performance**: Fast (5-10 sec), efficient (<100 MB memory per patient)

## 🎯 Risk Prediction Methodology

The risk prediction model assigns risk metrics using a weighted scoring system that combines multiple clinical and demographic factors.

### Risk Scoring System

The model uses two approaches:
1. **Rule-Based Approach (Default)**: Weighted sum of clinical factors
2. **Gemini AI-Based Approach (Optional)**: Advanced AI analysis of discharge summaries

### Scoring Factors & Weights

| Factor | Weight | Threshold | Description |
|--------|--------|-----------|-------------|
| **Age** | +0.30 | Age ≥ 75 | High risk age |
| | +0.15 | Age ≥ 65 | Medium risk age |
| **High-Risk Keywords** | +0.40 | ≥3 keywords | Very high condition risk |
| | +0.30 | ≥2 keywords | High condition risk |
| | +0.20 | ≥1 keyword | Medium condition risk |
| **Abnormal Lab Values** | +0.20 | ≥3 abnormal labs | High lab risk |
| | +0.10 | ≥1 abnormal lab | Medium lab risk |
| **Multiple Diagnoses** | +0.15 | ≥5 diagnoses | High diagnosis complexity |
| | +0.10 | ≥3 diagnoses | Medium diagnosis complexity |
| **Complexity Score** | +0.10 | >0.7 | High document complexity |
| | +0.05 | >0.5 | Medium document complexity |
| **Urgency Indicator** | +0.10 | ≥2 | High urgency |
| | +0.05 | ≥1 | Medium urgency |

### Risk Level Classification

After calculating the total weighted score (maximum 1.0):

- **HIGH Risk**: Score ≥ 0.7
- **MEDIUM Risk**: Score ≥ 0.4 and < 0.7
- **LOW Risk**: Score < 0.4

### High-Risk Keywords (23 conditions)

Critical conditions that contribute significantly to risk assessment:
- **Critical Conditions**: sepsis, shock, cardiac arrest, stroke, myocardial infarction
- **Organ Failures**: respiratory failure, renal failure, liver failure, multiorgan failure
- **Severe Indicators**: severe, critical, acute, emergency, icu, intensive care
- **Procedures**: ventilator, intubation, dialysis, transfusion, surgery
- **Complications**: complication, infection, bleeding, hemorrhage

### Medium-Risk Keywords (14 conditions)

Chronic and moderate conditions:
- **Chronic Conditions**: hypertension, diabetes, copd, asthma, chf, congestive heart failure
- **Infections**: pneumonia, uti, urinary tract infection
- **Other**: dehydration, electrolyte imbalance, anemia, fever, infection

### Example Risk Calculation

For a 28-year-old male patient with multiple stab wounds:
- **Age (28)**: +0.00 (below threshold)
- **High-risk keywords (5 found)**: +0.40 (surgery, emergency, bleeding, etc.)
- **Abnormal labs (382)**: +0.20 (≥3 labs)
- **Diagnoses (9)**: +0.15 (≥5 diagnoses)
- **Complexity score (high)**: +0.10
- **Urgency indicator (high)**: +0.10

**Total Score**: 0.00 + 0.40 + 0.20 + 0.15 + 0.10 + 0.10 = **0.95**

**Risk Level**: **HIGH** (0.95 ≥ 0.7) ✓

### Risk Factors Extracted

The model extracts the following factors from each discharge record:
- `age`: Patient age at admission
- `abnormal_labs`: Count of abnormal laboratory values
- `diagnosis_count`: Number of diagnoses
- `text_length`: Length of discharge summary text
- `has_medications`: Whether medications are documented
- `has_follow_up`: Whether follow-up care is scheduled
- `complexity_score`: Document complexity metric (0-1)
- `urgency_indicator`: Urgency level indicator (0-3)
- `high_risk_keywords`: Count of high-risk condition keywords found
- `medium_risk_keywords`: Count of medium-risk condition keywords found
- `total_risk_keywords`: Total keywords identified

### Gemini AI-Based Risk Prediction

When enabled, the Gemini AI approach:
- Analyzes full discharge summary text (up to 3000 characters)
- Considers clinical context, relationships, and nuance
- Returns structured risk assessment with:
  - Risk level (LOW/MEDIUM/HIGH)
  - Risk score (0.0-1.0)
  - Key risk factors identified
  - Clinical recommendations
- Falls back to rule-based method if unavailable

### Usage

```bash
# Test risk prediction with MIMIC data
python scripts/test_complete_model.py

# Test with specific record
python scripts/test_complete_model.py --index 5

# Test with specific HADM ID
python scripts/test_complete_model.py --hadm-id 149188
```

## 🗂️ Project Structure

This repository follows standard MLOps best practices with clear separation of concerns:

```
lab-lens/
├── 📁 data/                          # Data storage (gitignored)
│   ├── raw/                          # Raw data from sources
│   ├── processed/                     # Processed/cleaned data
│   └── external/                      # External datasets
│
├── 📁 data_preprocessing/            # Data preprocessing pipeline
│   ├── configs/                       # Preprocessing configurations
│   ├── scripts/                       # Preprocessing scripts
│   │   ├── data_acquisition.py        # Data acquisition from BigQuery
│   │   ├── preprocessing.py           # Data cleaning
│   │   ├── validation.py              # Data validation
│   │   ├── feature_engineering.py     # Feature engineering
│   │   ├── bias_detection.py          # Bias detection
│   │   └── main_pipeline.py           # Main orchestration
│   ├── notebooks/                     # Exploration notebooks
│   └── tests/                         # Preprocessing tests
│
├── 📁 model_development/             # Model training and development
│   ├── configs/                       # Training configurations
│   ├── scripts/                       # Training scripts
│   │   ├── train_gemini.py            # Model training
│   │   ├── hyperparameter_tuning.py  # Hyperparameter optimization
│   │   └── model_validation.py       # Model validation
│   ├── notebooks/                     # Training notebooks
│   └── experiments/                   # Experiment results
│
├── 📁 model_deployment/              # Model deployment
│   ├── api/                           # FastAPI application
│   │   ├── app.py                     # API endpoints
│   │   └── summarizer.py              # Summarization model
│   ├── web/                           # Web interface (Streamlit)
│   │   └── file_qa_web.py            # Streamlit web app
│   └── scripts/                       # Deployment scripts
│
├── 📁 monitoring/                     # Monitoring and observability
│   ├── metrics.py                      # Metrics collection
│   └── logging/                        # Logging configurations
│
├── 📁 infrastructure/                 # Infrastructure as code
│   ├── docker/                         # Docker configurations
│   │   ├── Dockerfile
│   │   ├── Dockerfile.cloudrun
│   │   └── cloudbuild.yaml
│   └── ci_cd/                         # CI/CD workflows
│       └── .github/workflows/
│
├── 📁 src/                            # Source code library
│   ├── rag/                           # RAG system
│   │   ├── rag_system.py              # Core RAG implementation
│   │   ├── file_qa.py                 # File Q&A system
│   │   ├── patient_qa.py              # Patient Q&A interface
│   │   └── document_processor.py      # Document processing
│   └── utils/                         # Shared utilities
│       ├── logging_config.py          # Logging configuration
│       ├── error_handling.py          # Error handling
│       └── medical_utils.py           # Medical utilities
│
├── 📁 notebooks/                      # Jupyter notebooks
│   ├── exploration/                   # Data exploration
│   └── experiments/                   # Experiment notebooks
│
├── 📁 tests/                          # Test suite
│   ├── unit/                          # Unit tests
│   ├── integration/                   # Integration tests
│   └── e2e/                           # End-to-end tests
│
├── 📁 docs/                           # Documentation
│   ├── deployment/                    # Deployment guides
│   └── api/                           # API documentation
│
└── 📁 scripts/                        # Utility scripts
```

**For detailed structure information, see [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)**

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.12+
- **Google Cloud Platform**: Account with BigQuery access
- **PhysioNet Credentialing**: Required for MIMIC-III access
- **Memory**: Minimum 8GB RAM recommended
- **Storage**: 10GB+ free space for data processing

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/lab-lens.git
cd lab-lens

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Mac/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r data-pipeline/requirements.txt
```

### GCP Authentication

```bash
# Install gcloud CLI (if needed)
brew install --cask google-cloud-sdk  # Mac
# or download from https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Running the Pipeline

```bash
# Run complete pipeline
python data-pipeline/scripts/main_pipeline.py

# Run with custom configuration
python data-pipeline/scripts/main_pipeline.py --config data-pipeline/configs/pipeline_config.json

# Run specific steps only
python data-pipeline/scripts/main_pipeline.py --skip-preprocessing --skip-validation

# Run with custom paths
python data-pipeline/scripts/main_pipeline.py \
  --input-path /path/to/raw/data \
  --output-path /path/to/processed/data \
  --logs-path /path/to/logs
```

## 📈 Pipeline Components

### 1. Data Acquisition
- **Source**: Google BigQuery MIMIC-III tables
- **Method**: Cloud-based querying (no local download)
- **Output**: 5,000+ discharge summaries with demographics
- **Performance**: ~30 seconds for 5,000 records

### 2. Data Preprocessing
- **Text Processing**: Section extraction, abbreviation expansion
- **Feature Engineering**: 26+ new features created
- **Quality Control**: Comprehensive text cleaning and validation
- **Output**: Structured dataset ready for ML modeling

### 3. Data Validation
- **Schema Validation**: Required columns verification
- **Quality Metrics**: Completeness, duplicates, outliers
- **Score**: 95/100 overall validation score
- **Coverage**: 0% missing critical fields

### 4. Bias Detection
- **Demographic Analysis**: Gender, ethnicity, age, insurance
- **Statistical Testing**: Significance tests for differences
- **Visualization**: Automated bias plot generation
- **Score**: 5.88 overall bias score (lower is better)

### 5. Automated Bias Mitigation
- **Detection**: Automatic bias threshold monitoring
- **Strategies**: Stratified sampling, oversampling, weighting
- **Compliance**: Real-time bias score tracking
- **Reporting**: Comprehensive mitigation reports

## 🔧 Configuration

### Pipeline Configuration

The pipeline can be configured using JSON configuration files:

```json
{
  "pipeline_config": {
    "input_path": "data/raw",
    "output_path": "data/processed",
    "logs_path": "logs",
    "enable_preprocessing": true,
    "enable_validation": true,
    "enable_bias_detection": true,
    "enable_automated_bias_handling": true
  },
  "bias_detection_config": {
    "alert_thresholds": {
      "gender_cv_max": 5.0,
      "ethnicity_cv_max": 10.0,
      "overall_bias_score_max": 10.0
    }
  }
}
```

### Environment Variables

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export MIMIC_PROJECT_ID="your-gcp-project-id"
export LOG_LEVEL="INFO"
```

## 📊 Results & Metrics

### Data Quality Metrics
- **Validation Score**: 95/100 (Excellent)
- **Schema Valid**: ✅ All required columns present
- **Missing Text**: 0 records
- **Duplicate Records**: 0
- **Average Text Length**: 9,558 characters

### Bias Analysis Results
- **Gender Bias**: Not statistically significant (p=0.21)
- **Ethnicity Variation**: 8.4% coefficient of variation
- **Age Group Variation**: 7.2% coefficient of variation
- **Overall Bias Score**: 5.88 (scale 0-100, lower is better)

### Performance Metrics
- **Query Time**: ~30 seconds for 5,000 records
- **Processing Time**: ~45 seconds for full preprocessing
- **Total Pipeline Runtime**: ~3 minutes end-to-end
- **Memory Usage**: <2GB peak usage

## 🛡️ Error Handling & Logging

### Comprehensive Error Management
- **Custom Exceptions**: Specialized error types for different operations
- **Error Recovery**: Automatic retry mechanisms and fallback strategies
- **Context Preservation**: Detailed error context and stack traces
- **Graceful Degradation**: Pipeline continues despite non-critical errors

### Centralized Logging System
- **Structured Logging**: JSON-formatted logs with metadata
- **Performance Metrics**: Automatic timing and resource usage tracking
- **Data Metrics**: Record counts and processing statistics
- **Log Rotation**: Automatic log file rotation and archival

### Log Levels
- **DEBUG**: Detailed execution information
- **INFO**: General pipeline progress and metrics
- **WARNING**: Non-critical issues and fallbacks
- **ERROR**: Processing errors with recovery attempts
- **CRITICAL**: System failures requiring intervention

## 🔍 Monitoring & Alerting

### Real-time Monitoring
- **Pipeline Status**: Live pipeline execution monitoring
- **Data Quality**: Continuous data quality assessment
- **Bias Metrics**: Real-time bias score tracking
- **Performance**: Resource usage and timing metrics

### Automated Alerts
- **Bias Thresholds**: Alerts when bias scores exceed limits
- **Data Quality**: Notifications for validation failures
- **Performance**: Alerts for slow processing or resource issues
- **System Health**: Overall system status monitoring

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_preprocessing.py
python -m pytest tests/test_validation.py
python -m pytest tests/test_bias_detection.py
```

### Integration Tests
```bash
# Run integration tests
python -m pytest tests/integration/

# Test complete pipeline
python tests/test_pipeline_integration.py
```

## 🔍 RAG (Retrieval-Augmented Generation) System Testing

The RAG system enables patients to ask questions about their discharge summaries using semantic search and AI-powered answer generation.

### Overview

- **Purpose**: Allow patients to ask questions about their discharge summary
- **Mode**: Single-patient mode (loads only one patient's record)
- **Features**: Semantic search, patient-friendly answers, source citations

### Quick Start

#### Single-Patient Q&A (Recommended)

Test with a specific patient's discharge summary:

```bash
# Interactive Q&A for a specific patient
python scripts/patient_qa_single.py --hadm-id 130656
```

This loads **only that patient's record** and allows interactive Q&A.

#### View Patient Record First

```bash
# View the discharge summary before asking questions
python scripts/patient_qa_single.py --hadm-id 130656 --view
```

#### Ask a Single Question

```bash
# Ask one question
python scripts/patient_qa_single.py \
  --hadm-id 130656 \
  --question "What are my diagnoses?"
```

### Step-by-Step Testing Guide

#### Step 1: List Available Records

Find available patient HADM IDs:

```bash
python scripts/test_rag_with_record.py --list-records
```

This shows available HADM IDs from your processed discharge summaries.

#### Step 2: Test with a Specific Patient

```bash
# Interactive mode (best for testing)
python scripts/patient_qa_single.py --hadm-id 130656
```

Then you can:
- Type questions interactively
- Type `help` for example questions
- Type `summary` to see patient record
- Type `exit` to quit

#### Step 3: Ask Example Questions

Try questions like:
- "What are my diagnoses?"
- "What medications do I need to take?"
- "What happened during my hospital stay?"
- "What are my discharge instructions?"
- "When is my follow-up appointment?"
- "What should I watch for at home?"

### Advanced Testing

#### View and Test Workflow

```bash
# Step 1: View patient record
python scripts/test_rag_with_record.py --view-record 130656

# Step 2: Test RAG with default questions
python scripts/test_rag_with_record.py --test 130656

# Step 3: Test with custom questions
python scripts/test_rag_with_record.py --test 130656 \
  --questions "What are my diagnoses?" "What medications do I need?"
```

#### Python API Usage

```python
from src.rag.patient_qa import PatientQA

# Initialize with single patient (loads only that patient's record)
qa = PatientQA(
    data_path="data-pipeline/data/processed/processed_discharge_summaries.csv",
    hadm_id=130656  # Single-patient mode
)

# Ask questions
result = qa.ask_question("What are my diagnoses?")
print(result['answer'])
print(f"Sources: {len(result['sources'])} sections found")
```

### Prerequisites

1. **Dependencies**:
   ```bash
   pip install sentence-transformers faiss-cpu google-generativeai
   ```

2. **Google API Key**:
   ```bash
   export GOOGLE_API_KEY="your-api-key"
   # Or use
   python scripts/setup_gemini_api_key.py
   ```

3. **Processed Data**: Ensure discharge summaries are processed:
   ```bash
   python data-pipeline/scripts/main_pipeline.py
   ```

### Single-Patient Mode Benefits

✅ **Fast**: Only processes one patient's record (5-10 seconds vs 2-5 minutes)  
✅ **Efficient**: Minimal memory usage (<100 MB vs 2-4 GB)  
✅ **Secure**: Only loads that specific patient's data  
✅ **Simple**: No filtering needed - all data is from that patient  

### Performance

- **First run** (create embeddings): 5-10 seconds for one patient
- **Subsequent runs** (cached): 1-2 seconds
- **Question answering**: 2-5 seconds per question
- **Memory usage**: <100 MB (vs 2-4 GB for all records)

### Example Output

```
======================================================================
SINGLE PATIENT Q&A - HADM ID: 130656
======================================================================

Loading only this patient's discharge summary...
✅ RAG System ready!

❓ Your question: What are my diagnoses?

======================================================================
ANSWER
======================================================================

Based on your discharge summary, your primary diagnoses include...

📚 Sources: 5 relevant sections found
   Top relevance score: 0.852
======================================================================
```

### Troubleshooting

**Error: No record found with HADM ID**
- Verify HADM ID exists: `python scripts/test_rag_with_record.py --list-records`

**Error: GOOGLE_API_KEY not found**
```bash
export GOOGLE_API_KEY="your-api-key"
```

**Error: Missing dependencies**
```bash
pip install sentence-transformers faiss-cpu
```

**Process crashes during embedding generation**
- This is normal for large datasets - use single-patient mode instead

### Documentation

- **RAG Guide**: See [docs/RAG_GUIDE.md](docs/RAG_GUIDE.md) for complete RAG system documentation
- **API Setup**: See [docs/API_SETUP.md](docs/API_SETUP.md) for Gemini API configuration
- **All Documentation**: See [docs/README.md](docs/README.md) for complete documentation index

## 📚 Documentation

All documentation is organized in the [`docs/`](docs/) directory. Key guides:

### Getting Started
- **[API Setup Guide](docs/API_SETUP.md)** - Setting up Gemini API keys
- **[RAG Guide](docs/RAG_GUIDE.md)** - Patient Q&A system documentation
- **[BigQuery Setup](docs/BIGQUERY_SETUP.md)** - Setting up BigQuery access

### User Guides
- **[Model Testing Guide](docs/MODEL_TESTING_GUIDE.md)** - How to test models
- **[Model Development Guide](docs/MODEL_DEVELOPMENT_GUIDE.md)** - Model development
- **[File Q&A Guide](docs/FILE_QA_GUIDE.md)** - File-based Q&A system
- **[Web Interface Guide](docs/WEB_INTERFACE_GUIDE.md)** - Web interface usage

### Complete Documentation Index
See [docs/README.md](docs/README.md) for the full documentation index.

## 🔮 Future Roadmap

### Phase 1: Enhanced ML Integration
- [x] Gemini 1.5 Pro integration for medical text summarization
- [ ] Advanced NLP models for text summarization
- [ ] Multi-modal model training pipeline
- [ ] Model versioning and experiment tracking

### Phase 2: Production Deployment
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline integration
- [ ] Production monitoring dashboard

### Phase 3: Advanced Features
- [ ] Real-time data processing
- [ ] Stream processing for new admissions
- [ ] Interactive bias monitoring dashboard
- [ ] Automated model retraining

### Phase 4: Scalability & Performance
- [ ] Distributed processing with Dask/Ray
- [ ] GPU acceleration for model training
- [ ] Cloud-native deployment options
- [ ] Auto-scaling based on workload

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ data-pipeline/scripts/
isort src/ data-pipeline/scripts/
```

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Data Usage
- **MIMIC-III**: Licensed under PhysioNet Credentialed Health Data License 1.5.0
- **MedMNIST**: Licensed under Apache License 2.0
- **Project Code**: Licensed under MIT License

## 📞 Support & Contact

### Getting Help
- **Documentation**: Check the comprehensive documentation in `/docs`
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join community discussions in GitHub Discussions
- **Email**: Contact the team at lab-lens-support@example.com

### Team Contacts
- **Project Lead**: Asad Ullah Waraich
- **Technical Lead**: Shahid Kamal
- **Data Science Lead**: Shruthi Kashetty
- **Infrastructure Lead**: Dhruv Rameshbhai Gajera

---

## 🏆 Key Achievements

✅ **Successfully processed 5,000+ discharge summaries**  
✅ **Achieved 95/100 data validation score**  
✅ **Implemented comprehensive bias detection**  
✅ **Automated bias mitigation strategies**  
✅ **Built production-ready MLOps pipeline**  
✅ **Comprehensive error handling and logging**  
✅ **Real-time monitoring and alerting**  
✅ **RAG-Powered Patient Q&A System** - Single-patient mode for efficient Q&A  
✅ **Scalable and maintainable architecture**  

*Developed as part of MLOps Course Project - Fall 2025*
