# DSView Agent Documentation

## Overview
DSView is an intelligent data science news monitoring and knowledge management platform that automatically ingests, processes, and organizes data science content using LLMs and graph representation in Obsidian.

## Core Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Content       │    │   Extraction    │    │   Knowledge     │
│   Ingestion     │───▶│   Pipeline      │───▶│   Management    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ • URL Loading   │    │ • LLM Analysis  │    │ • Obsidian      │
│ • PDF Processing│    │ • Topic Extract │    │ • Graph Build   │
│ • Content Clean │    │ • Link Extract  │    │ • Note Writing  │
│ • Validation    │    │ • Summarization │    │ • Cross-linking │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Components

### 1. Content Ingestion (`dsview/ingestion/`)
- **Purpose**: Load and validate content from various sources
- **Key Files**:
  - `ingest_source.py`: Main ingestion logic
  - `content_loader.py`: Load content from URLs, PDFs, etc.
- **Supported Sources**: URLs, PDFs, documentation, blog posts, repositories

### 2. Extraction Pipeline (`dsview/extraction/`)
- **Purpose**: Process content using LLMs to extract insights
- **Key Files**:
  - `content_extraction.py`: Main extraction workflow
  - `models/`: Extraction models and schemas
- **Processes**:
  - Topic extraction (25+ data science categories)
  - Content summarization
  - Relevant link extraction
  - Entity resolution and deduplication

### 3. Database Layer (`dsview/db/`)
- **Purpose**: Store and manage all content and metadata
- **Key Files**:
  - `schemas/`: SQLModel database schemas
  - `ingest/`: Ingestion operations
  - `query/`: Data retrieval operations
- **Main Tables**:
  - `InputContent`: Source URLs and metadata
  - `ExtractionResult`: Processed content with summaries
  - `ExtractionTopic`: Identified topics
  - `ExtractionLink`: Relevant external links
  - `ContentTopicRelation`: Content-topic mapping

### 4. LLM Integration (`dsview/model_utils/`)
- **Purpose**: Abstract LLM provider interactions
- **Key Files**:
  - `providers/`: Provider-specific implementations
    - `mistral.py`, `openai.py`, `anthropic.py`, `ollama.py`
  - `llm_model.py`: Core LLM model interface
  - `model_provider.py`: Provider factory
- **Supported Providers**: Mistral, OpenAI, Anthropic, Ollama

### 5. Obsidian Integration (`dsview/obsidian/`)
- **Purpose**: Generate and manage Obsidian vault
- **Key Files**:
  - `write_notes.py`: Create structured notes
  - `sync_vault.py`: Synchronize vault with database
  - `obsidian_utils.py`: Utility functions
- **Output**: 1000+ structured notes with cross-references

### 6. Evaluation (`dsview/evaluation/`)
- **Purpose**: Quality assessment and metrics
- **Key Files**:
  - `topics_extraction.py`: Topic extraction accuracy
  - `links_extraction.py`: Link extraction validation
  - `description_generation.py`: Summary quality (ROUGE scores)
  - `entity_resolution.py`: Deduplication accuracy
  - `semantic_score.py`: Content similarity metrics

### 7. API (`dsview/api.py`)
- **Purpose**: REST API for content ingestion and management
- **Key Endpoints**:
  - `POST /ingest`: Add new content
  - `PATCH /relevance`: Update content relevance
- **Framework**: FastAPI

### 8. CLI (`dsview/cli.py`)
- **Purpose**: Command-line interface for operations
- **Key Commands**:
  - `ingest`: Add new content
  - `retry-failed`: Reprocess failed ingestions
  - `regen-vault`: Regenerate Obsidian vault
  - `backup`: Database backup
  - `reset-db`: Reset database

### 9. Dashboards (`dsview/interface/`)
- **Purpose**: Web interfaces for content exploration
- **Key Components**:
  - `dashboard/`: Marimo-based analytics dashboards
    - Content dashboard
    - Search vault
    - Upload dashboard
    - Extraction dashboard
    - Embedding dashboard
    - Database explorer
  - `labelling/`: Streamlit labelling interface

## Data Flow

```
1. Content Ingestion
   └─ URL/PDF → Content Loading → Validation → Database

2. Extraction Pipeline
   └─ Raw Content → LLM Analysis → Topic Extraction
     └─ Link Extraction → Summarization → Entity Resolution

3. Knowledge Management
   └─ Processed Content → Graph Building → Obsidian Notes
     └─ Cross-linking → Vault Synchronization

4. Quality Assessment
   └─ Evaluation Metrics → Feedback Loop → Pipeline Improvement
```

## Configuration

### Environment Variables (`.env`)
- `DATABASE_URL`: PostgreSQL or SQLite connection
- `MISTRAL_API_KEY`, `OPENAI_API_KEY`, etc.: LLM provider keys
- `VAULT_PATH`: Obsidian vault location
- `CONF_DIR`: Configuration directory
- `PROMPT_DIR`: LLM prompt templates directory

### Configuration Files
- `config/model.yaml`: LLM model configurations
- `config/extraction.yaml`: Extraction pipeline settings
- `prompts/`: LLM prompt templates

## Development Notes

### Project Structure
```
dsview/
├── dsview/                 # Main package
│   ├── api.py             # FastAPI server
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration management
│   ├── extraction/        # Content extraction pipeline
│   ├── db/               # Database models and operations
│   ├── interface/        # Web dashboards and UIs
│   ├── obsidian/         # Obsidian vault management
│   ├── model_utils/      # LLM provider abstractions
│   └── evaluation/       # Quality assessment tools
├── config/               # YAML configuration files
├── prompts/             # LLM prompts for extraction
├── tests/               # Test suite
└── dsview_vault/        # Generated Obsidian vault
```

### Key Dependencies
- **Core**: Python 3.11+, FastAPI, SQLModel
- **LLMs**: Mistral, OpenAI, Anthropic, Ollama
- **Database**: PostgreSQL (recommended) or SQLite
- **UI**: Marimo (dashboards), Streamlit (labelling)
- **Knowledge Management**: Obsidian

### Testing
- **Framework**: pytest
- **Coverage**: Minimum 80% required
- **Key Test Files**:
  - `tests/test_extraction_models.py`
  - `tests/test_obsidian.py`
  - `tests/test_api.py`

## Agent-Specific Notes

### For Content Processing Agents
- Focus on `dsview/extraction/` and `dsview/model_utils/`
- Key functions: topic extraction, summarization, link extraction
- Configuration in `config/extraction.yaml`

### For Knowledge Management Agents
- Focus on `dsview/obsidian/` and `dsview/db/`
- Key functions: note generation, cross-linking, vault synchronization
- Output format: Obsidian markdown with YAML frontmatter

### For Quality Assessment Agents
- Focus on `dsview/evaluation/`
- Key metrics: ROUGE scores, topic accuracy, link relevance
- Feedback loop: Results inform pipeline improvements

### For API/Integration Agents
- Focus on `dsview/api.py` and `dsview/cli.py`
- Key endpoints: `/ingest`, `/relevance`
- Authentication: API key based

## Performance Considerations

### Scaling
- **Database**: Use PostgreSQL with connection pooling
- **LLM API**: Implement rate limiting and retry logic
- **Processing**: Batch operations for efficiency
- **Storage**: Consider object storage for large documents

### Optimization Points
- **Caching**: Redis for extraction results
- **Parallel Processing**: Concurrent LLM API calls
- **Batch Operations**: Bulk database operations
- **Model Selection**: Balance quality vs. cost

## Error Handling

### Common Issues
- **API Key Issues**: Validate provider credentials
- **Database Connection**: Check connection strings
- **Content Ingestion**: Validate URLs and formats
- **Obsidian Vault**: Check permissions and paths
- **Memory**: Monitor resource usage

### Debugging Tools
- **Logging**: Set `LOG_LEVEL=DEBUG` in `.env`
- **MLflow**: Experiment tracking
- **Dashboard**: Extraction status monitoring
- **CLI**: Individual component testing
