# DSView 🔍

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com)

**DSView** is an intelligent data science news monitoring and knowledge management platform that automatically ingests, processes, and organizes data science content using Large Language Models (LLMs) and graph representation in Obsidian.

## 🌟 Overview

DSView transforms how data scientists stay current with rapidly evolving trends, tools, and research. It automatically:

- 📥 **Ingests content** from URLs, blogs, papers, and repositories
- 🤖 **Extracts insights** using state-of-the-art LLMs (Mistral, OpenAI, Anthropic, Ollama)
- 🏷️ **Categorizes and tags** content with 25+ predefined data science topics
- 🔗 **Builds knowledge graphs** connecting related concepts, tools, and resources
- 📝 **Generates Obsidian notes** for seamless knowledge management
- 📊 **Provides analytics dashboards** for content exploration and insights

## ✨ Key Features

### 🔄 Automated Content Processing
- **Multi-source ingestion**: URLs, PDFs, documentation, blog posts, repositories
- **Smart content extraction**: Summaries, topics, links, and metadata
- **Intelligent categorization**: 6 content types and 25+ technical tags
- **Entity resolution**: Automatic deduplication and relationship mapping

### 🧠 LLM-Powered Analysis
- **Multi-provider support**: Mistral, OpenAI, Anthropic, Ollama
- **Structured extraction**: Topics, summaries, relevant links, descriptions
- **Semantic understanding**: Context-aware content analysis
- **Configurable models**: Different models for different tasks

### 🗂️ Knowledge Management
- **Obsidian integration**: Automatic vault generation with 1000+ structured notes
- **Graph visualization**: Interactive network of topics and content relationships
- **Smart organization**: Hierarchical categorization by type and topic
- **Cross-linking**: Automatic linking between related content and concepts

### 📊 Analytics & Monitoring
- **Interactive dashboards**: Built with Marimo for content exploration
- **Search capabilities**: Full-text search across your knowledge base
- **Progress tracking**: Monitor ingestion, extraction, and processing status
- **Quality metrics**: Content relevance scoring and evaluation tools

### 🏷️ Content Categories

**Content Types:**
- 📝 Blog Posts - Industry insights and tutorials
- 🎓 Courses - Learning materials and educational content
- 📋 Documentation - Technical docs and best practices
- 🏢 Product Pages - Tools and platform overviews
- 💻 Repositories - Open-source projects and code
- 🔬 Scientific Articles - Research papers and publications

**Topic Areas (25+ categories):**
- 🧠 Machine Learning (Deep Learning, Supervised/Unsupervised Learning)
- 🔤 NLP (Large Language Models, Text Processing)
- 👁️ Computer Vision (Image Processing, Object Detection)
- ⚙️ MLOps (Model Deployment, Monitoring, DevOps)
- 🏗️ Data Engineering (Pipelines, Big Data, Cloud Computing)
- 📊 Analytics (Visualization, Statistics, Time Series)
- 🛠️ Tools & Libraries (Python packages, development tools)

## 📋 Requirements

- **Python**: 3.11 or higher
- **Database**: PostgreSQL (recommended) or SQLite
- **LLM API Keys**: At least one of:
  - Mistral AI (recommended)
  - OpenAI
  - Anthropic
  - Or local Ollama installation
- **Memory**: 4GB+ RAM recommended
- **Storage**: 1GB+ for Obsidian vault and database

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/tdurouchoux/dsview.git
cd dsview

# Create environment file
cp .env.example .env
# Edit .env with your API keys and configuration

# Start the services
docker-compose up -d

# Access the interfaces
# - Dashboard: http://localhost:2718
# - API: http://localhost:8000
# - Labelling Interface: http://localhost:8501

# Try ingesting your first content
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"link": "https://towardsdatascience.com/some-article", "source": "manual"}'
```

### Local Development

```bash
# Clone and setup
git clone https://github.com/tdurouchoux/dsview.git
cd dsview

# Install dependencies with uv (recommended)
uv sync
# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# OR .venv\Scripts\activate  # Windows

# Alternative: Install with pip
pip install -e .

# Setup environment
cp .env.example .env
# Configure your API keys and database settings in .env

# Initialize the database (if using PostgreSQL)
# Make sure PostgreSQL is running and database exists

# Run the CLI to see available commands
dsview --help

# Try ingesting a single URL
dsview ingest "https://example.com/article" --source "test"

# Start the dashboard
dsview-ui

# Or start the API server directly
uvicorn dsview.api:app --reload --host 0.0.0.0 --port 8000
```

## 🛠️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Copy the example file
cp .env.example .env

# Edit with your configuration
nano .env  # or your preferred editor
```

**Required Variables:**
```bash
# At least one LLM provider API key
MISTRAL_API_KEY=your_mistral_key
# OR
OPENAI_API_KEY=your_openai_key
# OR  
ANTHROPIC_API_KEY=your_anthropic_key

# Database connection
DATABASE_URL=postgresql://user:password@localhost:5432/dsview
# OR for development
DATABASE_URL=sqlite:///./dsview.db
```

**Optional Variables:**
```bash
# Obsidian Vault Path (default: ./dsview_vault)
VAULT_PATH=./dsview_vault

# Configuration Directories
CONF_DIR=./config
PROMPT_DIR=./prompts

# Server Configuration
API_PORT=8000
DASHBOARD_PORT=2718
```

### Model Configuration

DSView supports multiple LLM providers. Edit `config/model.yaml` to configure:

```yaml
configs:
  - model_type: DEFAULT
    model_config:
      chat_model: mistral-small-latest
      embedding_model: mistral-embed
      provider: MISTRAL
      token_limit: 100000
```

## 📖 Usage

### Command Line Interface

```bash
# Ingest a single URL
dsview ingest "https://example.com/article" --source "blog"

# Batch process failed ingestions
dsview retry-failed

# Rebuild processing for content range
dsview rebuild --start 100 --end 200

# Generate/regenerate Obsidian vault
dsview regen-vault

# Backup database
dsview backup

# Reset components (with confirmations)
dsview reset-db
dsview reset-vault
dsview reset-labelling
```

### API Endpoints

```bash
# Ingest content via API
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"link": "https://example.com", "source": "manual"}'

# Update content relevance
curl -X PATCH "http://localhost:8000/relevance" \
  -H "Content-Type: application/json" \
  -d '{"link": "https://example.com", "relevance": 8}'
```

### Dashboard Features

Access the dashboard at `http://localhost:2718` for:

- **Content Dashboard**: Browse and filter ingested content
- **Search Vault**: Full-text search across your knowledge base
- **Upload Dashboard**: Bulk upload and manage content
- **Extraction Dashboard**: Monitor processing pipeline status
- **Embedding Dashboard**: Explore content similarity and clustering
- **Database Explorer**: Direct database query interface

## 🏗️ Architecture

### Core Components

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

### Database Schema

- **InputContent**: Source URLs and metadata
- **ExtractionResult**: Processed content with summaries and insights
- **ExtractionTopic**: Identified topics and their relationships
- **ExtractionLink**: Relevant external links
- **ContentTopicRelation**: Content-topic mapping
- **LabelledContent**: Manual annotations for training/evaluation

### Processing Pipeline

1. **Content Loading**: Download and parse content from various sources
2. **Text Extraction**: Extract clean text from HTML, PDFs, etc.
3. **LLM Analysis**: Parallel API calls for topic extraction, summarization, link extraction
4. **Entity Resolution**: Deduplicate and merge similar topics/concepts
5. **Knowledge Graph**: Build relationships between content and topics
6. **Obsidian Generation**: Create structured notes with cross-references
7. **Quality Evaluation**: Score relevance and extraction quality

## 🧪 Evaluation & Quality

DSView includes comprehensive evaluation modules:

- **Topic Extraction Evaluation**: Measure accuracy of topic identification
- **Link Extraction Evaluation**: Validate relevant link discovery
- **Description Generation**: Assess summary quality using ROUGE scores
- **Entity Resolution**: Test deduplication accuracy
- **Semantic Scoring**: Content similarity and relevance metrics

## 🔧 Development

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

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_extraction_models.py
pytest tests/test_obsidian.py

# Run with coverage
pytest --cov=dsview
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

## 💡 Examples & Use Cases

### Common Workflows

**1. Daily Data Science News Monitoring**
```bash
# Ingest content from popular data science blogs
dsview ingest "https://towardsdatascience.com/latest-ml-trends" --source "tds"
dsview ingest "https://blog.paperswithcode.com/new-research" --source "pwc"

# Generate updated Obsidian vault
dsview regen-vault
```

**2. Research Paper Collection**
```bash
# Ingest arXiv papers
dsview ingest "https://arxiv.org/abs/2301.xxxxx" --source "arxiv" --relevance 9

# Ingest GitHub repositories
dsview ingest "https://github.com/openai/whisper" --source "github"
```

**3. Tool Discovery and Tracking**
```bash
# Track new ML tools and libraries
dsview ingest "https://huggingface.co/transformers" --source "huggingface"
dsview ingest "https://docs.ray.io/en/latest/" --source "documentation"
```

### Batch Processing
```bash
# Process a list of URLs from a file
while read url; do
  dsview ingest "$url" --source "batch_import"
done < urls.txt

# Retry any failed ingestions
dsview retry-failed
```

## 🔧 Troubleshooting

### Common Issues

**1. API Key Issues**
```bash
# Check if your API keys are configured
echo $MISTRAL_API_KEY  # Should not be empty
echo $OPENAI_API_KEY

# Test API connectivity
curl -H "Authorization: Bearer $MISTRAL_API_KEY" \
  "https://api.mistral.ai/v1/models"
```

**2. Database Connection Problems**
```bash
# For PostgreSQL, test connection
psql $DATABASE_URL -c "SELECT 1;"

# Check if tables exist
dsview --help  # Should initialize tables on first run

# Reset database if corrupted
dsview reset-db
```

**3. Content Ingestion Failures**
```bash
# Check failed ingestions
dsview retry-failed

# View logs for debugging
tail -f run_dsview.log

# Test specific URL manually
curl -I "https://problem-url.com"
```

**4. Obsidian Vault Issues**
```bash
# Regenerate vault if corrupted
dsview reset-vault
dsview regen-vault

# Check vault permissions
ls -la dsview_vault/
```

**5. Memory/Performance Issues**
```bash
# Monitor resource usage
htop

# Reduce batch size in config
# Edit config/extraction.yaml to reduce concurrent requests

# Use smaller models for large-scale processing
# Edit config/model.yaml to use lighter models
```

### Debugging Tips

- **Enable debug logging**: Set `LOG_LEVEL=DEBUG` in your `.env` file
- **Check MLflow logs**: Access MLflow UI to see experiment tracking
- **Monitor API responses**: Enable request logging in your LLM provider dashboard
- **Validate content extraction**: Use the extraction dashboard to review results
- **Test individual components**: Use the CLI commands to test each pipeline stage

## 🐳 Deployment

### Docker Services

The project includes multiple Docker containers:

- **API Service** (`api.Dockerfile`): FastAPI server for content ingestion
- **Dashboard Service** (`dashboard.Dockerfile`): Marimo-based analytics interface  
- **Labels Service** (`labels.Dockerfile`): Streamlit labelling interface

### Production Deployment

**Docker Compose Production**
```bash
# Use production docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Set up reverse proxy (nginx example)
# Point domain to localhost:2718 for dashboard
# Point api.domain to localhost:8000 for API
```

**Environment Variables for Production**
```bash
# Use production database
DATABASE_URL=postgresql://user:pass@db-server:5432/dsview_prod

# Configure secure API keys
MISTRAL_API_KEY=prod_api_key
API_SECRET_KEY=your_secure_secret

# Set production logging
LOG_LEVEL=INFO
```

### Kubernetes

Kubernetes manifests are available in the `kubernetes/` directory for production deployment.

### Scaling Considerations

- **Database**: Use PostgreSQL with connection pooling
- **LLM API Limits**: Implement rate limiting and retry logic
- **Storage**: Consider object storage for large document archives
- **Caching**: Redis for caching extraction results

## 📊 Monitoring & Observability

- **MLflow Integration**: Track model performance and experiments
- **Comprehensive Logging**: Structured logs with Rich formatting
- **Database Monitoring**: Query performance and health checks
- **Processing Metrics**: Ingestion rates, success/failure tracking

## 📚 API Reference

### REST API Endpoints

**Content Ingestion**
```http
POST /ingest
Content-Type: application/json

{
  "link": "https://example.com/article",
  "source": "manual",
  "already_read": false,
  "read_priority": 5,
  "relevance": 7,
  "upload_date": "2024-01-01"
}
```

**Update Content Relevance**
```http
PATCH /relevance?link=https://example.com&relevance=9
```

### CLI Commands Reference

```bash
# Content Management
dsview ingest <URL> [--source TEXT] [--priority INT] [--relevance INT]
dsview retry-failed [--ignore LIST]
dsview rebuild [--start INT] [--end INT]

# Vault Operations
dsview regen-vault
dsview reset-vault

# Database Operations
dsview backup [--output-dir PATH]
dsview restore-db [--backup-dir PATH]
dsview reset-db
dsview reset-labelling

# Health Checks
dsview --version
dsview --help
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- **Code style**: Using Ruff for formatting and linting
- **Test coverage**: Minimum 80% coverage required
- **Documentation**: Docstrings for all public functions
- **Feature requests**: Use GitHub issues with feature template
- **Bug reports**: Include reproduction steps and logs

### Development Setup
```bash
# Install development dependencies
uv sync --group dev

# Run tests
pytest

# Format code
ruff format .

# Lint code
ruff check .

# Type checking
pyright
```

## ❓ FAQ

**Q: Which LLM provider should I use?**
A: Mistral AI is recommended for the best balance of cost, performance, and structured output. OpenAI GPT-4 offers highest quality but at higher cost. Ollama is great for local/offline use.

**Q: How much does it cost to run?**
A: Costs vary by usage. Typical monthly usage (100 articles): ~$5-20 with Mistral, ~$20-50 with OpenAI. Local Ollama is free but requires GPU.

**Q: Can I use this without Obsidian?**
A: Yes! The dashboard provides full content exploration. Obsidian integration is optional but recommended for knowledge management.

**Q: How do I add custom content types or topics?**
A: Edit `config/extraction.yaml` to add new categories. Retrain by running `dsview rebuild` on existing content.

**Q: Is my data secure?**
A: Content is processed by your chosen LLM provider. For sensitive content, use local Ollama models or ensure compliance with your provider's privacy policy.

**Q: Can I run this on a server?**
A: Yes! Use Docker compose with appropriate security configurations. See deployment section for details.

## 🔗 Links

- **Documentation**: [GitHub Wiki](https://github.com/tdurouchoux/dsview/wiki)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/tdurouchoux/dsview/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tdurouchoux/dsview/discussions)
- **Docker Hub**: [Container Images](https://hub.docker.com/r/tdurouchoux/dsview)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with FastAPI, SQLModel, and Marimo
- LLM providers: Mistral AI, OpenAI, Anthropic, Ollama
- Obsidian for knowledge management
- The open-source data science community

---

**DSView** - Transform information overload into organized knowledge. Stay ahead in the rapidly evolving world of data science! 🚀