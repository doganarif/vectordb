# VectorDB

<div align="center">
In-memory vector database with pluggable indexing algorithms, metadata filtering, and a FastAPI-based REST API.
</div>

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Python SDK](#python-sdk)
- [Indexing Algorithms](#indexing-algorithms)
- [Testing](#testing)

## Features

### Core Capabilities

- Multiple Indexing Algorithms: Linear, KD-Tree, and LSH (Locality Sensitive Hashing)
- Flexible Similarity Metrics: Cosine similarity and Euclidean distance
- Metadata Filtering: Filter search results by custom metadata
- Persistence: Snapshot and restore functionality for data durability
- Thread-Safe: Custom reader-writer locks for concurrent operations
- In-Memory: Fast access with in-memory storage
- RESTful API: Full CRUD operations via FastAPI
- Python SDK: Native client library for seamless integration
- Embeddings API: Integrated Cohere support for text embeddings

### Key Benefits

- **Production Ready**: Docker support, health checks, and graceful shutdown
- **Scalable**: Per-library indices and lightweight design
- **Extensible**: Modular design for easy algorithm additions
- **Developer Friendly**: Comprehensive API, SDK, and Postman collection

## Architecture

### System Overview

The VectorDB system follows a layered architecture with clear separation of concerns. Each layer has specific responsibilities and communicates through well-defined interfaces.

```mermaid
graph TB
    subgraph "Client Layer"
        A["Python SDK"]
        B["REST API Clients"]
        C["Postman/cURL"]
    end

    subgraph "API Layer"
        D["FastAPI Server"]
        E["Request Validation"]
        F["Response Serialization"]
    end

    subgraph "Service Layer"
        G["VectorDB Service"]
        H["Library Service"]
        I["Document Service"]
        J["Chunk Service"]
        K["Index Service"]
        L["Snapshot Service"]
    end

    subgraph "Index Layer"
        M["Linear Index"]
        N["KD-Tree Index"]
        O["LSH Index"]
    end

    subgraph "Storage Layer"
        P["In-Memory Repository"]
        Q["Reader-Writer Locks"]
    end

    subgraph "Persistence"
        R["JSON Snapshots"]
        S["File System"]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    G --> I
    G --> J
    G --> K
    G --> L
    K --> M
    K --> N
    K --> O
    H --> P
    I --> P
    J --> P
    L --> P
    P --> Q
    L --> R
    R --> S

    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style D fill:#fff3e0
    style G fill:#f3e5f5
    style M fill:#e8f5e9
    style N fill:#e8f5e9
    style O fill:#e8f5e9
    style P fill:#fce4ec
    style L fill:#e0f2f1
    style R fill:#f5f5f5
```

### Data Model Hierarchy

```mermaid
erDiagram
    LIBRARY ||--o{ DOCUMENT : contains
    DOCUMENT ||--o{ CHUNK : contains
    LIBRARY ||--o| INDEX : has

    LIBRARY {
        string id PK
        string name
        string description
        json metadata
    }

    DOCUMENT {
        string id PK
        string library_id FK
        string title
        string description
        json metadata
    }

    CHUNK {
        string id PK
        string document_id FK
        string text
        float[] embedding
        json metadata
    }

    INDEX {
        string library_id FK
        string algorithm
        string metric
        json index_data
    }
```

### Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Service
    participant Repository
    participant Index
    participant Lock

    Client->>API: Search Request
    API->>API: Validate Input
    API->>Service: Process Search
    Service->>Lock: Acquire Read Lock
    Lock-->>Service: Lock Granted
    Service->>Repository: Get Library Data
    Repository-->>Service: Return Data
    Service->>Index: Query Index
    Index-->>Service: Return Results
    Service->>Service: Apply Filters
    Service->>Lock: Release Lock
    Service-->>API: Search Results
    API-->>Client: JSON Response
```

### Concurrency Model

The system uses a custom Reader-Writer lock implementation that:

- Allows multiple concurrent read operations for high throughput
- Ensures exclusive write access for data consistency
- Implements writer priority to prevent starvation
- Uses context managers for clean resource management

# Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/vectordb.git
cd vectordb

# Run with Docker
docker-compose up

# API available at http://localhost:8000
```

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

make run

uvicorn app.main:app --reload --port 8000
```

### Quick Example

```python
from sdk.client import VectorDBClient

# Initialize client
client = VectorDBClient(base_url="http://localhost:8000")

# Create a library
library = client.create_library("my-vectors", description="Demo library")

# Create a document
doc = client.create_document(
    library["id"],
    title="Sample Document",
    metadata={"category": "demo"}
)

# Add vector chunks
chunk = client.create_chunk(
    library["id"],
    doc["id"],
    text="The Eiffel Tower is in Paris",
    embedding=[0.1, 0.2, 0.3],  # Your embedding vector
    metadata={"language": "en"}
)

# Build an index for fast search (uses PUT under the hood)
client.build_index(library["id"], algorithm="lsh", metric="cosine")

# Search for similar vectors
results = client.search(
    library["id"],
    vector=[0.1, 0.15, 0.3],
    k=5,
    metadata_filters={"language": "en"}
)

print(f"Found {len(results['results'])} similar vectors")
```

# Configuration

Environment variables for customization:

| Variable         | Default | Description                            |
| ---------------- | ------- | -------------------------------------- |
| `ENV`            | `local` | Environment (local/staging/production) |
| `DATA_DIR`       | `data`  | Directory for snapshots                |
| `COHERE_API_KEY` | -       | API key for embeddings (optional)      |

## Production Deployment

### Multi-Worker Limitations

The default in-memory repository is **per-process** and not suitable for multi-worker deployments. When running with Gunicorn or similar WSGI servers with multiple workers:

- Each worker maintains its own separate data copy
- Data will diverge across workers
- Snapshots will not be shared between processes

### Solutions

1. **Single Worker Mode** (Quick fix):

   ```bash
   gunicorn app.main:app --workers 1
   ```

2. **Persistent Repository** (Recommended):
   Implement a file-based or database-backed repository, then configure via:

   ```python
   from app.repositories import FileRepository  # or PostgresRepository
   from app.services import set_repository

   # At startup
   repository = FileRepository(data_dir="/persistent/data")
   set_repository(repository)
   ```

3. **External Storage**:
   Use Redis, PostgreSQL, or another shared storage backend for production deployments.

| Variable         | Default  | Description               |
| ---------------- | -------- | ------------------------- |
| `DEFAULT_METRIC` | `cosine` | Default similarity metric |
| `DEFAULT_INDEX`  | `linear` | Default index algorithm   |
| `LSH_NUM_PLANES` | `16`     | LSH hash bit count        |
| `LSH_NUM_TABLES` | `4`      | LSH table count           |
| `LOG_LEVEL`      | `INFO`   | Logging verbosity         |

# API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Core Endpoints

| Method              | Endpoint                                 | Description                      |
| ------------------- | ---------------------------------------- | -------------------------------- |
| **Libraries**       |
| POST                | `/libraries/`                            | Create a new library             |
| GET                 | `/libraries/`                            | List all libraries               |
| GET                 | `/libraries/{id}`                        | Get library details              |
| PATCH               | `/libraries/{id}`                        | Update library                   |
| DELETE              | `/libraries/{id}`                        | Delete library                   |
| **Documents**       |
| POST                | `/libraries/{id}/documents`              | Create document                  |
| GET                 | `/libraries/{id}/documents`              | List documents                   |
| GET                 | `/libraries/{id}/documents/{doc_id}`     | Get document details             |
| PATCH               | `/libraries/{id}/documents/{doc_id}`     | Update document                  |
| DELETE              | `/libraries/{id}/documents/{doc_id}`     | Delete document                  |
| **Chunks**          |
| POST                | `/libraries/{id}/chunks`                 | Create chunk                     |
| GET                 | `/libraries/{id}/chunks`                 | List chunks                      |
| GET                 | `/libraries/{id}/chunks/{chunk_id}`      | Get chunk details                |
| PATCH               | `/libraries/{id}/chunks/{chunk_id}`      | Update chunk                     |
| DELETE              | `/libraries/{id}/chunks/{chunk_id}`      | Delete chunk                     |
| **Index & Search**  |
| PUT                 | `/libraries/{id}/index`                  | Create/replace index             |
| GET                 | `/libraries/{id}/index`                  | Get index info                   |
| DELETE              | `/libraries/{id}/index`                  | Clear index                      |
| POST                | `/libraries/{id}/chunks/search`          | Search vectors                   |
| **Admin/Snapshots** |
| GET                 | `/admin/snapshots`                       | List all snapshots               |
| POST                | `/admin/snapshots`                       | Create snapshot                  |
| GET                 | `/admin/snapshots/{snapshot_id}`         | Get snapshot details             |
| POST                | `/admin/snapshots/{snapshot_id}/restore` | Restore from snapshot (sync 200) |
| DELETE              | `/admin/snapshots/{snapshot_id}`         | Delete snapshot                  |
| **Utilities**       |
| GET                 | `/health`                                | Health check                     |
| POST                | `/embeddings`                            | Generate embeddings              |

### Example API Calls

```bash
# Create a library
curl -X POST http://localhost:8000/libraries/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "product-embeddings",
    "description": "Product description vectors",
    "metadata": {"version": "1.0"}
  }'

# Create a document
curl -X POST http://localhost:8000/libraries/{library_id}/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Product Catalog",
    "description": "Product descriptions",
    "metadata": {"category": "electronics"}
  }'

# Create a chunk with embedding
curl -X POST http://localhost:8000/libraries/{library_id}/chunks \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "{document_id}",
    "text": "High-quality wireless headphones",
    "embedding": [0.1, 0.2, 0.3, 0.4],
    "metadata": {"category": "electronics"}
  }'

# Build an index (create/replace)
curl -X PUT http://localhost:8000/libraries/{library_id}/index \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "kdtree",
    "metric": "euclidean"
  }'

# Search for similar vectors
curl -X POST http://localhost:8000/libraries/{library_id}/chunks/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4],
    "k": 10,
    "metadata_filters": {"category": "electronics"}
  }'

# Generate embeddings (requires COHERE_API_KEY)
curl -X POST http://localhost:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "text": "High-quality wireless headphones with noise cancellation"
  }'

# Create a snapshot
curl -X POST http://localhost:8000/admin/snapshots \
  -H "Content-Type: application/json" \
  -d '{
    "name": "backup_before_migration"
  }'

# List all snapshots
curl -X GET http://localhost:8000/admin/snapshots

# Restore from snapshot (synchronous)
curl -X POST http://localhost:8000/admin/snapshots/{snapshot_id}/restore
```

# Python SDK

### Installation

### Complete Example

```python
from sdk.client import VectorDBClient

class VectorDBManager:
    def __init__(self, base_url="http://localhost:8000"):
        self.client = VectorDBClient(base_url=base_url)

    def setup_library(self, name: str) -> str:
        """Create and configure a new library"""
        library = self.client.create_library(
            name=name,
            description=f"{name} vector collection",
            metadata={"created_by": "sdk"}
        )
        return library["id"]

    def add_vectors(self, library_id: str, vectors: list, texts: list):
        """Add multiple vectors with text"""
        doc = self.client.create_document(
            library_id,
            title="Batch Import",
            metadata={"type": "bulk"}
        )

        chunks = []
        for vector, text in zip(vectors, texts):
            chunk = self.client.create_chunk(
                library_id,
                doc["id"],
                text=text,
                embedding=vector.tolist() if hasattr(vector, 'tolist') else vector,
                metadata={"source": "batch"}
            )
            chunks.append(chunk)

        return chunks

    def similarity_search(self, library_id: str, query_vector: list, k: int = 5):
        """Perform similarity search"""
        # Build index if needed
        self.client.build_index(library_id, algorithm="lsh", metric="cosine")

        # Search
        results = self.client.search(
            library_id,
            vector=query_vector,
            k=k
        )

        return results

# Usage
manager = VectorDBManager()
lib_id = manager.setup_library("demo")

vectors = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
texts = ["Document 1", "Document 2", "Document 3"]

# Add vectors
manager.add_vectors(lib_id, vectors, texts)

# Search
query = [0.15, 0.25, 0.35]
results = manager.similarity_search(lib_id, query, k=2)
print(f"Found {len(results['results'])} similar vectors")
```

# Indexing Algorithms

### Algorithm Comparison

| Algorithm   | Build Time | Search Time | Memory |
| ----------- | ---------- | ----------- | ------ |
| **Linear**  | O(1)       | O(n)        | O(n)   |
| **KD-Tree** | O(n log n) | O(log n)\*  | O(n)   |
| **LSH**     | O(n×t×p)   | O(t×m)      | O(n×t) |

\*Average case; worst case O(n) for KD-Tree

### Supported Metric Combinations

| Algorithm | Cosine Similarity | Euclidean Distance |
| --------- | ----------------- | ------------------ |
| Linear    | ✅                | ✅                 |
| KD-Tree   | ❌                | ✅                 |
| LSH       | ✅                | ❌                 |

# Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest app/tests/test_api_libraries.py

# Run integration tests
pytest app/tests/test_indexing_and_search.py -v
```

### Test Coverage

- Unit tests for all services
- Integration tests for API endpoints
- Persistence validation

### Project Structure

```
vectordb/
├── app/
│   ├── api/            # FastAPI routers
│   │   └── routers/    # API endpoints
│   ├── core/           # Core configuration
│   ├── domain/         # Business entities
│   │   ├── dto/        # Data transfer objects
│   │   └── models/     # Domain models
│   ├── repositories/   # Data access layer
│   ├── services/       # Business logic
│   ├── vector_index/   # Index implementations
│   └── tests/          # Test suite
├── sdk/                # Python client library
├── scripts/            # Utility scripts
└── postman/            # API collection
```

## Notes

- Indices are built per-library and cached in-memory. Snapshot persistence saves data and index metadata; indices are rebuilt on load.
- The `/embeddings` endpoint requires a valid `COHERE_API_KEY` and proxies to Cohere with retry logic. Without the key, it returns 503.
