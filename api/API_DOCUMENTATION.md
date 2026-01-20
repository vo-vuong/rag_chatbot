# RAG Chatbot API Documentation

API documentation for the RAG Chatbot backend service.

## Overview

- **Base URL**: `http://localhost:8000`
- **API Prefix**: `/api/v1`
- **Version**: `1.0.0`
- **Content-Type**: `application/json` (except multipart/form-data for upload preview)

### Default Configuration

All retrieval parameters use centralized constants from `config/constants.py`:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `DEFAULT_NUM_RETRIEVAL` | `5` | Number of text chunks to retrieve |
| `DEFAULT_SCORE_THRESHOLD` | `0.7` | Minimum similarity score for text search |
| `DEFAULT_IMAGE_NUM_RETRIEVAL` | `1` | Number of images to retrieve |
| `DEFAULT_IMAGE_SCORE_THRESHOLD` | `0.6` | Minimum similarity score for image search |

These defaults are used in:
- Request models (`api/models/requests.py`)
- Service layer (`api/services/`)
- Session state (`backend/session_manager.py`)
- UI components (`ui/`)

To change defaults globally, modify values in `config/constants.py`.

### Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check with Qdrant status |
| `/api/v1/chat/query` | POST | Synchronous chat with RAG |
| `/api/v1/chat/query/stream` | POST | SSE streaming chat |
| `/api/v1/rag/search` | POST | Vector search without LLM |
| `/api/v1/upload/preview` | POST | Process document and return preview |
| `/api/v1/upload/save` | POST | Save processed chunks to Qdrant |

## Endpoints

### 1. Health Check

Check API and Qdrant database status.

| Property | Value |
|----------|-------|
| **Method** | `GET` |
| **Path** | `/api/v1/health` |
| **Auth** | None |

#### Request

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

#### Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "qdrant_connected": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"healthy"` or `"degraded"` |
| `version` | string | API version |
| `qdrant_connected` | boolean | Qdrant DB connection status |

---

### 2. Chat Query (Synchronous)

Process a chat query and return complete response.

| Property | Value |
|----------|-------|
| **Method** | `POST` |
| **Path** | `/api/v1/chat/query` |
| **Auth** | None |

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | User's question |
| `session_id` | string | Yes | - | Unique session identifier |
| `mode` | string | No | `"rag"` | `"rag"` or `"llm_only"` |
| `top_k` | integer | No | `5` | Number of documents to retrieve |
| `score_threshold` | float | No | `0.7` | Minimum similarity score |

#### Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "session_id": "user-123-session-1",
    "mode": "rag",
    "top_k": 5,
    "score_threshold": 0.7
  }'
```

#### Example Response

```json
{
  "response": "Machine learning is a subset of artificial intelligence...",
  "route": "text_only",
  "route_reasoning": "Query asks about concepts, no images needed",
  "retrieved_chunks": [
    {
      "text": "Machine learning enables computers to learn from data...",
      "score": 0.89,
      "source_file": "ml_guide.pdf",
      "page_number": 5,
      "element_type": "text"
    },
    {
      "text": "ML algorithms can be supervised or unsupervised...",
      "score": 0.82,
      "source_file": "ml_guide.pdf",
      "page_number": 12,
      "element_type": "text"
    }
  ],
  "images": [],
  "timestamp": "2026-01-13T10:30:00.000000"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | Generated answer |
| `route` | string | Query route: `"text_only"`, `"image_only"`, or `"text_and_image"` |
| `route_reasoning` | string | Explanation for routing decision |
| `retrieved_chunks` | array | Retrieved text chunks with scores, source_file, page_number (optional), element_type |
| `images` | array | Retrieved images with metadata |
| `timestamp` | string | ISO timestamp |

---

### 3. Chat Query (Streaming)

Process a chat query with Server-Sent Events (SSE) streaming.

| Property | Value |
|----------|-------|
| **Method** | `POST` |
| **Path** | `/api/v1/chat/query/stream` |
| **Auth** | None |
| **Response Type** | `text/event-stream` |

#### Request Body

Same as [Chat Query (Synchronous)](#2-chat-query-synchronous).

#### Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/chat/query/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "query": "Explain neural networks",
    "session_id": "user-123-session-1"
  }'
```

#### SSE Event Types

| Event | Data Structure | Description |
|-------|---------------|-------------|
| `route` | `{"route": "text_only", "reasoning": "..."}` | Query routing info |
| `context` | `[{"text": "...", "score": 0.85}]` | Retrieved context chunks |
| `token` | `"partial response text..."` | Streamed response tokens |
| `done` | `{"timestamp": "2026-01-13T10:30:00"}` | Stream complete |
| `error` | `"error message"` | Error occurred |

#### Example SSE Response

```
data: {"event": "route", "data": {"route": "text_only", "reasoning": "Query is about concepts"}}

data: {"event": "context", "data": [{"text": "Neural networks are...", "score": 0.91}]}

data: {"event": "token", "data": "Neural networks are computational "}

data: {"event": "token", "data": "models inspired by biological "}

data: {"event": "done", "data": {"timestamp": "2026-01-13T10:30:00.123456"}}
```

#### JavaScript Example

```javascript
const eventSource = new EventSource('/api/v1/chat/query/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'What is deep learning?',
    session_id: 'session-123'
  })
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  switch (data.event) {
    case 'token':
      console.log('Token:', data.data);
      break;
    case 'done':
      console.log('Complete');
      eventSource.close();
      break;
  }
};
```

---

### 4. RAG Search

Search the vector database directly without LLM generation.

| Property | Value |
|----------|-------|
| **Method** | `POST` |
| **Path** | `/api/v1/rag/search` |
| **Auth** | None |

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query |
| `collection_type` | string | No | `"text"` | `"text"` or `"image"` |
| `top_k` | integer | No | `5` | Number of results |
| `score_threshold` | float | No | `0.7` | Minimum similarity score |

#### Example Request (Text Search)

```bash
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "transformer architecture",
    "collection_type": "text",
    "top_k": 5,
    "score_threshold": 0.7
  }'
```

#### Example Response (Text)

```json
{
  "route": "text_only",
  "reasoning": "Query is about technical concepts",
  "results": [
    {
      "text": "The Transformer architecture uses self-attention mechanisms...",
      "score": 0.92,
      "source_file": "transformers.pdf",
      "page_number": 3,
      "element_type": "text"
    },
    {
      "text": "Attention is computed as a weighted sum of values...",
      "score": 0.87,
      "source_file": "transformers.pdf",
      "page_number": 8,
      "element_type": "text"
    }
  ]
}
```

#### Example Request (Image Search)

```bash
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural network diagram",
    "collection_type": "image",
    "top_k": 3
  }'
```

#### Example Response (Image)

```json
{
  "route": "image_only",
  "reasoning": "Query requests visual content",
  "results": [
    {
      "caption": "Diagram showing layers of a neural network",
      "image_path": "/data/images/nn_diagram.png",
      "score": 0.88,
      "page_number": 15,
      "source_file": "neural_nets.pdf"
    }
  ]
}
```

---

### 5. Upload Document (Two-Step Workflow)

The upload functionality uses a two-step workflow: preview then save.

#### Step 1: Preview Upload

Process document and return preview data without saving to Qdrant.

| Property | Value |
|----------|-------|
| **Method** | `POST` |
| **Path** | `/api/v1/upload/preview` |
| **Content-Type** | `multipart/form-data` |
| **Auth** | None |

**Request Body (Form Data)**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | File | Yes | - | Document file (PDF/DOCX/CSV) |
| `language` | string | No | `"en"` | Document language (`"en"` or `"vi"`) |
| `processing_mode` | string | No | `"fast"` | `"fast"` (no OCR) or `"ocr"` |
| `csv_columns` | string | No | - | Comma-separated column names for CSV |
| `vision_failure_mode` | string | No | `"graceful"` | `"graceful"`, `"strict"`, or `"skip"` |

**Example Request**

```bash
curl -X POST "http://localhost:8000/api/v1/upload/preview" \
  -F "file=@document.pdf" \
  -F "language=en" \
  -F "processing_mode=fast" \
  -F "vision_failure_mode=graceful"
```

**Example Response**

```json
{
  "status": "success",
  "file_name": "document.pdf",
  "file_type": "pdf",
  "chunks": [
    {
      "text": "Chapter 1: Introduction...",
      "source_file": "document.pdf",
      "page_number": 1,
      "element_type": "text",
      "chunk_index": 0,
      "file_type": "pdf",
      "headings": ["Chapter 1"],
      "token_count": 245
    }
  ],
  "images": [
    {
      "caption": "Diagram showing system architecture",
      "image_path": "/extracted_images/abc123.png",
      "page_number": 5,
      "source_file": "document.pdf",
      "image_hash": "abc123...",
      "image_metadata": {"width": 800, "height": 600}
    }
  ],
  "total_chunks_count": 42,
  "total_images_count": 5,
  "processing_time_seconds": 12.34,
  "full_chunks_data": [...],
  "full_images_data": [...]
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"success"` or `"error"` |
| `file_name` | string | Uploaded filename |
| `file_type` | string | File extension (pdf/docx/csv) |
| `chunks` | array | Preview chunks (max 50) |
| `images` | array | Preview images |
| `total_chunks_count` | integer | Total number of chunks |
| `total_images_count` | integer | Total number of images |
| `processing_time_seconds` | float | Processing duration |
| `full_chunks_data` | array | Complete chunk data for save step |
| `full_images_data` | array | Complete image data for save step |

#### Step 2: Save Upload

Save processed chunks and images to Qdrant.

| Property | Value |
|----------|-------|
| **Method** | `POST` |
| **Path** | `/api/v1/upload/save` |
| **Content-Type** | `application/json` |
| **Auth** | None |

**Request Body (JSON)**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_name` | string | Yes | Original filename |
| `file_type` | string | Yes | File type (pdf/docx/csv) |
| `language` | string | Yes | Document language |
| `chunks` | array | Yes | Chunk data from preview step |
| `images` | array | No | Image data from preview step |

**Example Request**

```bash
curl -X POST "http://localhost:8000/api/v1/upload/save" \
  -H "Content-Type: application/json" \
  -d '{
    "file_name": "document.pdf",
    "file_type": "pdf",
    "language": "en",
    "chunks": [...],
    "images": [...]
  }'
```

**Example Response**

```json
{
  "status": "success",
  "file_name": "document.pdf",
  "chunks_count": 42,
  "images_count": 5,
  "message": "Successfully saved 42 chunks and 5 images",
  "text_collection": "rag_docs",
  "image_collection": "rag_images"
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"success"` or `"error"` |
| `file_name` | string | Filename |
| `chunks_count` | integer | Number of chunks saved |
| `images_count` | integer | Number of images saved |
| `message` | string | Status message |
| `text_collection` | string | Qdrant collection for text chunks |
| `image_collection` | string | Qdrant collection for images |

#### Vision Failure Modes

| Mode | Behavior |
|------|----------|
| `graceful` | Use fallback caption on failure, continue processing |
| `strict` | Abort entire upload on any caption failure |
| `skip` | Skip failed images, continue with successful ones |

---

## Error Handling

All endpoints return standard HTTP error responses.

### Error Response Format

```json
{
  "detail": "Error message describing the issue"
}
```

### Common HTTP Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `400` | Bad Request - Invalid input |
| `500` | Internal Server Error |

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Chat query
response = requests.post(
    f"{BASE_URL}/chat/query",
    json={
        "query": "What is machine learning?",
        "session_id": "my-session-123",
        "mode": "rag",
        "top_k": 3
    }
)
print(response.json())

# RAG search
response = requests.post(
    f"{BASE_URL}/rag/search",
    json={
        "query": "attention mechanism",
        "collection_type": "text",
        "top_k": 5
    }
)
print(response.json())

# Upload document (two-step workflow)
# Step 1: Preview
with open("document.pdf", "rb") as f:
    preview_response = requests.post(
        f"{BASE_URL}/upload/preview",
        files={"file": ("document.pdf", f)},
        data={
            "language": "en",
            "processing_mode": "fast",
            "vision_failure_mode": "graceful"
        }
    )
preview_data = preview_response.json()
print(f"Preview: {preview_data['total_chunks_count']} chunks, {preview_data['total_images_count']} images")

# Step 2: Save
save_response = requests.post(
    f"{BASE_URL}/upload/save",
    json={
        "file_name": preview_data["file_name"],
        "file_type": preview_data["file_type"],
        "language": "en",
        "chunks": preview_data["full_chunks_data"],
        "images": preview_data["full_images_data"]
    }
)
print(save_response.json())
```

---

## Configuration

The API uses environment variables for configuration (via `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |

---

## Running the API

```bash
# Activate environment
conda activate rag_chatbot

# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

OpenAPI documentation available at: `http://localhost:8000/docs`
