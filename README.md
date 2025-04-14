# QAnvas - Canvas Learning Management System Q&A Assistant

QAnvas is an intelligent Q&A system for the Canvas Learning Management System that combines advanced data retrieval capabilities with natural language processing to provide users with a powerful tool for accessing course information.

## Project Overview

QAnvas consists of two main modules:

1. **MongoDB Database Module** - Responsible for collecting and storing course data from the Canvas API
2. **Canvas QA Intelligent Q&A Module** - Provides Q&A functionality based on advanced language models

This project aims to simplify the process of retrieving information from Canvas learning management system, enabling users to easily access course content, assignment information, announcements, and more through natural language queries.

## Core Features

### Data Collection and Storage
- Retrieve course information, folders, files, assignments, and announcements via Canvas API
- Store data in a structured MongoDB database
- Support local storage and retrieval of file content
- Handle paginated results and nested folder structures
- Automatically track resource updates and version management

### Intelligent Q&A System
- Use Google Gemini 2.0 Flash model for natural language processing
- Vector search technology for finding semantically related content
- Intelligently determine search scope and relevant courses
- PDF document analysis and content extraction
- Complex query processing based on ReAct agent architecture

## Technical Architecture

### Database Structure
MongoDB database `canvas_qa_system` contains the following collections:
- `courses`: Stores basic course information
- `folders`: Stores folder structures
- `files`: Stores file metadata and local paths
- `assignments`: Stores assignment information
- `announcements`: Stores announcement information
- `query_logs`: Stores query logs

### System Components
- **LLM Engine**: Uses Google Gemini 2.0 Flash model
- **Vector Search**: Uses SentenceTransformer to generate text embeddings
- **MongoDB Integration**: Connects to a MongoDB database storing Canvas data
- **Langchain Framework**: Used for building ReAct agents and tool integration

## Project Structure

```
QAnvas/
├── README.md
├── MongoDB_code/
│   ├── main.py                  # Main program entry point
│   ├── core/
│   │   ├── api/                 # Canvas API interfaces
│   │   ├── db/                  # MongoDB database connections and collection definitions
│   │   ├── processors/          # Data processors
│   │   └── storage/             # File storage management
│   └── MongoDB_Module_Summary.md
├── Agent_code/
│   ├── main.py                  # QA agent main program entry point
│   ├── config/
│   │   └── settings.py          # Configurations such as database connections, API keys, etc.
│   ├── prompts/
│   │   └── templates.py         # Prompt templates for agent decision-making and responses
│   ├── models/
│   │   └── data_models.py       # Data models for search paths and results
│   ├── tools/
│   │   ├── canvas_search.py     # Canvas content search functionality
│   │   └── pdf_analyzer.py      # PDF document analysis functionality
│   └── Agent_Module_Summary.md
```

## Key Implementation Details

### Pagination Handling
Canvas API interfaces typically return paginated data, which our system handles by:
- Automatically detecting and processing Canvas API's `Link` header information
- Recursively or iteratively requesting all available pages
- Merging paginated results into a single dataset
- Implementing backpressure mechanisms to prevent request overload

### Vector Search Implementation
```python
# Generate embedding for query
query_embedding = self.model.encode([query], prompt_name="query", 
                                   show_progress_bar=False)[0]

# Generate embeddings for documents
doc_embeddings = []
for batch in batches_of(texts_to_embed, 32):
    batch_embeddings = self.model.encode(batch, show_progress_bar=False)
    doc_embeddings.extend(batch_embeddings)
    
# Compute similarity and sort results
similarities = np.dot(normalized_doc_embeddings, normalized_query_embedding)
top_indices = np.argsort(-similarities)[:top_k]
```

### ReAct Agent Implementation
```python
self.agent = create_react_agent(
    llm=self.llm,
    tools=self.tools,
    prompt=REACT_PROMPT
)

self.agent_executor = AgentExecutor(
    agent=self.agent,
    tools=self.tools,
    memory=self.memory,
    verbose=True,
    handle_parsing_errors=True
)
```

## Usage Flow

### Data Collection
1. Configure Canvas API access
2. Run the MongoDB module to collect course data
3. Verify that data has been successfully stored in MongoDB

### Q&A Function
```python
agent = CanvasQAAgent()
query = "When is my data analysis assignment due?"
answer = agent.answer_query(query)
print(answer)
```

## Optimization and Performance
- Batch processing to improve embedding generation efficiency
- Support for GPU acceleration of embedding models (if CUDA is available)
- Creating indexes on MongoDB collections to improve query performance
- Splitting long documents into chunks for better processing
- Implementing caching mechanisms to reduce repeated API requests
- Using connection pools to optimize database connections

## Installation and Configuration

### Prerequisites
- Python 3.8+
- MongoDB 4.4+
- Canvas API access token

### Installation Steps
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables or `.env` file:
   ```
   CANVAS_API_URL=https://your-canvas-instance.com/api/v1
   CANVAS_API_TOKEN=your_canvas_api_token
   MONGODB_CONNECTION_STRING=mongodb://localhost:27017
   GOOGLE_API_KEY=your_google_api_key
   ```
4. Run the data collection module: `python MongoDB_code/main.py`
5. Start the Q&A agent: `python Agent_code/main.py`

## Summary

QAnvas provides a powerful natural language interface for the Canvas learning management system, enabling users to easily access course-related information. It combines modern LLM technology, vector search, and structured database queries to offer an intelligent assistant solution for educational environments. By determining the most relevant search paths and using precise search tools, the agent can deliver accurate and helpful responses tailored to user queries. 