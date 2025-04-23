# QAnvas - Canvas Learning Management System Q&A Assistant

> QAnvas is an intelligent Q&A system for the Canvas Learning Management System that combines advanced data retrieval capabilities with natural language processing to provide users with a powerful tool for accessing course information.

## 1. Project Overview

QAnvas consists of two main modules:

1. **MongoDB Database Module** - Responsible for collecting and storing course data from the Canvas API
2. **Canvas QA Intelligent Q&A Module** - Provides Q&A functionality based on advanced language models

This project aims to simplify the process of retrieving information from Canvas learning management system, enabling users to easily access course content, assignment information, announcements, and more through natural language queries.



## 2. Core Features

### 2.1 Data Collection and Storage
- Retrieve course information, folders, files, assignments, and announcements via Canvas API
- Store data in a structured MongoDB database
- Support local storage and retrieval of file content
- Handle paginated results and nested folder structures
- Automatically track resource updates and version management

### 2.2 Intelligent Q&A System
- Use Google Gemini 2.0 Flash model for natural language processing
- Vector search technology for finding semantically related content
- Intelligently determine search scope and relevant courses
- PDF document analysis and content extraction
- Complex query processing based on ReAct agent architecture



## 3. Technical Architecture

### 3.1 Database Structure
MongoDB database `canvas_qa_system` contains the following collections:
- `courses`: Stores basic course information
- `folders`: Stores folder structures
- `files`: Stores file metadata and local paths
- `assignments`: Stores assignment information
- `announcements`: Stores announcement information
- `query_logs`: Stores query logs

### 3.2 System Components
- **LLM Engine**: Uses Google Gemini 2.0 Flash model
- **Vector Search**: Uses SentenceTransformer to generate text embeddings
- **MongoDB Integration**: Connects to a MongoDB database storing Canvas data
- **Langchain Framework**: Used for building ReAct agents and tool integration



## 4. Project Structure

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
│   │   └── multi_intents_decompostion.py      # Query analysis and spliting
│   └── Agent_Module_Summary.md
```



## 5. Key Implementation Details

### 5.1 Query Decomposition
Our system supports **multi-intent query decomposition**, allowing users to issue complex, natural language queries involving multiple information needs. For example, consider the user query:

> _“I missed Prof. Wong’s last lecture, so I’m not sure if we’re still presenting next Monday or if it got postponed. Also, I heard the final is now open-book — is that true? And where are the updated slides uploaded?”_

This is automatically decomposed into the following sub-queries:

1. **`presentation_schedule`** — _Are we still presenting next Monday for Prof. Wong's class, or has it been postponed?_
2. **`exam_format`** — _Is the final exam for Prof. Wong's class open-book?_
3. **`lecture_notes`** — _Where are the updated lecture slides for Prof. Wong's class uploaded?_

Each sub-query is enriched through contextual disambiguation (e.g., resolving “it” to "presentation", “the final” to the course exam) and routed to the appropriate handler. This enables:
- Precise handling of implicit references and cross-sentence dependencies
- Parallel querying and result synthesis from multiple APIs
- A more natural and fluent user experience with minimal need for query rewriting

### 5.2 Pagination Handling
Canvas API interfaces typically return paginated data, which our system handles by:
- Automatically detecting and processing Canvas API's `Link` header information
- Recursively or iteratively requesting all available pages
- Merging paginated results into a single dataset
- Implementing backpressure mechanisms to prevent request overload

### 5.3 Vector Search Implementation
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

### 5.4 ReAct Agent Implementation
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



## 6. Usage Flow

### 6.1 Data Collection
1. Configure Canvas API access
2. Run the MongoDB module to collect course data
3. Verify that data has been successfully stored in MongoDB

### 6.2 Q&A Function
```python
agent = CanvasQAAgent()
query = "When is my data analysis assignment due?"
answer = agent.answer_query(query)
print(answer)
```



## 7. Installation and Usage

**Prerequisites**:
- Python 3.9+
- MongoDB 4.4+
- Canvas API access token

**Get Started**:

1. Clone the repository
   ```
   git clone https://github.com/LuminousDy/PLP-PM-2025-2-27-GRP6-QAnvas.git
   cd PLP-PM-2025-2-27-GRP6-QAnvas
   ```
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables or `.env` file (get token from Canvas -> Account -> Settings -> Approved integrations -> New Access Token):
   ```
   CANVAS_API_URL=https://your-canvas-instance.com/api/v1
   CANVAS_API_TOKEN=your_canvas_api_token
   MONGODB_CONNECTION_STRING=mongodb://localhost:27017
   GOOGLE_API_KEY=your_google_api_key
   ```
4. Launch MongoDB service on port 27017
5. Run the data collection module: `python MongoDB_code/main.py`
6. Start the Q&A agent: `python Agent_code/main.py`



## 8. Optimization and Performance
- Batch processing to improve embedding generation efficiency
- Support for GPU acceleration of embedding models (if CUDA is available)
- Creating indexes on MongoDB collections to improve query performance
- Splitting long documents into chunks for better processing
- Implementing caching mechanisms to reduce repeated API requests
- Using connection pools to optimize database connections

## 9. Multi-Intent Query Decomposition with T5 (Optional)

This repository contains the code and training instructions for fine-tuning a T5-based model to perform **multi-intent query decomposition** — converting natural user queries into structured sub-queries, each with its associated semantic intent.

---

## 📊 Dataset

The dataset is available for download from:

🔗 [Google Drive Link](https://drive.google.com/file/d/12jXaMwgx4YbYZTgi9ntA1pDhUcGMrODq/view?usp=sharing)

## Model
The finetuned model for download from:

🔗 [T5 specific for query decompostion-after finetuned](https://drive.google.com/drive/folders/1MZUM5gPLVdk6LgtQ8nEBnXUyiMmcYbg6?usp=drive_link)

and put it under the 
'''
QAnvas/
├──Multi_Intents_code
│ ├── checkpoints/

'''
## 10. Summary

QAnvas provides a powerful natural language interface for the Canvas learning management system, enabling users to easily access course-related information. It combines modern LLM technology, vector search, and structured database queries to offer an intelligent assistant solution for educational environments. By determining the most relevant search paths and using precise search tools, the agent can deliver accurate and helpful responses tailored to user queries. 
