# Canvas QA Agent Module Summary

## 1. Overview

The Canvas QA Agent is an intelligent question-answering agent designed specifically for the Canvas Learning Management System. It combines a large language model (Gemini) with a MongoDB database to provide users with an intelligent assistant capable of answering questions about Canvas course content. The agent can automatically determine the search scope based on user queries, retrieve relevant information, and generate accurate responses.

## 2. Core Features

### 2.1 Intelligent Q&A

- Receive and analyze user questions  
- Identify the most relevant data modules and courses  
- Perform searches to retrieve relevant information  
- Generate natural language responses  

### 2.2 Intelligent Search

- Use vector search technology to find semantically related content in MongoDB  
- Support searching for different types of resources such as announcements, assignments, and files  
- Filter search results based on course names  
- Combine vector similarity matching to improve search accuracy  

### 2.3 PDF Analysis

- Support analysis of PDF document content  
- Split documents into manageable text blocks  
- Extract pages and content relevant to the query  

## 3. Technical Architecture

### 3.1 Architecture Components

- **LLM Engine**: Uses the Google Gemini 2.0 Flash model for natural language processing  
- **Vector Search**: Uses SentenceTransformer to generate text embeddings  
- **MongoDB Integration**: Connects to a MongoDB database storing Canvas data  
- **Langchain Framework**: Used for building ReAct agents and tool integration  

### 3.2 File Organization

The module includes the following main files and directories:

- `main.py`: Main program entry point, defines the CanvasQAAgent class and implements core logic  
- `config/`: Configuration directory  
  - `settings.py`: Contains configurations such as database connections, API keys, and model settings  
- `prompts/`: Prompt templates directory  
  - `templates.py`: Defines prompt templates for agent decision-making and responses  
- `models/`: Data models directory  
  - `data_models.py`: Defines data models for search paths and results  
- `tools/`: Tools directory  
  - `canvas_search.py`: Implements Canvas content search functionality  
  - `pdf_analyzer.py`: Implements PDF document analysis functionality  

## 4. Key Implementation Details

### 4.1 ReAct Agent Implementation

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
'''

### 4.2 Search Path Determination

Search path is determined by LLM analyzing the user query and deciding which modules and courses should be searched:

```python
def _determine_search_path(self, query: str) -> SearchPath:
    # Format prompt with query
    prompt = SEARCH_PATH_TEMPLATE.format(query=query)
    
    # Get response from LLM
    response = self.llm.predict(prompt)
    
    # Clean response to ensure valid JSON
    # ... handle response ...
    
    # Parse JSON and create SearchPath object
    data = json.loads(json_str)
    return SearchPath(**data)
```

### 4.3 Vector Search Implementation

Uses SentenceTransformer to create embeddings for query and documents, then computes similarity:

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

## 5. Usage Flow

1. Initialize a CanvasQAAgent instance, connecting to MongoDB database and Gemini API
2. Receive user query
3. Agent analyzes the query and determines search scope
4. Execute search to obtain relevant information
5. Generate and return response

```python
agent = CanvasQAAgent()
query = "When is my data analysis assignment due?"
answer = agent.answer_query(query)
print(answer)
```

## 6. Optimization and Performance

-Uses batching to improve embedding generation efficiency
-Supports GPU acceleration for embedding models (if CUDA is available)
-Creates indexes on MongoDB collections to improve query performance
-Splits long documents into chunks for better processing

## 7. Summary

The Canvas QA Agent provides a powerful natural language interface that allows users to easily access information related to Canvas courses. It combines modern LLM technology, vector search, and structured database queries to offer an intelligent assistant solution for educational environments. By determining the most relevant search path and using precise search tools, the agent can deliver accurate and helpful responses tailored to user queries.