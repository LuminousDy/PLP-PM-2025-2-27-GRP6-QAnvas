"""
Canvas search implementation
"""
from typing import Dict, List, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from models.data_models import SearchPath, SearchResult
from config.settings import (
    EMBEDDING_MODEL_NAME, 
    MONGODB_URI, 
    MONGODB_DB_NAME, 
    AVAILABLE_MODULES,
    AVAILABLE_COURSES,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    COLLECTION_NAMES,
    COLLECTION_FIELDS
)
import os
from pymongo import MongoClient, ASCENDING, TEXT
import json
import re

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CanvasSearcher:
    """Tool for searching Canvas content"""
    
    def __init__(self):
        """Initialize MongoDB connection and models"""
        try:
            # Initialize MongoDB connection
            self.client = MongoClient(MONGODB_URI)
            self.db = self.client[MONGODB_DB_NAME]
            
            # Verify database connection and collections
            db_names = self.client.list_database_names()
            if MONGODB_DB_NAME not in db_names:
                raise Exception(f"Database {MONGODB_DB_NAME} not found. Available databases: {db_names}")
            
            collection_names = self.db.list_collection_names()
            print(f"Available collections: {collection_names}")
            
            # Initialize embeddings model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize LLM for path decision
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME,
                google_api_key=GEMINI_API_KEY,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            
            # Ensure text indexes exist for all collections
            for collection_name in COLLECTION_NAMES.values():
                if collection_name in self.db.list_collection_names():
                    collection = self.db[collection_name]
                    try:
                        collection.create_index([("$**", TEXT)], background=True)
                        collection.create_index([("course_id", ASCENDING)])
                        print(f"Created indexes for collection: {collection_name}")
                    except Exception as e:
                        print(f"Error creating indexes for {collection_name}: {str(e)}")
            
            print("Successfully connected to MongoDB and initialized models")
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def search(self, query: str) -> Dict[str, Any]:
        """
        Search Canvas content based on query using both text and vector search
        """
        try:
            # Determine which collections to search based on query
            search_path = self._determine_search_path(query)
            if not search_path:
                return {"error": "Could not determine search path", "search_path": None, "results": []}

            results = []
            collections_to_search = [COLLECTION_NAMES[m] for m in search_path.get("modules", [])]
            courses_to_search = search_path.get("courses", [])
            
            # Debug print
            print(f"Searching in collections: {collections_to_search}")
            print(f"Searching for courses: {courses_to_search}")
            
            # Get course IDs if courses are specified
            course_ids = []
            if courses_to_search:
                courses_collection = self.db[COLLECTION_NAMES["courses"]]
                for course_name in courses_to_search:
                    # Try exact match first
                    course_doc = courses_collection.find_one({"course_name": course_name})
                    if course_doc:
                        course_ids.append(course_doc["_id"])
                        print(f"Found exact match for course: {course_name}")
                        continue
                    
                    # If no exact match, try partial match
                    base_name = course_name.split('(')[0].strip()
                    pattern = re.compile(f".*{re.escape(base_name)}.*", re.IGNORECASE)
                    course_docs = courses_collection.find({"course_name": pattern})
                    
                    for doc in course_docs:
                        course_ids.append(doc["_id"])
                        print(f"Found partial match: {doc['course_name']} for {course_name}")
            
            print(f"Found course IDs: {course_ids}")
            
            for collection_name in collections_to_search:
                if collection_name not in self.db.list_collection_names():
                    print(f"Collection {collection_name} not found in database")
                    continue
                    
                collection = self.db[collection_name]
                try:
                    # Build query filter for course_id
                    query_filter = {"course_id": {"$in": course_ids}} if course_ids else {}
                    
                    # Get documents matching course filter
                    matching_docs = list(collection.find(query_filter))
                    print(f"Found {len(matching_docs)} documents in {collection_name} matching course filter")
                    
                    if not matching_docs:
                        print(f"No documents found in {collection_name} for the specified courses")
                        if not course_ids:  # If no course filter, try searching all documents
                            matching_docs = list(collection.find())
                            print(f"Searching all documents in {collection_name}, found {len(matching_docs)}")
                        else:
                            continue
                        
                    # Prepare texts for embedding search based on collection type
                    texts_to_embed = []
                    doc_map = {}  # Map to store original documents
                    
                    for doc in matching_docs:
                        if collection_name == COLLECTION_NAMES["announcements"]:
                            text = f"{doc.get('title', '')} {doc.get('message', '')}"
                            fields_to_keep = COLLECTION_FIELDS["announcements"]
                        elif collection_name == COLLECTION_NAMES["assignments"]:
                            text = f"{doc.get('name', '')} {doc.get('description', '')}"
                            fields_to_keep = COLLECTION_FIELDS["assignments"]
                        elif collection_name == COLLECTION_NAMES["files"]:
                            text = f"{doc.get('filename', '')} {doc.get('folder_path', '')}"
                            fields_to_keep = COLLECTION_FIELDS["files"]
                        else:
                            continue
                            
                        if text.strip():  # Only process non-empty texts
                            texts_to_embed.append(text)
                            # Create a filtered document with only the required fields
                            filtered_doc = {k: doc.get(k) for k in fields_to_keep if k in doc}
                            filtered_doc["source"] = collection_name
                            doc_map[text] = filtered_doc
                    
                    if not texts_to_embed:
                        print(f"No valid texts to embed found in {collection_name}")
                        continue
                        
                    print(f"Processing {len(texts_to_embed)} texts for embedding in {collection_name}")
                    
                    # Generate embeddings for query and texts
                    try:
                        query_embedding = self.embeddings.embed_query(query)
                        doc_embeddings = self.embeddings.embed_documents(texts_to_embed)
                        
                        # Calculate cosine similarity
                        from numpy import dot
                        from numpy.linalg import norm
                        import numpy as np
                        
                        similarities = []
                        for idx, doc_embedding in enumerate(doc_embeddings):
                            # Convert to numpy arrays if they aren't already
                            query_embedding_np = np.array(query_embedding)
                            doc_embedding_np = np.array(doc_embedding)
                            
                            # Calculate cosine similarity
                            similarity = dot(query_embedding_np, doc_embedding_np) / (
                                norm(query_embedding_np) * norm(doc_embedding_np)
                            )
                            similarities.append((similarity, texts_to_embed[idx]))
                        
                        # Sort by similarity and get top 2
                        similarities.sort(reverse=True)
                        top_results = similarities[:2]
                        
                        # Add top results to final results
                        for similarity, text in top_results:
                            if text in doc_map:
                                result_doc = doc_map[text].copy()  # Create a copy to avoid modifying the original
                                result_doc["score"] = float(similarity)
                                result_doc["matched_text"] = text
                                # Convert ObjectId to string
                                if "_id" in result_doc:
                                    result_doc["_id"] = str(result_doc["_id"])
                                if "course_id" in result_doc:
                                    result_doc["course_id"] = str(result_doc["course_id"])
                                results.append(result_doc)
                                print(f"Added result from {collection_name} with score {similarity}")
                                
                    except Exception as e:
                        print(f"Error during embedding search for {collection_name}: {str(e)}")
                        continue
                        
                except Exception as e:
                    print(f"Error processing collection {collection_name}: {str(e)}")
                    continue

            print(f"Final results count: {len(results)}")
            return {
                "error": None,
                "search_path": search_path,
                "results": results  # Already limited to top 2 per collection
            }
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return {"error": str(e), "search_path": None, "results": []}

    def _determine_search_path(self, query: str) -> Dict[str, List[str]]:
        """Determine which modules and courses to search based on the query using LLM"""
        try:
            # Construct prompt for LLM
            prompt = f"""Given a user query about Canvas content, determine which modules and courses to search in.

Available Modules:
{', '.join(AVAILABLE_MODULES)}

Available Courses:
{', '.join(AVAILABLE_COURSES)}

User Query: {query}

Analyze the query and determine:
1. Which modules would contain relevant information? (can be multiple or none)
2. Which specific courses should be searched? (can be multiple or none)

Return your decision as a valid JSON object with this format:
{{
    "modules": ["module1", "module2"],
    "courses": ["course1", "course2"],
    "reasoning": "explanation for your choices"
}}

Only return the JSON object, no other text."""

            # Get LLM response
            response = self.llm.predict(prompt)
            
            # Clean the response to ensure it's valid JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            # Parse JSON response
            try:
                result = json.loads(response)
                
                # Validate modules
                if not isinstance(result.get("modules", []), list):
                    result["modules"] = []
                result["modules"] = [m for m in result["modules"] if m in AVAILABLE_MODULES]
                
                # Validate courses
                if not isinstance(result.get("courses", []), list):
                    result["courses"] = []
                result["courses"] = [c for c in result["courses"] if c in AVAILABLE_COURSES]
                
                # Ensure reasoning exists
                if not isinstance(result.get("reasoning", ""), str):
                    result["reasoning"] = "No reasoning provided"
                
                return result
                
            except json.JSONDecodeError:
                print(f"Error parsing LLM response: {response}")
                # Fallback to default behavior
                return {
                    "modules": AVAILABLE_MODULES.copy(),
                    "courses": [],
                    "reasoning": "Failed to parse LLM response, using all modules as fallback"
                }
                
        except Exception as e:
            print(f"Error in path determination: {str(e)}")
            # Fallback to default behavior
            return {
                "modules": AVAILABLE_MODULES.copy(),
                "courses": [],
                "reasoning": f"Error in LLM path determination: {str(e)}"
            }