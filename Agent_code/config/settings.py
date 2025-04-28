"""
Configuration settings for the Canvas QA Agent
"""
import os
from typing import List
from pymongo import MongoClient
import argparse
# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/DINGYI_CANVAS_DATA")
MONGODB_DB_NAME = "canvas_qa_system"  # Changed to match your actual database name

# Collection names (all lowercase as per MongoDB best practices)
COLLECTION_NAMES = {
    "courses": "courses",
    "announcements": "announcements",
    "assignments": "assignments",
    "files": "files"
}

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key")

# File paths
PDF_BASE_PATH = "Files_Database"

# Model settings
GEMINI_MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"

# Vector search settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 3

# Available modules (must match collection names)
AVAILABLE_MODULES = [
    "announcements",
    "assignments",
    "files"
]

def get_available_courses() -> List[str]:
    """Dynamically load available courses from MongoDB"""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        db = client[MONGODB_DB_NAME]
        courses_collection = db[COLLECTION_NAMES["courses"]]
        courses = list(courses_collection.find({}, {"course_name": 1, "_id": 0}))
        return [course["course_name"] for course in courses]
    except Exception as e:
        print(f"Error loading courses from database: {str(e)}")
        return ["ISY5004", "MTech Program"]  # Fallback default courses
    finally:
        client.close()

# Available courses (dynamically loaded from database)
AVAILABLE_COURSES = get_available_courses()

# MongoDB collection fields mapping
COLLECTION_FIELDS = {
    "courses": ["_id", "course_name", "course_code"],
    "announcements": ["_id", "course_id", "title", "message", "posted_at", "url"],
    "assignments": ["_id", "course_id", "name", "description", "due_at", "points_possible", "html_url"],
    "files": ["_id", "course_id", "file_name", "folder_path", "size", "url", "created_at", "updated_at", 
              "storage_status", "local_path"]
}

def get_args():
    parser = argparse.ArgumentParser(description="Inference multiple intents")
    # === Output and Logging ===
    parser.add_argument('--max_length', type=int, default=128, help="Maximum length of the generated sequence") 
    parser.add_checkpoints('--model_path', type=str, default="/media/labpc2x2080ti/data/Mohan_Workspace/checkpoint-225000",)
    return parser.parse_args()
