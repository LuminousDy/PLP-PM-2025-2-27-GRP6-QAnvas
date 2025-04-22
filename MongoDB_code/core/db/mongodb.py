from pymongo import MongoClient, errors
import os

# Initialize MongoDB connection
try:
    mongo_client = MongoClient(os.getenv("MONGODB_URI"), serverSelectionTimeoutMS=3000)
    mongo_client.server_info()
    db = mongo_client["canvas_qa_system"]
except errors.ServerSelectionTimeoutError as e:
    print("Unable to connect to MongoDB. The service may not be running.")
    print("Error message:", e)
    db = None
    exit(1)
except Exception as e:
    print("An error occurred while connecting to MongoDB.")
    print("Error message:", e)
    db = None
    exit(1)

# Collections for different data types
course_collection = db["courses"]
file_collection = db["files"]
folder_collection = db["folders"]
assignment_collection = db["assignments"]
announcement_collection = db["announcements"]
query_log_collection = db["query_logs"]

# Create indexes for better query performance
folder_collection.create_index([("course_id", 1), ("full_path", 1)])
file_collection.create_index([("course_id", 1), ("folder_path", 1)])
file_collection.create_index([("course_id", 1), ("canvas_id", 1)]) 