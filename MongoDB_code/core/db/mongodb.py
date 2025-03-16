from pymongo import MongoClient

# Initialize MongoDB connection
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["canvas_qa_system"]

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