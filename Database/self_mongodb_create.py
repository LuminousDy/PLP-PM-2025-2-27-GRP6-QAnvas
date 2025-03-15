import os
from pymongo import MongoClient
import requests
from time import sleep
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

class FileStorage:
    def __init__(self, base_path: str = "./storage"):
        """
        Initialize file storage system
        base_path: Base storage path, defaults to ./storage directory
        """
        self.base_path = Path(base_path)
        self.max_file_size = 100 * 1024 * 1024  # 100MB default limit
        
        # Create storage directory structure
        self.create_storage_structure()
        
    def create_storage_structure(self):
        """Create storage directory structure"""
        # Main storage directory
        self.base_path.mkdir(exist_ok=True)
            
    def get_storage_path(self, course_id: str, folder_path: str, filename: str) -> Path:
        """
        Generate storage path based on course ID and folder path
        """
        # Create course-specific directory
        course_path = self.base_path / str(course_id)
        course_path.mkdir(exist_ok=True)
        
        # Create folder path
        if folder_path:
            full_path = course_path
            for folder in folder_path.strip('/').split('/'):
                full_path = full_path / folder
                full_path.mkdir(exist_ok=True)
        else:
            full_path = course_path
            
        return full_path / filename
    
    def store_file(self, file_url: str, course_id: str, folder_path: str, 
                   filename: str, headers: Dict) -> Dict:
        """
        Download and store file
        """
        try:
            # Get file size
            response = requests.head(file_url, headers=headers)
            file_size = int(response.headers.get('content-length', 0))
            
            if file_size > self.max_file_size:
                return {
                    "status": "size_limit",
                    "file_size": file_size
                }
            
            # Determine storage path
            file_path = self.get_storage_path(course_id, folder_path, filename)
            
            # Download file in chunks
            response = requests.get(file_url, headers=headers, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
            return {
                "status": "success",
                "stored_path": str(file_path),
                "file_size": file_size
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Initialize MongoDB connection
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["canvas_qa_system"]

# Collections for different data types
course_collection = db["courses"]
file_collection = db["files"]
folder_collection = db["folders"]  # New collection for folders
assignment_collection = db["assignments"]
announcement_collection = db["announcements"]
query_log_collection = db["query_logs"]

# Create indexes for better query performance
folder_collection.create_index([("course_id", 1), ("full_path", 1)])
file_collection.create_index([("course_id", 1), ("folder_path", 1)])
file_collection.create_index([("course_id", 1), ("canvas_id", 1)])

# API Configuration
BASE_URL = "https://canvas.nus.edu.sg/api/v1"
HEADERS = {"Authorization": f"Bearer {os.getenv('DINGYI_CANVAS_API_KEY')}"}
PAGE_SIZE = 100
RATE_LIMIT_DELAY = 0.1  # Delay between API calls to avoid rate limiting

def get_paginated_results(url: str) -> List[Dict[Any, Any]] or None:
    """
    Generic function to get paginated results from Canvas API.
    - For announcements (using discussion_topics endpoint with only_announcements param), returns empty list on 404
    - Returns None on 403 permission denied to stop crawling that resource
    
    Args:
        url: Base API endpoint URL
        
    Returns:
        List of results from all pages; or None (indicating no permission)
    """
    results = []
    page = 1
    
    while True:
        separator = "&" if "?" in url else "?"
        paginated_url = f"{url}{separator}page={page}&per_page={PAGE_SIZE}"
        try:
            response = requests.get(paginated_url, headers=HEADERS)
            response.raise_for_status()
            page_results = response.json()
            if not page_results:
                break
                
            results.extend(page_results)
            page += 1
            sleep(RATE_LIMIT_DELAY)
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                status = e.response.status_code
                if status == 404 and "discussion_topics" in url and "only_announcements=true" in url:
                    return []
                if status == 403:
                    print(f"Permission denied for URL: {paginated_url}")
                    return None
            print(f"Error fetching data from {paginated_url}: {str(e)}")
            break
            
    return results

def get_course_data(course_id: int) -> Dict[str, Any] or None:
    """
    Get all relevant data for a specific course.
    If any resource returns no permission (None), stop crawling this course.
    
    Args:
        course_id: Canvas course ID
        
    Returns:
        Dictionary containing course data; or None (indicating no permission)
    """
    endpoints = {
        'details': f"{BASE_URL}/courses/{course_id}",
        'folders': f"{BASE_URL}/courses/{course_id}/folders",  # Add folders endpoint
        'files': f"{BASE_URL}/courses/{course_id}/files",
        'assignments': f"{BASE_URL}/courses/{course_id}/assignments",
        'announcements': f"{BASE_URL}/courses/{course_id}/discussion_topics?only_announcements=true",
        'users': f"{BASE_URL}/courses/{course_id}/users",
        'quizzes': f"{BASE_URL}/courses/{course_id}/quizzes"
    }
    
    course_data = {}
    
    # Get course details
    try:
        response = requests.get(endpoints['details'], headers=HEADERS)
        response.raise_for_status()
        course_data['details'] = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching course details for course {course_id}: {str(e)}")
        return None
        
    # Get folders first to establish structure
    folders_data = get_paginated_results(endpoints['folders'])
    if folders_data is None:
        print(f"Permission denied for folders in course {course_id}. Stopping crawl for this course.")
        return None
    course_data['folders'] = folders_data
        
    # Get other course resources
    for resource, url in endpoints.items():
        if resource not in ['details', 'folders']:
            data = get_paginated_results(url)
            if data is None:
                print(f"Permission denied for resource '{resource}' in course {course_id}. Stopping crawl for this course.")
                return None
                
            # If this is files data, associate with folders
            if resource == 'files':
                for file_item in data:
                    folder_id = file_item.get('folder_id')
                    if folder_id:
                        # Find the corresponding folder in folders_data
                        for folder in folders_data:
                            if folder['id'] == folder_id:
                                file_item['folder'] = folder
                                break
                    
            course_data[resource] = data
            
    return course_data

def process_folder_structure(course_id: Any, folders_data: List[Dict]) -> None:
    """
    Process and store course folder structure
    
    Args:
        course_id: MongoDB course ID
        folders_data: List of folder data from Canvas API
    """
    def process_folder(folder: Dict, parent_path: str = "") -> Dict:
        folder_path = f"{parent_path}/{folder['name']}" if parent_path else folder['name']
        
        folder_doc = {
            "course_id": course_id,
            "canvas_id": folder.get("id"),
            "name": folder.get("name"),
            "full_path": folder_path,
            "parent_folder_id": folder.get("parent_folder_id"),
            "folders_count": len(folder.get("folders", [])),
            "files_count": len(folder.get("files", [])),
            "created_at": folder.get("created_at"),
            "updated_at": folder.get("updated_at"),
            "stored_at": datetime.now()
        }
        
        # Store folder information
        folder_collection.insert_one(folder_doc)
        
        # Process subfolders recursively
        for subfolder in folder.get("folders", []):
            process_folder(subfolder, folder_path)
            
        return folder_doc

    # Process all top-level folders
    for folder in folders_data:
        process_folder(folder)

def store_file(file_item: Dict, course_id: Any) -> None:
    """
    Store file metadata and content with folder information
    
    Args:
        file_item: File item data from Canvas API
        course_id: MongoDB course ID
    """
    # Get file storage instance
    storage = FileStorage()
    
    # Build file path information
    folder_path = file_item.get("folder", {}).get("full_name", "")
    filename = file_item.get("filename", "")
    
    # Basic file information
    file_doc = {
        "course_id": course_id,
        "canvas_id": file_item["id"],
        "filename": filename,
        "display_name": file_item.get("display_name", ""),
        "folder_path": folder_path,
        "folder_id": file_item.get("folder_id"),
        "content_type": file_item.get("content-type", ""),
        "size": file_item.get("size", 0),
        "url": file_item.get("url", ""),
        "created_at": file_item.get("created_at", ""),
        "updated_at": file_item.get("updated_at", ""),
        "stored_at": datetime.now()
    }
    
    # Download and store file if URL exists
    if file_doc["url"]:
        storage_result = storage.store_file(
            file_doc["url"], 
            str(course_id),
            folder_path,
            filename,
            HEADERS
        )
        file_doc["storage_status"] = storage_result["status"]
        if "stored_path" in storage_result:
            file_doc["local_path"] = storage_result["stored_path"]
        if "file_size" in storage_result:
            file_doc["downloaded_size"] = storage_result["file_size"]
    
    # Store in database
    file_collection.insert_one(file_doc)

def store_assignment(assignment_item: Dict, course_id: Any) -> None:
    """
    Store assignment data in MongoDB
    
    Args:
        assignment_item: Assignment item data from Canvas API
        course_id: MongoDB course ID
    """
    assignment_collection.insert_one({
        "course_id": course_id,
        "canvas_id": assignment_item["id"],
        "name": assignment_item.get("name", ""),
        "description": assignment_item.get("description", ""),
        "due_at": assignment_item.get("due_at", ""),
        "points_possible": assignment_item.get("points_possible", 0),
        "submission_types": assignment_item.get("submission_types", []),
        "html_url": assignment_item.get("html_url", ""),
        "stored_at": datetime.now()
    })

def store_announcement(announcement_item: Dict, course_id: Any) -> None:
    """
    Store announcement data in MongoDB
    
    Args:
        announcement_item: Announcement item data from Canvas API
        course_id: MongoDB course ID
    """
    announcement_collection.insert_one({
        "course_id": course_id,
        "canvas_id": announcement_item["id"],
        "title": announcement_item.get("title", ""),
        "message": announcement_item.get("message", ""),
        "posted_at": announcement_item.get("posted_at", ""),
        "url": announcement_item.get("url", ""),
        "stored_at": datetime.now()
    })

def store_course_data(course_data: Dict[str, Any]) -> str:
    """
    Store course metadata and related resources in MongoDB
    
    Args:
        course_data: Dictionary containing course details and resources
        
    Returns:
        course_id: MongoDB ID of stored course
    """
    # Store course details in MongoDB with nested structure
    course_document = {
        "course_name": course_data["details"]["name"],
        "canvas_id": course_data["details"]["id"],
        "course_code": course_data["details"].get("course_code", ""),
        "start_at": course_data["details"].get("start_at", ""),
        "end_at": course_data["details"].get("end_at", ""),
        "resources": {
            "files_count": 0,
            "folders_count": 0,
            "assignments_count": 0,
            "announcements_count": 0
        },
        "stored_at": datetime.now()
    }
    
    course_id = course_collection.insert_one(course_document).inserted_id
    
    # Process and store folder structure first
    if "folders" in course_data:
        process_folder_structure(course_id, course_data["folders"])
        course_document["resources"]["folders_count"] = len(course_data["folders"])
    
    # Store different resource types with appropriate handlers
    resource_handlers = {
        "files": store_file,
        "assignments": store_assignment,
        "announcements": store_announcement
    }
    
    for resource_type, handler in resource_handlers.items():
        if resource_type in course_data:
            items = course_data[resource_type]
            for item in items:
                handler(item, course_id)
            
            # Update resource counts in course document
            course_collection.update_one(
                {"_id": course_id},
                {"$set": {f"resources.{resource_type}_count": len(items)}}
            )
    
    return str(course_id)

def get_all_available_courses() -> List[Dict]:
    """
    Get all courses accessible by the current user
    """
    url = f"{BASE_URL}/courses"
    courses = get_paginated_results(url)
    
    if courses is None:
        print("Failed to get courses. Please check API key and permissions.")
        return []
        
    # Only keep active courses
    active_courses = [
        course for course in courses 
        if course.get('workflow_state') == 'available'
    ]
    
    return active_courses

def main():
    # Get all course IDs from the system
    print("Getting available courses...")
    available_courses = get_all_available_courses()
    all_courses_data = {}

    if not available_courses:
        print("No courses found or unable to access courses.")
    else:
        print(f"Found {len(available_courses)} available courses")
        
        # Get and store data for each course
        for course in available_courses:
            course_id = course['id']
            print(f"\nProcessing course: {course['name']} (ID: {course_id})")
            
            course_data = get_course_data(course_id)
            if course_data:
                all_courses_data[course_id] = course_data
                stored_id = store_course_data(course_data)
                print(f"Course {course_id} stored with database ID: {stored_id}")
            else:
                print(f"Failed to get data for course {course_id}")

    # Print storage statistics
    print(f"\nTotal courses stored: {len(all_courses_data)}")
    for course_id, data in all_courses_data.items():
        print(f"Course {course_id}: {data['details']['name']}")

if __name__ == "__main__":
    main()