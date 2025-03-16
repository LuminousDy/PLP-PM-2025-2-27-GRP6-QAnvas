from datetime import datetime
from typing import List, Dict, Any
from core.storage.file_storage import FileStorage
from core.db.mongodb import (
    folder_collection, file_collection, 
    assignment_collection, announcement_collection,
    course_collection
)
from core.api.canvas_api import HEADERS

def process_folder_structure(course_id: Any, folders_data: List[Dict]) -> None:
    """
    Process and store course folder structure
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