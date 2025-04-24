import logging
from datetime import datetime
from typing import List, Dict, Any
from core.storage.file_storage import FileStorage
from core.db.mongodb import (
    folder_collection, file_collection,
    assignment_collection, announcement_collection,
    course_collection, quiz_collection,
)
from core.api.canvas_api import HEADERS, get_file_download_url

# Configure logging
logging.basicConfig(
    filename='data_processors.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
def is_course_in_db(course_id: int) -> bool:
    return bool(course_collection.find_one({"course_id": course_id}))

def add_folder_to_db(db_id: Any, folders_data: List[Dict]) -> None:
    """
    Process and store course folder structure
    """
    def process_folder(folder: Dict, parent_path: str = "") -> Dict:
        folder_path = f"{parent_path}/{folder['name']}" if parent_path else folder['name']
        
        existing = folder_collection.find_one({
            "course_id": folder.get("course_id"),
            "canvas_id": folder.get("id"),
            "full_path": folder_path,
        })
        update = False
        if existing:
            if folder.get("updated_at") == existing.get("updated_at"):
                # Skip if the folder has not been updated
                logging.info(f"Skip folder: {folder_path}")
                return False
            else:
                update = True
        
        folder_doc = {
            # "db_id": db_id,
            "canvas_id": folder.get("id"),
            "course_id": folder.get("course_id"),
            "name": folder.get("name"),
            "full_path": folder_path,
            "parent_folder_id": folder.get("parent_folder_id"),
            "folders_count": len(folder.get("folders", [])),
            "files_count": len(folder.get("files", [])),
            "created_at": folder.get("created_at"),
            "updated_at": folder.get("updated_at"),
            "stored_at": datetime.now(),
        }
        
        if update:
            folder_collection.update_one(
                {"_id": existing["_id"]},
                {"$set": folder_doc}
            )
            logging.info(f"Updated folder: {folder_path}")
        else:
            folder_collection.insert_one(folder_doc)
            logging.info(f"Stored folder: {folder_path}")
        return True

    # Process all top-level folders
    update_status = False
    for folder in folders_data:
        update_status = process_folder(folder) or update_status

    return update_status

def store_file_and_add_to_db(file_item: Dict, db_id: Any) -> None:
    """
    Store file metadata and content with folder information
    """
    course_id = file_item.get("course_id")
    canvas_id = file_item.get("id")
    update = False
    existing = file_collection.find_one({
        "course_id": course_id,
        "canvas_id": canvas_id,
    })
    if existing:
        if file_item.get("updated_at") == existing.get("updated_at"):
            logging.info(f"Skip file: {file_item['filename']}")
            return False
        else:
            update = True

    # Get file storage instance
    storage = FileStorage()
    folder_path = file_item.get("folder", {}).get("full_name", "")
    filename = file_item.get("filename", "")
    logging.info(f"Processing file: {filename} (ID: {canvas_id})")
    
    # Basic file information
    file_doc = {
        # "db_id": db_id,
        "canvas_id": canvas_id,
        "course_id": course_id,
        "filename": filename,
        "display_name": file_item.get("display_name", ""),
        "folder_path": folder_path,
        "folder_id": file_item.get("folder_id"),
        "content_type": file_item.get("content-type", ""),
        "size": file_item.get("size", 0),
        "url": file_item.get("url", ""),
        "created_at": file_item.get("created_at", ""),
        "updated_at": file_item.get("updated_at", ""),
        "stored_at": datetime.now(),
    }
    
    # Get authenticated download URL through Canvas API
    if canvas_id and course_id:
        # Get the authenticated file download URL from Canvas API
        auth_download_url = get_file_download_url(canvas_id, course_id)
        
        if auth_download_url:
            logging.info(f"Got authenticated download URL for {filename}")
            
            # Download the file using the authenticated URL
            storage_result = storage.store_file(
                auth_download_url,
                str(db_id),
                folder_path,
                filename,
                {} # This URL already contains authentication, no additional headers needed
            )
            
            file_doc["storage_status"] = storage_result["status"]
            file_doc["used_auth_url"] = True
            
            if storage_result["status"] == "success":
                logging.info(f"Successfully downloaded {filename} using authenticated URL")
                if "stored_path" in storage_result:
                    file_doc["local_path"] = storage_result["stored_path"]
                if "file_size" in storage_result:
                    file_doc["downloaded_size"] = storage_result["file_size"]
            else:
                logging.info(f"Failed to download using authenticated URL: {storage_result.get('error', 'Unknown error')}")
                if "error" in storage_result:
                    file_doc["storage_error"] = storage_result["error"]
        else:
            logging.info(f"Could not get authenticated download URL for file {filename}")
            file_doc["storage_status"] = "error"
            file_doc["storage_error"] = "Failed to get authenticated download URL"
    else:
        logging.info(f"Missing Canvas file ID or course ID for {filename}")
        file_doc["storage_status"] = "error"
        file_doc["storage_error"] = "Missing Canvas file ID or course ID"
    

    if update:
        file_collection.update_one(
            {"_id": existing["_id"]},
            {"$set": file_doc}
        )
        logging.info(f"Updated file: {filename} (ID: {canvas_id})")
    else:
        file_collection.insert_one(file_doc)
        logging.info(f"Stored file: {filename} (ID: {canvas_id})")

    return True

def add_assignment_to_db(assignment_item: Dict, db_id: Any) -> bool:
    """
    Store assignment data in MongoDB
    """
    existing = assignment_collection.find_one({
        "course_id": assignment_item.get("course_id"),
        "canvas_id": assignment_item["id"]
    })
    update = False
    if existing:
        if assignment_item.get("updated_at") == existing.get("updated_at"):
            logging.info(f"Skip assignment: {assignment_item['name']}")
            return False
        else:
            update = True

    assignment_doc = {
        # "db_id": db_id,
        "course_id": assignment_item.get("course_id"),
        "canvas_id": assignment_item["id"],
        "name": assignment_item.get("name", ""),
        "description": assignment_item.get("description", ""),
        "due_at": assignment_item.get("due_at", ""),
        "points_possible": assignment_item.get("points_possible", 0),
        "submission_types": assignment_item.get("submission_types", []),
        "html_url": assignment_item.get("html_url", ""),
        "updated_at": assignment_item.get("updated_at", ""),
        "stored_at": datetime.now(),
    }
    if update == True:
        assignment_collection.update_one(
            {"_id": existing["_id"]},
            {"$set": assignment_doc}
        )
        logging.info(f"Updated assignment: {assignment_item['name']}")
    else:
        assignment_collection.insert_one(assignment_doc)
        logging.info(f"Stored assignment: {assignment_item['name']}")
    
    return True

def add_announcement_to_db(announcement_item: Dict, db_id: Any) -> None:
    """
    Store announcement data in MongoDB
    """
    existing = announcement_collection.find_one({
        "course_id": announcement_item.get("course_id"),
        "canvas_id": announcement_item["id"]
    })
    if existing:
        logging.info(f"Skip announcement: {announcement_item['title']}")
        return False

    announcement_collection.insert_one({
        # "db_id": db_id,
        "canvas_id": announcement_item["id"],
        "course_id": announcement_item.get("course_id"),
        "title": announcement_item.get("title", ""),
        "message": announcement_item.get("message", ""),
        "posted_at": announcement_item.get("posted_at", ""),
        "url": announcement_item.get("url", ""),
        "stored_at": datetime.now(),
    })
    logging.info(f"Stored announcement: {announcement_item['title']}")
    return True

def add_quiz_to_db(quiz_item: Dict, db_id: Any) -> None:
    """
    Store quiz data in MongoDB
    """
    existing = quiz_collection.find_one({
        "course_id": quiz_item.get("course_id"),
        "canvas_id": quiz_item["id"]
    })
    if existing:
        logging.info(f"Skip quiz: {quiz_item['title']}")  
        return False
    
    quiz_collection.insert_one({
        # "db_id": db_id,
        "course_id": quiz_item.get("course_id"),
        "canvas_id": quiz_item["id"],
        "title": quiz_item.get("title", ""),
        "description": quiz_item.get("description", ""),
        "due_at": quiz_item.get("due_at", ""),
        "html_url": quiz_item.get("html_url", ""),
        "stored_at": datetime.now(),
    })
    logging.info(f"Stored quiz: {quiz_item['title']}")
    return True

def store_course_data(course_data: Dict[str, Any]) -> str:
    """
    Store course metadata and related resources in MongoDB
    """
    course_id = course_data["details"]["id"]
    course_name = course_data["details"]["name"]

    # Store course details in MongoDB with nested structure
    course_document = {
        "course_name": course_name,
        "course_id": course_id,
        "course_code": course_data["details"].get("course_code", ""),
        "start_at": course_data["details"].get("start_at", ""),
        "end_at": course_data["details"].get("end_at", ""),
        "resources": {
            "files_count": 0,
            "folders_count": 0,
            "assignments_count": 0,
            "announcements_count": 0
        },
        # "updated_at": datetime.now(),
        "status": "in_progress", # done, in_progress, error
    }
    
    # Check if course already exists in the database
    if is_course_in_db(course_id):
        logging.info(f"Course {course_data['details']['id']} already exists. Updating...")
        # print(f"Course {course_data['details']['id']} already exists. Updating...")
        db_id = course_collection.find_one({"course_id": course_id})["_id"]
        # course_collection.update_one({"course_id": course_data["details"]["id"]}, {"$set": course_document})
    else:
        logging.info(f"Storing new course {course_data['details']['id']}...")
        # print(f"Storing new course {course_data['details']['id']}...")
        db_id = course_collection.insert_one(course_document).inserted_id
    
    # Process and store folder structure first
    if "folders" in course_data:
        add_folder_to_db(db_id, course_data["folders"])
        course_document["resources"]["folders_count"] = len(course_data["folders"])
    
    # Store different resource types with appropriate handlers
    resource_handlers = {
        "assignments": add_assignment_to_db,
        "announcements": add_announcement_to_db,
        "quizzes": add_quiz_to_db,
        "files": store_file_and_add_to_db,
    }
    
    for resource_type, handler in resource_handlers.items():
        if resource_type in course_data:
            items = course_data[resource_type]
            for item in items:
                handler(item, db_id)
            
            # Update resource counts in course document
            course_collection.update_one(
                {"_id": db_id},
                {"$set": {f"resources.{resource_type}_count": len(items)}}
            )
    
    # Update course status to "done"
    course_collection.update_one(
        {"_id": db_id},
        {"$set": {"status": "done"}}
    )
    return str(db_id)