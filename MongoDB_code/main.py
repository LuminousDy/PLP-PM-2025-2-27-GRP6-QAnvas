import os, logging
from pathlib import Path
from .core.api.canvas_api import get_all_available_courses, get_course_data
from .core.data_processors import store_course_data
from .core.mongodb import meta_col
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

load_dotenv()
def setup_environment():
    """Setup necessary directories and environment variables"""
    # Get the project root directory
    logging.info("Setting up environment.")
    project_root = Path(__file__).resolve().parent.parent
    
    # Ensure Database directory exists
    database_dir = project_root / "Files_Database"
    database_dir.mkdir(exist_ok=True)
    
    # Check for API key
    if not os.getenv('CANVAS_API_TOKEN'):
        raise EnvironmentError(
            "CANVAS_API_TOKEN environment variable not set. "
            "Please set it before running the script."
        )

def get_last_updated_time():
    doc = meta_col.find_one({"type": "update_status"})
    return doc.get("last_updated") if doc else None

def update_last_updated_time():
    meta_col.update_one(
        {"type": "update_status"},
        {"$set": {"last_updated": datetime.utcnow()}},
        upsert=True
    )
def main():
    # Setup environment
    setup_environment()
    
    # Get all course IDs from the system
    print("Fetching... This may take a while.")
    available_courses = get_all_available_courses()
    all_courses_data = {}

    if not available_courses:
        print("No courses found or unable to access courses.")
    else:
        print(f"Found {len(available_courses)} available courses.")
        
        # Get and store data for each course
        for course in tqdm(available_courses, desc="Progress", unit="course"):
            course_id = course['id']
            logging.info(f"\nProcessing course: {course['name']} (ID: {course_id})")
            # Get course data
            course_data = get_course_data(course_id)
            if course_data:
                all_courses_data[course_id] = course_data
                db_id = store_course_data(course_data)
                logging.info(f"Course {course_id} stored with database ID: {db_id}")
            else:
                logging.info(f"Failed to get data for course {course_id}")

    # Print storage statistics
    print(f"\nTotal courses stored: {len(all_courses_data)}")
    for course_id, data in all_courses_data.items():
        print(f"Course {course_id}: {data['details']['name']}")
    update_last_updated_time()

if __name__ == "__main__":
    main()