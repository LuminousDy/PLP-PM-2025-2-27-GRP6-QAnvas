import os
from pathlib import Path
from core.api.canvas_api import get_all_available_courses, get_course_data
from core.processors.data_processors import store_course_data
from dotenv import load_dotenv
load_dotenv()
def setup_environment():
    """Setup necessary directories and environment variables"""
    # Get the project root directory
    print("Setting up environment...")
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

def main():
    # Setup environment
    setup_environment()
    
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