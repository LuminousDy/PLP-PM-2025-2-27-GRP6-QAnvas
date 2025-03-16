import os
import requests
from time import sleep
from typing import List, Dict, Any

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
    """
    endpoints = {
        'details': f"{BASE_URL}/courses/{course_id}",
        'folders': f"{BASE_URL}/courses/{course_id}/folders",
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