import os
import requests
import logging
from time import sleep
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    filename='canvas_api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# API Configuration
BASE_URL = os.getenv('CANVAS_API_URL')
HEADERS = {"Authorization": f"Bearer {os.getenv('CANVAS_API_TOKEN')}"}
if not BASE_URL or not HEADERS["Authorization"]:
    raise EnvironmentError(
        "CANVAS_API_URL or CANVAS_API_TOKEN environment variable not set. "
        "Please set them before running the script."
    )
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
                    logging.info(f"Permission denied for URL: {paginated_url}")
                    return None
            logging.error(f"Error fetching data from {paginated_url}: {str(e)}")
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
        logging.error(f"Error fetching course details for course {course_id}: {str(e)}")
        return None
        
    # Get folders first to establish structure
    folders_data = get_paginated_results(endpoints['folders'])
    if folders_data is None:
        logging.info(f"Permission denied for folders in course {course_id}. Stopping crawl for this course.")
        return None
    course_data['folders'] = folders_data
        
    # Get other course resources
    for resource, url in endpoints.items():
        if resource not in ['details', 'folders']:
            data = get_paginated_results(url)
            if data is None:
                logging.info(f"Permission denied for resource '{resource}' in course {course_id}. Stopping crawl for this course.")
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
                    
                    # Ensure each file record contains the course ID for later download use
                    file_item['course_id'] = course_id
                    
                    # Get download URL for each file
                    try:
                        logging.info(f"Getting download URL for file: {file_item.get('display_name', 'Unknown')} (ID: {file_item.get('id')})")
                        # Even if a URL already exists, fetch again to ensure complete download information
                        file_id = file_item.get('id')
                        if file_id:
                            # Get file details, including download link
                            file_url = f"{BASE_URL}/courses/{course_id}/files/{file_id}"
                            logging.info(f"Requesting file details from: {file_url}")
                            # Using Session for better connection handling
                            session = requests.Session()
                            session.headers.update(HEADERS)
                            file_response = session.get(file_url)
                            
                            if file_response.ok:
                                file_details = file_response.json()
                                
                                # Store all available URLs for this file
                                if 'url' in file_details:
                                    file_item['url'] = file_details['url']
                                    logging.info(f"Got URL: {file_details['url']}")
                                if 'download_url' in file_details:
                                    file_item['download_url'] = file_details['download_url']
                                    logging.info(f"Got download_url: {file_details['download_url']}")
                                # Check for additional Canvas API specific fields
                                for key in ['preview_url', 'canvadoc_url']:
                                    if key in file_details:
                                        file_item[key] = file_details[key]
                                
                                # If we still don't have a download URL, construct a direct one
                                if not file_item.get('download_url') and not file_item.get('url'):
                                    # Try to construct a direct download URL using the file ID
                                    direct_download_url = f"{BASE_URL}/courses/{course_id}/files/{file_id}/download"
                                    logging.info(f"Constructed direct download URL: {direct_download_url}")
                                    file_item['constructed_download_url'] = direct_download_url
                                    
                                logging.info(f"Available URLs for {file_item.get('display_name', 'Unknown')}: {[k for k in file_item.keys() if 'url' in k]}")
                            else:
                                logging.info(f"Failed to get file details. Status: {file_response.status_code}, Response: {file_response.text[:200]}")
                            sleep(RATE_LIMIT_DELAY)  # Respect API rate limit
                    except Exception as e:
                        logging.error(f"Error getting file details for file {file_item.get('id')}: {str(e)}")
                    
            course_data[resource] = data
            
    return course_data

def get_all_available_courses() -> List[Dict]:
    """
    Get all courses accessible by the current user
    """
    url = f"{BASE_URL}/courses"
    courses = get_paginated_results(url)
    
    if courses is None:
        logging.info("Failed to get courses. Please check API key and permissions.")
        return []
        
    # Only keep active courses
    active_courses = [
        course for course in courses 
        if course.get('workflow_state') == 'available'
    ]
    
    return active_courses 

def get_file_download_url(file_id: int, course_id: int) -> str or None:
    """
    Get the authenticated download URL for a file
    This is the authenticated URL provided by Canvas API
    """
    try:
        # Use the Canvas API's dedicated file download interface
        download_url = f"{BASE_URL}/courses/{course_id}/files/{file_id}?include[]=download_url"
        logging.info(f"Getting authenticated download URL for file ID {file_id}")
        
        response = requests.get(download_url, headers=HEADERS)
        if not response.ok:
            logging.info(f"Failed to get download URL. Status: {response.status_code}")
            return None
            
        file_data = response.json()
        
        # Get the authenticated download link from Canvas API response
        if "url" in file_data:
            # This URL already contains authentication information
            authenticated_url = file_data.get("url")
            logging.info(f"Got authenticated URL: {authenticated_url[:100]}...")
            return authenticated_url
        else:
            logging.info("No URL found in file data")
            return None
    except Exception as e:
        logging.error(f"Error getting download URL: {str(e)}")
        return None 