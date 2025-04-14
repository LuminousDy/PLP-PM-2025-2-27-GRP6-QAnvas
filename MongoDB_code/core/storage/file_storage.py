import requests
import os
from pathlib import Path
from typing import Dict

class FileStorage:
    def __init__(self, base_path: str = None):
        """
        Initialize file storage system
        base_path: Base storage path, defaults to ../../Files_Database directory
        """
        if base_path is None:
            # Get the current file's directory
            current_dir = Path(__file__).resolve().parent
            # Navigate to project root and then to Files_Database directory
            base_path = current_dir.parent.parent.parent / "Files_Database"
            
        self.base_path = Path(base_path)
        self.max_file_size = 300 * 1024 * 1024  # 300MB default limit
        
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
            print(f"Downloading file: {filename} from URL: {file_url}")
            
            # Determine storage path
            file_path = self.get_storage_path(course_id, folder_path, filename)
            print(f"Storage path: {file_path}")
            
            # Create a session for maintaining cookies and auth state
            session = requests.Session()
            
            # If headers are provided, use them
            if headers:
                session.headers.update(headers)
            
            print(f"Starting download from {file_url}")
            # Canvas API authenticated URL automatically handles authentication, no extra headers needed
            response = session.get(file_url, stream=True, allow_redirects=True, verify=True)
            
            if not response.ok:
                print(f"Download failed: Status code {response.status_code}")
                return {
                    "status": "error",
                    "error": f"Download request failed with status code {response.status_code}"
                }
            
            # Get content length if available
            content_length = response.headers.get('content-length')
            file_size = int(content_length) if content_length else 0
            print(f"File size from headers: {file_size} bytes")
            
            if file_size > self.max_file_size and file_size > 0:
                print(f"File too large: {file_size} bytes (max: {self.max_file_size})")
                return {
                    "status": "size_limit",
                    "file_size": file_size
                }
            
            # Write the file in chunks
            with open(file_path, 'wb') as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                print(f"Downloaded {downloaded_size} bytes for file {filename}")
            
            # Verify file was actually downloaded
            actual_size = os.path.getsize(file_path)
            if actual_size == 0:
                print(f"WARNING: Downloaded file is empty: {file_path}")
                return {
                    "status": "error",
                    "error": "Downloaded file is empty"
                }
            
            print(f"Successfully downloaded file to {file_path} ({actual_size} bytes)")
            return {
                "status": "success",
                "stored_path": str(file_path),
                "file_size": actual_size
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Request error downloading {filename}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
        except IOError as e:
            print(f"IO error writing file {filename}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
        except Exception as e:
            print(f"Unexpected error downloading {filename}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 