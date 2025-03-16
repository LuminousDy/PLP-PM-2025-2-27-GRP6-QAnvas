import requests
import os
from pathlib import Path
from typing import Dict

class FileStorage:
    def __init__(self, base_path: str = None):
        """
        Initialize file storage system
        base_path: Base storage path, defaults to ../../Database/Original_files directory
        """
        if base_path is None:
            # Get the current file's directory
            current_dir = Path(__file__).resolve().parent
            # Navigate to project root and then to Database/Original_files directory
            base_path = current_dir.parent.parent.parent / "Database" / "Original_files"
            
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