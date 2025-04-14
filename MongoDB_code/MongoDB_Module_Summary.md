# MongoDB Text Database Module Summary

## 1. Overview

The MongoDB text database module is a data collection and storage solution designed specifically for the Canvas Learning Management System. It aims to retrieve course data from the Canvas platform and store it in a structured manner in a MongoDB database for subsequent querying and data analysis. This module supports storing various course resources, including files, folders, assignments, and announcements.

## 2. Core Features

### 2.1 Data Collection

- Retrieve the list of available courses via the Canvas API
- Perform in-depth crawling for each course to obtain the following information:
  - Course details (name, code, start and end dates, etc.)
  - Folder structure
  - File content and metadata
  - Assignment information
  - Announcement information
  - User information
  - Quiz information

### 2.2 Data Storage

- Store structured data using a MongoDB database
- Store file content in the local file system
- Maintain hierarchical relationships between folders and files
- Record metadata for all resources (creation time, update time, size, etc.)

### 2.3 Data Processing

- Associate files with their respective folders
- Handle nested folder structures
- Download and store file content
- Provide data statistics and summaries

## 3. Technical Architecture

### 3.1 Database Structure

The MongoDB database `canvas_qa_system` contains the following collections:

- `courses`: Stores basic course information
- `folders`: Stores folder structures
- `files`: Stores file metadata and local paths
- `assignments`: Stores assignment information
- `announcements`: Stores announcement information
- `query_logs`: Stores query logs

### 3.2 File Organization

The module includes the following main files and directories:

- `main.py`: Main program entry point, responsible for setting up the environment and invoking related functions
- `core/`: Core functionality directory
  - `api/`: Canvas API interfaces
  - `db/`: MongoDB database connections and collection definitions
  - `processors/`: Data processors
  - `storage/`: File storage management

### 3.3 API Interfaces

Data is retrieved via the Canvas API, with main endpoints including:

- `/courses`: Retrieve all available courses
- `/courses/{id}`: Retrieve course details
- `/courses/{id}/folders`: Retrieve course folders
- `/courses/{id}/files`: Retrieve course files
- `/courses/{id}/assignments`: Retrieve course assignments
- `/courses/{id}/discussion_topics?only_announcements=true`: Retrieve course announcements

## 4. Key Implementation Details

### 4.1 Pagination Handling

Handle paginated results from the Canvas API: