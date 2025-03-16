"""
Main entry point for the Canvas QA Agent
"""
from typing import Dict
import json
import os
import google.generativeai as genai
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from config.settings import (
    MONGODB_URI, 
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME
)
from tools.canvas_search import CanvasSearcher
from tools.pdf_analyzer import PDFAnalyzer
from prompts.templates import SEARCH_PATH_TEMPLATE, FINAL_ANSWER_TEMPLATE, REACT_PROMPT
from models.data_models import SearchPath

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CanvasQAAgent:
    """Canvas QA Agent implementation"""
    
    def __init__(self):
        """Initialize the agent"""
        # Initialize MongoDB connection
        self.searcher = CanvasSearcher()
        
        # Initialize Gemini model
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
            
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize tools
        self.tools = [
            Tool(
                name="CanvasSearch",
                func=self.searcher.search,
                description="Search for information in Canvas"
            )
        ]
        
        # Create agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=REACT_PROMPT
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _determine_search_path(self, query: str) -> SearchPath:
        """Determine search path based on query"""
        try:
            # Format the prompt with the query
            prompt = SEARCH_PATH_TEMPLATE.format(query=query)
            
            # Get response from LLM
            response = self.llm.predict(prompt)
            
            # Clean the response to ensure it's valid JSON
            response = response.strip()
            
            # Remove any markdown code block markers
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            # Remove any leading/trailing whitespace and newlines
            response = response.strip()
            
            # Try to find JSON object in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise json.JSONDecodeError("No JSON object found", response, 0)
            
            json_str = response[start_idx:end_idx]
            
            # Parse JSON and create SearchPath object
            data = json.loads(json_str)
            
            # Validate required fields
            if not isinstance(data.get('modules', []), list):
                data['modules'] = []
            if not isinstance(data.get('courses', []), list):
                data['courses'] = []
            if not isinstance(data.get('reasoning', ''), str):
                data['reasoning'] = "No reasoning provided"
            
            return SearchPath(**data)
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing search path: {str(e)}")
            print(f"Raw response: {response}")
            return SearchPath(modules=[], courses=[], reasoning="Parse failed")
    
    def _search_canvas(self, query: str) -> Dict:
        """Tool implementation for Canvas search"""
        try:
            # Use searcher directly with query
            results = self.searcher.search(query)
            return results
        except Exception as e:
            return {
                "error": str(e),
                "search_path": None,
                "results": []
            }
    
    def _analyze_pdf(self, query: str, file_path: str) -> Dict:
        """Tool implementation for PDF analysis"""
        try:
            return self.pdf_analyzer.analyze(file_path, query)
        except Exception as e:
            return {
                "error": str(e),
                "source": file_path,
                "content": [],
                "page_numbers": []
            }
    
    def answer_query(self, query: str) -> str:
        """Process user query and generate response using the agent"""
        try:
            # Use agent_executor to process the query
            response = self.agent_executor.invoke(
                {"input": query},
                {"callbacks": None}
            )
            return response["output"]
        except Exception as e:
            return f"Error processing query: {str(e)}"

def main():
    agent = CanvasQAAgent()
    
    # Interactive testing
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        print("\nProcessing your query...\n")
        answer = agent.answer_query(query)
        print(f"\nAnswer: {answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()