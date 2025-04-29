"""
Main entry point for the Canvas QA Agent
"""
from typing import Dict, Optional, List, Any
import json
import os
import uuid
import pickle
from datetime import datetime
import google.generativeai as genai
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from .tools.multi_intents_decomposition import prompt_analyze

from .config.settings import (
    MONGODB_URI, 
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME
)
from .tools.canvas_search import CanvasSearcher
from .tools.pdf_analyzer import PDFAnalyzer
from .prompts.templates import SEARCH_PATH_TEMPLATE, FINAL_ANSWER_TEMPLATE, REACT_PROMPT, CONTEXT_ENHANCED_PROMPT
from .models.data_models import SearchPath, Conversation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Directory for storing conversation history
CONVERSATION_DIR = os.path.join(os.path.dirname(__file__), "conversations")
os.makedirs(CONVERSATION_DIR, exist_ok=True)
model = AutoModelForSeq2SeqLM.from_pretrained("path_to_your_model")
tokenizer = AutoTokenizer.from_pretrained("path_to_your_model")

class ConversationManager:
    """Manager for conversation history"""
    
    def __init__(self):
        """Initialize conversation manager"""
        self.active_conversations = {}
        self.load_saved_conversations()
    
    def load_saved_conversations(self):
        """Load saved conversations from disk"""
        try:
            for filename in os.listdir(CONVERSATION_DIR):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(CONVERSATION_DIR, filename)
                    with open(file_path, 'rb') as f:
                        conversation = pickle.load(f)
                        conversation_id = str(conversation.id)
                        self.active_conversations[conversation_id] = conversation
            print(f"Loaded {len(self.active_conversations)} saved conversations")
        except Exception as e:
            print(f"Error loading saved conversations: {str(e)}")
    
    def get_conversation(self, conversation_id: Optional[str] = None) -> Conversation:
        """Get or create a conversation"""
        if conversation_id and conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        
        # Create new conversation
        new_conversation = Conversation()
        new_id = str(new_conversation.id)
        self.active_conversations[new_id] = new_conversation
        return new_conversation
    
    def save_conversation(self, conversation: Conversation):
        """Save conversation to disk"""
        try:
            conversation_id = str(conversation.id)
            file_path = os.path.join(CONVERSATION_DIR, f"{conversation_id}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(conversation, f)
        except Exception as e:
            print(f"Error saving conversation: {str(e)}")
    
    def add_message(self, conversation_id: str, role: str, content: str):
        """Add message to conversation and save"""
        conversation = self.get_conversation(conversation_id)
        conversation.add_message(role, content)
        self.save_conversation(conversation)

class CanvasQAAgent:
    """Canvas QA Agent implementation"""
    
    def __init__(self):
        """Initialize the agent"""
        # Initialize MongoDB connection
        self.searcher = CanvasSearcher()
        self.pdf_analyzer = PDFAnalyzer()
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager()
        
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
                func=self._search_canvas,
                description="Search for information in Canvas courses, including announcements, assignments, and files"
            ),
            Tool(
                name="PDFAnalyzer",
                func=self._analyze_pdf_wrapper,
                description="Analyze PDF documents to extract relevant information. Format: 'file_path::query'"
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
    
    def _determine_search_path(self, query: str, conversation_history: Optional[str] = None) -> SearchPath:
        """Determine search path based on query and conversation history"""
        try:
            # Format the prompt with the query and conversation history
            if conversation_history:
                context_prompt = CONTEXT_ENHANCED_PROMPT.format(
                    query=query,
                    conversation_history=conversation_history
                )
            else:
                context_prompt = SEARCH_PATH_TEMPLATE.format(query=query)
            
            # Get response from LLM
            response = self.llm.predict(context_prompt)
            
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
    
    def _analyze_pdf_wrapper(self, input_str: str) -> Dict:
        """Wrapper for PDF analysis tool that handles parsing the input string"""
        try:
            # Check if input has the expected format "file_path::query"
            if "::" in input_str:
                file_path, query = input_str.split("::", 1)
                file_path = file_path.strip()
                query = query.strip()
            else:
                # If no separator, assume it's a file path and use a generic query
                file_path = input_str.strip()
                query = "Extract the key information from this document"
            
            return self._analyze_pdf(query, file_path)
        except Exception as e:
            return {
                "error": f"Error parsing PDF analyzer input: {str(e)}. Expected format: 'file_path::query'",
                "source": input_str,
                "content": [],
                "page_numbers": []
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
    
    def answer_query(self, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Process user query with conversation context and generate response using the agent"""
        try:
            # Get or create conversation
            conversation = self.conversation_manager.get_conversation(conversation_id)
            conversation_id = str(conversation.id)
            
            # Add user query to conversation
            self.conversation_manager.add_message(conversation_id, "user", query)
            
            # Get conversation history for context
            conversation_history = conversation.get_context_for_query()
            
            # Override agent memory with conversation history
            self.memory.clear()
            for msg in conversation.messages[:-1]:  # All messages except the current query
                if msg.role == "user":
                    self.memory.chat_memory.add_user_message(msg.content)
                else:
                    self.memory.chat_memory.add_ai_message(msg.content)
            
            # Use agent_executor to process the query with context
            response = self.agent_executor.invoke(
                {"input": query},
                {"callbacks": None}
            )
            
            answer = response["output"]
            
            # Add assistant response to conversation
            self.conversation_manager.add_message(conversation_id, "assistant", answer)
            
            # Return response with conversation ID
            return {
                "answer": answer,
                "conversation_id": conversation_id
            }
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            # If we have a conversation ID, record the error
            if conversation_id:
                self.conversation_manager.add_message(conversation_id, "assistant", error_msg)
            return {
                "answer": error_msg,
                "conversation_id": conversation_id
            }
            
def main():
    agent = CanvasQAAgent()
    
    # Interactive testing
    print("\nCanvas QA Agent started. Type 'quit' to exit.")
    print("For a new conversation, just start asking questions.")
    print("To continue a conversation, enter: 'session:your_session_id'\n")
    
    current_conversation_id = None
    
    while True:
        # Show current session ID if we have one
        if current_conversation_id:
            prompt = f"\n[Session: {current_conversation_id}] Enter your question (or 'quit' to exit): "
        else:
            prompt = "\nEnter your question (or 'quit' to exit): "
            
        query = input(prompt)
        sub_querys_stores=prompt_analyze(query, model,tokenizer)
        for i, sub_query in enumerate(sub_querys_stores['sub_queries']):
            # Check for exit command
            query = sub_query['text']
            if query.lower() == 'quit':
                break
                
            # Check for session command
            if query.lower().startswith('session:'):
                parts = query.split(':', 1)
                if len(parts) > 1 and parts[1].strip():
                    current_conversation_id = parts[1].strip()
                    print(f"Switched to session: {current_conversation_id}")
                else:
                    print("Invalid session ID format. Please use 'session:your_session_id'")
                continue
            
            # Check for new session command
            if query.lower() == 'new':
                current_conversation_id = None
                print("Started a new conversation session")
                continue
                
            print("\nProcessing your query...\n")
            
            # Process the query
            result = agent.answer_query(query, current_conversation_id)
            
            # Update conversation ID if this was a new conversation
            if not current_conversation_id:
                current_conversation_id = result["conversation_id"]
                print(f"\nNew conversation started with ID: {current_conversation_id}")
            
            # Display the answer
            print(f"\nAnswer: {result['answer']}\n")
            print("-" * 50)

if __name__ == "__main__":
    main()