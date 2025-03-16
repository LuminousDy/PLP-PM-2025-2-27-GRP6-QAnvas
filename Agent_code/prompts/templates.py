"""
Prompt templates for the Canvas QA Agent
"""
from langchain.prompts import PromptTemplate

SEARCH_PATH_TEMPLATE = PromptTemplate(
    input_variables=["query"],
    template="""Analyze the following query to determine search scope:
    Query: {query}
    
    1. Modules to search in (can select multiple):
    - Announcements
    - Assignments
    - Files
    
    2. Relevant courses (can select multiple):
    - ISY5004
    - MTech Program
    [Other courses...]
    
    Please return a valid JSON object in the following format:
    {{
        "modules": ["module1", "module2"],
        "courses": ["course1", "course2"],
        "reasoning": "reason for selection"
    }}
    
    Make sure the response is a valid JSON object with no additional text or formatting.
    """
)

FINAL_ANSWER_TEMPLATE = PromptTemplate(
    input_variables=["query", "results"],
    template="""Based on the following information, answer the user query:
    Query: {query}
    Search Results: {results}
    
    Please provide a complete and accurate answer with source citations.
    """
)

REACT_PROMPT = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    template="""You are a helpful Canvas assistant that helps students find information about their courses.
    
You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""
)
