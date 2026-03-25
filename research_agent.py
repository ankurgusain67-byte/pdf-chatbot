import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0)

search_tool = DuckDuckGoSearchRun()

@tool
def calculator(expression: str) -> str:
    '''Useful for maths calculations. Input should be a valid 
    Python math expression like "2 + 2" or "150 * 0.18".'''
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f'Error: {e}'

@tool
def get_current_year(query: str) -> str:
    '''Use this when the user asks about the current year or today's date.'''
    return 'The current year is 2026.'

tools = [calculator, get_current_year]

agent = create_react_agent(llm, tools)

print('Research Agent ready! Type exit to quit.')

while True:
    question = input('You: ').strip()
    if question.lower() == 'exit':
        break
    result = agent.invoke({'messages': [('user', question)]})
    print(f'\nFinal Answer: {result["messages"][-1].content}\n')
    print('-' * 50)