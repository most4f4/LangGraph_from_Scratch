from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    message: List[HumanMessage]

llm = ChatOpenAI(model="gpt-4", temperature=0)

def process_message(state: AgentState) -> None:
    response = llm.invoke(state["message"])
    print("Agent Response:", response.content)

    return state

graph = StateGraph(AgentState)
graph.add_node("process_message", process_message)
graph.add_edge(START, "process_message")
graph.add_edge("process_message", END)

agent = graph.compile()

user_input = input("Enter your message: ")
while user_input.lower() != "exit":
    agent.invoke({"message": [HumanMessage(content=user_input)]})
    user_input = input("Enter your message: ")