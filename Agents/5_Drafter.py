from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

load_dotenv()

# This is a global variable storing document content
document_content = ""

class AgentState(TypedDict):
    """Defines the structure of the agent's state."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """This tool updates the document with new content."""
    global document_content # declare the variable as global to modify it
    document_content = content
    return f"Document successfully updated. Current content is: {document_content}"

@tool
def save(filename: str) -> str:
    """This tool saves the current document content to a file.
    
    Args:
        filename (str): The name of the text file to save the content to."""
    global document_content

    if not filename.endswith(".txt"):
       filename += ".txt"
    
    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f"\n 💾 Document successfully saved to {filename}.")
        return f"Document successfully saved to {filename}."
    except Exception as e:
        return f"Error saving document: {e}"

tools = [update, save]

model = ChatOpenAI(model="gpt-4").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
        You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
        
        - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
        - If the user wants to save and finish, you need to use the 'save' tool.
        - Make sure to always show the current document state after modifications.
        
        The current document content is:{document_content}
        """)
    
    # If there are no previous messages, start the conversation with a greeting.
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    # Otherwise, get the user's input from the last message.
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    # Combine system prompt, previous messages, and the new user message.
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    # Get the model's response.
    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    # Add the new messages to the state and return it.
    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determines whether the agent should continue or end the conversation."""

    messages = state["messages"]

    # If there are no messages, continue
    if not messages:
        return "continue"

    # Scan messages in reverse (latest first)
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
        return "continue"


def print_messages(messages):
    """Function to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")


graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()