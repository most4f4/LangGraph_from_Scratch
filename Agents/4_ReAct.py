from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict # For type annotations and structured data types.
from langchain_core.messages import BaseMessage # The foundation of all messages in LangChain.
from langchain_core.messages import ToolMessage # Passes data back from tools to the LLM agent after tool execution.
from langchain_core.messages import SystemMessage # Used to provide context or instructions to the LLM.
from langchain_openai import ChatOpenAI 
from langchain_core.tools import tool # Decorator to define tools for the agent.
from langgraph.graph.message import add_messages # Reducer function to add messages to the state.
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode # Prebuilt node for tool execution.

# Load environment variables from a .env file.
load_dotenv()

# Define the state structure for the ReAct agent.
class AgentState(TypedDict):
    """Defines the structure of the agent's state."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define tools that adds two integers.
@tool
def add(a: int, b: int) -> int:
    """This is a tool that adds two integers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """This is a tool that multiplies two integers."""
    return a * b

@tool
def subtract(a: int, b: int) -> int:
    """This is a tool that subtracts two integers."""
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """This is a tool that divides two integers."""
    if b == 0:
        return float('inf')  # Handle division by zero
    return a / b


# Create a list of available tools for the agent.
tools = [add, multiply, subtract, divide]

# Initialize the language model and bind the tools to it.
model = ChatOpenAI(model="gpt-4").bind_tools(tools)

# Define the processing function for the ReAct agent.
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# define the function to determine if the agent should continue or end.
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    # check if the last message contains any tool calls
    if not last_message.tool_calls: 
        return "end"
    return "continue"

# Build the state graph for the ReAct agent.
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)

graph.add_edge("tools", "our_agent")
app = graph.compile()

# define a helper function to print the stream of messages
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message) # if it’s a raw tuple like ("user", "text")
        else:
            message.pretty_print()  # if it’s an object like ToolMessage or AIMessage

# Example usage of the ReAct agent
inputs = {
    "messages": [("user", "Add 30 + 12, then multiply the result by 6. Also tell me a joke.")],
}

# Stream the agent's response and print it
print_stream(app.stream(inputs, stream_mode="values")) # stream_mode can be "keys", "values", or "items"

