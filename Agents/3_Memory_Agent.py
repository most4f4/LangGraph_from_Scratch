from dotenv import load_dotenv
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Define the state structure for the agent with memory
class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]

# Initialize the language model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define the processing function for the agent with memory
def process_message(state: AgentState) -> AgentState:
    """Process the incoming message and update the state with the response."""
    response = llm.invoke(state["message"])
    print("Agent Response:", response.content)
    
    # Append the AI's response to the message history
    state["message"].append(AIMessage(content=response.content))

    # debugging print to show memory state
    print("Current Conversation History:")
    for message in state["message"]:
        print(f" - {message.content}")

    return state

# Build the state graph for the agent with memory
graph = StateGraph(AgentState)
graph.add_node("process_message", process_message)
graph.add_edge(START, "process_message")
graph.add_edge("process_message", END)
agent = graph.compile()

def load_conversation_from_file(filename: str) -> List[Union[HumanMessage, AIMessage]]:
    """Load conversation history from a file."""
    history = []
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("You: "):
                    content = line.replace("You: ", "")
                    history.append(HumanMessage(content=content))
                elif line.startswith("AI: "):
                    content = line.replace("AI: ", "")
                    history.append(AIMessage(content=content))
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error loading conversation history: {e}")
    return history

# Initialize conversation history to maintain context
conversation_history = load_conversation_from_file("logging.txt")

# Example usage of the agent with memory
user_input = input("Enter your message: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"message": conversation_history})

    # Update the conversation history with the latest state, because it includes the AI response
    conversation_history = result["message"]

    user_input = input("Enter your message: ")


# Write the conversation history to a file
with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")


# -----------------------------------------------------------------------
# Note on state and memory
# -----------------------------------------------------------------------
# When you call: result = agent.invoke({"message": conversation_history})
# You’re actually passing a reference (not a copy) of the conversation_history list into the state dictionary.
# That means: state["message"] → conversation_history
# What happens next inside the agent

# Inside the process_message() function:

# response = llm.invoke(state["message"])
# state["message"].append(AIMessage(content=response.content))

# When it appends the AI message, it’s appending to that same list that was passed in — which is the same list as conversation_history.
# So even though you didn’t explicitly append to conversation_history inside this function, 
# it’s still being modified because both state["message"] and conversation_history are references to the same list.