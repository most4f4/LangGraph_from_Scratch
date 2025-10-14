import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

#---------------------------------- Environment Setup -------------------------------------
load_dotenv()

# ------------------------------ Model and Embeddings Setup -------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --------------------------------- Paths and Directories ---------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(current_dir, "chroma_db")
pdf_path = os.path.join(current_dir, "Stock_Market_Performance_2024.pdf")
collection_name = "pdf_collection"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"The file {pdf_path} does not exist. Please provide a valid PDF file.")

# ----------------------------------- Load PDF Document -----------------------------------
pdf_loader = PyPDFLoader(pdf_path)

try:
    documents = pdf_loader.load()
    print(f"Loaded {len(documents)} pages from the PDF.")
except Exception as e:
    raise RuntimeError(f"Failed to load PDF: {e}")

# ----------------------------------- Text Splitting --------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# ------------------------------- Create or Load Vector Store -----------------------------
try:
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created vector store with {len(texts)} documents.")
except Exception as e:
    print(f"Error creating vector store: {e}")
    raise RuntimeError(f"Failed to create vector store: {e}")

# ------------------------------------ Retriever Setup ------------------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieve top 5 similar documents
)

# ------------------------------------ Tool Definition ------------------------------------
@tool
def retrieve(query: str) -> str:
    """This tool retrieves relevant information from the PDF based on the user's query."""
    
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the document."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Result {i+1}:\n{doc.page_content}\n")

    return "\n\n".join(results)


tools = [retrieve]
model = llm.bind_tools(tools)

# -------------------------------- Agent State Definition ---------------------------------
class AgentState(TypedDict):
    """Defines the structure of the agent's state."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# -------------------------------- Control Flow Logic -------------------------------------
def should_continue(state: AgentState) -> bool:
    """Determines whether the agent should continue or end the conversation."""
    last_message = state["messages"][-1]
    # Continue if the last message has tool calls
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0


# -------------------------- System Prompt and Tools Dictionary ---------------------------
system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

# ------------------------------------- Agent Logic ---------------------------------------
def call_llm(state: AgentState) -> AgentState:
    """Calls the LLM with the current state and system prompt."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = model.invoke(messages)
    return {'messages': [message]}

# --------------------------------- Tool Execution Logic ----------------------------------
tools_dict = {our_tool.name: our_tool for our_tool in tools} 

def take_action(state: AgentState) -> AgentState:
    """Executes tool calls from LLM's response"""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"🔧 Calling tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tools_dict:
            print(f"Tool {t['name']} does not exist.")
            result = "Incorrect tool name, Please retry and select a valid tool from list of available tools."

        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))} characters")

        # Append the tool message and the result to the state
        results.append(ToolMessage(content=str(result), name=t['name'], tool_call_id=t['id']))

    print(f"🔧 Finished all tool calls.")
    return {'messages': results}


# ------------------------------------- Build the Graph -----------------------------------
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()

# ------------------------------------- Run the Agent --------------------------------------
def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()