# LangGraph from Scratch

A comprehensive repository demonstrating LangGraph concepts from basic graphs to advanced agents with memory, tools, and RAG capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Concepts Covered](#concepts-covered)
- [Agents](#agents)
- [Graphs](#graphs)
- [Getting Started](#getting-started)
- [Documentation](#documentation)

---

## 🎯 Overview

This repository provides hands-on examples of building stateful, multi-step applications with LangGraph. From simple greeting bots to complex RAG agents, each example builds upon previous concepts to demonstrate the full power of LangGraph.

**What is LangGraph?**
LangGraph is a framework for building stateful, multi-step applications with LLMs using a graph-based approach where:

- **Nodes** = Functions that process data
- **Edges** = Paths between nodes
- **State** = Shared data passed between nodes

---

## 🛠️ Installation

### Prerequisites

- Python 3.13+
- OpenAI API Key

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/most4f4/LangGraph_from_Scratch.git
cd LangGraph_from_Scratch
```

2. **Create virtual environment**

```bash
python -m venv langchain
source langchain/bin/activate  # On Windows: langchain\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
   Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## 📁 Repository Structure

```
LangGraph_from_Scratch/
├── Agents/
│   ├── 1_Agent_Bot.py                 # Basic agent
│   ├── 2_Memory_Agent.py              # Agent with memory
│   ├── 3_Memory_Agent.py              # Agent with file persistence
│   ├── 4_ReAct.py                     # ReAct agent with tools
│   ├── 4_React-agent-explanation.md   # ReAct documentation
│   ├── 5_Drafter.py                   # Document drafting agent
│   ├── 6_RAG_Agent.py                 # RAG agent
│   └── rag_agent_docs.md              # RAG documentation
├── Graphs/
│   ├── 1_basic_single_input.ipynb     # Single input graph
│   ├── 2_multiple_inputs.ipynb        # Multiple inputs
│   ├── 3_Sequential_agent.ipynb       # Sequential processing
│   ├── 4_Conditional_agent.ipynb      # Conditional routing
│   └── 5_Looping.ipynb                # Looping logic
├── Exercise/
│   ├── 1_Graph1.ipynb                  # Basic exercise
│   ├── 2_Graph2.ipynb                  # Intermediate exercise
│   ├── 3_Graph3.ipynb                  # Sequential processing
│   ├── 4_Graph4.ipynb                  # Conditional routing
│   └── 5_Graph5.ipynb                  # Looping logic
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 💡 Concepts Covered

### Core Concepts

1. **State Management**

   - TypedDict structures
   - Message reducers
   - State persistence

2. **Graph Architecture**

   - Node creation
   - Edge definition
   - Conditional routing
   - Looping mechanisms

3. **Agent Patterns**
   - ReAct (Reasoning + Acting)
   - Memory management
   - Tool usage
   - RAG (Retrieval-Augmented Generation)

---

## 🤖 Agents

### 1. Basic Agent Bot (`1_Agent_Bot.py`)

Simple conversational agent that processes messages without memory.

**Features:**

- Single-turn conversations
- Direct LLM invocation
- Basic graph structure

**Usage:**

```bash
python Agents/1_Agent_Bot.py
```

---

### 2. Memory Agent (`2_Memory_Agent.py`)

Agent with conversation history tracking.

**Features:**

- Multi-turn conversations
- In-memory history
- Conversation logging to file

**Key Concepts:**

- State accumulation
- Message history management
- File I/O for persistence

**Usage:**

```bash
python Agents/2_Memory_Agent.py
```

---

### 3. Persistent Memory Agent (`3_Memory_Agent.py`)

Enhanced memory agent with file loading and history limits.

**Features:**

- Load previous conversations
- Conversation history limits (last 10 messages)
- Automatic file persistence

**Usage:**

```bash
python Agents/3_Memory_Agent.py
```

---

### 4. ReAct Agent (`4_ReAct.py`)

Agent with tool usage capabilities following the ReAct pattern.

**Features:**

- Tool binding (add, multiply, subtract, divide)
- Multi-step reasoning
- Streaming responses

**Key Concepts:**

- Tool definition with `@tool` decorator
- `ToolNode` for execution
- Conditional tool routing

**Example:**

```python
inputs = {
    "messages": [("user", "Add 30 + 12, then multiply the result by 6")]
}
```

**Output Flow:**

1. AI decides to call `add` tool
2. Tool executes: 30 + 12 = 42
3. AI decides to call `multiply` tool
4. Tool executes: 42 × 6 = 252
5. AI provides final answer

**Documentation:** See `4_React-agent-explanation.md` for detailed explanation

---

### 5. Drafter Agent (`5_Drafter.py`)

Document creation and editing assistant.

**Features:**

- Document creation
- Content updates
- File saving
- Interactive editing loop

**Key Concepts:**

- Global state management
- Tool chaining
- Conditional loop termination

**Usage:**

```bash
python Agents/5_Drafter.py
```

---

### 6. RAG Agent (`6_RAG_Agent.py`)

Retrieval-Augmented Generation agent for PDF question answering.

**Features:**

- PDF document loading
- Vector database (ChromaDB)
- Semantic search
- Multi-step retrieval
- Citation of sources

**Architecture:**

```
User Query → LLM → Decide: Need Search?
                ↓ Yes
            Search PDF → Retrieve Chunks
                ↓
            LLM → Read & Answer
                ↓
            Final Response
```

**Key Concepts:**

- Text splitting (1000 char chunks, 100 overlap)
- Embeddings (text-embedding-3-small)
- Similarity search (top 5 results)
- Tool-based retrieval

**Usage:**

```bash
python Agents/6_RAG_Agent.py
```

**Example Query:**

```
What is your question: How was the S&P500 performing in 2024?
```

**Complete Documentation:** See `rag_agent_docs.md`

---

## 📊 Graphs

### 1. Basic Single Input (`1_basic_single_input.ipynb`)

Simplest graph with one node and one input.

**Concepts:**

- StateGraph creation
- Node definition
- Entry/finish points

---

### 2. Multiple Inputs (`2_multiple_inputs.ipynb`)

Graph handling multiple state fields.

**State:**

```python
class AgentState(TypedDict):
    values: List[int]
    name: str
    result: str
```

---

### 3. Sequential Agent (`3_Sequential_agent.ipynb`)

Linear processing through multiple nodes.

**Flow:**

```
first_node → second_node → END
```

**Concepts:**

- Node chaining
- State modification across nodes
- Linear execution

---

### 4. Conditional Agent (`4_Conditional_agent.ipynb`)

Decision-based routing with conditional edges.

**Flow:**

```
       START
         ↓
    router_node
       ↙  ↘
  add     subtract
    ↓      ↓
       END
```

**Concepts:**

- Decision functions
- Conditional routing
- Path selection based on state

---

### 5. Looping (`5_Looping.ipynb`)

Agent that loops until a condition is met.

**Flow:**

```
greet → random_number ↺ (loop 5 times) → END
```

**Concepts:**

- Loop conditions
- Counter-based termination
- Iterative processing

---

## 🚀 Getting Started

### Quick Start

1. **Try a basic agent:**

```bash
python Agents/1_Agent_Bot.py
```

2. **Explore Jupyter notebooks:**

```bash
jupyter notebook Graphs/1_basic_single_input.ipynb
```

3. **Run the RAG agent:**

```bash
# Ensure you have Stock_Market_Performance_2024.pdf in Agents/
python Agents/6_RAG_Agent.py
```

### Learning Path

**Beginner:**

1. Start with Graph examples (1-3)
2. Progress to Agent Bot (1-2)
3. Understand state management

**Intermediate:** 4. Explore conditional routing (Graph 4) 5. Learn ReAct pattern (Agent 4) 6. Implement looping (Graph 5)

**Advanced:** 7. Build with Drafter agent (Agent 5) 8. Master RAG implementation (Agent 6) 9. Customize for your use case

---

## 📚 Documentation

### Key Documentation Files

- **`4_React-agent-explanation.md`** - Detailed ReAct pattern explanation
- **`rag_agent_docs.md`** - Complete RAG agent guide with workflow examples

### Important Concepts

#### State Reducers

```python
messages: Annotated[Sequence[BaseMessage], add_messages]
```

The `add_messages` reducer appends new messages instead of replacing them.

#### Tool Binding

```python
model = llm.bind_tools(tools)  # Enables LLM to request tool calls
```

#### Conditional Edges

```python
graph.add_conditional_edges(
    "source_node",
    decision_function,
    {
        "path1": "destination_node1",
        "path2": "destination_node2"
    }
)
```

---

## 🔧 Common Issues & Solutions

### Issue: Vector store recreated every time

**Solution:** Check if `chroma_db` directory exists and load instead of creating:

```python
if os.path.exists(persist_directory):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
```

### Issue: LLM doesn't call tools

**Solution:** Use `model` (tools bound) not `llm`:

```python
message = model.invoke(messages)  # ✅ Correct
message = llm.invoke(messages)    # ❌ Wrong
```

### Issue: Agent loops infinitely

**Solution:** Verify `should_continue` returns False when done:

```python
def should_continue(state):
    last_message = state["messages"][-1]
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Add new agent examples
- Improve documentation
- Fix bugs
- Suggest enhancements

---

## 📄 License

This project is for educational purposes. Please ensure you comply with OpenAI's usage policies when using their API.

---

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [LangChain](https://github.com/langchain-ai/langchain)
- Powered by OpenAI GPT models

---

## 📬 Contact

For questions or suggestions, please open an issue in the repository.

---

**Happy Building! 🎉**
