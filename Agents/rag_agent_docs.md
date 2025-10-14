# RAG Agent with LangGraph - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup and Dependencies](#setup-and-dependencies)
4. [Component Breakdown](#component-breakdown)
5. [Complete Workflow Example](#complete-workflow-example)
6. [Graph Flow](#graph-flow)
7. [Key Concepts](#key-concepts)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This is a **Retrieval-Augmented Generation (RAG) Agent** that answers questions about stock market performance in 2024 by searching through a PDF document. It uses LangGraph to orchestrate the conversation flow between the LLM and the retrieval tool.

### What is RAG?

RAG combines:

- **Retrieval**: Finding relevant information from a knowledge base (PDF)
- **Generation**: Using an LLM to generate answers based on retrieved information

### What is LangGraph?

LangGraph is a framework for building stateful, multi-step applications with LLMs. It represents your application as a graph where:

- **Nodes** = Functions that process data
- **Edges** = Paths between nodes
- **State** = Shared data passed between nodes

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     RAG Agent Flow                      │
└─────────────────────────────────────────────────────────┘

      User Input
         │
         ▼
┌─────────────────┐
│   START (llm)   │  ← Entry point
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│     call_llm()          │  → Sends messages to LLM
│  (with system prompt)   │  → LLM decides: answer or search?
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  should_continue()?     │  → Checks if LLM wants to use tools
└────────┬────────────────┘
         │
    ┌────┴────┐
    │         │
   YES       NO
    │         │
    ▼         ▼
┌──────────┐  END
│retriever │  (return answer)
│ _agent   │
└────┬─────┘
     │
     ▼
┌─────────────────────────┐
│   take_action()         │  → Executes retrieve tool
│  (searches PDF)         │  → Returns relevant chunks
└────────┬────────────────┘
         │
         │ (Loop back with results)
         ▼
    ┌─────────┐
    │   llm   │  → LLM reads results & formulates answer
    └─────────┘
         │
         ▼
    (Repeat until no more tools needed)
```

---

## Component Breakdown

### 1. Vector Store Setup

**Purpose:** Convert PDF into searchable chunks

```python
# Load PDF
pdf_loader = PyPDFLoader(pdf_path)
documents = pdf_loader.load()  # Returns list of pages

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Max characters per chunk
    chunk_overlap=100   # Overlap prevents context loss
)
texts = text_splitter.split_documents(documents)

# Create vector database
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,  # Converts text to numbers
    persist_directory=persist_directory
)
```

**What happens:**

- PDF pages → Split into ~1000 character chunks
- Each chunk → Converted to vector (numerical representation)
- Vectors → Stored in ChromaDB for similarity search

**Console Output:**

```
Loaded 9 pages from the PDF.
Created vector store with 24 documents.
```

---

### 2. Retriever Tool

```python
@tool
def retrieve(query: str) -> str:
    """Searches PDF for relevant information"""
    docs = retriever.invoke(query)  # Finds top 5 similar chunks

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Result {i+1}:\n{doc.page_content}\n")

    return "\n\n".join(results)
```

**What it does:**

1. Takes search query (e.g., "S&P 500 performance 2024")
2. Converts query to vector
3. Finds 5 most similar document chunks
4. Returns text content

---

### 3. Agent State

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

**State holds conversation history:**

- `HumanMessage`: User input
- `AIMessage`: LLM response (may contain tool calls)
- `ToolMessage`: Results from tool execution
- `SystemMessage`: Instructions for LLM

---

### 4. Node Functions

#### **call_llm(state) → state**

```python
def call_llm(state: AgentState) -> AgentState:
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = model.invoke(messages)  # model has tools bound
    return {'messages': [message]}
```

**Process:**

1. Add system prompt to messages
2. Send to LLM
3. LLM returns text answer OR tool call request
4. Add response to state

---

#### **should_continue(state) → bool**

```python
def should_continue(state: AgentState) -> bool:
    last_message = state["messages"][-1]
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0
```

**Logic:**

- Has tool calls? → `True` (go to retriever)
- Just text? → `False` (end, show answer)

---

#### **take_action(state) → state**

```python
def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
        results.append(ToolMessage(content=str(result), name=t['name'], tool_call_id=t['id']))

    return {'messages': results}
```

**Process:**

1. Extract tool calls from LLM message
2. Execute each tool
3. Wrap results in ToolMessage
4. Return to state

---

### 5. Graph Construction

```python
graph = StateGraph(AgentState)

# Add processing nodes
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

# Conditional routing
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)

# Loop back to llm
graph.add_edge("retriever_agent", "llm")

# Set entry point
graph.set_entry_point("llm")

# Compile
rag_agent = graph.compile()
```

---

## Complete Workflow Example

### User Query: "How was the S&P500 performing in 2024?"

---

### **STEP 1: User Input**

```bash
=== RAG AGENT===
What is your question: How was the SMP500 performing in 2024?
```

**State:**

```python
{
    "messages": [
        HumanMessage(content="How was the SMP500 performing in 2024?")
    ]
}
```

---

### **STEP 2: First call_llm() Execution**

**What happens:**

1. System prompt is prepended
2. Messages sent to LLM
3. LLM analyzes: "I need to search the PDF for S&P 500 data"
4. LLM creates tool call request

**Messages sent to LLM:**

```python
[
    SystemMessage(content="You are an intelligent AI assistant..."),
    HumanMessage(content="How was the SMP500 performing in 2024?")
]
```

**LLM Response:**

```python
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "retrieve",
            "args": {"query": "S&P 500 performance 2024"},
            "id": "call_abc123"
        }
    ]
)
```

**Updated State:**

```python
{
    "messages": [
        HumanMessage(content="How was the SMP500 performing in 2024?"),
        AIMessage(content="", tool_calls=[...])
    ]
}
```

---

### **STEP 3: should_continue() Decision**

**Check:**

```python
last_message = AIMessage(tool_calls=[...])
has_tool_calls = len(tool_calls) > 0  # True
return True  # Continue to retriever_agent
```

**Decision:** Route to `retriever_agent`

---

### **STEP 4: take_action() Execution**

**Console Output:**

```bash
🔧 Calling tool: retrieve with query: S&P 500 performance 2024
Result length: 4813 characters
🔧 Finished all tool calls.
```

**What happens:**

1. Extract tool call: `retrieve("S&P 500 performance 2024")`
2. Execute retriever:
   - Converts query to vector
   - Searches ChromaDB
   - Finds 5 most similar chunks
   - Returns ~4813 characters of text

**Retrieved Content (example):**

```
Result 1:
The S&P 500 index delivered a total return of approximately 25% in 2024...

Result 2:
The tech-heavy Nasdaq Composite outpaced the broader market with a nearly 29% increase...

Result 3:
Smaller-cap stocks, such as those in the S&P 500 Equal-Weight index and the Russell 2000...

Result 4:
A key theme for the year was the dominance of mega-cap technology stocks...

Result 5:
This marked the second consecutive year of over 20% returns for the S&P 500...
```

**Updated State:**

```python
{
    "messages": [
        HumanMessage(content="How was the SMP500 performing in 2024?"),
        AIMessage(content="", tool_calls=[...]),
        ToolMessage(
            content="Result 1:\nThe S&P 500 index delivered a total return of approximately 25%...\n\nResult 2:...",
            name="retrieve",
            tool_call_id="call_abc123"
        )
    ]
}
```

---

### **STEP 5: Loop Back to call_llm()**

**What happens:**

1. Graph automatically routes back to `llm` node
2. System prompt prepended again
3. Full conversation (including retrieved data) sent to LLM

**Messages sent to LLM:**

```python
[
    SystemMessage(content="You are an intelligent AI assistant..."),
    HumanMessage(content="How was the SMP500 performing in 2024?"),
    AIMessage(content="", tool_calls=[...]),
    ToolMessage(content="Result 1: The S&P 500 index delivered...\n\nResult 2:...")
]
```

**LLM Response:**

```python
AIMessage(
    content="In 2024, the S&P 500 index delivered a total return of approximately 25%, with around 23% in price terms. This marked the second consecutive year of over 20% returns for the S&P 500, a feat not observed since the late 1990s. The strong performance was part of a broader rally in the U.S. stock market, although the gains were not evenly distributed across all sectors. The tech-heavy Nasdaq Composite outpaced the broader market with a nearly 29% increase, while smaller-cap stocks, such as those in the S&P 500 Equal-Weight index and the Russell 2000, rose about 10-11% in 2024. A key theme for the year was the dominance of mega-cap technology stocks, often referred to as the 'Magnificent 7,' which includes companies like Apple, Microsoft, Alphabet (Google), Amazon, and Meta (Source: Stock Market Performance in 2024, U.S. Market Overview).",
    tool_calls=[]  # No more tool calls
)
```

**Updated State:**

```python
{
    "messages": [
        HumanMessage(content="How was the SMP500 performing in 2024?"),
        AIMessage(content="", tool_calls=[...]),
        ToolMessage(content="Result 1:..."),
        AIMessage(content="In 2024, the S&P 500 index delivered...", tool_calls=[])
    ]
}
```

---

### **STEP 6: should_continue() Decision**

**Check:**

```python
last_message = AIMessage(content="In 2024, the S&P 500...", tool_calls=[])
has_tool_calls = len(tool_calls) > 0  # False
return False  # End execution
```

**Decision:** Route to `END`

---

### **STEP 7: Return Answer to User**

**Console Output:**

```bash
=== ANSWER ===
In 2024, the S&P 500 index delivered a total return of approximately 25%, with around 23% in price terms. This marked the second consecutive year of over 20% returns for the S&P 500, a feat not observed since the late 1990s. The strong performance was part of a broader rally in the U.S. stock market, although the gains were not evenly distributed across all sectors. The tech-heavy Nasdaq Composite outpaced the broader market with a nearly 29% increase, while smaller-cap stocks, such as those in the S&P 500 Equal-Weight index and the Russell 2000, rose about 10-11% in 2024. A key theme for the year was the dominance of mega-cap technology stocks, often referred to as the "Magnificent 7," which includes companies like Apple, Microsoft, Alphabet (Google), Amazon, and Meta (Source: Stock Market Performance in 2024, U.S. Market Overview).
```

**Code:**

```python
result = rag_agent.invoke({"messages": messages})
print(result['messages'][-1].content)
```

---

### **Workflow Summary**

```
1. User Input
   ↓
2. call_llm() → LLM decides to search
   ↓
3. should_continue() → Returns True
   ↓
4. take_action() → Searches PDF, retrieves chunks
   ↓
5. Loop back to call_llm() → LLM reads chunks & answers
   ↓
6. should_continue() → Returns False
   ↓
7. END → Show answer to user
```

**Total Iterations:** 2 LLM calls, 1 tool call

---

## Graph Flow

### Visual Representation

```
User: "How was S&P500 in 2024?"
    │
    ▼
┌─────────────────────────────────────┐
│ ITERATION 1                         │
├─────────────────────────────────────┤
│ [llm] call_llm()                    │
│   → LLM: "I need to search"         │
│   → Output: AIMessage(tool_calls)   │
├─────────────────────────────────────┤
│ [decision] should_continue()        │
│   → has tool_calls? YES             │
│   → Route to retriever_agent        │
├─────────────────────────────────────┤
│ [retriever_agent] take_action()     │
│   → Execute: retrieve(query)        │
│   → Output: ToolMessage(results)    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ ITERATION 2                         │
├─────────────────────────────────────┤
│ [llm] call_llm()                    │
│   → LLM reads retrieved data        │
│   → Output: AIMessage(answer)       │
├─────────────────────────────────────┤
│ [decision] should_continue()        │
│   → has tool_calls? NO              │
│   → Route to END                    │
└─────────────────────────────────────┘
    │
    ▼
Display final answer
```

---

## Key Concepts

### 1. Why bind_tools?

```python
model = llm.bind_tools(tools)
```

- Tells LLM about available tools
- Enables LLM to request tool calls
- Without this, LLM wouldn't know about `retrieve`

---

### 2. Why ToolMessage?

```python
ToolMessage(content=result, name="retrieve", tool_call_id=t['id'])
```

- Structured format LLM understands
- Links result to specific tool call
- Distinguishes tool output from user input

---

### 3. Why the Loop?

```python
graph.add_edge("retriever_agent", "llm")
```

- Enables multi-step reasoning
- LLM can make multiple searches
- Example: Search → Read → Search again → Combine

---

### 4. State Management

```python
Annotated[Sequence[BaseMessage], add_messages]
```

- `add_messages` appends instead of replacing
- Complete conversation history preserved
- LLM always has full context

---

## Troubleshooting

### Issue: LLM doesn't call tools

**Symptom:**

```
=== ANSWER ===
To provide you with accurate information...
```

**Cause:** Using `llm` instead of `model`

**Solution:**

```python
# ❌ Wrong
message = llm.invoke(messages)

# ✅ Correct
message = model.invoke(messages)
```

---

### Issue: Tuple in query output

**Symptom:**

```
🔧 Calling tool: retrieve with query: ('S&P 500', 'No query provided')
```

**Cause:** Incorrect default value syntax

**Solution:**

```python
# ❌ Wrong
t['args'].get('query', ''), 'No query provided'

# ✅ Correct
t['args'].get('query', 'No query provided')
```

---

### Issue: Agent loops infinitely

**Symptom:** Never reaches END

**Cause:** `should_continue` always returns True

**Solution:** Verify logic:

```python
def should_continue(state: AgentState) -> bool:
    last_message = state["messages"][-1]
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0
```

---

## Summary

This RAG agent demonstrates:

✅ Stateful conversation management  
✅ Dynamic tool usage based on LLM decisions  
✅ Semantic search over documents  
✅ Multi-step reasoning with loops  
✅ Structured graph-based orchestration

**Key Insight:** The LLM decides when to search, what to search for, and when it has enough information to answer.
