# Building AI-Enhanced Applications with MCP Python SDK: A Comprehensive Tutorial

The Model Context Protocol (MCP) is a standardized way for applications to provide context to Large Language Models (LLMs). This comprehensive tutorial will guide you through using the Python SDK to build and integrate MCP servers into your applications, with special attention to creating a brain-inspired memory system.

## Table of Contents

1. [What is MCP?](#what-is-mcp)
2. [Installation and Environment Setup](#installation-and-environment-setup)
3. [Core Concepts](#core-concepts)
4. [Creating Your First MCP Server](#creating-your-first-mcp-server)
5. [Working with Resources](#working-with-resources)
6. [Implementing Tools](#implementing-tools)
7. [Creating Prompts](#creating-prompts)
8. [Using Context in Tools and Resources](#using-context-in-tools-and-resources)
9. [Running Your Server](#running-your-server)
10. [Authentication](#authentication)
11. [Building a Brain-Inspired Memory System](#building-a-brain-inspired-memory-system)
12. [Advanced Usage](#advanced-usage)

## What is MCP?

The Model Context Protocol (MCP) lets you build servers that expose data and functionality to LLM applications in a secure, standardized way. Think of it like a web API, but specifically designed for LLM interactions. MCP servers can:

- Expose data through **Resources** (like GET endpoints; they load information into the LLM's context)
- Provide functionality through **Tools** (like POST endpoints; they execute code or produce side effects)
- Define interaction patterns through **Prompts** (reusable templates for LLM interactions)

MCP defines three core primitives that servers can implement:

| Primitive | Control               | Description                                         | Example Use                  |
|-----------|-----------------------|-----------------------------------------------------|------------------------------|
| Prompts   | User-controlled       | Interactive templates invoked by user choice        | Slash commands, menu options |
| Resources | Application-controlled| Contextual data managed by the client application   | File contents, API responses |
| Tools     | Model-controlled      | Functions exposed to the LLM to take actions        | API calls, data updates      |

## Installation and Environment Setup

### Setting Up a New Project with uv

We'll use [uv](https://docs.astral.sh/uv/) to manage our Python project and virtual environment:

```bash
# Create a new project
uv init mcp-server-demo
cd mcp-server-demo

# Create a virtual environment
uv venv

# Activate the virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install MCP with CLI tools
uv add "mcp[cli]"
```

**Important Notes:**
- The virtual environment **must be activated** in each terminal session where you want to use the `mcp` command
- You'll see `(.venv)` at the beginning of your prompt when the environment is activated
- Always make sure your virtual environment is activated before running any `mcp` commands

Alternatively, for projects using pip for dependencies:
```bash
# Create virtual environment
python -m venv .venv

# Activate the virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install MCP
pip install "mcp[cli]"
```

### Verifying Installation

To verify that MCP is installed and available in your activated environment:

```bash
# Check if MCP is installed
uv pip list | grep mcp

# Test if the MCP command is available
mcp --help
```

## Core Concepts

Let's review the key concepts in MCP:

### Server

The FastMCP server is your core interface to the MCP protocol. It handles connection management, protocol compliance, and message routing:

```python
from mcp.server.fastmcp import FastMCP

# Create a named server
mcp = FastMCP("My App")

# Specify dependencies for deployment and development
mcp = FastMCP("My App", dependencies=["pandas", "numpy"])
```

### Resources

Resources are how you expose data to LLMs. They're similar to GET endpoints in a REST API - they provide data but shouldn't perform significant computation or have side effects:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "App configuration here"

@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Dynamic user data"""
    return f"Profile data for user {user_id}"
```

### Tools

Tools let LLMs take actions through your server. Unlike resources, tools are expected to perform computation and have side effects:

```python
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI given weight in kg and height in meters"""
    return weight_kg / (height_m**2)

@mcp.tool()
async def fetch_weather(city: str) -> str:
    """Fetch current weather for a city"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.text
```

### Prompts

Prompts are reusable templates that help LLMs interact with your server effectively:

```python
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("My App")

@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"

@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]
```

## Creating Your First MCP Server

Let's create a simple MCP server:

```python
# main.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

# Run the server when executed directly
if __name__ == "__main__":
    mcp.run()
```

With your virtual environment activated, you can install this server in [Claude Desktop](https://claude.ai/download) and interact with it right away:
```bash
# Make sure your virtual environment is activated
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install the server in Claude Desktop
mcp install main.py
```

Alternatively, you can test it with the MCP Inspector:
```bash
# Make sure your virtual environment is activated
mcp dev main.py
```

## Working with Resources

Resources in MCP are data sources that can be accessed by the LLM:

```python
# Simple static resource
@mcp.resource("config://app")
def get_config() -> str:
    """Return static configuration data."""
    return "App configuration information"

# Dynamic resource with parameters
@mcp.resource("user://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Get user profile data for a specific user."""
    return f"Profile data for user {user_id}"

# Binary resource (image)
@mcp.resource("image://logo", mime_type="image/png")
def get_logo() -> bytes:
    """Return the application logo as binary data."""
    with open("logo.png", "rb") as f:
        return f.read()

# JSON resource
@mcp.resource("stats://users", mime_type="application/json")
def get_user_stats() -> dict:
    """Return user statistics as JSON."""
    return {
        "total_users": 1250,
        "active_users": 867,
        "new_users_today": 42
    }
```

You can also use the Resource classes directly:

```python
from mcp.server.fastmcp.resources import TextResource, FileResource, DirectoryResource
from pathlib import Path

# Add a text resource directly
mcp.add_resource(
    TextResource(
        uri="info://welcome",
        name="Welcome Message",
        description="A welcome message for new users",
        text="Welcome to our application!"
    )
)

# Add a file resource
mcp.add_resource(
    FileResource(
        uri="file://docs/readme.md",
        path=Path("/absolute/path/to/readme.md"),
        mime_type="text/markdown"
    )
)

# Add a directory resource
mcp.add_resource(
    DirectoryResource(
        uri="dir://docs",
        path=Path("/absolute/path/to/docs"),
        recursive=True,
        pattern="*.md"
    )
)
```

## Implementing Tools

Tools in MCP are functions that can be called by the LLM to perform actions:

```python
# Simple tool without parameters
@mcp.tool()
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Tool with parameters
@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI given weight in kg and height in meters."""
    return weight_kg / (height_m ** 2)

# Async tool for network operations
@mcp.tool()
async def fetch_weather(city: str) -> str:
    """Fetch current weather for a city."""
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.text
```

### Tools with Complex Inputs

You can use Pydantic models to handle complex structured inputs:

```python
from typing import Annotated, List
from pydantic import BaseModel, Field

# Define a complex input model
class ShrimpTank(BaseModel):
    class Shrimp(BaseModel):
        name: Annotated[str, Field(max_length=10)]
        color: str
        size_cm: float
    
    shrimp: List[Shrimp]
    water_temperature: float
    tank_name: str

@mcp.tool()
def analyze_tank(
    tank: ShrimpTank,
    include_size_analysis: bool = True
) -> str:
    """Analyze a shrimp tank configuration."""
    result = f"Tank Analysis for {tank.tank_name}:\n"
    result += f"Water Temperature: {tank.water_temperature}°C\n"
    result += f"Number of Shrimp: {len(tank.shrimp)}\n"
    
    if include_size_analysis:
        avg_size = sum(s.size_cm for s in tank.shrimp) / len(tank.shrimp)
        result += f"Average Shrimp Size: {avg_size:.2f} cm\n"
    
    return result
```

### Tools with Image Output

MCP provides an `Image` class for handling image data:

```python
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image

mcp = FastMCP("Image Demo", dependencies=["matplotlib", "Pillow"])

@mcp.tool()
def generate_chart(data: list[float], title: str) -> Image:
    """Generate a chart from the given data."""
    import matplotlib.pyplot as plt
    import io
    
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    
    return Image(data=buffer.getvalue(), format="png")
```

## Creating Prompts

Prompts in MCP are templates that help LLMs interact with your server effectively:

```python
@mcp.prompt()
def code_review(code: str) -> str:
    """Create a prompt for code review."""
    return f"Please review this code and provide suggestions for improvement:\n\n{code}"

@mcp.prompt()
def analyze_data(data: str, focus: str = "trends") -> str:
    """Create a prompt for data analysis with optional focus."""
    return f"""
    Here is some data to analyze:
    
    {data}
    
    Please focus your analysis on {focus}.
    """

# Prompt returning multiple messages
from mcp.server.fastmcp.prompts.base import UserMessage, AssistantMessage

@mcp.prompt()
def debug_error(error: str) -> list:
    """Create a debugging conversation."""
    return [
        UserMessage("I'm getting this error:"),
        UserMessage(error),
        AssistantMessage("I'll help debug that. What have you tried so far?")
    ]
```

## Using Context in Tools and Resources

MCP provides a `Context` object that gives your tools access to MCP capabilities:

```python
from mcp.server.fastmcp import Context

@mcp.tool()
async def process_files(files: list[str], ctx: Context) -> str:
    """Process multiple files with progress tracking."""
    total_files = len(files)
    
    # Log information to the client
    await ctx.info(f"Starting to process {total_files} files")
    
    for i, file in enumerate(files):
        # Report progress
        await ctx.report_progress(i, total_files, message=f"Processing {file}")
        
        # Read a resource
        contents, mime_type = await ctx.read_resource(f"file://{file}")
        
        # Log specific information
        await ctx.debug(f"File {file} is {len(contents)} bytes")
    
    await ctx.report_progress(total_files, total_files, message="Processing complete")
    return f"Successfully processed {total_files} files"
```

Context provides these key features:

- **Logging**: `debug()`, `info()`, `warning()`, `error()`
- **Progress Reporting**: `report_progress()`
- **Resource Access**: `read_resource()`
- **Request Information**: `request_id`, `client_id`

## Running Your Server

### Development Mode

The fastest way to test and debug your server is with the MCP Inspector:

```bash
# Make sure your virtual environment is activated
mcp dev main.py

# Add dependencies
mcp dev main.py --with pandas --with numpy

# Mount local code
mcp dev main.py --with-editable .
```

### Claude Desktop Integration

Once your server is ready, install it in Claude Desktop:

```bash
# Make sure your virtual environment is activated
mcp install main.py

# Custom name
mcp install main.py --name "My Analytics Server"

# Environment variables
mcp install main.py -v API_KEY=abc123 -v DB_URL=postgres://...
mcp install main.py -f .env
```

### Direct Execution

You can also run the server directly:

```bash
# Make sure your virtual environment is activated
python main.py
# or
mcp run main.py
```

### Transport Options

MCP supports multiple transport protocols:

```python
# Run with stdio (default)
mcp.run()

# Run with SSE
mcp.run(transport="sse")

# Run with Streamable HTTP (recommended for production)
mcp.run(transport="streamable-http")
```

For production deployments, Streamable HTTP is recommended:

```python
# main.py
from mcp.server.fastmcp import FastMCP

# Create a FastMCP server
mcp = FastMCP("Production API")

# Add your tools/resources here

if __name__ == "__main__":
    # Run server with streamable_http transport
    mcp.run(transport="streamable-http")
```

You can also create a stateless server for better scalability:

```python
# Stateless server (no session persistence)
mcp = FastMCP("StatelessServer", stateless_http=True)
```

## Authentication

MCP provides built-in support for OAuth 2.0 authentication:

```python
from mcp.server.fastmcp import FastMCP
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from my_auth_provider import MyOAuthServerProvider

# Create server with authentication
mcp = FastMCP(
    "Secure API",
    auth_server_provider=MyOAuthServerProvider(),
    auth=AuthSettings(
        issuer_url="https://myapp.com",
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=["read:data", "write:data"],
            default_scopes=["read:data"],
        ),
        required_scopes=["read:data"],
    ),
)

@mcp.tool()
def get_sensitive_data(ctx: Context) -> str:
    """Get sensitive data (requires authentication)."""
    # Authentication is handled automatically by the framework
    return "This is sensitive data that only authenticated users can see"
```

## Building a Brain-Inspired Memory System

Now let's build a sophisticated memory system for Claude using MCP. This system mimics how human brains cluster, retrieve, and forget memories.

### Overview

The memory system is inspired by how human brains organize memories:

- **Memory nodes**: Text with vector embeddings that represent semantic meaning
- **Memory clustering**: Similar memories are automatically merged
- **Dynamic importance**: Important or frequently accessed memories are reinforced
- **Memory decay**: Less relevant memories gradually fade away
- **Memory pruning**: Less important memories are removed when capacity is reached

### Core Architecture

The system relies on several key components:

1. **Vector Embeddings**: Text is converted to mathematical representations that capture meaning
2. **PostgreSQL with pgvector**: Efficiently stores and queries vector embeddings
3. **Memory management algorithms**: Handle merging, reinforcement, decay, and pruning
4. **MCP interface**: Exposes memory tools to Claude Desktop

### Setup

First, let's set up the dependencies:

```bash
# Install MCP and dependencies
pip install "mcp[cli]"
pip install pydantic-ai-slim[openai] asyncpg numpy pgvector
```

You'll need PostgreSQL with the pgvector extension. The code assumes it's running at `localhost:54320` with username/password `postgres/postgres`.

### Implementation

Here's a simplified version of the memory system:

```python
# memory.py
"""
Brain-inspired memory system for Claude with MCP.
Uses vector embeddings and pgvector for efficient similarity search.
"""

import asyncio
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import asyncpg
import numpy as np
from openai import AsyncOpenAI
from pgvector.asyncpg import register_vector  
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from mcp.server.fastmcp import FastMCP

# Configuration
MAX_DEPTH = 5
SIMILARITY_THRESHOLD = 0.7
DECAY_FACTOR = 0.99
REINFORCEMENT_FACTOR = 1.1

DEFAULT_LLM_MODEL = "openai:gpt-4o"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

DB_DSN = "postgresql://postgres:postgres@localhost:54320/memory_db"
PROFILE_DIR = (
    Path.home() / ".fastmcp" / os.environ.get("USER", "anon") / "memory"
).resolve()
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize MCP server
mcp = FastMCP(
    "memory",
    dependencies=[
        "pydantic-ai-slim[openai]",
        "asyncpg",
        "numpy",
        "pgvector",
    ],
)

# Utility functions
def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_array = np.array(a, dtype=np.float64)
    b_array = np.array(b, dtype=np.float64)
    return np.dot(a_array, b_array) / (
        np.linalg.norm(a_array) * np.linalg.norm(b_array)
    )

async def do_ai[T](
    user_prompt: str,
    system_prompt: str,
    result_type: type[T] | Annotated,
    deps=None,
) -> T:
    agent = Agent(
        DEFAULT_LLM_MODEL,
        system_prompt=system_prompt,
        result_type=result_type,
    )
    result = await agent.run(user_prompt, deps=deps)
    return result.data

@dataclass
class Deps:
    openai: AsyncOpenAI
    pool: asyncpg.Pool

# Memory node model
class MemoryNode(BaseModel):
    id: Optional[int] = None
    content: str
    summary: str = ""
    importance: float = 1.0
    access_count: int = 0
    timestamp: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp()
    )
    embedding: list[float]

    @classmethod
    async def from_content(cls, content: str, deps: Deps):
        embedding = await get_embedding(content, deps)
        return cls(content=content, embedding=embedding)

    async def save(self, deps: Deps):
        async with deps.pool.acquire() as conn:
            if self.id is None:
                result = await conn.fetchrow(
                    """
                    INSERT INTO memories (content, summary, importance, access_count,
                        timestamp, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                    """,
                    self.content,
                    self.summary,
                    self.importance,
                    self.access_count,
                    self.timestamp,
                    self.embedding,
                )
                self.id = result["id"]
            else:
                await conn.execute(
                    """
                    UPDATE memories
                    SET content = $1, summary = $2, importance = $3,
                        access_count = $4, timestamp = $5, embedding = $6
                    WHERE id = $7
                    """,
                    self.content,
                    self.summary,
                    self.importance,
                    self.access_count,
                    self.timestamp,
                    self.embedding,
                    self.id,
                )

    async def merge_with(self, other, deps: Deps):
        self.content = await do_ai(
            f"{self.content}\n\n{other.content}",
            "Combine the following two texts into a single, coherent text.",
            str,
            deps,
        )
        self.importance += other.importance
        self.access_count += other.access_count
        self.embedding = [(a + b) / 2 for a, b in zip(self.embedding, other.embedding)]
        self.summary = await do_ai(
            self.content, "Summarize the following text concisely.", str, deps
        )
        await self.save(deps)
        # Delete the merged node
        if other.id is not None:
            await delete_memory(other.id, deps)

    def get_effective_importance(self):
        return self.importance * (1 + math.log(self.access_count + 1))

# Get embeddings from OpenAI
async def get_embedding(text: str, deps: Deps) -> list[float]:
    embedding_response = await deps.openai.embeddings.create(
        input=text,
        model=DEFAULT_EMBEDDING_MODEL,
    )
    return embedding_response.data[0].embedding

# Database functions
async def get_db_pool() -> asyncpg.Pool:
    async def init(conn):
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await register_vector(conn)

    pool = await asyncpg.create_pool(DB_DSN, init=init)
    return pool

async def delete_memory(memory_id: int, deps: Deps):
    async with deps.pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)

async def add_memory(content: str, deps: Deps):
    new_memory = await MemoryNode.from_content(content, deps)
    await new_memory.save(deps)

    similar_memories = await find_similar_memories(new_memory.embedding, deps)
    for memory in similar_memories:
        if memory.id != new_memory.id:
            await new_memory.merge_with(memory, deps)

    await update_importance(new_memory.embedding, deps)
    await prune_memories(deps)

    return f"Remembered: {content}"

async def find_similar_memories(embedding: list[float], deps: Deps) -> list[MemoryNode]:
    async with deps.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, summary, importance, access_count, timestamp, embedding
            FROM memories
            ORDER BY embedding <-> $1
            LIMIT 5
            """,
            embedding,
        )
    memories = [
        MemoryNode(
            id=row["id"],
            content=row["content"],
            summary=row["summary"],
            importance=row["importance"],
            access_count=row["access_count"],
            timestamp=row["timestamp"],
            embedding=row["embedding"],
        )
        for row in rows
    ]
    return memories

async def update_importance(user_embedding: list[float], deps: Deps):
    async with deps.pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, importance, access_count, embedding FROM memories"
        )
        for row in rows:
            memory_embedding = row["embedding"]
            similarity = cosine_similarity(user_embedding, memory_embedding)
            if similarity > SIMILARITY_THRESHOLD:
                new_importance = row["importance"] * REINFORCEMENT_FACTOR
                new_access_count = row["access_count"] + 1
            else:
                new_importance = row["importance"] * DECAY_FACTOR
                new_access_count = row["access_count"]
            await conn.execute(
                """
                UPDATE memories
                SET importance = $1, access_count = $2
                WHERE id = $3
                """,
                new_importance,
                new_access_count,
                row["id"],
            )

async def prune_memories(deps: Deps):
    async with deps.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, importance, access_count
            FROM memories
            ORDER BY importance DESC
            OFFSET $1
            """,
            MAX_DEPTH,
        )
        for row in rows:
            await conn.execute("DELETE FROM memories WHERE id = $1", row["id"])

async def display_memory_tree(deps: Deps) -> str:
    async with deps.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT content, summary, importance, access_count
            FROM memories
            ORDER BY importance DESC
            LIMIT $1
            """,
            MAX_DEPTH,
        )
    result = ""
    for row in rows:
        effective_importance = row["importance"] * (
            1 + math.log(row["access_count"] + 1)
        )
        summary = row["summary"] or row["content"]
        result += f"- {summary} (Importance: {effective_importance:.2f})\n"
    return result

# MCP tools
@mcp.tool()
async def remember(
    contents: list[str] = Field(
        description="List of observations or memories to store"
    ),
):
    """Store new memories in the system."""
    deps = Deps(openai=AsyncOpenAI(), pool=await get_db_pool())
    try:
        return "\n".join(
            await asyncio.gather(*[add_memory(content, deps) for content in contents])
        )
    finally:
        await deps.pool.close()

@mcp.tool()
async def read_profile() -> str:
    """Retrieve and display current memories."""
    deps = Deps(openai=AsyncOpenAI(), pool=await get_db_pool())
    profile = await display_memory_tree(deps)
    await deps.pool.close()
    return profile

# Database initialization
async def initialize_database():
    # Create database
    pool = await asyncpg.create_pool(
        "postgresql://postgres:postgres@localhost:54320/postgres"
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = 'memory_db'
                AND pid <> pg_backend_pid();
            """)
            await conn.execute("DROP DATABASE IF EXISTS memory_db;")
            await conn.execute("CREATE DATABASE memory_db;")
    finally:
        await pool.close()

    # Create tables
    pool = await asyncpg.create_pool(DB_DSN)
    try:
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await register_vector(conn)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    summary TEXT,
                    importance REAL NOT NULL,
                    access_count INT NOT NULL,
                    timestamp DOUBLE PRECISION NOT NULL,
                    embedding vector(1536) NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
                    USING hnsw (embedding vector_l2_ops);
            """)
    finally:
        await pool.close()

# Main entry point
if __name__ == "__main__":
    asyncio.run(initialize_database())
```

### Enhanced Memory System with Autonomous Features

Let's enhance the memory system with autonomous features:

```python
# Add these to your memory.py file

from mcp.server.fastmcp.prompts.base import UserMessage

# Enhanced autonomous memory tools
@mcp.tool()
async def enhance_with_memories(conversation_context: str) -> dict:
    """
    Automatically retrieve relevant memories and suggest new ones to store.
    """
    deps = Deps(openai=AsyncOpenAI(), pool=await get_db_pool())
    try:
        # Get conversation embedding
        context_embedding = await get_embedding(conversation_context, deps)
        
        # Find relevant memories
        relevant_memories = await find_similar_memories(context_embedding, deps)
        
        # Extract potential new memories
        new_memory_candidates = await do_ai(
            conversation_context,
            "Extract 3-5 key personal facts from this conversation that would be valuable to remember.",
            list[str],
            deps
        )
        
        return {
            "relevant_memories": [m.content for m in relevant_memories],
            "new_memory_candidates": new_memory_candidates
        }
    finally:
        await deps.pool.close()

# MCP prompts for autonomous memory usage
@mcp.prompt()
def conversation_with_memory() -> list:
    """A prompt that automatically incorporates memories."""
    return [
        UserMessage("""
        Before responding, retrieve and consider what you remember about me.
        Please incorporate this context naturally in your responses without explicitly 
        mentioning that you're using stored memories. At the end of our conversation, 
        please identify any important new information I've shared so it can be remembered.
        """)
    ]

@mcp.prompt()
def remember_conversation(conversation_text: str) -> list:
    """Extract and store key information from a conversation."""
    return [
        UserMessage(f"""
        Please analyze this conversation and extract 3-5 key facts or preferences that would 
        be useful to remember about me:
        
        {conversation_text}
        
        Format your response as a list of facts in bullet points.
        """)
    ]
```

### Using Local Models for Memory Systems

For running the memory system locally, you can use alternative models:

```python
# Add to memory.py for using Sentence Transformers instead of OpenAI embeddings

from sentence_transformers import SentenceTransformer
import numpy as np

# Update Deps class
@dataclass
class Deps:
    openai: AsyncOpenAI
    pool: asyncpg.Pool
    embedding_model: SentenceTransformer = None
    
    def __post_init__(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Replace the get_embedding function
async def get_embedding(text: str, deps: Deps) -> list[float]:
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(
        None, 
        lambda: deps.embedding_model.encode(text, normalize_embeddings=True)
    )
    return embedding.tolist()
```

For using local LLM models:

```python
import httpx

# Modified do_ai function for local LLM
async def do_ai[T](
    user_prompt: str,
    system_prompt: str,
    result_type: type[T] | Annotated,
    deps=None,
) -> T:
    # Use local Ollama endpoint
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3-mini",
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False
            }
        )
        result = response.json()["response"]
        # Parse result into expected type
        return parse_result(result, result_type)
```

### Using the Memory System with Claude Desktop

Initialize the database and install the MCP server in Claude Desktop:

```bash
# Initialize the database
python memory.py

# Install in Claude Desktop
mcp install memory.py
```

Once installed, you can use it in Claude Desktop:

1. **Storing Memories**:
   ```
   Claude, remember these things about me: ['I have two dogs named Max and Bella', 'I'm allergic to peanuts', 'I'm planning a trip to Japan next year']
   ```

2. **Retrieving Memories**:
   ```
   Claude, what do you remember about me?
   ```

## Advanced Usage

### Lifespan Management

MCP provides a lifespan context manager for managing resources over the server lifecycle:

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP, Context

# Define your application context
@dataclass
class AppContext:
    db_connection: Any
    api_client: Any

# Create a lifespan function
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Handle application lifecycle."""
    print("Server starting up - initializing resources")
    
    # Initialize resources
    db = await connect_to_database()
    api = await create_api_client()
    
    try:
        # Yield the context to the server
        yield AppContext(
            db_connection=db,
            api_client=api
        )
    finally:
        # Clean up resources
        print("Server shutting down - cleaning up resources")
        await db.disconnect()
        await api.close()

# Create server with lifespan
mcp = FastMCP("Database App", lifespan=app_lifespan)

# Use context in tools
@mcp.tool()
def query_database(query: str, ctx: Context) -> list:
    """Execute a database query."""
    # Access the database connection from the lifespan context
    db = ctx.request_context.lifespan_context.db_connection
    return db.execute(query)
```

### Building a Client Application

The SDK provides a high-level client interface for connecting to MCP servers:

```python
# client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Create server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",  # Executable
        args=["main.py"],  # Your MCP server script
        env=None,  # Optional environment variables
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")

            # Call a tool
            result = await session.call_tool("add", arguments={"a": 5, "b": 3})
            print(f"Result of add tool: {result.content[0].text}")
            
            # List available resources
            resources = await session.list_resources()
            print(f"Available resources: {[res.name for res in resources.resources]}")
            
            # Read a resource
            greeting = await session.read_resource("greeting://World")
            print(f"Greeting: {greeting.contents[0].text}")

if __name__ == "__main__":
    asyncio.run(main())
```

You can also use the Streamable HTTP transport for client applications:

```python
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

async def main():
    # Connect to a streamable HTTP server
    async with streamablehttp_client("http://localhost:3000/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            # Call a tool
            tool_result = await session.call_tool("add", {"a": 5, "b": 3})
            print("Result:", tool_result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

This tutorial has covered the basics of working with the MCP Python SDK. You've learned how to create MCP servers, implement resources, tools, and prompts, work with authentication, build a brain-inspired memory system, and create client applications. To explore more advanced features, check out the [official documentation](https://modelcontextprotocol.io) and the examples provided in the SDK repository.

Happy building with MCP!
