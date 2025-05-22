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

Now let's build a sophisticated memory system for Claude using MCP that runs entirely on local Hugging Face models. This system mimics how human brains cluster, retrieve, and forget memories, creating a more natural, context-aware assistant experience that doesn't rely on external APIs.

### Overview

The memory system is inspired by how human brains organize memories:

- **Memory nodes**: Text with vector embeddings that represent semantic meaning
- **Memory clustering**: Similar memories are automatically merged using local LLMs
- **Dynamic importance**: Important or frequently accessed memories are reinforced
- **Memory decay**: Less relevant memories gradually fade away
- **Memory pruning**: Less important memories are removed when capacity is reached

### Core Architecture

The system relies on several key components:

1. **Local Embedding Models**: Hugging Face sentence-transformers for semantic representation
2. **Local LLM Models**: Hugging Face transformers for text generation and summarization
3. **PostgreSQL with pgvector**: Efficiently stores and queries vector embeddings
4. **Memory management algorithms**: Handle merging, reinforcement, decay, and pruning
5. **MCP interface**: Exposes memory tools to Claude Desktop

### Setup

First, let's install all the required dependencies:

```bash
# Create a new project
uv init memory-system
cd memory-system

# Activate virtual environment
uv venv
source .venv/bin/activate  # On Unix/macOS
# .venv\Scripts\activate   # On Windows

# Install MCP and core dependencies
uv add "mcp[cli]"
uv add pydantic-ai asyncpg numpy pgvector

# Install Hugging Face dependencies
uv add transformers torch accelerate bitsandbytes sentence-transformers

# For GPU support (optional but recommended)
# uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

You'll also need PostgreSQL with the pgvector extension. For development, you can use Docker:

```bash
# Run PostgreSQL with pgvector
docker run --name postgres-memory \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=memory_db \
  -p 54320:5432 \
  -d pgvector/pgvector:pg16
```

### Complete Implementation

Let's build the complete memory system step by step. Create a file called `memory.py`:

```python
# memory.py
"""
Brain-inspired memory system for Claude with MCP.
Uses local Hugging Face models for both LLM and embeddings.
"""

import asyncio
import math
import os
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional, Union, List

import asyncpg
import numpy as np
import torch
from pgvector.asyncpg import register_vector
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import UserMessage, AssistantMessage

# ============================================================================
# CONFIGURATION
# ============================================================================

# Memory system parameters
MAX_DEPTH = 10  # Maximum number of memories to keep
SIMILARITY_THRESHOLD = 0.7  # Threshold for memory similarity
DECAY_FACTOR = 0.99  # How quickly memories decay
REINFORCEMENT_FACTOR = 1.1  # How much similar memories are reinforced
EMBEDDING_DIMENSIONS = 384  # For all-MiniLM-L6-v2

# Database configuration
DB_DSN = "postgresql://postgres:postgres@localhost:54320/memory_db"

# Model configurations for different hardware setups
MODEL_CONFIG = {
    # For 8-16GB RAM systems
    "low_memory": {
        "llm_model": "microsoft/Phi-3-mini-4k-instruct",
        "embedding_model": "all-MiniLM-L6-v2",
        "load_in_4bit": True,
        "max_new_tokens": 256
    },
    # For 16-32GB RAM systems
    "medium_memory": {
        "llm_model": "microsoft/Phi-3-small-8k-instruct",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "load_in_8bit": True,
        "max_new_tokens": 384
    },
    # For 32GB+ RAM systems
    "high_memory": {
        "llm_model": "microsoft/Phi-3-medium-4k-instruct",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "load_in_4bit": False,
        "max_new_tokens": 512
    }
}

# Select configuration based on available memory
MEMORY_CONFIG = "low_memory"  # Change this based on your system

# Profile directory for storing user-specific data
PROFILE_DIR = (
    Path.home() / ".fastmcp" / os.environ.get("USER", "anon") / "memory"
).resolve()
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# GLOBAL MODEL INSTANCES
# ============================================================================

# Global instances for models (loaded once at startup)
_embedding_model: Optional[SentenceTransformer] = None
_llm_model: Optional[AutoModelForCausalLM] = None
_llm_tokenizer: Optional[AutoTokenizer] = None
_text_generator: Optional[pipeline] = None

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

async def initialize_embedding_model() -> SentenceTransformer:
    """Initialize the embedding model."""
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
    
    config = MODEL_CONFIG[MEMORY_CONFIG]
    model_name = config["embedding_model"]
    
    print(f"Loading embedding model: {model_name}")
    
    # Load in executor to avoid blocking
    loop = asyncio.get_event_loop()
    _embedding_model = await loop.run_in_executor(
        None, 
        lambda: SentenceTransformer(model_name)
    )
    
    print(f"✓ Embedding model loaded: {model_name}")
    return _embedding_model

async def initialize_llm_model() -> None:
    """Initialize the LLM model and tokenizer."""
    global _llm_model, _llm_tokenizer, _text_generator
    
    if _text_generator is not None:
        return
    
    config = MODEL_CONFIG[MEMORY_CONFIG]
    model_name = config["llm_model"]
    
    print(f"Loading LLM model: {model_name}")
    
    def load_model():
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure model loading parameters
        model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
        
        # Add quantization if specified
        if config.get("load_in_4bit"):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif config.get("load_in_8bit"):
            model_kwargs["load_in_8bit"] = True
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        # Create text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            pad_token_id=tokenizer.eos_token_id
        )
        
        return model, tokenizer, generator
    
    # Load in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    _llm_model, _llm_tokenizer, _text_generator = await loop.run_in_executor(
        None, load_model
    )
    
    print(f"✓ LLM model loaded: {model_name}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_array = np.array(a, dtype=np.float64)
    b_array = np.array(b, dtype=np.float64)
    return np.dot(a_array, b_array) / (
        np.linalg.norm(a_array) * np.linalg.norm(b_array)
    )

def parse_response_to_type(response: str, result_type):
    """Parse the model response to the expected type."""
    response = response.strip()
    
    if result_type == str:
        return response
    elif result_type == list or result_type == List[str]:
        # Try to extract a list from the response
        try:
            # Look for bullet points or numbered lists
            lines = response.split('\n')
            items = []
            for line in lines:
                line = line.strip()
                if line.startswith(('- ', '* ', '• ')) or re.match(r'^\d+\.', line):
                    # Remove bullet point or number
                    clean_line = re.sub(r'^[-*•\d.]\s*', '', line).strip()
                    if clean_line:
                        items.append(clean_line)
            
            if items:
                return items
            
            # If no structured list found, try JSON parsing
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
                
        except (json.JSONDecodeError, Exception):
            pass
        
        # Fallback: split by newlines and filter
        return [line.strip() for line in response.split('\n') if line.strip()]
    
    elif result_type == dict:
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # Fallback to empty dict
        return {}
    
    # For other types, return the string and let the caller handle it
    return response

async def do_ai(
    user_prompt: str,
    system_prompt: str,
    result_type,
    deps=None,
):
    """Generate AI response using local Hugging Face model."""
    global _text_generator
    
    # Ensure model is loaded
    if _text_generator is None:
        await initialize_llm_model()
    
    # Format the prompt for the model
    formatted_prompt = f"""<|system|>
{system_prompt}
<|user|>
{user_prompt}
<|assistant|>
"""

    # Generate response using the local model
    loop = asyncio.get_event_loop()
    config = MODEL_CONFIG[MEMORY_CONFIG]
    
    def generate_text():
        outputs = _text_generator(
            formatted_prompt,
            max_new_tokens=config["max_new_tokens"],
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            return_full_text=False,
            pad_token_id=_llm_tokenizer.eos_token_id
        )
        return outputs[0]["generated_text"]
    
    # Run generation in executor to avoid blocking
    response = await loop.run_in_executor(None, generate_text)
    
    # Parse response to expected type
    parsed_response = parse_response_to_type(response, result_type)
    
    return parsed_response

async def get_embedding(text: str) -> list[float]:
    """Get embedding for text using local model."""
    global _embedding_model
    
    # Ensure model is loaded
    if _embedding_model is None:
        await initialize_embedding_model()
    
    # Generate embedding in executor to avoid blocking
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(
        None,
        lambda: _embedding_model.encode(text, normalize_embeddings=True)
    )
    
    return embedding.tolist()

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

async def get_db_pool() -> asyncpg.Pool:
    """Get database connection pool."""
    async def init(conn):
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await register_vector(conn)

    pool = await asyncpg.create_pool(DB_DSN, init=init)
    return pool

# ============================================================================
# DEPENDENCIES CLASS
# ============================================================================

@dataclass
class Deps:
    """Dependencies for memory operations."""
    pool: asyncpg.Pool

# ============================================================================
# MEMORY NODE MODEL
# ============================================================================

class MemoryNode(BaseModel):
    """Represents a single memory with content and metadata."""
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
    async def from_content(cls, content: str):
        """Create a memory node from content."""
        embedding = await get_embedding(content)
        return cls(content=content, embedding=embedding)

    async def save(self, deps: Deps):
        """Save memory node to database."""
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

    async def merge_with(self, other: 'MemoryNode', deps: Deps):
        """Merge this memory with another similar memory."""
        # Use local LLM to combine the memories
        self.content = await do_ai(
            f"Memory 1: {self.content}\n\nMemory 2: {other.content}",
            "Combine these two related memories into a single, coherent memory. Keep all important information but remove redundancy.",
            str,
            deps,
        )
        
        # Update metadata
        self.importance += other.importance
        self.access_count += other.access_count
        
        # Average the embeddings
        self.embedding = [(a + b) / 2 for a, b in zip(self.embedding, other.embedding)]
        
        # Generate summary using local LLM
        self.summary = await do_ai(
            self.content, 
            "Summarize this memory in one clear, concise sentence.",
            str,
            deps
        )
        
        # Save updated memory
        await self.save(deps)
        
        # Delete the merged memory
        if other.id is not None:
            await delete_memory(other.id, deps)

    def get_effective_importance(self):
        """Calculate effective importance including access count."""
        return self.importance * (1 + math.log(self.access_count + 1))

# ============================================================================
# MEMORY MANAGEMENT FUNCTIONS
# ============================================================================

async def delete_memory(memory_id: int, deps: Deps):
    """Delete a memory from the database."""
    async with deps.pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)

async def find_similar_memories(embedding: list[float], deps: Deps) -> list[MemoryNode]:
    """Find memories similar to the given embedding."""
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
    """Update importance scores based on relevance to new memory."""
    async with deps.pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, importance, access_count, embedding FROM memories"
        )
        
        for row in rows:
            memory_embedding = row["embedding"]
            similarity = cosine_similarity(user_embedding, memory_embedding)
            
            if similarity > SIMILARITY_THRESHOLD:
                # Reinforce similar memories
                new_importance = row["importance"] * REINFORCEMENT_FACTOR
                new_access_count = row["access_count"] + 1
            else:
                # Decay less relevant memories
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
    """Remove least important memories when limit exceeded."""
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

async def add_memory(content: str, deps: Deps) -> str:
    """Add a new memory to the system."""
    # Create new memory
    new_memory = await MemoryNode.from_content(content)
    await new_memory.save(deps)

    # Find and merge with similar memories
    similar_memories = await find_similar_memories(new_memory.embedding, deps)
    
    for memory in similar_memories:
        if memory.id != new_memory.id:
            similarity = cosine_similarity(new_memory.embedding, memory.embedding)
            if similarity > SIMILARITY_THRESHOLD:
                await new_memory.merge_with(memory, deps)

    # Update importance scores
    await update_importance(new_memory.embedding, deps)
    
    # Prune excess memories
    await prune_memories(deps)

    return f"Remembered: {content}"

async def display_memory_tree(deps: Deps) -> str:
    """Display current memories in order of importance."""
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
    
    if not rows:
        return "No memories stored yet."
    
    result = "Current Memories:\n"
    for i, row in enumerate(rows, 1):
        effective_importance = row["importance"] * (
            1 + math.log(row["access_count"] + 1)
        )
        summary = row["summary"] or row["content"]
        result += f"{i}. {summary} (Importance: {effective_importance:.2f})\n"
    
    return result

# ============================================================================
# MCP SERVER SETUP
# ============================================================================

# Initialize MCP server
mcp = FastMCP(
    "memory",
    dependencies=[
        "pydantic-ai",
        "asyncpg", 
        "numpy",
        "pgvector",
        "transformers",
        "torch",
        "accelerate",
        "bitsandbytes",
        "sentence-transformers"
    ],
)

# ============================================================================
# MCP TOOLS
# ============================================================================

@mcp.tool()
async def remember(
    contents: list[str] = Field(
        description="List of observations or memories to store"
    ),
) -> str:
    """Store new memories in the system."""
    deps = Deps(pool=await get_db_pool())
    try:
        results = await asyncio.gather(*[add_memory(content, deps) for content in contents])
        return "\n".join(results)
    finally:
        await deps.pool.close()

@mcp.tool()
async def read_profile() -> str:
    """Retrieve and display current memories."""
    deps = Deps(pool=await get_db_pool())
    try:
        profile = await display_memory_tree(deps)
        return profile
    finally:
        await deps.pool.close()

@mcp.tool()
async def enhance_with_memories(conversation_context: str) -> dict:
    """
    Automatically retrieve relevant memories and suggest new ones to store.
    """
    deps = Deps(pool=await get_db_pool())
    try:
        # Get conversation embedding
        context_embedding = await get_embedding(conversation_context)
        
        # Find relevant memories
        relevant_memories = await find_similar_memories(context_embedding, deps)
        
        # Extract potential new memories using local LLM
        new_memory_candidates = await do_ai(
            conversation_context,
            "Extract 3-5 key personal facts or preferences from this conversation that would be valuable to remember long-term. Return as a bullet-point list.",
            list,
            deps
        )
        
        return {
            "relevant_memories": [m.content for m in relevant_memories[:3]],
            "new_memory_candidates": new_memory_candidates
        }
    finally:
        await deps.pool.close()

@mcp.tool()
async def forget_memory(memory_description: str) -> str:
    """Remove memories that match the given description."""
    deps = Deps(pool=await get_db_pool())
    try:
        # Get embedding for the description
        description_embedding = await get_embedding(memory_description)
        
        # Find similar memories
        similar_memories = await find_similar_memories(description_embedding, deps)
        
        deleted_count = 0
        for memory in similar_memories:
            similarity = cosine_similarity(description_embedding, memory.embedding)
            if similarity > 0.8:  # High threshold for deletion
                await delete_memory(memory.id, deps)
                deleted_count += 1
        
        return f"Deleted {deleted_count} memories matching: {memory_description}"
    finally:
        await deps.pool.close()

# ============================================================================
# MCP PROMPTS
# ============================================================================

@mcp.prompt()
def conversation_with_memory() -> list:
    """A prompt that automatically incorporates memories."""
    return [
        UserMessage("""
        Before responding, use the read_profile tool to check what you remember about me.
        Incorporate this context naturally in your responses without explicitly 
        mentioning that you're using stored memories. 
        
        At the end of our conversation, use the enhance_with_memories tool to identify 
        any important new information that should be remembered.
        """)
    ]

@mcp.prompt()
def remember_conversation(conversation_text: str) -> list:
    """Extract and store key information from a conversation."""
    return [
        UserMessage(f"""
        Please analyze this conversation and use the remember tool to store 3-5 key 
        facts or preferences that would be useful to remember about the user:
        
        {conversation_text}
        
        Focus on personal details, preferences, goals, and important context that 
        would help in future conversations.
        """)
    ]

@mcp.prompt()
def memory_maintenance() -> list:
    """Prompt for reviewing and maintaining memories."""
    return [
        UserMessage("""
        Use the read_profile tool to review current memories. Then suggest which 
        memories might need updating, which could be forgotten, and what additional 
        context might be helpful to remember.
        """)
    ]

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

async def initialize_database():
    """Initialize the database and create necessary tables."""
    print("Initializing memory system...")
    
    # Initialize models first
    print("Loading AI models...")
    await initialize_embedding_model()
    await initialize_llm_model()
    print("✓ All AI models loaded successfully")
    
    # Create database
    print("Setting up database...")
    pool = await asyncpg.create_pool(
        "postgresql://postgres:postgres@localhost:54320/postgres"
    )
    try:
        async with pool.acquire() as conn:
            # Terminate existing connections to memory_db
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

    # Create tables with proper vector dimensions
    pool = await asyncpg.create_pool(DB_DSN)
    try:
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await register_vector(conn)

            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    summary TEXT,
                    importance REAL NOT NULL,
                    access_count INT NOT NULL,
                    timestamp DOUBLE PRECISION NOT NULL,
                    embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
                    USING hnsw (embedding vector_l2_ops);
            """)
    finally:
        await pool.close()
    
    print("✓ Database initialized successfully")
    print("✓ Memory system ready!")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Initialize database and models
    asyncio.run(initialize_database())
    
    # Start the MCP server
    print("\nStarting MCP server...")
    print("Use 'mcp install memory.py' to add this to Claude Desktop")
    mcp.run()
```

### Starting the Memory System

To set up and start your brain-inspired memory system:

1. **Initialize the database and models**:
   ```bash
   # Make sure your virtual environment is activated
   source .venv/bin/activate
   
   # Initialize the system (this will download models and set up the database)
   python memory.py
   ```

2. **Install in Claude Desktop**:
   ```bash
   # Install the memory system in Claude Desktop
   mcp install memory.py --name "Brain Memory System"
   ```

3. **Start using with Claude**:
   Once installed, you can interact with the memory system in Claude Desktop:
   
   - **Store memories**: "Remember that I prefer dark roast coffee and work remotely from home"
   - **Check memories**: "What do you remember about me?"
   - **Enhanced conversations**: Use the conversation_with_memory prompt for automatic memory integration

The system will automatically:
- Download and cache the AI models locally (first run will take longer)
- Merge similar memories to avoid redundancy
- Reinforce frequently accessed memories
- Gradually forget less important information
- Provide natural, context-aware conversations based on stored memories

### Hardware Requirements

- **Minimum (8-16GB RAM)**: Uses Phi-3-mini with 4-bit quantization
- **Recommended (16-32GB RAM)**: Uses larger models with 8-bit quantization  
- **Optimal (32GB+ RAM)**: Full precision models for best quality

The system automatically adapts to your hardware by adjusting the `MEMORY_CONFIG` setting in the code.

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
