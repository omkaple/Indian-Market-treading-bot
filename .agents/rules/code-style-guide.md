# Code Style Guide

Adhere to the following programming styles and code quality rules:

## 1. PEP 8 Standards & Variable Typing
- Write clean, self-documenting python code.
- Always use type annotations for function parameters and return values (e.g., `def my_func(val: int) -> float:`).
- Document all classes and functions with concise descriptive docstrings.

## 2. Heavy Operations & Containment
- Wrap all network-bound API calls (HTTP requests to Angel One, WebSocket feeds, and Ollama endpoints) in explicit `try-except` containment blocks.
- Wrap all local database (MongoDB) writes and lookups in robust `try-except` blocks.
- Log error messages with tracebacks when errors are caught.

## 3. Standard Logging
- Do not use print statements for operational logging in production modules.
- Use Python's standard `logging` library with format: `%(asctime)s [%(levelname)s] %(message)s`.
