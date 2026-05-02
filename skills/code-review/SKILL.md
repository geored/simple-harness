---
name: code-review
description: Reviews source code files for bugs, security issues, style problems, and best practices. Use when asked to review, audit, or check code quality.
allowed-tools: read_file list_files
---

## Instructions

1. Read the target file(s) using read_file
2. Analyze the code for:
   - **Bugs**: logic errors, off-by-one, null handling, race conditions
   - **Security**: injection, hardcoded secrets, unsafe operations
   - **Style**: naming, formatting, readability
   - **Performance**: unnecessary allocations, O(n²) where O(n) is possible
3. Cite specific line numbers or code snippets for each finding
4. Provide an overall rating: Good / Needs Work / Critical Issues
5. List actionable fix suggestions with priority (P0/P1/P2)
