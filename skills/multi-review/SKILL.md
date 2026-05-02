---
name: multi-review
description: Read all files in a directory and summarize each file's purpose. Use when asked to review or understand an entire project or directory.
allowed-tools: shell read_file
---

## Instructions

1. Use shell to find all files: find {directory} -type f -not -name '.*'
2. Read each file using read_file
3. For each file, write 2-3 sentences describing its purpose and key components
4. Provide an overall project summary at the end
