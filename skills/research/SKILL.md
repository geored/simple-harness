---
name: research
description: Search the web for a topic and provide a comprehensive summary with sources. Use when asked to research, investigate, or find information about any subject.
allowed-tools: web_search http_fetch
---

## Instructions

1. Use web_search to find relevant information about the given topic
2. Search with 2-3 different query variations for comprehensive coverage
3. If specific articles look promising, use http_fetch to get more detail
4. Summarize findings in a structured format:
   - **Key findings** (bullet points)
   - **Sources** (title + URL for each)
   - **Gaps** (note what couldn't be found or conflicting information)
