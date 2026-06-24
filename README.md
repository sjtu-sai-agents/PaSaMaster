<div align="center">
  <h1>
    <img src="assets/logo.png" alt="PaSaMaster logo" height="48" style="vertical-align: middle;">
    PaSaMaster: Towards Self-Evolving Agentic Literature Retrieval
  </h1>

  <img src="assets/teaser.png" alt="PaSaMaster teaser" width="900">
</div>

## Overview

PaSaMaster is a self-evolving agentic literature retrieval system that produces relevance-scored paper rankings with evidence-grounded recommendations through iterative intent analysis, retrieval, and ranking.

## Try PaSaMaster

Use PaSaMaster online at: [https://scimaster.bohrium.com/chat/paper-search](https://scimaster.bohrium.com/chat/paper-search)

## PaSaMaster CLI for Agentic Literature Retrieval

PaSaMaster can also be used from the command line through [`scimaster-cli`](https://www.npmjs.com/package/scimaster-cli?activeTab=readme), a SciMaster literature-search CLI powered by PaSaMaster. It provides both a `sci` command for direct terminal searches and a `sci-mcp` Model Context Protocol (MCP) server that can be plugged into your own agents, Claude Code, Cursor, or any MCP-compatible client.

This makes PaSaMaster available as a retrieval tool inside agent workflows: the agent can call `search_papers` with a research query and receive a structured, high-quality set of relevant papers, including metadata, abstracts, URLs, and BibTeX entries. This is useful for building research assistants, review-writing agents, paper recommendation systems, and other literature-aware AI applications that need stronger recall and relevance ranking than simple keyword search.

Install and run:

```bash
npm install -g scimaster-cli
sci init
sci search "CRISPR gene editing 2024" --limit 20 --mode low
```

Use it without installation:

```bash
npx -y scimaster-cli@latest search "machine learning protein" --limit 20
```

Register it as an MCP tool for your own agent:

```json
{
  "mcpServers": {
    "scimaster": {
      "command": "npx",
      "args": ["-y", "scimaster-cli"]
    }
  }
}
```

The MCP server exposes:

| Tool | Description |
|------|-------------|
| `search_papers` | Search for relevant papers with `query`, `limit`, and `mode`, returning structured paper results for downstream agent reasoning. |

## Evaluation

Evaluated on the PaSaMaster Benchmark across 38 scientific disciplines, PaSaMaster substantially improves over traditional keyword retrieval and generative LLM baselines, achieving stronger relevance ranking, zero source hallucination, and competitive performance at a fraction of the computational cost.

<div align="center">
  <img src="assets/table.png" alt="PaSaMaster evaluation results" width="900">
</div>
