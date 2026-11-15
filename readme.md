# 👁️ Mim (Mimir) — Local-First Tactical Desktop Memory & MCP Server

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-00E5FF.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![MCP Compatible](https://img.shields.io/badge/MCP-FastMCP%20Server-FFB000.svg?style=flat-square)](https://modelcontextprotocol.io/)
[![Storage](https://img.shields.io/badge/Storage-SQLite%20WAL%20%2B%20LanceDB-10B981.svg?style=flat-square)]()
[![Privacy](https://img.shields.io/badge/Zero--Trust-PII%20%2B%20Pixel%20Redaction-FF3B30.svg?style=flat-square)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

**Mim** is a local-first, zero-bloat temporal desktop flight recorder and **Model Context Protocol (MCP)** server built for software engineers and AI coding agents (**Google Antigravity**, **Cursor**, **Claude Desktop**, **Windsurf**).

AI coding agents are normally blind to everything happening outside your IDE—they cannot see the stack trace in your Chrome DevTools tab, the failed layout on `localhost:3000`, the documentation you read two hours ago, or the terminal window that just crashed. **Mim bridges that gap** while giving you a full-screen **Time-Scrubber Overlay (`Ctrl + Shift + Space`)** with instant point-and-click text extraction, 3-layer multi-day AI recall, and automatic workspace rollback snapshots.

---

## ⚡ Key Engineering Highlights

* **🔌 Native MCP Server for IDE Agents (`mim_mcp_server.py`):** Exposes 8 zero-latency tools over `stdio` (backed by a single-writer HTTP bridge on `127.0.0.1:8420` with automatic read-only SQLite fallback) so your IDE agent can query recent screen errors, search historical documentation, and take workspace backups before performing risky refactors.
* **🛡️ Zero-Trust Privacy & Pixel Blackout Redaction:**
  * **Process & Title Guard:** Automatically pauses ingestion whenever password managers (`Bitwarden`, `1Password`, `KeePass`), Incognito browsers, `.env` files, or personal messaging apps (`WhatsApp`, `Telegram`, `Discord`, `Gmail`) are focused.
  * **Dual-Layer Secret Scrubbing:** `ZeroTrustSanitizer` strips API keys (`sk-...`, `AIza...`, AWS, GitHub PATs, JWTs) and high-entropy secrets from scraped text *and* automatically burns a solid `#000000` **Pixel Blackout Box** directly over those exact UIA `[left, top, right, bottom]` coordinates on the `.webp` screenshot before it stays on disk.
  * **Capture Cloaking (`WDA_EXCLUDEFROMCAPTURE`):** Mim's own floating Eye Sentinel and full-screen Scrubber Overlay are cloaked at the Win32 DWM level and filtered by PID so Mim never records itself.
* **💾 Zero-Bloat 3-Tier Storage Engine (~220 MB Hard Disk Cap):**
  * **Hot Tier (Newest 150 Frames):** 100% native resolution WebP (`quality=85`) for crisp 1:1 active-window ROI crops during live multimodal AI analysis.
  * **Warm Tier (Up to 5,000 Frames / ~3–4 Days):** Re-compressed in place (`quality=50`, ~35 KB/frame) for smooth visual timeline scrubbing.
  * **Cold "Milestone" Tier (Months/Years of History):** Keeps 100% of your SQLite FTS5 searchable text and UIA bounding-box geometry forever, retains **1 real Milestone `.webp` thumbnail per unique window session**, and unlinks redundant intermediate images. When scrubbing across pruned intermediate frames from months ago, Mim dynamically renders a **Tactical Matrix Blueprint Wireframe** from the saved UIA nodes.
* **🔍 Hybrid Time-Decayed RAG (TD-RRF):** Combines **SQLite FTS5 BM25** lexical search with **batched LanceDB (`BAAI/bge-small-en-v1.5` ONNX)** semantic vector search using Time-Decayed Reciprocal Rank Fusion:
  $$\text{Score}(d) = \left( \frac{w_{\text{lex}}}{k + \text{rank}_{\text{lex}}(d)} + \frac{w_{\text{sem}}}{k + \text{rank}_{\text{sem}}(d)} \right) \times e^{-\lambda \cdot \Delta t_{\text{hours}}}$$
* **🧠 3-Layer Gemini 2.5 Flash Memory Assistant:** Answers both broad multi-day questions (*"What have I been working on over the last 3 days?"*) and needle-in-a-haystack queries (*"What was the hex color or library I looked at yesterday?"*) by combining:
  1. **Layer 1:** Multi-Day Hourly SQLite Activity Rollup (apps, window titles, and frame counts grouped by hour).
  2. **Layer 2:** All-Time Full-Database FTS5 Keyword Hits.
  3. **Layer 3:** Detailed Recent Screen Transcript + 1:1 High-DPI Active-Window Crops.

---

## 🏗️ System Architecture

```text
+-----------------------------------------------------------------------------------+
|                        WIN32 DESKTOP SUBSTRATE (Per-Monitor V2 DPI)               |
+-----------------------------------------------------------------------------------+
         |                                      |
         | 2 FPS MSS Grab + dHash Gate          | Throttled Win32 UIA Tree Walk (2.5s)
         v                                      v
+----------------------------------+   +--------------------------------------------+
|   VisionGrabber (Delta Filter)   |   |  UIAScraper + ZeroTrustSanitizer (Regex/H) |
|  Drops ~85% static idle frames   |   |  Extracts [L,T,R,B] boxes & redacts PII    |
+----------------------------------+   +--------------------------------------------+
         |                                      |
         +------------------+-------------------+
                            |
                            v
+-----------------------------------------------------------------------------------+
|                 IngestionCoordinator (Pixel Blackout Redaction)                   |
|   Burns #000000 censor bars onto .webp over any [REDACTED_*] UIA bounding boxes   |
+-----------------------------------------------------------------------------------+
         |                                      |
         | Single-Writer WAL Transaction        | Async Batched Queue (Flush @ 20)
         v                                      v
+----------------------------------+   +--------------------------------------------+
|  SQLiteManager (mim_temporal.db) |   |   LanceDBManager (FastEmbed ONNX 384-d)    |
|  - frames, ui_nodes, fts_ui_text |   |   - Semantic vector index + sync pruning   |
|  - Hot / Warm / Cold Milestone   |   +--------------------------------------------+
+----------------------------------+                    |
         |                                              |
         +----------------------+-----------------------+
                                |
                                v
+-----------------------------------------------------------------------------------+
|            HybridRAGEngine (TD-RRF) + VSSManager (Dirty-Aware Snapshots)          |
+-----------------------------------------------------------------------------------+
         |                                              |
         | FastAPI Local Daemon (127.0.0.1:8420)        | PyQt6 HUD (Ctrl+Shift+Space)
         v                                              v
+----------------------------------+   +--------------------------------------------+
|   mim_mcp_server.py (FastMCP)    |   |        ScrubberOverlayWindow + Eye         |
|  8 Tools for Antigravity/Cursor  |   |  Click-to-copy boxes, Blueprint Wireframes |
+----------------------------------+   +--------------------------------------------+