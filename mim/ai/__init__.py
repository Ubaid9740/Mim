"""AI integration subsystem for Mim: session summarization with Gemini and fallback heuristics."""

from mim.ai.gemini_summarizer import GeminiSummarizer

GeminiSessionSummarizer = GeminiSummarizer

__all__ = ["GeminiSummarizer", "GeminiSessionSummarizer"]
