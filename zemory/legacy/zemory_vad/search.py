"""Async background web search pipeline.

Flow: LLM uncertainty detection → extract query → DuckDuckGo → synthesize.
"""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI
from zemory_vad.config import SEARCH_MAX_RESULTS, SEARCH_MODEL


async def detect_uncertainty(
    client: AsyncOpenAI,
    assistant_text: str,
) -> bool:
    """Use LLM to classify whether the response indicates uncertainty."""
    resp = await client.chat.completions.create(
        model=SEARCH_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a classifier. Determine if the assistant's response "
                    "indicates it does NOT know the answer and needs to search. "
                    "Respond with ONLY 'yes' or 'no'."
                ),
            },
            {
                "role": "user",
                "content": f"Assistant response: {assistant_text}",
            },
        ],
        max_completion_tokens=500,
    )
    text = resp.choices[0].message.content or ""
    return text.strip().lower().startswith("y")


async def extract_query(
    client: AsyncOpenAI,
    user_transcript: str,
    assistant_text: str,
) -> str:
    """Extract a concise web search query from the conversation."""
    resp = await client.chat.completions.create(
        model=SEARCH_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract a concise web search query from this conversation. "
                    "Return ONLY the search query, nothing else. "
                    "Use the same language as the user's question."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User asked: {user_transcript}\n"
                    f"Assistant replied: {assistant_text}"
                ),
            },
        ],
        max_completion_tokens=500,
    )
    return (resp.choices[0].message.content or "").strip()


async def web_search(query: str) -> list[dict]:
    """Search DuckDuckGo (sync lib wrapped in to_thread)."""
    if not query:
        return []

    from duckduckgo_search import DDGS

    def _search() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=SEARCH_MAX_RESULTS))

    return await asyncio.to_thread(_search)


async def synthesize_results(
    client: AsyncOpenAI,
    query: str,
    results: list[dict],
) -> str:
    """Synthesize search results into a concise answer."""
    snippets = "\n".join(f"- {r['title']}: {r['body']}" for r in results)
    resp = await client.chat.completions.create(
        model=SEARCH_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Based on the search results below, provide a concise and "
                    "accurate answer to the query. Keep it under 3 sentences. "
                    "Respond in the same language as the query."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nSearch results:\n{snippets}",
            },
        ],
        max_completion_tokens=1000,
    )
    return (resp.choices[0].message.content or "").strip()


async def search_pipeline(
    client: AsyncOpenAI,
    user_transcript: str,
    assistant_text: str,
) -> str:
    """Full pipeline: extract query → web search → synthesize."""
    query = await extract_query(client, user_transcript, assistant_text)
    if not query:
        query = user_transcript  # fallback to user's original question
    results = await web_search(query)
    if not results:
        return f"'{query}'에 대한 검색 결과를 찾지 못했습니다."
    return await synthesize_results(client, query, results)
