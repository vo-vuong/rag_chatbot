"""
Agent-specific prompt templates.

Provides system prompts for agent workflows.
"""


def get_agent_system_prompt() -> str:
    """Get system prompt for agentic RAG workflow."""
    return """You are a helpful AI assistant with access to a knowledge base through the
        retrieve_documents tool.

When answering questions:
1. First, consider if you need to search the knowledge base
2. Use the retrieve_documents tool to find relevant information
3. Base your answers on the retrieved documents
4. If the documents don't contain the answer, clearly state this
5. Always cite your sources from the retrieved documents

Tool Usage:
- Use retrieve_documents when you need factual information from the uploaded documents
- The tool automatically routes to text or image collections
- You can specify search_type="text" or "image" for specific searches

Response Format:
- Be concise and accurate
- Respond in the same language as the user's question
- If referencing documents, mention the source file"""


def get_rag_context_prompt(context: str, query: str, chat_history: str = "") -> str:
    """
    Build RAG context prompt for final response generation.

    This is used after retrieval to format the final LLM prompt.
    """
    prompt_parts = []

    if chat_history:
        prompt_parts.append(f"Previous conversation:\n{chat_history}\n")

    prompt_parts.append(f"Context from knowledge base:\n{context}\n")
    prompt_parts.append(f"User question: {query}")

    return "\n".join(prompt_parts)


def format_retrieved_context(chunks: list, include_scores: bool = False) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Args:
        chunks: List of chunk dictionaries from retrieval tool
        include_scores: Whether to include relevance scores

    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant documents found in the knowledge base."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_file", "Unknown")
        content = chunk.get("content", "")
        page = chunk.get("page_number")

        header = f"[Document {i}]"
        if source != "Unknown":
            header += f" Source: {source}"
        if page:
            header += f", Page {page}"
        if include_scores:
            score = chunk.get("score", 0.0)
            header += f" (Score: {score:.2f})"

        context_parts.append(f"{header}\n{content}")

    return "\n\n".join(context_parts)
