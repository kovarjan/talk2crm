from __future__ import annotations

# The test for recent_contacts_context resolution via crm_data_tool has been removed
# because crm_data_tool no longer exists. The crm_query_tool does not perform
# context-based name resolution — that logic is now handled upstream by the LLM
# using rag_search_tool before issuing crm_query_tool calls.
