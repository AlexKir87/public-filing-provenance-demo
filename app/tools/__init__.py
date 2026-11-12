"""Read-only tools of the agent. Step 5 registers them on the Pydantic AI agent."""

from app.tools.compute import compute_ratio
from app.tools.edgar_tools import list_statement_lines, lookup_fact, search_filing_text
from app.tools.schemas import (
    ComputeRatioInput,
    ListStatementLinesInput,
    LookupFactInput,
    SearchFilingTextInput,
)

TOOL_NAMES = ("lookup_fact", "list_statement_lines", "search_filing_text", "compute_ratio")

__all__ = [
    "TOOL_NAMES",
    "ComputeRatioInput",
    "ListStatementLinesInput",
    "LookupFactInput",
    "SearchFilingTextInput",
    "compute_ratio",
    "list_statement_lines",
    "lookup_fact",
    "search_filing_text",
]
