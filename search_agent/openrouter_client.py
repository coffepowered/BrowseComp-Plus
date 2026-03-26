"""
OpenRouter search agent client.

Uses the OpenRouter Responses API (beta), which is a drop-in replacement for
OpenAI's Responses API.  All core logic (tool handling, conversation loop,
persistence) is reused from ``openai_client.py``; this file only overrides the
client setup (base URL + API key) and CLI defaults.

Reference: https://openrouter.ai/docs/api/reference/responses/overview

Example
-------
    OPENROUTER_API_KEY=sk-or-... python search_agent/openrouter_client.py \\
        --model qwen/qwen3-235b-a22b:free \\
        --searcher-type bm25 --index-path indexes/bm25/ \\
        --output-dir runs/bm25/qwen3-235b
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv
from rich import print as rprint

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Re-use all core logic from the OpenAI client
from openai_client import (
    SearchToolHandler,
    build_request,
    run_conversation_with_tools,
    _persist_response,
    _process_tsv_dataset,
)
from prompts import format_query
from searcher.searchers import SearcherType

load_dotenv()

# ── OpenRouter base URL ─────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Default model – free Qwen-3 MoE on OpenRouter ──────────────────────
DEFAULT_MODEL = "qwen/qwen3-235b-a22b:free"


def main():
    parser = argparse.ArgumentParser(
        description="Run BrowseComp-Plus benchmark via OpenRouter (Responses API).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--query",
        default="topics-qrels/queries.tsv",
        help="Single query string, or path to a TSV file (id<TAB>query).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenRouter model identifier.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10_000,
        help="Max tokens to generate per API call.",
    )
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort level (for reasoning-capable models).",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/bm25/openrouter",
        help="Directory to store run JSON files.",
    )
    parser.add_argument(
        "--query-template",
        choices=[
            "QUERY_TEMPLATE",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
        ],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
        help="Prompt template to wrap queries in.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (model default if not set).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling top-p (model default if not set).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Parallel threads for dataset mode.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per query on transient errors.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Max conversation rounds with function calls.",
    )

    # ── Searcher configuration ──────────────────────────────────────────
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Tokens per document snippet (Qwen3-0.6B tokenizer).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k search results per query.",
    )
    parser.add_argument(
        "--get-document",
        action="store_true",
        help="Also register the get_document tool.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for private datasets/models.",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        help="Hugging Face cache directory.",
    )

    # Let the chosen searcher add its own CLI args
    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    # ── Environment setup ───────────────────────────────────────────────
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Get a free key at https://openrouter.ai/keys"
        )

    # Point the standard OpenAI client at the OpenRouter Responses API
    client = openai.OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

    # ── Build searcher & tool handler ───────────────────────────────────
    searcher = searcher_class(args)
    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
    )

    tools_registered = ["search"]
    if args.get_document:
        tools_registered.append("get_document")
    print(
        f"Search agent started — model={args.model}, "
        f"searcher={searcher.search_type}, tools={tools_registered}"
    )

    # ── TSV dataset mode or single query ────────────────────────────────
    if isinstance(args.query, str):
        qstr = args.query.strip()
        if qstr.lower().endswith(".tsv"):
            potential_path = Path(qstr)
            try:
                if potential_path.is_file():
                    print(f"Processing TSV dataset {potential_path}")
                    _process_tsv_dataset(str(potential_path), client, args, tool_handler)
                    return
            except OSError:
                pass

    # Single query
    request_body = build_request(
        query=args.query,
        model=args.model,
        max_tokens=args.max_tokens,
        tool_handler=tool_handler,
        system_prompt=args.system,
        reasoning_effort=args.reasoning_effort,
        query_template=args.query_template,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("Sending request to OpenRouter Responses API with function calling...")
    response, combined_output, cumulative_usage, tool_outputs = (
        run_conversation_with_tools(
            client, request_body, tool_handler, args.max_iterations
        )
    )

    _persist_response(
        args.output_dir,
        request_body,
        response,
        combined_output,
        cumulative_usage,
        tool_outputs,
        query_id=None,
    )

    rprint(response)


if __name__ == "__main__":
    main()
