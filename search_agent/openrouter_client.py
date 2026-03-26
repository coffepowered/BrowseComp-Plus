"""
OpenRouter search agent client.

Uses the OpenAI Chat Completions API (compatible with OpenRouter) with
function-calling to drive a multi-turn search conversation.

Example
-------
    OPENROUTER_API_KEY=sk-or-... python search_agent/openrouter_client.py \
        --model qwen/qwen3-235b-a22b:free \
        --searcher-type bm25 --index-path indexes/bm25/ \
        --output-dir runs/bm25/qwen3-235b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv
from prompts import format_query
from rich import print as rprint
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import extract_retrieved_docids_from_result

from searcher.searchers import SearcherType

load_dotenv()

# ── OpenRouter base URL ─────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Default model – free Qwen-3 MoE on OpenRouter ──────────────────────
DEFAULT_MODEL = "qwen/qwen3-235b-a22b:free"


# ╭───────────────────────────────────────────────────────────────────────╮
# │  Tool handler (Chat Completions flavour)                              │
# ╰───────────────────────────────────────────────────────────────────────╯
class SearchToolHandler:
    """Wraps a searcher and exposes Chat-Completions-style tool definitions."""

    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # -- Chat Completions tool schema ------------------------------------
    def get_tool_definitions(self) -> list[dict]:
        """Return tool definitions in OpenAI **Chat Completions** format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": self.searcher.search_description(self.k),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_document",
                        "description": self.searcher.get_document_description(),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "docid": {
                                    "type": "string",
                                    "description": "Document ID to retrieve",
                                }
                            },
                            "required": ["docid"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return tools

    # -- Tool execution --------------------------------------------------
    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "search":
            return self._search(arguments["query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _search(self, query: str) -> str:
        candidates = self.searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    cand["snippet"] = self.tokenizer.decode(
                        tokens[: self.snippet_max_tokens], skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            entry: dict = {"docid": cand["docid"], "snippet": cand["snippet"]}
            if cand.get("score") is not None:
                entry["score"] = cand["score"]
            results.append(entry)

        return json.dumps(results, indent=2)

    def _get_document(self, docid: str) -> str:
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)


# ╭───────────────────────────────────────────────────────────────────────╮
# │  Conversation loop (Chat Completions + function calling)              │
# ╰───────────────────────────────────────────────────────────────────────╯
def run_conversation_with_tools(
    client: openai.OpenAI,
    messages: list[dict],
    model: str,
    tools: list[dict],
    tool_handler: SearchToolHandler,
    max_tokens: int = 10_000,
    max_iterations: int = 100,
    temperature: float | None = None,
    top_p: float | None = None,
    verbose: bool = False,
) -> tuple[list[dict], dict[str, int], dict, str]:
    """
    Drive a multi-turn tool-calling conversation via Chat Completions API.

    Returns
    -------
    messages : list[dict]
        Full message history (including tool calls / results).
    tool_usage : dict[str, int]
        Counts of each tool invoked.
    cumulative_usage : dict
        Aggregated token usage across all turns.
    status : str
        ``"completed"`` or ``"incomplete"``.
    """

    tool_usage: dict[str, int] = {}
    cumulative_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    for _iteration in range(max_iterations):
        request_kwargs: dict = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p

        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if verbose:
                rprint(f"[red]API error:[/red] {exc}")
            # Retry on next iteration
            continue

        # Accumulate usage
        if response.usage:
            cumulative_usage["input_tokens"] += response.usage.prompt_tokens or 0
            cumulative_usage["output_tokens"] += response.usage.completion_tokens or 0
            cumulative_usage["total_tokens"] += response.usage.total_tokens or 0

        choice = response.choices[0]
        assistant_msg = choice.message

        # Append assistant reply to history
        messages.append(assistant_msg.model_dump(mode="python"))

        # No tool calls → conversation finished
        if not assistant_msg.tool_calls:
            return messages, tool_usage, cumulative_usage, "completed"

        # Process each tool call
        for tc in assistant_msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
                result = tool_handler.execute_tool(fn_name, fn_args)
            except Exception as exc:
                result = json.dumps({"error": f"Error executing {fn_name}: {exc}"})

            tool_usage[fn_name] = tool_usage.get(fn_name, 0) + 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return messages, tool_usage, cumulative_usage, "incomplete"


# ╭───────────────────────────────────────────────────────────────────────╮
# │  Persistence (normalised output matching repo conventions)            │
# ╰───────────────────────────────────────────────────────────────────────╯
def _persist_response(
    out_dir: str,
    model: str,
    messages: list[dict],
    tool_usage: dict[str, int],
    cumulative_usage: dict,
    status: str,
    *,
    query_id: str | None = None,
    max_tokens: int | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    # Build normalised result list
    normalized_results: list[dict] = []
    tool_call_outputs: dict[str, str] = {}  # tool_call_id → output

    # First pass – collect tool outputs
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tool_call_outputs[msg.get("tool_call_id", "")] = msg.get("content", "")

    # Second pass – build normalised items
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            # Text content
            content = msg.get("content")
            if content:
                normalized_results.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": content,
                    }
                )
            # Tool calls
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
                normalized_results.append(
                    {
                        "type": "tool_call",
                        "tool_name": fn.get("name"),
                        "arguments": fn.get("arguments"),
                        "output": tool_call_outputs.get(tc_id),
                    }
                )

    normalized_record = {
        "metadata": {
            "model": model,
            "output_dir": str(out_dir),
            "max_tokens": max_tokens,
        },
        "query_id": query_id,
        "tool_call_counts": tool_usage,
        "usage": cumulative_usage,
        "status": status,
        "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
        "result": normalized_results,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(str(out_dir), f"run_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(normalized_record, f, indent=2, default=str)

    print(f"Saved response to {filename} | tool call counts: {tool_usage}")


# ╭───────────────────────────────────────────────────────────────────────╮
# │  Dataset processing                                                    │
# ╰───────────────────────────────────────────────────────────────────────╯
def _process_tsv_dataset(
    tsv_path: str,
    client: openai.OpenAI,
    args,
    tool_handler: SearchToolHandler,
):
    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()

    # Load queries
    queries: list[tuple[str, str]] = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries.append((row[0].strip(), row[1].strip()))

    # Skip already-processed queries
    processed_ids: set[str] = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved:
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue

    remaining = [(qid, qt) for qid, qt in queries if qid not in processed_ids]
    print(
        f"Processing {len(remaining)} remaining queries "
        f"(skipping {len(processed_ids)}) from {dataset_path} …"
    )

    tools = tool_handler.get_tool_definitions()
    completed_lock = threading.Lock()
    completed_count = [0]

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        formatted = format_query(qtext, args.query_template)
        messages: list[dict] = [{"role": "user", "content": formatted}]

        try:
            messages, tool_usage, cumulative_usage, status = (
                run_conversation_with_tools(
                    client,
                    messages,
                    model=args.model,
                    tools=tools,
                    tool_handler=tool_handler,
                    max_tokens=args.max_tokens,
                    max_iterations=args.max_iterations,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    verbose=args.verbose,
                )
            )

            if status == "completed":
                with completed_lock:
                    completed_count[0] += 1
                    if pbar:
                        pbar.set_postfix(completed=completed_count[0])

            _persist_response(
                str(out_dir),
                args.model,
                messages,
                tool_usage,
                cumulative_usage,
                status,
                query_id=qid,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            print(f"[Error] Query id={qid} failed: {exc}")
            raise

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar)
    else:
        errors: list[str] = []
        with (
            ThreadPoolExecutor(max_workers=args.num_threads) as executor,
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar,
        ):
            futures = {
                executor.submit(_handle_single_query, qid, qtext, pbar): qid
                for qid, qtext in remaining
            }
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    future.result()
                except Exception as exc:
                    errors.append(f"Query {futures[future]}: {exc}")
        if errors:
            print(f"\n[Error] {len(errors)} queries failed:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)


# ╭───────────────────────────────────────────────────────────────────────╮
# │  CLI entry point                                                       │
# ╰───────────────────────────────────────────────────────────────────────╯
def main():
    parser = argparse.ArgumentParser(
        description="Run BrowseComp-Plus benchmark via OpenRouter (Chat Completions API with function calling).",
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
        "--max-tokens",
        type=int,
        default=10_000,
        help="Max tokens to generate per API call.",
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
        "--max-iterations",
        type=int,
        default=100,
        help="Max conversation rounds with function calls.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
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

    # Let the chosen searcher add its own args
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
    print(f"Processing single query: {args.query!r}")
    formatted = format_query(args.query, args.query_template)
    messages: list[dict] = [{"role": "user", "content": formatted}]
    tools = tool_handler.get_tool_definitions()

    messages, tool_usage, cumulative_usage, status = run_conversation_with_tools(
        client,
        messages,
        model=args.model,
        tools=tools,
        tool_handler=tool_handler,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        top_p=args.top_p,
        verbose=args.verbose,
    )

    _persist_response(
        args.output_dir,
        args.model,
        messages,
        tool_usage,
        cumulative_usage,
        status,
        max_tokens=args.max_tokens,
    )

    rprint(messages)


if __name__ == "__main__":
    main()
