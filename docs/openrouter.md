# Running with OpenRouter

[OpenRouter](https://openrouter.ai/) provides a single API gateway to hundreds
of models — including many **free** and very cheap options.  Because OpenRouter
exposes an OpenAI-compatible Chat Completions endpoint, we ship a dedicated
client (`search_agent/openrouter_client.py`) that works out of the box.

## Prerequisites

1. Follow the main [README](../README.md) to download the decrypted dataset,
   set up the environment, and download indexes.
2. Get an OpenRouter API key at <https://openrouter.ai/keys> (free sign-up).
3. Export the key:
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-…"
   ```
   Or add it to a `.env` file in the project root (the client loads `.env`
   automatically via `python-dotenv`).

## Recommended Models

| Model | OpenRouter ID | Notes |
|-------|--------------|-------|
| **Qwen3-235B-A22B** (free) | `qwen/qwen3-235b-a22b:free` | Free MoE thinking model by Alibaba. Good function-calling. Default. |
| **DeepSeek-V3-0324** | `deepseek/deepseek-chat-v3-0324` | Very cheap model by DeepSeek with strong reasoning. |
| **DeepSeek-R1-0528** | `deepseek/deepseek-r1-0528` | Cheap reasoning/thinking model by DeepSeek. |
| **GPT-4.1 mini** | `openai/gpt-4.1-mini` | Affordable OpenAI mini model. |
| **Gemini 2.5 Flash** | `google/gemini-2.5-flash-preview-05-20` | Cheap and fast. |

> **Tip:** Browse all available models at <https://openrouter.ai/models>.
> Append `:free` to any model ID that offers a free tier (e.g.
> `qwen/qwen3-235b-a22b:free`).

## Quick Start — BM25 Retriever

The simplest setup uses the pre-built BM25 index (no GPU needed for retrieval):

```bash
# 1. Download the BM25 index (if you haven't already)
bash scripts_build_index/download_indexes.sh

# 2. Run with the free Qwen3-235B model
python search_agent/openrouter_client.py \
    --model "qwen/qwen3-235b-a22b:free" \
    --searcher-type bm25 \
    --index-path indexes/bm25/ \
    --output-dir runs/bm25/qwen3-235b-openrouter \
    --num-threads 5
```

To run with DeepSeek-V3 instead:

```bash
python search_agent/openrouter_client.py \
    --model "deepseek/deepseek-chat-v3-0324" \
    --searcher-type bm25 \
    --index-path indexes/bm25/ \
    --output-dir runs/bm25/deepseek-v3-openrouter \
    --num-threads 5
```

## Qwen3-Embedding (Dense Retrieval)

Pair OpenRouter models with the Qwen3-Embedding FAISS index for dense retrieval.
This requires a GPU for the embedding model:

```bash
# Download the Qwen3-Embedding index (if you haven't already)
bash scripts_build_index/download_indexes.sh

python search_agent/openrouter_client.py \
    --model "qwen/qwen3-235b-a22b:free" \
    --searcher-type faiss \
    --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
    --model-name "Qwen/Qwen3-Embedding-8B" \
    --normalize \
    --output-dir runs/qwen3-8/qwen3-235b-openrouter \
    --num-threads 5
```

> You may swap `--model-name` and `--index-path` to smaller variants such as
> `Qwen/Qwen3-Embedding-4B` with `indexes/qwen3-embedding-4b/corpus.shard*.pkl`.

## Custom Retriever

You can also use a custom retriever (see [Custom Retriever](custom_retriever.md))
with the OpenRouter client:

```bash
python search_agent/openrouter_client.py \
    --model "qwen/qwen3-235b-a22b:free" \
    --searcher-type custom \
    --output-dir runs/custom/qwen3-235b-openrouter
```

## Single-Query Mode

For quick testing, pass a query string instead of the TSV file:

```bash
python search_agent/openrouter_client.py \
    --model "qwen/qwen3-235b-a22b:free" \
    --searcher-type bm25 \
    --index-path indexes/bm25/ \
    --query "What is the tallest building in the world?" \
    --output-dir runs/test
```

## Evaluation

Evaluate results using the standard evaluation script:

```bash
python scripts_evaluation/evaluate_run.py \
    --input_dir runs/bm25/qwen3-235b-openrouter \
    --tensor_parallel_size 1
```

> Replace `--tensor_parallel_size` with the number of GPUs available for the
> Qwen3-32B judge model.

## All CLI Options

```text
python search_agent/openrouter_client.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen/qwen3-235b-a22b:free` | OpenRouter model identifier |
| `--query` | `topics-qrels/queries.tsv` | Query string or TSV file path |
| `--searcher-type` | *(required)* | `bm25`, `faiss`, `reasonir`, or `custom` |
| `--output-dir` | `runs/bm25/openrouter` | Where to save run JSON files |
| `--max-tokens` | `10000` | Max tokens per API response |
| `--query-template` | `QUERY_TEMPLATE_NO_GET_DOCUMENT` | Prompt template |
| `--temperature` | *(model default)* | Sampling temperature |
| `--top_p` | *(model default)* | Nucleus sampling top-p |
| `--num-threads` | `1` | Parallel query threads |
| `--max-iterations` | `100` | Max tool-calling rounds |
| `--snippet-max-tokens` | `512` | Tokens per snippet |
| `--k` | `5` | Top-k results per search |
| `--get-document` | `false` | Also enable the `get_document` tool |
| `--verbose` | `false` | Verbose logging |
