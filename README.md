# RAG Knowledge Assistant

A retrieval-augmented generation (RAG) assistant built with Python.

This is project #4 in the `ai-engineering` series. It ties together the LLM from project #1 and the semantic search from project #3 to build a system that can answer questions using a specific set of documents. Just as importantly, it includes an automated evaluation pipeline to measure how well the system actually works.

## Architecture

**The RAG Pipeline:**

```text
User Question → Query Embedding → Semantic Search → Top-K Chunks → Context Builder → LLM → Grounded Answer
```

**The Evaluation Pipeline:**

```text
(RAG Output + Retrieved Context) → LLM Judge → Structured Evaluation (Grounded, Relevant, Correct, Score)
```

## Features

- Full RAG pipeline (ingestion, search, generation, and source attribution)
- Automated retrieval evaluation (Accuracy and Mean Reciprocal Rank)
- LLM-as-a-judge answer evaluation (checks for grounding, relevance, and correctness)
- Pydantic-structured evaluation outputs
- Configurable `top-k` retrieval to test how context depth affects answers

## Tech Stack

- Python
- OpenRouter
- Sentence Transformers (from Project 3)
- NumPy
- Pydantic

## Project Structure

```text
04-rag-knowledge-assistant/
├── data/
│   ├── documents/       # Source text files (python.txt, sql.txt, etc.)
│   └── index/           # Generated embeddings and metadata (gitignored)
├── evaluation/
│   ├── questions.json   # Test questions and expected sources
│   ├── evaluate_retrieval.py
│   ├── evaluate_answers.py
│   ├── judge.py         # The LLM-as-a-judge logic
│   ├── schemas.py       # Pydantic models for structured eval output
│   └── results/         # Where eval reports get saved
├── documents.py         # Chunking and loading
├── embeddings.py        # Vector generation
├── search.py            # Semantic search
├── rag.py               # Ties search and LLM together
├── llm.py               # LLM API calls
├── main.py              # CLI entry point
├── README.md
└── requirements.txt
```

## Running the Project

### The App

From the repo root:

```bash
python projects/04-rag-knowledge-assistant/main.py
```

Ask it a question, and it will retrieve the relevant context, generate an answer, and cite its sources. Type `/exit` to quit.

### Retrieval Evaluation

To test how well the search is working:

```bash
python projects/04-rag-knowledge-assistant/evaluation/evaluate_retrieval.py
```

This tests different `top_k` values and outputs accuracy and MRR scores.

### Answer Evaluation

To test how well the final generated answers are:

```bash
python projects/04-rag-knowledge-assistant/evaluation/evaluate_answers.py
```

This runs the full RAG pipeline and sends the outputs to the LLM judge, saving a detailed report to `evaluation/results/latest.json`.

## How it Works

### RAG vs. Traditional LLMs

A standard LLM app is basically a closed-book test: you ask a question, and the model guesses the answer based purely on its training data.

RAG turns it into an open-book test. Before the LLM generates an answer, we search our external knowledge base for relevant documents and inject them into the prompt. This drastically reduces hallucinations and lets the model answer questions about private or highly specific data.

### Top-K Retrieval

When you ask a question, we don't just grab the single best matching chunk. We grab the top `k` chunks (e.g., `top_k=3`) and pass all of them to the LLM as context.

We actually evaluate different `k` values in this project. Finding the sweet spot is a core part of RAG tuning—too few chunks and the LLM misses context; too many and you waste tokens or confuse the model with irrelevant noise.

### Evaluating Retrieval (Accuracy & MRR)

How do you know if your search is actually good? We use a test dataset of questions with known correct source files.

- **Accuracy:** Did the correct file show up in the top `k` results at all? (e.g., 9 out of 10 = 90%).
- **Mean Reciprocal Rank (MRR):** Accuracy doesn't tell you _where_ the right answer showed up. MRR fixes that. If the correct chunk is rank 1, the score is 1.0. If it's rank 2, it's 0.5. Rank 3 is 0.33. A higher MRR means your most relevant docs are actually at the top of the list, not buried at the bottom.

### Evaluating Answers (LLM-as-a-Judge)

Good retrieval doesn't guarantee a good answer. The LLM might still ignore the context or make things up. To catch this, we use a second LLM to "judge" the first LLM's output.

The judge looks at the question, the retrieved context, and the generated answer, and scores it on:

- **Grounding:** Is the answer actually supported by the provided context, or did the LLM hallucinate outside info?
- **Relevance:** Does it actually answer the user's question?
- **Correctness:** Is the information factually right based on the context?

We use Pydantic to force the judge to output structured JSON, making it easy to parse and aggregate the scores.

### The Catch with LLM Judges

LLM-as-a-judge is a great tool, but it's not a perfect source of truth. Judges can be biased, miss subtle errors, or prefer certain writing styles. In a real production environment, you'd combine automated metrics, LLM evaluation, and regular human spot-checks.

## What's Next?

This project gives us a fully functional, measurable RAG system. In the future, we'll look at improving it by adding:

- Similarity score thresholds (ignoring chunks that are too dissimilar)
- Better handling of unanswerable questions
- Advanced chunking strategies
- Evaluation dashboards and result history
- More robust hallucination detection
