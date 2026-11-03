# Football RAG

A retrieval-augmented generation pipeline that answers questions using a mix of PDFs and scraped web pages. I built it around the 2026 World Cup as a test domain since it's live and changing right now, which makes it a good stress test for hallucination (the model shouldn't invent things that haven't happened yet).

## How it works

- `sources.py` holds the list of URLs to pull from; PDFs are read from `data/`
- `pipeline.py` loads the PDFs, scrapes the URLs with BeautifulSoup, splits everything into chunks, embeds the chunks with a HuggingFace sentence-transformer model, and stores them in a FAISS index. The index and the chunk texts are cached in `index/` and rebuilt automatically when the sources change
- `retrieval.py` does the retrieval: BM25 keyword search + FAISS vector search, fused with Reciprocal Rank Fusion, then reranked with a cross-encoder (see below)
- `main.py` is a simple CLI loop to ask it questions
- `test.py` runs the original 10 test questions through the full pipeline and scores the answers
- `evaluate.py` compares retrieval modes on 19 labelled questions (no API key needed)

The prompt tells the model to only answer from the retrieved context and say "I don't know" if the context doesn't cover it. That's the main defense against hallucination here.

## Hybrid retrieval

The first version used vector search only. That works for questions phrased in meaning ("who is leading the scoring charts") but is weak on exact tokens: names, numbers and rare words like "Saïss" or "48 teams", which a plain keyword search would catch. The retriever now combines both:

```text
question ──┬──▶ BM25 keyword search   (top 20) ──┐
           │                                     ├──▶ Reciprocal Rank Fusion ──▶ (optional cross-encoder rerank) ──▶ top 3 ──▶ LLM (Groq)
           └──▶ FAISS vector search   (top 20) ──┘
```

- **BM25** (implemented from scratch in `retrieval.py`) scores chunks by exact term overlap, weighting rare terms higher.
- **Reciprocal Rank Fusion** merges the two ranked lists with `score = Σ 1 / (60 + rank)`. It only uses ranks, so the two retrievers' very different score scales never need to be calibrated.
- **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads the question and each candidate chunk together, which is more accurate than comparing two separate embeddings, but too slow to run over every chunk. So it only reorders the 20 fused candidates.

The mode can be switched with the `RETRIEVAL_MODE` environment variable (`vector`, `bm25`, `hybrid`, `hybrid_rerank`). The default is `hybrid`, without the reranker, because it scored best in the evaluation below.

## Retrieval evaluation

`evaluate.py` runs every mode on 19 questions labelled with the source that should answer them (`eval/questions.json`: the 9 answerable original questions plus 10 retrieval-only ones) and reports Hit@1, Hit@3 and MRR@10.

Results from 3 November 2026 (912 chunks from the web sources only, no PDFs; CPU):

| Mode | Hit@1 | Hit@3 | MRR@10 | ms / query |
|---|---|---|---|---|
| vector | 0.95 | 1.00 | 0.974 | 13 |
| bm25 | 0.74 | 0.95 | 0.826 | 2 |
| hybrid | **1.00** | **1.00** | **1.000** | 15 |
| hybrid_rerank | 0.89 | 0.95 | 0.932 | 1616 |

Fusing BM25 with vector search fixed the one question vector search ranked second (Hit@1 0.95 → 1.00) at almost no extra cost. The cross-encoder made things worse: it pushed the right chunk out of the top 3 for "who is Morocco's captain" and demoted another to second place, while making each query about 100× slower. A reranker trained on MS MARCO web search doesn't seem to suit these long Wikipedia chunks, so plain `hybrid` is the better choice on this data.

Caveats: 19 questions is a small set, so one question moves Hit@k by 0.05. The Wikipedia pages are scraped live, so the numbers can shift as the pages change. Run `python evaluate.py` to reproduce (writes `eval/retrieval_results.md`).

## Testing the answers

I wrote 10 questions covering a few different things: basic facts, squad/roster info, live stats, general tournament facts, and one deliberate trap question about a World Cup that hasn't been played, so the right answer is "I don't know". `test.py` scores them: keyword checks give PASS/FAIL, and questions whose answer changed during the tournament are marked MANUAL.

### Latest run (3 November 2026, `openai/gpt-oss-120b`, `hybrid` retrieval)

Groq retired Llama 3.3 70B, so the pipeline now uses `openai/gpt-oss-120b` (set `GROQ_MODEL` to change it).

**Automatic checks: 5/6 passed, 4 checked by hand.**

- **Trap question:** the original trap was "who won the 2026 World Cup", but the tournament has now finished and the scraped Wikipedia page says Spain won, so "Spain" became the correct, grounded answer. I replaced it with "who won the 2030 World Cup", and the model correctly declined.
- **Real miss:** asked which club Mbappé plays for, it answered PSG. His Wikipedia page says he joined Real Madrid in 2024, but the three retrieved chunks all came from the PSG section of his career. The grounding prompt can't fix stale context, so this is a retrieval problem: long biographies contain several "current club" contexts from different years.
- **Manual answers:** "I don't know" for Morocco's 2026 captain, Cape Verde's squad and Messi's 2026 goals, and "Mbappé" for the Golden Boot, which matches the scraped tracker.

### First run (Llama 3.3 70B, vector retrieval, before the tournament)

**8/10 clearly correct, 1 with an incorrect supporting fact, 1 correctly declined to answer.** The model didn't invent a 2026 winner, and it said "I don't know" where the sources lacked the data. The one miss: asked who Morocco's captain is, it said Achraf Hakimi captained Morocco at the 2022 World Cup. Romain Saïss was the captain; Hakimi only wore the armband when Saïss was out. I checked this against outside sources rather than trusting the output.

Together the two runs show the limits of grounding: the prompt stops the model from inventing facts that aren't in the sources, but it can't stop it from confidently restating a source that is ambiguous or out of date.

## A chunking problem I ran into

I first tried `chunk_size=500`. With the full source list (84 URLs + 7 PDFs) that produced about 18,000 chunks, and embedding that many on CPU basically never finished in a reasonable time — I had to kill it twice. Bumping to `chunk_size=1500` cut that down to roughly a third as many chunks, which made it actually usable for iterating locally.

While testing, I also just used a smaller subset of sources (12 URLs instead of 84) so I could run the eval quickly without waiting 20+ minutes every time I changed something. The full source list still lives in `sources.py`, just used for the "real" run rather than every single test.

## Running it

```bash
pip install -r requirements.txt
```

Add a `.env` file:
```
GROQ_API_KEY=your_key_here
```
(see `.env.example` for the optional `RETRIEVAL_MODE` and `GROQ_MODEL`)

Then:
```bash
python main.py                    # ask questions interactively
python test.py                    # score the 10 original questions end-to-end
python evaluate.py                # compare retrieval modes (no API key needed)
python -m unittest discover tests # unit tests for BM25, fusion and metrics
```

The first run downloads the embedding and reranking models and builds the index; later runs load it from `index/`.

## Stack

LangChain, HuggingFace embeddings, FAISS, BM25 (from scratch), cross-encoder reranking (sentence-transformers), Groq (openai/gpt-oss-120b), BeautifulSoup, PyPDF

## What's not done yet

- No fact-verification step — the model can restate an inaccurate claim from a source confidently if the source itself is ambiguous (see the captain example above)
- The scraped sources are mostly Wikipedia plus a handful of news sites. A live stats API (football-data.org, for example) would be more reliable than scraping for anything time-sensitive
- The evaluation set is small (19 retrieval questions) and labelled by hand, so differences of one or two questions between modes aren't meaningful on their own