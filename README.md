# RAG Pipeline

A retrieval-augmented generation pipeline that answers questions using a mix of PDFs and scraped web pages. I built it around the 2026 World Cup as a test domain since it's live and changing right now, which makes it a good stress test for hallucination (the model shouldn't invent things that haven't happened yet).

## How it works

- `sources.py` holds the list of PDFs (local files) and URLs to pull from
- `pipeline.py` does the actual work: loads the PDFs, scrapes the URLs with BeautifulSoup, splits everything into chunks, embeds the chunks with a HuggingFace sentence-transformer model, and stores them in a FAISS index (cached to disk, so it doesn't re-embed every run)
- `main.py` is a simple CLI loop to ask it questions
- `test.py` runs a fixed list of test questions automatically instead of typing them one by one

The prompt tells the model to only answer from the retrieved context and say "I don't know" if the context doesn't cover it. That's the main defense against hallucination here.

## Testing it

I wrote 10 questions covering a few different things: basic facts, squad/roster info, live stats, general tournament facts, and one deliberate trap question ("who won the 2026 World Cup" — it hasn't happened yet in the data used, so the right answer is "I don't know").

Result: 9 out of 10 were correct or correctly said "I don't know" instead of guessing. The one interesting case was the Morocco captain question — instead of making something up, it referenced Hakimi's 2022 captaincy but didn't claim it applies to 2026, which is the kind of careful hedging you actually want.

The most important one is the "who won" question — it didn't invent a winner, which is really the whole point of the grounding prompt.

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

Then:
```bash
python main.py     # ask questions interactively
python test.py     # run the fixed test set
```

## Stack

LangChain, HuggingFace embeddings, FAISS, Groq (Llama 3.3 70B), BeautifulSoup, PyPDF

## What's not done yet

- Retrieval is pure vector search right now — no keyword/hybrid search, so it can miss exact-match queries a plain keyword search would catch
- The scraped sources are mostly Wikipedia plus a handful of news sites. A live stats API (football-data.org, for example) would be more reliable than scraping for anything time-sensitive
- If a source changes, the FAISS index doesn't auto-invalidate — you have to delete `index/faiss_index` manually and rebuild