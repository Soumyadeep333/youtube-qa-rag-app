# 🎬 YouTube Video Q&A using RAG

Ask questions about any YouTube video. Paste the video ID, and the app answers using only the video's transcript.

**Live demo:** [ADD YOUR STREAMLIT LINK HERE]
*(Access code available on request.)*

![App screenshot](screenshot.png)

## What it does

- Takes a YouTube video ID
- Gets the transcript of the video
- Lets you chat and ask questions about the video
- Answers only from the transcript. If the answer is not in the video, it says "I don't know" instead of making things up

## How it works

This project uses **RAG (Retrieval-Augmented Generation)**.

1. **Get transcript:** fetch the English captions with `youtube-transcript-api`
2. **Clean text:** remove timestamps and extra spaces
3. **Split into chunks:** 1000 characters each, with 200 overlap
4. **Create embeddings:** convert each chunk into numbers using Gemini (`gemini-embedding-001`)
5. **Store in vector database:** save the embeddings in FAISS
6. **Retrieve:** for each question, find the 4 most similar chunks
7. **Generate answer:** send the chunks and the question to Gemini (`gemini-3.6-flash`) to write the answer

```
Video ID → Transcript → Chunks → Embeddings → FAISS
                                                 ↓
        Question → Search top 4 chunks → Gemini → Answer
```

## Tech stack

- Python
- LangChain
- Google Gemini API (embeddings + LLM)
- FAISS (vector store)
- Streamlit (web app)
- youtube-transcript-api

## Run on your computer

1. Clone the repo
   ```bash
   git clone https://github.com/Soumyadeep333/youtube-qa-rag-app.git
   cd youtube-qa-rag-app
   ```
2. Install packages
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free Gemini API key from https://aistudio.google.com/apikey
4. Set the key and run
   ```bash
   export GOOGLE_API_KEY="your-key-here"
   streamlit run app.py
   ```
   (On Windows PowerShell: `$env:GOOGLE_API_KEY="your-key-here"`)

You can also skip step 4's export and type the key in the app sidebar.

## Settings for deployment (Streamlit Cloud)

Add these in **Settings → Secrets**:

```
GOOGLE_API_KEY = "your-key-here"
ACCESS_CODE = "your-access-code"   # optional, locks the app
```

## Limits (to protect the free API quota)

- Long videos are refused (maximum 80 chunks)
- Maximum 10 questions per visit
- Embedding takes about 1 to 2 minutes because of the free-tier rate limit. The app waits and retries automatically

## Known issues

- Only videos with English captions work
- YouTube may block transcript requests from some cloud servers
- The free Gemini quota is shared by all users of the app

## Author

**Soumyadeep**
GitHub: [Soumyadeep333](https://github.com/Soumyadeep333)
