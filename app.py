import os
import re
import time

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from google.genai.errors import ClientError

st.set_page_config(page_title="YouTube Q&A (RAG)", page_icon="🎬")
st.title("🎬 YouTube Video Q&A")
st.caption("Paste a YouTube video ID, then ask questions about its content.")

# ---------- API key ----------
def get_secret_key():
    try:
        return st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        return ""


api_key = (
    os.environ.get("GOOGLE_API_KEY")
    or get_secret_key()
    or st.sidebar.text_input(
        "Google API key", type="password", help="Get one from https://ai.google.dev"
    )
)
if not api_key:
    st.info("Enter your Google API key in the sidebar to continue.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key


# ---------- Helpers ----------
def get_transcript(video_id: str) -> str:
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id, languages=["en"])
    text = " ".join(snippet.text for snippet in fetched)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vector_store(chunks, embeddings, batch_size=20, wait_between=15, status=None):
    vector_store = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        for attempt in range(5):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                break
            except ClientError as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    if status:
                        status.write(f"Rate limit hit, waiting 60s (attempt {attempt+1}/5)...")
                    time.sleep(60)
                else:
                    raise
        else:
            raise RuntimeError("Failed to embed a batch after 5 retries.")
        if status:
            status.write(f"Embedded {min(i + batch_size, len(chunks))} / {len(chunks)} chunks")
        if i + batch_size < len(chunks):
            time.sleep(wait_between)
    return vector_store


@st.cache_resource(show_spinner=False)
def process_video(video_id: str):
    transcript = get_transcript(video_id)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    with st.status("Embedding transcript chunks...", expanded=True) as status:
        vector_store = build_vector_store(chunks, embeddings, status=status)
        status.update(label="Done embedding.", state="complete")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever, len(chunks)


def build_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=["context", "question"],
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel(
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    )
    return parallel_chain | prompt | llm | StrOutputParser()


# ---------- UI ----------
video_id = st.text_input("YouTube video ID", placeholder="e.g. Gfr50f6ZBvo")

if "video_id" not in st.session_state:
    st.session_state.video_id = None
    st.session_state.chat = []

if st.button("Load video", type="primary") and video_id:
    try:
        with st.spinner("Fetching transcript..."):
            retriever, n_chunks = process_video(video_id)
        st.session_state.retriever = retriever
        st.session_state.video_id = video_id
        st.session_state.chat = []
        st.success(f"Video loaded — {n_chunks} chunks indexed. Ask a question below.")
    except TranscriptsDisabled:
        st.error("Captions are disabled for this video.")
    except NoTranscriptFound:
        st.error("No English transcript found for this video.")
    except VideoUnavailable:
        st.error("This video is unavailable.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

if st.session_state.video_id:
    st.divider()
    st.subheader(f"Ask about video: {st.session_state.video_id}")

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    question = st.chat_input("Ask a question about the video...")
    if question:
        st.session_state.chat.append(("user", question))
        with st.chat_message("user"):
            st.write(question)

        chain = build_chain(st.session_state.retriever)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = chain.invoke(question)
                except Exception as e:
                    answer = f"Error: {e}"
                st.write(answer)
        st.session_state.chat.append(("assistant", answer))
