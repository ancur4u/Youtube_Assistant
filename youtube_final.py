import os, re, base64, numpy as np
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import yt_dlp

load_dotenv()
st.set_page_config(page_title="🎥 YouTube RAG Chatbot", layout="wide")
st.title("🎥 YouTube Video Assistant")

# Initialize session
if "videos" not in st.session_state: st.session_state.videos = {}
if "active_video" not in st.session_state: st.session_state.active_video = None

# Sidebar controls
st.sidebar.header("🔧 Controls")

# Allow user to enter their OpenAI API Key
api_key = st.sidebar.text_input("🔑 Enter OpenAI API Key", type="password")
if api_key:
    st.session_state["OPENAI_API_KEY"] = api_key

if not st.session_state.get("OPENAI_API_KEY"):
    st.warning("⚠️ Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

new_video_input = st.sidebar.text_input("📅 Add YouTube URL or ID")
if st.sidebar.button("➕ Load Video"):
    if new_video_input:
        vid = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", new_video_input)
        vid_id = vid.group(1) if vid else new_video_input.strip()
        st.session_state.active_video = vid_id
        if vid_id not in st.session_state.videos:
            st.session_state.videos[vid_id] = {}
        st.rerun()

if st.session_state.videos:
    selected = st.sidebar.selectbox("🎬 Loaded Videos", list(st.session_state.videos.keys()),
                                     index=list(st.session_state.videos.keys()).index(st.session_state.active_video))
    st.session_state.active_video = selected

    persona = st.sidebar.selectbox("🧐 Persona", ["Friendly Bot", "Professor", "Entertainer"])
    summary_style = st.sidebar.radio("📘 Summary Format", ["Bullet Points", "Paragraph"])
    summary_length = st.sidebar.selectbox("📏 Summary Length", ["Short (3)", "Medium (5)", "Detailed (7+)"])
    if st.sidebar.button("🧹 Clear This Video's History"):
        st.session_state.videos[selected]["history"] = []
        st.rerun()

    persona_prompts = {
        "Friendly Bot": "You are a friendly assistant. Be casual, helpful and clear.",
        "Professor": "You are a wise professor. Be precise, educational and informative.",
        "Entertainer": "You are a witty assistant. Make your answers fun, yet insightful."
    }

    def load_metadata(vid):
        with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True, 'forcejson': True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
            return {
                "title": info.get("title", "Unknown Title"),
                "author": info.get("uploader", "Unknown Author"),
                "length": int(info.get("duration", 0)),
                "publish_date": info.get("upload_date", "Unknown")
            }

    video_data = st.session_state.videos[selected]
    if not video_data.get("loaded", False):
        try:
            metadata = load_metadata(selected)
            mins, secs = divmod(metadata["length"], 60)
            transcript_list = YouTubeTranscriptApi.get_transcript(selected, languages=["en", "hi"])
            transcript = " ".join([t['text'] for t in transcript_list])
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(transcript)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.session_state["OPENAI_API_KEY"])
            vector_store = FAISS.from_texts(chunks, embeddings)
            video_data.update({
                "metadata": metadata,
                "duration": f"{mins}m {secs}s",
                "transcript_list": transcript_list,
                "chunks": chunks,
                "vector_store": vector_store,
                "summary": "",
                "history": [],
                "loaded": True
            })
        except TranscriptsDisabled:
            st.error("⚠️ Transcript is not available.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    metadata = video_data["metadata"]
    st.markdown(f"""### ℹ️ **Video Metadata**
- **📺 Title**: `{metadata['title']}`
- **👤 Author**: `{metadata['author']}`
- **⏱ Duration**: {video_data['duration']}
- **📅 Published on**: `{metadata['publish_date']}`""")

    tab1, tab2, tab3, tab4 = st.tabs(["📘 Summary", "💬 Chat", "📄 Transcript", "🗂️ History & Export"])

    with tab1:
        st.subheader("📘 Video Summary")
        if st.button("📝 Generate Summary"):
            bullets = {"Short (3)": 3, "Medium (5)": 5, "Detailed (7+)": 7}[summary_length]
            prompt = f"Summarize in {bullets} bullet points:" if summary_style == "Bullet Points" else "Summarize in a paragraph:"
            combined = "\n\n".join(video_data["chunks"])
            video_data["summary"] = ChatOpenAI(model="gpt-4o-mini", openai_api_key=st.session_state["OPENAI_API_KEY"]).invoke(f"{prompt}\n\n{combined}").content
        if video_data.get("summary"):
            st.success(video_data["summary"])

    with tab2:
        st.subheader("💬 Ask a Question")
        default_question = st.session_state.pop("follow_up_question", "") if "follow_up_question" in st.session_state else ""
        auto_submit = st.session_state.pop("auto_submit", False)

        with st.form("qa_form", clear_on_submit=True):
            question = st.text_input("Ask about the video:", value=default_question)
            ask_button = st.form_submit_button("Get Answer")

        if auto_submit and question:
            ask_button = True

        if ask_button and question:
            with st.spinner("💭 Thinking..."):
                retriever = video_data["vector_store"].as_retriever(search_type="similarity", search_kwargs={"k": 4})
                retrieved = retriever.invoke(question)
                used_chunks = [doc.page_content for doc in retrieved]
                context = "\n\n".join(used_chunks)
                history = video_data["history"]
                chat_hist = "\n".join([f"User: {q['question']}\nAssistant: {q['answer']}" for q in history[-3:]])
                persona_prompt = persona_prompts[persona]
                full_prompt = f"""{persona_prompt}

Transcript:
{context}

{chat_hist}

User: {question}
Assistant:"""
                llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=st.session_state["OPENAI_API_KEY"])
                answer = llm.invoke(full_prompt).content
                q_embed = OpenAIEmbeddings(openai_api_key=st.session_state["OPENAI_API_KEY"]).embed_query(question)
                d_embed = OpenAIEmbeddings(openai_api_key=st.session_state["OPENAI_API_KEY"]).embed_documents(used_chunks)
                confidence = np.mean([np.dot(q_embed, e) for e in d_embed])
                suggestions = llm.invoke(f"{persona_prompt}\nSuggest 3 follow-up questions for:\n{question}").content

                video_data["history"].append({
                    "question": question,
                    "answer": answer,
                    "context": context,
                    "confidence": confidence,
                    "chunks": used_chunks,
                    "suggestions": suggestions
                })

        if video_data["history"]:
            latest = video_data["history"][-1]
            st.markdown(f"""<div style='background:#f0faff;padding:16px;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,0.1);'>
<b>🧑‍💬 You:</b> {latest["question"]}<br>
<b>🤖 {persona}:</b> {latest["answer"]}<br>
<span style='color:gray;'>Confidence: {latest["confidence"]:.2f}</span>
</div>""", unsafe_allow_html=True)

            with st.expander("📚 Retrieved Context"):
                st.code(latest["context"])

            with st.expander("💡 Suggestions"):
                for line in latest["suggestions"].splitlines():
                    if line.strip().startswith(tuple("1234567890")):
                        follow = line.split(".", 1)[-1].strip()
                        if st.button(follow, key=f"{selected}_{follow}"):
                            st.session_state.follow_up_question = follow
                            st.session_state.auto_submit = True
                            st.rerun()

            with st.expander("🕒 Timestamps"):
                for chunk in latest["chunks"]:
                    for entry in video_data["transcript_list"]:
                        if entry['text'].strip() in chunk:
                            t = entry['start']
                            st.markdown(f"[{t:.2f}s](https://www.youtube.com/watch?v={selected}&t={int(t)}s) — {entry['text']}")
                            break

    with tab3:
        st.subheader("📄 Transcript Viewer")
        keyword = st.text_input("Search transcript:")
        with st.expander("📝 Transcript", expanded=True):
            for entry in video_data["transcript_list"]:
                if keyword.lower() in entry['text'].lower():
                    t = entry['start']
                    st.markdown(f"[{t:.2f}s](https://www.youtube.com/watch?v={selected}&t={int(t)}s) — {entry['text']}")

    with tab4:
        st.subheader("🗂️ Full Q&A History")
        for i, qa in enumerate(video_data["history"]):
            st.markdown(f"""<div style='background:#f6fafd;padding:12px;border-radius:10px;margin-bottom:8px;'>
<b>Q{i+1}:</b> {qa['question']}<br>
<b>A{i+1}:</b> {qa['answer']}<br>
<span style='color:gray;'>Confidence: {qa['confidence']:.2f}</span>
</div>""", unsafe_allow_html=True)

        pdf_path = f"{selected}_chat_history.pdf"
        try:
            font_path = "fonts/DejaVuSans.ttf"
            pdfmetrics.registerFont(TTFont("DejaVu", font_path))
            font_name = "DejaVu"
        except:
            font_name = "Helvetica"

        c = canvas.Canvas(pdf_path, pagesize=A4)
        c.setFont(font_name, 12)
        width, height = A4
        y = height - 20 * mm
        for label, val in [("Title", metadata['title']), ("Author", metadata['author']), ("Duration", video_data['duration']), ("Published", metadata['publish_date'])]:
            c.drawString(10 * mm, y, f"{label}: {val}"); y -= 10 * mm
        for i, qa in enumerate(video_data["history"]):
            c.drawString(10 * mm, y, f"Q{i+1}: {qa['question']}"); y -= 8 * mm
            c.drawString(10 * mm, y, f"A{i+1}: {qa['answer']}"); y -= 8 * mm
            c.drawString(10 * mm, y, "Context:"); y -= 6 * mm
            for line in qa["context"].splitlines():
                if y < 20 * mm: c.showPage(); y = height - 20 * mm; c.setFont(font_name, 12)
                c.drawString(12 * mm, y, line[:120]); y -= 6 * mm
            y -= 10 * mm
        c.save()
        with open(pdf_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="{selected}_chat_history.pdf">📅 Download PDF</a>', unsafe_allow_html=True)
