import streamlit as st
import speech_recognition as sr
import tempfile
import os
import subprocess
from langchain_core.runnables import RunnableLambda
from googleapiclient.discovery import build
from dotenv import load_dotenv
import sys
import time
import threading
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# === Load API key from .env ===
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Set HuggingFace cache directory
os.environ['HF_HOME'] = 'D:/NLP/huggingface_cache'

# === YouTube Client ===
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# === Initialize RAG Components ===
@st.cache_resource
def initialize_rag_components():
    """Initialize RAG components (cached for performance)"""
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    
    llm = HuggingFaceEndpoint(
        repo_id='meta-llama/Llama-3.3-70B-Instruct',
        task='text-generation'
    )
    
    model = ChatHuggingFace(llm=llm)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    return embeddings, model, text_splitter

# === Streamlit UI Setup ===
st.set_page_config(page_title="🎤 Voice YouTube Search & RAG", layout="wide")
st.title("🎧 Voice-Based YouTube Search & Intelligent Q&A")

# Initialize session state
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None
if 'transcript_ready' not in st.session_state:
    st.session_state.transcript_ready = False
if 'processing_transcript' not in st.session_state:
    st.session_state.processing_transcript = False
if 'video_title' not in st.session_state:
    st.session_state.video_title = None

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎤 Voice Search")
    duration = st.slider("🎙️ Voice record duration", 2, 10, 5)

with col2:
    st.subheader("🎬 Video Controls")
    st.info("📢 How to use:")
    st.markdown("""
    - **🎤 Voice Search**: Search for videos using your voice
    - **🎬 Select Video**: Choose from search results
    - **❓ Ask Question**: Click the question button to ask about video content
    """)

# === Enhanced Voice to Text Function ===
def voice_to_text_enhanced(path, timeout=10, phrase_time_limit=None):
    recognizer = sr.Recognizer()
    
    # Adjust recognizer settings for better accuracy
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.non_speaking_duration = 0.5
    
    with sr.AudioFile(path) as source:
        # Remove background noise
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.record(source)
    
    try:
        # Try Google Speech Recognition first
        result = recognizer.recognize_google(audio, language='en-US')
        return result
    except sr.UnknownValueError:
        # If Google fails, try with different settings
        try:
            result = recognizer.recognize_google(audio, language='en-US', show_all=True)
            if result and len(result) > 0:
                return result[0]['transcript']
        except:
            pass
        raise sr.UnknownValueError("Could not understand audio")

voice_to_text_runnable = RunnableLambda(voice_to_text_enhanced)

# === YouTube Search Function ===
def search_youtube(query):
    request = youtube.search().list(
        q=query, part="snippet", maxResults=5, type="video"
    )
    response = request.execute()
    return [
        {"title": item["snippet"]["title"], "videoId": item["id"]["videoId"]}
        for item in response["items"]
    ]

search_youtube_runnable = RunnableLambda(search_youtube)

# === Enhanced Audio Recording Function ===
def record_audio_enhanced(recognizer, mic, duration, purpose="search"):
    """Enhanced audio recording with better settings"""
    with mic as source:
        st.info(f"🎙️ Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        if purpose == "search":
            st.info(f"🎙️ Recording your search query for {duration} seconds...")
            audio = recognizer.listen(source, phrase_time_limit=duration)
        elif purpose == "selection":
            st.info("🎧 Say your selection clearly (e.g., 'one', 'two', 'three')...")
            audio = recognizer.listen(source, phrase_time_limit=3, timeout=10)
        else:  # question
            st.info("🎧 Ask your question about the video...")
            audio = recognizer.listen(source, phrase_time_limit=8, timeout=15)
    
    return audio

# === Voice Question Function ===
def ask_voice_question():
    """Function to handle voice questions"""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.non_speaking_duration = 0.5
    
    mic = sr.Microphone()
    
    try:
        # Record question
        question_audio = record_audio_enhanced(recognizer, mic, duration=8, purpose="question")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(question_audio.get_wav_data())
            question_audio_path = f.name
        
        # Convert question to text
        question = voice_to_text_enhanced(question_audio_path)
        st.success(f"❓ Your question: **{question}**")
        
        # Get answer from RAG
        with st.spinner("🧠 Thinking about your question..."):
            answer = st.session_state.rag_chain.invoke(question)
        
        st.success(f"💬 Answer: **{answer}**")
        
        # Speak the answer
        st.info("🔊 Speaking the answer...")
        subprocess.run([sys.executable, "speak.py", answer])
        
        # Clean up
        os.remove(question_audio_path)
        
    except sr.UnknownValueError:
        st.error("❌ Could not understand your question. Please try again.")
        subprocess.run([sys.executable, "speak.py", "I couldn't understand your question. Please try again."])
    except Exception as e:
        st.error(f"⚠️ Error processing question: {e}")
        subprocess.run([sys.executable, "speak.py", "There was an error processing your question."])

# === RAG Setup Function ===
def setup_rag_for_video(video_id):
    """Setup RAG pipeline for the given video"""
    try:
        # Get video transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        
        # Initialize RAG components
        embeddings, model, text_splitter = initialize_rag_components()
        
        # Create text chunks
        chunks = text_splitter.create_documents([transcript])
        
        # Create vector store
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        # Setup retriever
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        
        # Setup prompt
        prompt = PromptTemplate(
            template="""You are a helpful assistant.
              Answer ONLY from the provided transcript context.
              If the context is insufficient, just say you don't know.
              Keep your answer simple and clear and also explain with the help of real world example.

              {context}
              Question: {question}""",
              input_variables=['context', 'question']
        )
        
        # Helper function to format documents
        def format_docs(retrieved_docs):
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text
        
        # Create the RAG chain
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | model | parser
        
        return main_chain, True
        
    except TranscriptsDisabled:
        st.error("❌ No captions available for this video.")
        return None, False
    except Exception as e:
        st.error(f"⚠️ Error setting up RAG: {e}")
        return None, False

# === Voice Search Section ===
if st.button("🎤 Start Voice Search"):
    recognizer = sr.Recognizer()
    
    # Enhanced recognizer settings
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.non_speaking_duration = 0.5
    
    mic = sr.Microphone()

    try:
        # === Record Voice for Search ===
        search_audio = record_audio_enhanced(recognizer, mic, duration, "search")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(search_audio.get_wav_data())
            search_audio_path = f.name

        # === Convert Voice to Text ===
        query = voice_to_text_runnable.invoke(search_audio_path)
        st.success(f"🔎 You searched for: **{query}**")

        # === Get YouTube Search Results ===
        results = search_youtube_runnable.invoke(query)

        # === Display & Speak All Titles ===
        st.subheader("🔊 Speaking video titles...")
        to_speak = []
        for i, video in enumerate(results):
            spoken = f"Option {i + 1}: {video['title']}"
            st.markdown(f"**{spoken}**")
            to_speak.append(spoken)

        # Final voice prompt
        to_speak.append("Which option would you like to select? Say the number clearly.")

        # === Call speak.py to say everything ===
        full_text = " ".join(to_speak)
        subprocess.run([sys.executable, "speak.py", full_text])
        time.sleep(1)  # Give more time for TTS to complete

        # === Listen for User Selection with Multiple Attempts ===
        selection_attempts = 0
        max_attempts = 3
        selection = None
        
        while selection_attempts < max_attempts:
            try:
                selection_attempts += 1
                st.info(f"🎧 Listening for your selection... (Attempt {selection_attempts}/{max_attempts})")
                
                # Record selection audio
                sel_audio = record_audio_enhanced(recognizer, mic, duration, "selection")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f2:
                    f2.write(sel_audio.get_wav_data())
                    sel_audio_path = f2.name

                # === Convert Selection Voice to Text ===
                selection = voice_to_text_enhanced(sel_audio_path).strip().lower()
                st.success(f"✅ You said: **{selection}**")
                
                # Clean up temp file
                os.remove(sel_audio_path)
                break
                
            except sr.UnknownValueError:
                if selection_attempts < max_attempts:
                    st.warning(f"⚠️ Couldn't understand. Try speaking more clearly. Attempt {selection_attempts + 1}/{max_attempts}")
                    subprocess.run([sys.executable, "speak.py", f"Couldn't understand. Try speaking more clearly."])
                    time.sleep(1)
                else:
                    st.error("❌ Could not understand your voice after multiple attempts.")
                    subprocess.run([sys.executable, "speak.py", "Could not understand your voice after multiple attempts."])
                    break
            except Exception as e:
                st.error(f"⚠️ Error during selection: {e}")
                break

        if selection:
            # === Enhanced Matching with More Options ===
            spoken_map = {
                "one": 0, "1": 0, "first": 0, "option one": 0,
                "two": 1, "2": 1, "to": 1, "too": 1, "second": 1, "option two": 1,
                "three": 2, "3": 2, "third": 2, "option three": 2,
                "four": 3, "4": 3, "for": 3, "fourth": 3, "option four": 3,
                "five": 4, "5": 4, "fifth": 4, "option five": 4,
            }

            # Try to find the selection in the spoken text
            index = None
            for key, value in spoken_map.items():
                if key in selection:
                    index = value
                    break

            if index is not None and index < len(results):
                video = results[index]
                video_id = video["videoId"]
                video_title = video["title"]
                
                # Update session state
                st.session_state.current_video_id = video_id
                st.session_state.video_title = video_title
                
                st.success(f"🎬 Selected: {video_title}")
                
                # Audio feedback
                subprocess.run([sys.executable, "speak.py", f"Selected video: {video_title}. Video is now playing."])
                
                # Force rerun to show video immediately
                st.rerun()
                
            else:
                st.error(f"❌ Couldn't match your selection '{selection}' to any video. Try saying 'one', 'two', etc.")
                subprocess.run([sys.executable, "speak.py", f"Couldn't match your selection. Try saying one, two, three, four, or five."])

        # === Cleanup temp files ===
        os.remove(search_audio_path)

    except sr.UnknownValueError:
        st.error("❌ Could not understand your voice during search.")
        subprocess.run([sys.executable, "speak.py", "Could not understand your voice during search."])
    except sr.RequestError as e:
        st.error(f"⚠️ Speech Recognition API error: {e}")
        subprocess.run([sys.executable, "speak.py", "There was an error with speech recognition."])
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        subprocess.run([sys.executable, "speak.py", "There was an error processing your request."])

# === Display Video Section ===
if st.session_state.current_video_id:
    st.subheader("🎬 Video Player")
    
    # Show video title
    if st.session_state.video_title:
        st.markdown(f"**🎥 Now Playing:** {st.session_state.video_title}")
    
    # Display video immediately
    st.video(f"https://www.youtube.com/watch?v={st.session_state.current_video_id}")
    
    # Setup RAG if not already done and not currently processing
    if not st.session_state.transcript_ready and not st.session_state.processing_transcript:
        st.session_state.processing_transcript = True
        st.info("⏳ Processing video transcript for Q&A... You can watch the video while we prepare!")
        
        with st.spinner("🔧 Setting up Q&A system..."):
            rag_chain, success = setup_rag_for_video(st.session_state.current_video_id)
            if success:
                st.session_state.rag_chain = rag_chain
                st.session_state.transcript_ready = True
                st.session_state.processing_transcript = False
                st.success("✅ Video is ready for questions!")
                subprocess.run([sys.executable, "speak.py", "Video is ready for questions."])
                st.rerun()
            else:
                st.session_state.processing_transcript = False
                st.session_state.transcript_ready = False
    
    # Show processing status
    elif st.session_state.processing_transcript:
        st.info("⏳ Processing video transcript for Q&A... You can watch the video while we prepare!")
    
    # Show Ask Question button only when RAG is ready
    elif st.session_state.transcript_ready and st.session_state.rag_chain:
        st.success("✅ Video is ready for questions!")
        st.markdown("---")
        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            if st.button("🎤 Ask Question About Video", 
                        key="ask-question-main", 
                        help="Click to ask a voice question about the video",
                        use_container_width=True):
                ask_voice_question()
    
    # Show error state if processing failed
    else:
        st.warning("⚠️ Transcript processing failed. Video playback available but Q&A unavailable.")

# === Manual Q&A Section (backup) ===
if st.session_state.current_video_id and st.session_state.transcript_ready and st.session_state.rag_chain:
    st.markdown("---")
    st.subheader("📝 Manual Question Input (Backup)")
    
    # Manual text input as backup
    manual_question = st.text_input("Enter your question about the video:")
    
    if manual_question and st.button("📝 Submit Text Question"):
        try:
            with st.spinner("🧠 Thinking about your question..."):
                answer = st.session_state.rag_chain.invoke(manual_question)
            
            st.success(f"💬 Answer: **{answer}**")
            
            # Speak the answer
            st.info("🔊 Speaking the answer...")
            subprocess.run([sys.executable, "speak.py", answer])
            
        except Exception as e:
            st.error(f"⚠️ Error processing question: {e}")

# === Status Information ===
st.sidebar.header("📊 Status")
st.sidebar.info(f"Video Selected: {'✅' if st.session_state.current_video_id else '❌'}")
st.sidebar.info(f"Processing: {'⏳' if st.session_state.processing_transcript else '❌'}")
st.sidebar.info(f"RAG Ready: {'✅' if st.session_state.transcript_ready else '❌'}")

if st.session_state.current_video_id:
    st.sidebar.markdown(f"**Current Video ID:** {st.session_state.current_video_id}")
if st.session_state.video_title:
    st.sidebar.markdown(f"**Video Title:** {st.session_state.video_title}")

# === Instructions ===
st.sidebar.header("📋 Instructions")
st.sidebar.markdown("""
### How to Use:
1. **🎤 Voice Search**: Search for videos using your voice
2. **🎬 Select Video**: Choose from the search results
3. **🎥 Watch**: Video starts playing immediately
4. **❓ Ask Questions**: Button appears when transcript is ready

### Features:
- **Instant Playback**: Video starts immediately after selection
- **Background Processing**: Transcript processes while you watch
- **Voice Questions**: Ask questions about video content using voice
- **Audio Answers**: Get spoken responses to your questions
- **Manual Backup**: Text input available as backup option
""")

# === Accessibility Notes ===
st.sidebar.header("♿ Accessibility Features")
st.sidebar.markdown("""
- **Voice Search**: Search for videos using voice
- **Instant Playback**: No waiting for transcript processing
- **Voice Questions**: Ask questions using voice
- **Audio Feedback**: Spoken answers and confirmations
- **Manual Backup**: Text input when voice fails
- **Simple Interface**: Large, clear buttons
- **Progress Indicators**: Visual feedback during processing
""")

# Auto-refresh for background processing updates
if st.session_state.processing_transcript:
    time.sleep(1)
    st.rerun()