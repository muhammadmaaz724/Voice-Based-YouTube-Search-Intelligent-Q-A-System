# 🎧 Voice-Based YouTube Search & Intelligent Q&A System

An advanced voice-driven application that combines YouTube API integration, speech recognition, and Retrieval-Augmented Generation (RAG) to enable hands-free video search and intelligent questioning about video content.

---

## 🚀 Project Overview

This project enables users to interact with YouTube content through voice. It allows voice-based video search, voice selection, and intelligent Q&A based on video transcripts. The system leverages advanced models like Meta's Llama-3.3-70B and GTE-Large embeddings to deliver highly contextual answers. All components are built into an intuitive, accessible interface that responds to voice commands and provides real-time spoken feedback.

---

## 🔑 Key Features

### 🎤 Voice-First Interface
- **Voice Search**: Use natural speech to search YouTube.
- **Voice Selection**: Choose among the top 5 video results using voice ("one", "two", etc.).
- **Voice Q&A**: Ask questions about the video content using your voice.
- **Audio Feedback**: System responds via text-to-speech confirmations and answers.

### 🧠 Advanced AI Integration
- **RAG Pipeline**: Retrieval-Augmented Generation architecture enables contextual question answering.
- **Llama-3.3-70B**: Meta’s state-of-the-art language model powers natural language responses.
- **GTE-Large Embeddings**: HuggingFace embeddings for high-quality semantic search.
- **FAISS Vector Store**: Enables fast similarity search across transcript chunks.

### 🎬 Seamless Video Experience
- **Instant Playback**: Plays selected video immediately.
- **Transcript Processing in Background**: Extracts and processes captions while the video plays.
- **Real-Time Progress**: Sidebar updates show background processing status.
- **English Transcript Support**: Supports English captions from YouTube Transcript API.

### ♿ Accessibility Features
- **Voice-Only Interaction**: Designed for full operability through voice commands.
- **Visual Feedback**: Progress indicators and status messages.
- **Fallback Options**: Manual text input available when voice fails.
- **Audio Confirmations**: All actions confirmed through text-to-speech.

---

## ⚙️ Technical Architecture

### 🧩 Core Components

- **Speech Recognition**: Uses Google Speech Recognition API with optimized thresholds and noise filtering.
- **YouTube Search**: Powered by YouTube Data API v3 for retrieving video metadata.
- **Captions Extraction**: Uses YouTube Transcript API to extract available English transcripts.
- **Embedding Engine**: Text chunks are embedded with GTE-Large from HuggingFace.
- **Semantic Search**: FAISS is used to retrieve relevant transcript chunks based on user queries.
- **Language Generation**: Meta’s Llama-3.3-70B returns natural language answers.
- **Text-to-Speech**: Converts final answer into spoken output via TTS.

---

## 🔄 Processing Pipeline

1. **Voice Input**: User speaks a query or command.
2. **Speech-to-Text**: Converted to text using Google Speech Recognition.
3. **YouTube Search**: Top 5 videos retrieved based on query.
4. **Voice Selection**: User picks video using voice.
5. **Transcript Extraction**: Captions retrieved and split into chunks.
6. **Embedding**: Chunks embedded and stored in FAISS.
7. **Q&A**: Voice question is matched with transcript chunks, passed to LLM for answer.
8. **Audio Output**: Spoken answer is delivered back to the user.

---

## 📚 Model Details

### 💬 Language Model
- **Name**: Meta Llama-3.3-70B-Instruct
- **API**: HuggingFace Inference API
- **Task**: Natural language generation for context-aware Q&A
- **Strength**: Multi-turn understanding, high factual accuracy

### 📊 Embedding Model
- **Name**: `thenlper/gte-large`
- **Purpose**: Generates semantic vector representations of transcript chunks
- **Strength**: High-dimensional accuracy for dense retrieval

### ✂️ Text Chunking Strategy
- **Splitter**: `RecursiveCharacterTextSplitter`
- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters
- **Goal**: Maintain contextual continuity for better retrieval

---

## 💼 Use Cases

### 🎓 Educational Content
- Ask questions about lectures or tutorials.
- Summarize complex topics via Q&A.
- Study with interactive video assistants.

### ♿ Accessibility Support
- Ideal for users with visual or motor impairments.
- Fully voice-operable interface.
- Audio feedback for all interactions.

### 🔍 Content Research
- Summarize key points from videos.
- Extract factual data from tutorials or talks.
- Rapid content exploration without manual searching.

### 🧑‍💼 Professional Applications
- Analyze meetings, webinars, or training videos.
- Support research with academic video content.
- Assistive learning from instructional media.

---

## 🧠 Performance Optimizations

### ⚡ Caching & Speed
- RAG pipeline cached for faster reuse.
- Session state used to avoid redundant processing.

### ✅ Robust Error Handling
- Retry logic for speech recognition.
- Text input fallback for voice commands.
- Graceful degradation if transcripts aren't available.

### 💾 Resource Management
- Optimized vector storage with FAISS.
- Controlled API usage with retry logic.
- Concurrent processing for background tasks.

---

## 🛣️ Future Enhancements
- Multi-language transcript support.
- Automatic video summarization.
- Persistent conversation memory.
- Batch video analysis for playlists or series.

