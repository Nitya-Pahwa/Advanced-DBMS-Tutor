# DBMS Concept Graph Tutor
An **AI-powered DBMS learning assistant** that combines:
- Retrieval-Augmented Generation (RAG)
- Concept Graph Visualization
- Multi-session Chat Interface

Built using **LangGraph + FAISS + Streamlit + Groq LLM**

## Features

### AI-Powered DBMS Question Answering

* Ask any DBMS-related question
* Context-aware answers using RAG
* Source-grounded responses from DBMS PDFs
* Automatic question classification
* Multi-session conversation support

### Concept Graph Visualization

For conceptual questions, the system automatically generates an interactive knowledge graph.

**Supported Question Types**

* Definitions
* Comparisons
* Relationships

**Visualization**

* Nodes represent DBMS concepts
* Directed edges represent relationships
* Interactive graph rendered using PyVis

### Voice-to-Voice Learning Assistant

Students can interact with the tutor entirely through voice.

#### Speech-to-Text

* Powered by OpenAI Whisper
* Supports recorded audio input
* Automatic transcription of DBMS questions

#### Text-to-Speech

* Powered by Microsoft Edge TTS
* Natural AI voice responses
* Audio playback directly in Streamlit

#### Voice Workflow

Voice Input

→ Whisper Transcription

→ LangGraph RAG Pipeline

→ Answer Generation

→ Edge TTS Synthesis

→ Spoken Response

### Adaptive MCQ Quiz Generator

Generate topic-specific DBMS quizzes instantly.

#### Supported Topics

* SQL
* Normalization
* Transactions
* Indexing
* Concurrency Control
* ER Modeling
* Any DBMS concept

#### Quiz Features

* Dynamic LLM-generated questions
* 5 or 10 question modes
* Easy, Medium, and Hard questions
* SQL syntax-based MCQs
* Automatic answer validation
* Instant scoring
* Retake support

### Multi-Session Chat History

* Persistent chat sessions
* Switch between conversations
* Session-wise query history
* Context preservation across interactions

### Source Citation Support

Every answer includes:

* Retrieved document sources
* Relevant PDF references
* Supporting text snippets

This improves transparency and helps students verify information.


## Architecture

### Question Answering Pipeline

User Query

→ Query Classifier

→ FAISS Retriever

→ Context Extraction

→ Reasoning Agent

→ Answer Generator

→ Final Response

### Concept Graph Pipeline

Question

→ Answer Generation

→ Relationship Extraction

→ Concept Nodes & Edges

→ PyVis Visualization

### Voice Assistant Pipeline

User Speech

→ Whisper STT

→ Query Processing

→ RAG Retrieval

→ LLM Reasoning

→ Answer Generation

→ Edge TTS

→ Audio Response

### Quiz Generation Pipeline

Topic Selection

→ Quiz Generator Agent

→ Groq LLaMA 3.1

→ JSON Validation

→ Quiz Renderer

→ Evaluation Engine

→ Score Calculation

### LangGraph Workflow

User Input

↓

Classifier Node

↓

Retriever Node (FAISS)

↓

Reasoning Node

↓

Generation Node

↓

Graph Builder (Optional)

↓

Voice Module (Optional)

↓

Quiz Module (Optional)

↓
<br> C:\Users\Hp\Desktop\CS_Projects\Agentic AI\DBMS Advanced\architecture.png </br>

Final Output

## Concept Graph

For questions like:
- Definitions
- Comparisons
- Relationships  

The system generates a **knowledge graph**:
- Nodes = DBMS concepts  
- Edges = relationships  

## Tech Stack

### AI & Orchestration

* LLM: Groq (LLaMA 3.1 8B Instant)
* Workflow Engine: LangGraph
* Framework: LangChain

### Retrieval

* Vector Database: FAISS
* Embeddings: Sentence Transformers

### Frontend

* Streamlit

### Visualization

* PyVis
* NetworkX

### Voice Processing

* Whisper (Speech-to-Text)
* Edge TTS (Text-to-Speech)
* streamlit-mic-recorder

### Quiz System

* Groq API
* JSON-based MCQ generation
* Automatic answer evaluation

### Utilities

* Python
* Regex
* JSON Parsing


## Project Structure  
```bash
DBMS-Tutor/
│
├── Nodes/
│   ├── classify.py
│   ├── retrieve.py
│   ├── reason.py
│   ├── generate.py
│   ├── safe_llm.py
│   └── quiz_generator.py
│
├── voice_handler.py
├── graph_flow.py
├── preprocess_index.py
├── app.py
│
├── data/
│   ├── pdfs/
│   └── faiss_index/
│
├── requirements.txt
└── README.md 
```

## User Interface

The application is organized into three interactive tabs:

### 💬 Q&A Tab
- Ask DBMS questions
- View generated answers
- Explore concept graphs
- Inspect retrieved source snippets

### 🎙️ Voice Tab
- Record a DBMS question directly from the browser
- Automatic speech transcription
- Spoken AI-generated answer
- Voice-to-voice learning experience

### 📝 Quiz Tab
- Select any DBMS topic
- Generate 5 or 10 MCQs
- Submit answers
- Get instant score and feedback

- 
## Setup Instructions

1. Clone repo
```bash
[git clone https://github.com/your-username/dbms-concept-tutor.git
cd dbms-concept-tutor](https://github.com/Nitya-Pahwa/DBMS-Tutor.git)
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add environment variables
```bash
GROQ_API_KEY=your_api_key_here
```

4. Build vector database
```bash
python preprocess_index.py
```

5. Run app
```bash
streamlit run app.py
```

## Dataset

DBMS PDFs (books, notes, slides)  
Converted into embeddings for retrieval       

## Glimpses of the Project  

<img width="1920" height="896" alt="Screenshot (2413)" src="https://github.com/user-attachments/assets/aa19025b-63b4-4c0e-8f83-a0660f62ff44" />
<img width="1920" height="900" alt="Screenshot (2414)" src="https://github.com/user-attachments/assets/2b15f27b-29b3-4a3c-86da-cab52aef5638" />


<img width="1920" height="900" alt="Screenshot (2415)" src="https://github.com/user-attachments/assets/46d12461-3027-4836-9a67-313a9695fabb" />

<img width="1920" height="1080" alt="Screenshot (2425)" src="https://github.com/user-attachments/assets/750cfb85-4209-4b11-90f2-eead070959d2" />
<img width="1920" height="901" alt="Screenshot (2424)" src="https://github.com/user-attachments/assets/d4505a16-ce51-4786-b004-fb1810905315" />
<img width="1920" height="909" alt="Screenshot (2423)" src="https://github.com/user-attachments/assets/26aaa6d2-e8a4-4627-bf82-1818d824e0ec" />
<img width="1920" height="1080" alt="Screenshot (2422)" src="https://github.com/user-attachments/assets/83790bc6-cea5-4343-9fbd-aba5e934c58e" />
<img width="1920" height="915" alt="Screenshot (2421)" src="https://github.com/user-attachments/assets/76589190-dae1-4a86-9dc2-d48f3ead844c" />
<img width="1920" height="905" alt="Screenshot (2420)" src="https://github.com/user-attachments/assets/74e266e0-3cc5-421a-b9dc-dca5b9242ea6" />
<img width="1920" height="1080" alt="Screenshot (2419)" src="https://github.com/user-attachments/assets/13a60490-f6e6-41b3-bb2d-759e933f189a" />
<img width="1920" height="915" alt="Screenshot (2418)" src="https://github.com/user-attachments/assets/c27416a1-f1c0-47fe-a46f-528025efdc78" />
<img width="1920" height="907" alt="Screenshot (2417)" src="https://github.com/user-attachments/assets/b9cd6a3c-bc18-42eb-b9fe-cb9a2664aa87" />
<img width="1920" height="896" alt="Screenshot (2416)" src="https://github.com/user-attachments/assets/2e5c1bac-bd23-4424-9f64-35dcc1e9958b" />


















