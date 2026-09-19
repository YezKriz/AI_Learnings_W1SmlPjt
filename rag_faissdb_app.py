"""
=============================================================
        LOCAL FAISS RAG APPLICATION
        Using OpenAI GPT-5-mini
=============================================================

FLOW:

PDF
 ↓
Load PDF
 ↓
Split into chunks
 ↓
Create embeddings locally
 ↓
Store embeddings in LOCAL FAISS
 ↓
User asks question
 ↓
Search FAISS
 ↓
Retrieve relevant chunks
 ↓
Send retrieved context + question to GPT-5-mini
 ↓
Generate answer

IMPORTANT:

FAISS                 -> LOCAL
Embeddings            -> LOCAL
PDF                   -> LOCAL
GPT-5-mini            -> OpenAI API

=============================================================
"""


# ============================================================
# 1. IMPORTS
# ============================================================

# PDF loader
from langchain_community.document_loaders import PyPDFLoader

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local Hugging Face embedding model
from langchain_huggingface import HuggingFaceEmbeddings

# FAISS local vector database
from langchain_community.vectorstores import FAISS

# OpenAI Chat Model
from langchain_openai import ChatOpenAI

# Prompt template
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 2. CONFIGURATION
# ============================================================

# PDF file
PDF_FILE = "Study_Guide_Spotify_Like_Web_Application_Architecture.pdf"

# Folder where FAISS will be stored
FAISS_FOLDER = "spotify_faiss_db"

# Local embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenAI model
OPENAI_MODEL = "gpt-5-mini"


# ============================================================
# 3. LOAD PDF
# ============================================================

def load_pdf():

    """
    PURPOSE:

    Read the PDF.

    PyPDFLoader converts the PDF pages into
    LangChain Document objects.
    """

    print("\n[1] Loading PDF...")

    loader = PyPDFLoader(PDF_FILE)

    documents = loader.load()

    print(f"PDF loaded successfully.")
    print(f"Number of pages: {len(documents)}")

    return documents


# ============================================================
# 4. SPLIT PDF INTO CHUNKS
# ============================================================

def split_into_chunks(documents):

    """
    PURPOSE:

    A complete PDF can be too large to search efficiently.

    Therefore we split it into smaller pieces.

    chunk_size = 800 characters

    chunk_overlap = 150 characters

    Example:

        PDF
         |
         +--- Chunk 1
         +--- Chunk 2
         +--- Chunk 3
         +--- Chunk 4

    Overlap helps maintain context between chunks.
    """

    print("\n[2] Splitting PDF into chunks...")

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Number of chunks created: {len(chunks)}")

    return chunks


# ============================================================
# 5. CREATE LOCAL EMBEDDINGS
# ============================================================

def create_embeddings():

    """
    PURPOSE:

    Embeddings convert text into numerical vectors.

    Example:

        "Redis is used for caching"

                     ↓

        [0.12, -0.45, 0.78, ...]

    Similar meanings will have similar vectors.

    IMPORTANT:

    The Hugging Face embedding model runs locally.
    """

    print("\n[3] Loading local embedding model...")

    embeddings = HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    return embeddings


# ============================================================
# 6. CREATE OR LOAD FAISS DATABASE
# ============================================================

def create_or_load_faiss(chunks, embeddings):

    """
    PURPOSE:

    FAISS is our vector database.

    FIRST RUN:

        PDF
         ↓
        Chunks
         ↓
        Embeddings
         ↓
        FAISS
         ↓
        Save to disk


    NEXT RUN:

        Load existing FAISS database.

    Therefore, we don't need to recreate the
    vector database every time.
    """

    print("\n[4] Checking FAISS database...")

    try:

        # ----------------------------------------------------
        # Try to load existing database
        # ----------------------------------------------------

        vector_db = FAISS.load_local(

            FAISS_FOLDER,

            embeddings,

            allow_dangerous_deserialization=True
        )

        print("Existing FAISS database loaded.")

    except Exception:

        # ----------------------------------------------------
        # Create new FAISS database
        # ----------------------------------------------------

        print("FAISS database not found.")

        print("Creating new FAISS database...")

        vector_db = FAISS.from_documents(

            chunks,

            embeddings
        )

        # ----------------------------------------------------
        # Save database locally
        # ----------------------------------------------------

        vector_db.save_local(

            FAISS_FOLDER
        )

        print(
            f"FAISS database saved locally in: "
            f"{FAISS_FOLDER}"
        )

    return vector_db


# ============================================================
# 7. CREATE RETRIEVER
# ============================================================

def create_retriever(vector_db):

    """
    PURPOSE:

    Retriever searches FAISS for the most relevant
    pieces of information.

    Example:

        User:

        "Why is Redis used?"

                 ↓

              Retriever

                 ↓

              FAISS

                 ↓

        Relevant chunks returned

    k = 4 means retrieve the top 4 relevant chunks.
    """

    print("\n[5] Creating retriever...")

    retriever = vector_db.as_retriever(

        search_kwargs={
            "k": 4
        }
    )

    return retriever


# ============================================================
# 8. CREATE OPENAI LLM
# ============================================================

def create_llm():

    """
    PURPOSE:

    Create OpenAI GPT-5-mini.

    FAISS remains LOCAL.

    Only the retrieved context and user's question
    are sent to OpenAI.
    """

    print("\n[6] Connecting to OpenAI...")

    llm = ChatOpenAI(

        model=OPENAI_MODEL,

        temperature=0
    )

    print("OpenAI GPT-5-mini ready.")

    return llm


# ============================================================
# 9. CREATE RAG PROMPT
# ============================================================

def create_prompt():

    """
    PURPOSE:

    Tell the LLM how to answer.

    We explicitly instruct the model to use
    the retrieved context and not invent information.
    """

    prompt = ChatPromptTemplate.from_template(

        """
You are a helpful study assistant.

Your job is to answer the user's question using
ONLY the information provided in the CONTEXT.

Rules:

1. Use the retrieved context.
2. Do not invent information.
3. If the answer is not available in the context,
   say:

   "I could not find that information in the
   provided document."

4. Explain the answer in simple language.
5. Use examples when the context provides them.

================ CONTEXT ================

{context}

================ QUESTION ================

{question}

================ ANSWER ================
"""
    )

    return prompt


# ============================================================
# 10. FORMAT DOCUMENTS
# ============================================================

def format_documents(documents):

    """
    PURPOSE:

    Convert retrieved LangChain Documents into
    a single text string.

    This becomes the CONTEXT given to the LLM.
    """

    context = ""

    for document in documents:

        context += document.page_content

        context += "\n\n-------------------------\n\n"

    return context


# ============================================================
# 11. RAG QUESTION ANSWERING
# ============================================================

def ask_question(
    question,
    retriever,
    llm,
    prompt
):

    """
    PURPOSE:

    This function performs the complete RAG operation.

    STEP 1:
        User asks question.

    STEP 2:
        Search FAISS.

    STEP 3:
        Retrieve relevant chunks.

    STEP 4:
        Combine chunks into context.

    STEP 5:
        Create prompt.

    STEP 6:
        Send context + question to GPT-5-mini.

    STEP 7:
        Return answer.
    """

    print("\nSearching local FAISS database...")

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    documents = retriever.invoke(question)

    print(
        f"Retrieved {len(documents)} relevant chunks."
    )

    # --------------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------------

    context = format_documents(documents)

    # --------------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------------

    messages = prompt.invoke({

        "context": context,

        "question": question
    })

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    response = llm.invoke(messages)

    return response.content, documents


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print("       SPOTIFY ARCHITECTURE - RAG APPLICATION")
    print("=" * 65)

    # --------------------------------------------------------
    # LOAD PDF
    # --------------------------------------------------------

    documents = load_pdf()

    # --------------------------------------------------------
    # SPLIT PDF
    # --------------------------------------------------------

    chunks = split_into_chunks(documents)

    # --------------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------------

    embeddings = create_embeddings()

    # --------------------------------------------------------
    # CREATE / LOAD FAISS
    # --------------------------------------------------------

    vector_db = create_or_load_faiss(

        chunks,

        embeddings
    )

    # --------------------------------------------------------
    # CREATE RETRIEVER
    # --------------------------------------------------------

    retriever = create_retriever(

        vector_db
    )

    # --------------------------------------------------------
    # CREATE OPENAI LLM
    # --------------------------------------------------------

    llm = create_llm()

    # --------------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------------

    prompt = create_prompt()

    # --------------------------------------------------------
    # APPLICATION READY
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("                 RAG APPLICATION READY")
    print("=" * 65)

    print("\nAsk questions about your Spotify architecture PDF.")

    print("\nExample questions:")

    print("1. What is the role of Redis?")

    print("2. Why is Kafka used?")

    print("3. Explain the music streaming flow.")

    print("4. What is the purpose of CDN?")

    print("5. How does the recommendation system work?")

    print("6. What is RAG in the Spotify AI Assistant?")

    print("\nType 'exit' to close the application.")

    # ========================================================
    # QUESTION LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # GET USER QUESTION
        # ----------------------------------------------------

        question = input("\nYou: ")

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if question.lower() in [

            "exit",
            "quit",
            "q"
        ]:

            print("\nRAG application stopped.")

            break

        # ----------------------------------------------------
        # EMPTY QUESTION
        # ----------------------------------------------------

        if not question.strip():

            print("enter your question here")

            continue

        # ----------------------------------------------------
        # RUN RAG
        # ----------------------------------------------------

        try:

            answer, source_documents = ask_question(

                question,

                retriever,

                llm,

                prompt
            )

            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            print("\n")
            print("=" * 65)
            print("                         ANSWER")
            print("=" * 65)

            print(answer)

            # ------------------------------------------------
            # DISPLAY SOURCE PAGES
            # ------------------------------------------------

            print("\n")
            print("=" * 65)
            print("                  RETRIEVED SOURCE PAGES")
            print("=" * 65)

            pages = set()

            for document in source_documents:

                page = document.metadata.get("page")

                if page is not None:

                    # PyPDFLoader page numbering starts at 0
                    pages.add(page + 1)

            if pages:

                print(
                    "Relevant PDF pages:",
                    sorted(pages)
                )

            else:

                print(
                    "Source page information unavailable."
                )

        # ----------------------------------------------------
        # ERROR HANDLING
        # ----------------------------------------------------

        except Exception as error:

            print("\n")
            print("=" * 65)
            print("                         ERROR")
            print("=" * 65)

            print(error)

            print("\nPossible causes:")

            print(
                "1. OPENAI_API_KEY is not configured."
            )

            print(
                "2. Required Python package is missing."
            )

            print(
                "3. PDF file is not in the same folder."
            )

            print(
                "4. Internet connection is unavailable."
            )


# ============================================================
# 13. START APPLICATION
# ============================================================

if __name__ == "__main__":

    main()
