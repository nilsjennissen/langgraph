# 🚀 Advanced Adaptive RAG Chatbot with Flexible LLM Integration (Langgraph)

## 🧭 Project Overview

This project implements a sophisticated chatbot leveraging Retrieval-Augmented Generation (RAG) with flexible Language Model (LLM) integration. It's designed to provide accurate, context-aware responses to user queries by combining document retrieval, question rewriting, and multi-stage answer generation and grading.

Key features:
- 🔄 Flexible LLM integration supporting OpenAI, Ollama, and other providers via LangChain
- 🧠 Intelligent question routing between vectorstore and direct LLM calls
- 🔍 Ensemble retrieval combining keyword (BM25) and semantic search
- ✍️ Dynamic question rewriting for improved retrieval
- 🎭 Multi-stage answer generation with self-evaluation
- 🔧 Easily customizable for different domains and document types

## 🚧 Prerequisites

- Python 3.10+
- API keys for chosen LLM providers (e.g., OpenAI, Anthropic)
- (Optional) Ollama for local LLM support

## 🎛 Project Setup

1. Clone the repository:
```bash
git clone https://github.com/nilsjennissen/langgraph-rag-chatbot.git
cd langgraph-rag-chatbot
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Set up environment variables:
```bash
cp .env.example .env

CopyEdit `.env` and add your API keys:
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
Add other API keys as needed
```
5. Prepare your document corpus:
- Place PDF documents in the `./docs` folder
- Alternatively, modify the `load_documents()` function in `app.py` to support other document types or sources

6. Customize the system prompt:
- Edit the `SYSTEM_PROMPT` constant in `app.py` to align with your specific use case or domain

7. Run the chatbot:
```bash
python app.py
```
```bash
streamlit run app.py
```
## 📦 Project Structure
```bash
langgraph-pdf-chat/
│
├── app.py                    # Main entrypoint for the chatbot
├── app_streamlit.py          # Streamlit interface for the chatbot
├── .gitignore                # Git ignore file
├── .pre-commit-config.yaml   # Pre-commit configuration file
├── requirements.txt          # Project dependencies
├── .env                      # Environment variables (create from .env.example)
├── .env.example              # Example environment variable file
├── notebooks/                # Directory for unit tests (optional)
│   ├── pdf-chat.ipynb        # Jupyter notebook for testing the chatbot (optional)
│   └── ...
├── docs/                     # Folder for storing PDF documents (or other supported formats)
│   ├── doc1.pdf
│   ├── doc2.pdf
│   └── ...
├── tests/                    # Directory for unit tests (optional)
│   ├── test1.py
│   └── ...
│
└── README.md                 # Project documentation
```

## 🔀 Reproducing the Graph

The chatbot's logic is implemented using LangGraphs GraphState. Here's how to reproduce and customize the graph.

1. Define the graph state:
   ```python
    class GraphState(TypedDict):
       question: str
       generation: str
       documents: List[str]
       retries: int

    # Implement node functions
    # retrieve: Fetch relevant documents
    # generate: Produce an answer using retrieved documents
    # grade_documents: Evaluate document relevance
    # transform_query: Rewrite the query for better retrieval
    # normal_llm: Direct LLM call without retrieval
    # route_question: Decide between vectorstore and normal LLM
    # decide_to_generate: Choose between generation and query transformation
    # grade_generation: Evaluate the generated answer


    # Create and compile the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("normal_llm", normal_llm)
    workflow.add_node("retrieve", retrieve)
    # ... Add other nodes ...

    # Add edges
    workflow.add_conditional_edges(
        START, route_question,
        {"normal_llm": "normal_llm", "vectorstore": "retrieve"}
    )
    workflow.add_edge("normal_llm", END)
    # ... Add other edges ...

    # Compile the graph
    app = workflow.compile()

    # Run the graph:
    inputs = {"question": user_input}
   
    for output in app.stream(inputs):
       for key, value in output.items():
           if "generation" in value:
               result = value["generation"]
    # Process output```


Customize the graph by modifying node functions, adding new nodes, or changing the edge connections to alter the chatbot's behavior.
# 🔧 LLM Provider Configuration
To switch between LLM providers:

Update the LLM initialization in main.py:
```python
# For OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# For Ollama (local)
from langchain_community.chat_models import ChatOllama
llm = ChatOllama(model="llama2")

# For LangChain Groq
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# For HuggingFace Endpoint
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat_model = ChatHuggingFace(llm=llm)

# For HuggingFace Pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

chat_model = ChatHuggingFace(llm=llm)
```


Ensure the corresponding API key is set in your .env file.
Update the requirements.txt file if necessary to include the appropriate LangChain integration package.

# 🗄️ Data
The chatbot processes documents stored in the ./docs/ folder to build its knowledge base. By default, it supports PDF files, but you can extend load_documents() in main.py to handle additional formats or data sources.

# 🛠 Customization

System Prompt: Modify the SYSTEM_PROMPT in main.py to tailor the chatbot's behavior and domain expertise.  
LLM Provider: Change the LLM initialization as described in the "LLM Provider Configuration" section.  
Document Loading: Extend the load_documents() function to support additional file types or data sources.  
Retrieval Strategy: Adjust the weights in the EnsembleRetriever to fine-tune the balance between keyword and semantic search.  
Evaluation Criteria: Modify the grading prompts to implement custom evaluation logic for document relevance and answer quality.  

# 📚 References

[Ollama](https://ollama.com/)
[OpenAI API Documentation](https://platform.openai.com/docs/overview)  
[LangChain Documentation](https://python.langchain.com/v0.2/docs/introduction/)  
[LangGraph Documentation](https://langchain-ai.github.io/langgraph/)




# 🏆 Conclusion
This Advanced RAG Chatbot showcases the power of combining retrieval-augmented generation with flexible LLM integration. Its modular design and customization options make it adaptable to various domains and LLM providers, providing a solid foundation for building sophisticated question-answering systems.
# 🤝 Contributions
Contributions are welcome! Please feel free to submit a Pull Request with improvements, bug fixes, or new features. For major changes, please open an issue first to discuss what you would like to change.
# 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
