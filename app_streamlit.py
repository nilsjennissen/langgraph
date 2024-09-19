import argparse
import glob
import logging
import os
import pprint
import random
from typing import Dict, List, TypedDict

import pandas as pd
import rich
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, validator, PositiveInt, ValidationError


llm_model = "gpt4o"

# Load PDFs from folder docs
pdf_files = glob.glob("./docs/*.pdf")
logging.info(f"Found {len(pdf_files)} PDF files")
print(f"Found {len(pdf_files)} PDF files")


def initialize_embeddings(json=False):
    if "OPENAI_API_KEY" not in os.environ:
        embeddings = OllamaEmbeddings(model=llm_model)

    else:
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    return embeddings


def load_and_process_pdfs(folder_path):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files")
    print(f"Found {len(pdf_files)} PDF files")

    docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs.extend(loader.load())

    df_docs = pd.DataFrame(
        [
            {
                "page_content": doc.page_content,
                "source": doc.metadata["source"],
                "metadata": doc.metadata,
            }
            for doc in docs
        ]
    )

    # Concatenate pages belonging to the same document
    concat_docs = []
    current_doc = None
    for doc in docs:
        if current_doc is None:
            current_doc = doc
        elif current_doc.metadata["source"] == doc.metadata["source"]:
            current_doc.page_content += "\n\n" + doc.page_content
        else:
            concat_docs.append(current_doc)
            current_doc = doc

    concat_docs.append(current_doc)
    docs = concat_docs
    logging.info(f"Loaded {len(docs)} documents")
    print(f"Loaded {len(docs)} documents")

    return docs


# 3. Setup Document Processing
def setup_document_processing(docs):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs)
    logging.info(f"Split {len(docs)} documents into {len(doc_splits)} chunks")
    print(f"Split {len(docs)} documents into {len(doc_splits)} chunks")

    return doc_splits


# Setup retrievers and LLMs
embeddings = initialize_embeddings()

# Load and process PDFs
docs = load_and_process_pdfs("./docs")

# Setup document processing
doc_splits = setup_document_processing(docs)


def initialize_retrievers():
    keyword_retriever = BM25Retriever.from_documents(doc_splits, similarity_top_k=2)
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )
    vectorstore = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "top_k": 2},
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore, keyword_retriever], weights=[0.2, 0.8]
    )

    return ensemble_retriever


ensemble_retriever = initialize_retrievers()

# Router LLM (Question Routing)
llm = ChatOpenAI(model=llm_model, format="json", temperature=0)
question_router_prompt = PromptTemplate(
    template="""You are an expert at routing a user question to a vectorstore or normal LLM call.
    Use the vectorstore for questions on LLM osram lamps, bulbs, products and specifications.
    You do not need to be stringent with the keywords in the question related to these topics.
    Otherwise, use normal LLM call. Give a binary choice 'normal_llm' or 'vectorstore' based on the question.
    Return the a JSON with a single key 'datasource' and no preamble or explanation.
    Question to route: '''{question}'''""",
    input_variables=["question"],
)
question_router = question_router_prompt | llm | JsonOutputParser()

# Normal LLM
llm = ChatOpenAI(model=llm_model, temperature=0)
prompt = PromptTemplate(
    template="""You are a question-answering system for Osram products. Respond politely and in a customer-oriented manner.
    If you don't know the answer, refer to the specifics of the question. What exactly is the customer looking for?
    Return a JSON with a single key 'generation' and no preamble or explanation. Be open and talkative.
    Here is the user question: '''{question}'''""",
    input_variables=["question"],
)
answer_normal = prompt | llm | StrOutputParser()

# Question Re-writer
llm = ChatOpenAI(model=llm_model, temperature=0)
question_rewriter_prompt = PromptTemplate(
    template="""You are a question re-writer that converts an input question to a better version optimized for vectorstore retrieval.
    Question: '''{question}'''.
    Improved question:""",
    input_variables=["question"],
)
question_rewriter = question_rewriter_prompt | llm | StrOutputParser()

# Generation (RAG Prompt)
llm = ChatOpenAI(model=llm_model, temperature=0)
rag_prompt = PromptTemplate(
    template="""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: '''{question}'''

    Here is the retrieved document:
    ------------
    Context: {context}
    ------------
    Answer:""",
    input_variables=["question", "context"],
)
rag_chain = rag_prompt | llm | StrOutputParser()

# Grading
llm = ChatOpenAI(model=llm_model, temperature=0)
retrieval_grader_prompt = PromptTemplate(
    template="""You are a grader evaluating the relevance of a retrieved document to a user question.
    Here is the retrieved document:
    ------------
    {document}
    ------------
    Here is the user question: '''{question}'''
    If the document contains keywords or matching product codes that are related to the user's question, rate it as relevant.
    It doesn't need to be a strict test. The goal is to filter out erroneous retrievals.
    Give a binary rating of 'yes' or 'no' to indicate whether the document is relevant to the question.
    Provide the binary rating as JSON with a single key 'score' and without any preamble or explanation.""",
    input_variables=["question", "document"],
)
retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

llm = ChatOpenAI(model=llm_model, temperature=0)
hallucination_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in facts of the document. \n
    Here are the documents:
    ----------
    {documents}
    ----------
    Here is the answer: '''{generation}'''
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    Always answer with 'yes'""",
    input_variables=["generation", "documents"],
)
hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

llm = ChatOpenAI(model=llm_model, temperature=0)
answer_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question.
    Here is the answer:
    -------
    {generation}
    -------
    Here is the question: '''{question}'''
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    Always reply with 'yes'.""",
    input_variables=["generation", "question"],
)
answer_grader = answer_grader_prompt | llm | JsonOutputParser()


def chatbot_page():

    st.title("Osram Product Chatbot")
    st.write("You can ask questions about Osram products or documents.")

    # Define the graph state
    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[str]
        retries: int

    # Function definitions for the state graph
    def retrieve(state):
        """Retrieve documents for the question."""
        logging.info("🗄️ Retrieving documents...")
        question = state["question"]
        documents = ensemble_retriever.invoke(question)
        logging.info(f"Retrieved {len(documents)} documents")
        return {"documents": documents, "question": question}

    def generate(state):
        """Generate an answer using retrieved documents."""
        logging.info("🤖 Generating answer...")
        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"context": documents, "question": question})
        logging.info(f"Generated answer: {generation[:20]}")
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state):
        """Grade the relevance of retrieved documents."""
        logging.info("💎 Grading documents...")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = [
            d
            for d in documents
            if retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )["score"]
            == "yes"
        ]
        logging.info(f"Filtered {len(documents) - len(filtered_docs)} documents")
        return {"documents": filtered_docs, "question": question}

    def transform_query(state):
        """Re-write the query to improve retrieval."""
        logging.info("📝  Transforming the query...")
        question = state["question"]
        better_question = question_rewriter.invoke({"question": question})
        logging.info(f"Improved question: {better_question}")
        return {"documents": state["documents"], "question": better_question}

    def normal_llm(state):
        logging.info("💭  Calling normal LLM...")
        question = state["question"]
        answer = answer_normal.invoke({"question": question})
        logging.info(f"Answer: {answer[:20]}")
        return {"question": question, "generation": answer}

    def route_question(state):
        """Route the question to either vectorstore or normal LLM."""
        logging.info("⚖️  Routing the question...")
        question = state["question"]
        source = question_router.invoke({"question": question})
        logging.info(f"Routing to: {source}")
        return "normal_llm" if source["datasource"] == "normal_llm" else "vectorstore"

    def decide_to_generate(state):
        """Decide whether to generate or rephrase the query."""
        logging.info("🗯️  Deciding to generate or rephrase the query...")
        return "transform_query" if not state["documents"] else "generate"

    def grade_generation(state):
        """Grade the generation and its relevance."""
        logging.info("🔍 Grading the generation...")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        hallucination_score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )["score"]
        logging.info(f"Grounded in the documents: {hallucination_score}")
        return "useful" if hallucination_score == "yes" else "not supported"

    # Create and compile the state graph
    workflow = StateGraph(GraphState)
    workflow.add_node("normal_llm", normal_llm)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_conditional_edges(
        START, route_question, {"normal_llm": "normal_llm", "vectorstore": "retrieve"}
    )
    workflow.add_edge("normal_llm", END)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate", grade_generation, {"not supported": "generate", "useful": END}
    )

    app = workflow.compile()

    user_input = st.text_area("Ask a question about Osram products", height=100)

    if user_input:
        with st.spinner("Answering your question..."):
            # Running the graph with the user's question
            inputs = {"question": user_input}
            result = None

            try:
                for output in app.stream(inputs):
                    for key, value in output.items():
                        if "generation" in value:
                            result = value["generation"]

            except Exception as e:
                st.error(f"Error occurred: {e}")

        if result:
            st.write(f"\nAnswer: {result}\n")
        else:
            print("\nSorry, I couldn't find an answer to that question.\n")


chatbot_page()
