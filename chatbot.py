import bs4
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from langchain.schema import AIMessage


class ConversationalRAG:
    def __init__(self, links: List[str], model="gpt-3.5-turbo", temperature=0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.links = links
        self.vectorstore = self._create_vectorstore()
        self.retriever = self.vectorstore.as_retriever()
        self.history_aware_retriever = self._create_history_aware_retriever()
        self.qa_chain = self._create_qa_chain()
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.qa_chain
        )
        self.store = {}

    def _create_vectorstore(self):
        all_splits = []
        for link in self.links:
            loader = WebBaseLoader(link)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
        return Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

    def _create_history_aware_retriever(self):
        contextualize_q_system_prompt = (
            "You are assisting with fine-tuning tasks in Natural Language Processing. "
            "Given a chat history and the latest user question, which might reference previous context, "
            "reformulate the question to be standalone and comprehensible without the chat history. "
            "If you do not know the answer, do not answer the question. Simply indicate that you do not know the answer."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

    def _create_qa_chain(self):
        system_prompt = (
            "You are an assistant specialized in Natural Language Processing fine-tuning tasks. "
            "Use the retrieved context to answer the question accurately. If you do not know the answer, "
            "indicate that you don't know. If the question requires a code fragment to illustrate the answer, "
            "provide the relevant code."
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return create_stuff_documents_chain(self.llm, qa_prompt)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def conversational_rag_chain(self, session_id: str, input_message: str):
        history = self.get_session_history(session_id)
        conversational_rag = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        return conversational_rag.invoke(
            {"input": input_message},
            config={"configurable": {"session_id": session_id}},
        )

    def get_response(self, session_id: str, input_message: str) -> str:
        response = self.conversational_rag_chain(session_id, input_message)
        return response["answer"]

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        hist = []
        history = self.get_session_history(session_id)

        if not history.messages:
            return []
        messages = history.messages
        for msg in messages:
            if isinstance(msg, AIMessage):
                prefix = "AI"
            else:
                prefix = "user"
            content_with_breaks = msg.content.replace("\n", "<br>")
            hist.append({"role": prefix, "content": content_with_breaks})
        return hist
