import streamlit as st

from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.tools import tool

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.types import interrupt, Command

from backend import graph_factory

@st.cache_resource
def init_graph():
    return graph_factory()
graph = init_graph()


#@st.cache_data
def draw_chat():
    if "chat_messages" in st.session_state:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])


def draw_confirmation_dialog():
    @st.dialog("Confirm action")
    def confirm_weather_check_dialog(query: str):
        st.write(query)
        feedback = ""
        if st.button("Accept"):
            feedback = "accept"
        if st.button("Deny"):
            feedback = "deny"
        if feedback != "":
            result = graph.resume(feedback)
            st.session_state["chat_messages"] = result["messages"]
            if result["interrupt"]:
                st.session_state["interrupt"] = result["interrupt"][0].value
            else:
                st.session_state.pop("interrupt", None)
            st.rerun()
    if "interrupt" in st.session_state:
        confirm_weather_check_dialog(st.session_state["interrupt"]["query"])


#If user provides input, run it through a graph.
prompt = st.chat_input("Ask AI...")
if prompt:
    # Run until completion or interrupt
    result = graph.invoke(prompt)
    st.session_state["chat_messages"] = result["messages"]
    if result["interrupt"]:
        st.session_state["interrupt"] = result["interrupt"][0].value

draw_chat()
draw_confirmation_dialog()
