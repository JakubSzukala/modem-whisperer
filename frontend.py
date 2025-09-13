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

@tool
def get_weather(city: str) -> str:
    """Use this tool to get weather data in given city"""
    response = interrupt({"city" : city})
    if response == "accept":
        return f"It's always sunny in {city}"
    elif response == "deny":
        return f"You lack privileges to check weather in {city}"
    else:
        raise ValueError(f"Unknown feedback: {response}")


@st.cache_resource
def init_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools([get_weather])

llm = init_llm()
checkpointer = InMemorySaver()

@st.cache_data
def chat(state: MessagesState):
    return {"messages" : [llm.invoke(state["messages"])]}


@st.cache_resource
def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node(chat)
    builder.add_node(ToolNode(tools=[get_weather])) # Node to call tools
    builder.add_edge(START, "chat")
    builder.add_conditional_edges(
        "chat",
        tools_condition, # If last msg has tool_calls it will route to "tools" node
        {"tools" : "tools", END : END} # Optional mapping
    )
    builder.add_edge("tools", "chat")
    return builder.compile(checkpointer=checkpointer)

graph = build_graph()
#st.text(graph.get_graph().draw_ascii())

#@st.cache_data
def draw_chat():
    if "chat_messages" in st.session_state:
        for message in st.session_state.chat_messages:
            persona = None
            if isinstance(message, HumanMessage):
                persona = "user"
            if isinstance(message, AIMessage) and message.content:
                persona = "assistant"
            if persona:
                with st.chat_message(persona):
                    st.write(message.content)

config = {"configurable": {"thread_id": "hello123"}}
runnable_cfg = RunnableConfig(**config)

def draw_feedback():
    @st.dialog("Accept weather check")
    def get_weather_lookup_feedback(city):
        st.write(f"Do you allow for weather lookup in {city}?")
        feedback = ""
        if st.button("Accept"):
            feedback = "accept"
        if st.button("Deny"):
            feedback = "deny"
        if feedback != "":
            response = graph.invoke(Command(resume=feedback), config=runnable_cfg)
            st.session_state["chat_messages"] = response["messages"]
            if "__interrupt__" in response:
                st.session_state["interrupt"] = response["__interrupt__"][0].value["city"]
            else:
                st.session_state.pop("interrupt", None)
            st.rerun()

    if "interrupt" in st.session_state:
        get_weather_lookup_feedback(st.session_state["interrupt"])


#If user provides input, run it through a graph.
prompt = st.chat_input("Ask AI...")
if prompt:
    # Run until completion or interrupt
    response = graph.invoke(
        {"messages" : [HumanMessage(content=prompt)]},
        config=runnable_cfg)

    st.session_state["chat_messages"] = response["messages"]

    if "__interrupt__" in response:
        st.session_state["interrupt"] = response["__interrupt__"][0].value["city"]

draw_chat()
draw_feedback()
