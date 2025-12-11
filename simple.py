from typing import Literal, Any, TypedDict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode

"""
Initialize tools.
"""
@tool
def modem_send_at_receive_resp(at_command: str) -> str:
    """Send AT command to cellular modem and get the response"""
    response = input(f"{at_command} ->")
    return response
tools = [modem_send_at_receive_resp]

"""
Initialize modem and bind tools to it. This will make model aware of what tools
it can use when acomplishing tasks.
"""
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: MessagesState):
    """
    Node invoking LLM. State is just list of messages.
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

"""
Tools node. It extracts tool call from AI message (simple dict, with function
name, args, id and type), executes proper tool from a list and wraps response
int Tool message.
"""
tools_node = ToolNode(tools, handle_tool_errors=True)

"""
Create and compile a graph.
"""
workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

"""
tools_condition, from documentation:
This utility function implements the standard conditional logic for ReAct-style
agents: if the last `AIMessage` contains tool calls, route to the tool execution
node; otherwise, end the workflow.
"""
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
workflow.add_edge("agent", END) # Is this necessary?

app = workflow.compile()

for chunk in app.stream(
    {"messages": [("human", "Check GSM registration status of cellular modem and to which operator it is registered.")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
