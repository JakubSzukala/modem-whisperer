from typing import Literal, Any, TypedDict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.base import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.schema.runnable.config import RunnableConfig

from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.types import interrupt, Command


GoogleModel = Literal["gemini-2.5-pro", "gemini-2.5-flash"]

class WhispererChatMessage(TypedDict):
    """
    Simplified langgraph message.
    """
    role: Literal["user", "assistant", "tool"]
    content: str


class WhispererChatResult(TypedDict):
    """
    Chat consisting of simplified messages, used by frontend.
    """
    messages: List[WhispererChatMessage]
    interrupt: Any


class WhispererConfirmationInterrupt(TypedDict):
    """
    Simple object containing query to display to a user in the frontend.
    """
    query: str


class Graph:
    def __init__(self, llm: BaseChatModel) -> None:
        """
        Given model, build an inference graph.
        """
        # Initialize model
        self.llm = llm.bind_tools([StructuredTool.from_function(self._get_weather)])

        # Construct graph
        builder = StateGraph(MessagesState)
        builder.add_node(self._chat)
        builder.add_node(ToolNode(tools=[self._get_weather]))
        builder.add_edge(START, "_chat")
        builder.add_conditional_edges(
            "_chat",
            tools_condition,
            {"tools" : "tools", END : END}
        )
        builder.add_edge("tools", "_chat")
        config = {"configurable": {"thread_id": "hello123"}}
        self.runnable_cfg = RunnableConfig(**config) # type: ignore
        self.graph = builder.compile(checkpointer=InMemorySaver())


    def invoke(self, human_message: str) -> WhispererChatResult:
        """
        Run the graph with given input and return the result
        """
        # Runt the graph
        response = self.graph.invoke(
            {"messages" : [HumanMessage(content=human_message)]},
            self.runnable_cfg
        )

        # Convert messages to whisperer native format
        messages = [
            self._langgraph_to_whisperer_message(msg) for msg in response["messages"]
        ]

        # Return messages and optionally interrupt
        return {
            "messages" : messages,
            "interrupt" : response.get("__interrupt__", None)
        }


    def resume(self, value: Any):
        """
        Resume interrupted graph and return the result
        """
        # Resume interrupted graph
        response = self.graph.invoke(
            Command(resume=value),
            self.runnable_cfg
        )

        # Convert messages to whisperer native format
        messages = [
            self._langgraph_to_whisperer_message(msg) for msg in response["messages"]
        ]

        # Return messages and optionally another interrupt
        return {
            "messages" : messages,
            "interrupt" : response.get("__interrupt__", None)
        }


    def _get_weather(self, city: str) -> str:
        """
        Use this tool to get weather data in given city
        """
        confirmation_interrupt: WhispererConfirmationInterrupt = {
            "query" : f"Do You allow for weather lookup in the city {city}?"
        }
        response = interrupt(confirmation_interrupt)
        if response == "accept":
            return f"It's always sunny in {city}"
        elif response == "deny":
            return f"You lack privileges to check weather in {city}"
        else:
            raise ValueError(f"Unknown feedback: {response}")


    def _chat(self, state: MessagesState):
        return {"messages" : [self.llm.invoke(state["messages"])]}


    def _langgraph_to_whisperer_message(
            self,
            message: HumanMessage | AIMessage | ToolMessage
    ) -> WhispererChatMessage:
        """
        Convert langgraph message to simplified format used by frontend.
        """
        role: Literal["user", "assistant", "tool"] = "tool"
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage) and message.content:
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = "tool"
        return {"role" : role, "content" : message.content} # type: ignore


def graph_factory(model: GoogleModel="gemini-2.5-flash") -> Graph:
    return Graph(ChatGoogleGenerativeAI(model=model))


