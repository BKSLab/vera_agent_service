from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.core.settings import McpSettings
from app.graph.edges import route_after_analyze_intent
from app.graph.nodes.analyze_intent import create_analyze_intent_node
from app.graph.nodes.call_kb_search import create_call_kb_search_node
from app.graph.nodes.generate_direct import create_generate_direct_node
from app.graph.nodes.generate_with_context import create_generate_with_context_node
from app.graph.state import AgentState


def build_graph(chat_model: ChatOpenAI, kb_search_tool: BaseTool, mcp_settings: McpSettings) -> StateGraph:
    """Собирает граф агента итерации 1 (`AGENT_VERA_ARCHITECTURE.md`,
    раздел "Граф агента Веры (итерация 1)"):

    ```
    START -> analyze_intent -> [call_kb_search -> generate_with_context] -> END
                             -> [generate_direct]                        -> END
    ```

    Компилируется **без** checkpointer'а — Redis checkpointer подключается
    отдельно в Этапе 5 (`graph.compile(checkpointer=...)`), чтобы граф
    оставался тестируемым независимо от Redis.
    """
    builder = StateGraph(AgentState)
    builder.add_node('analyze_intent', create_analyze_intent_node(chat_model, kb_search_tool))
    builder.add_node('call_kb_search', create_call_kb_search_node(kb_search_tool, mcp_settings))
    builder.add_node('generate_with_context', create_generate_with_context_node(chat_model))
    builder.add_node('generate_direct', create_generate_direct_node(chat_model))

    builder.add_edge(START, 'analyze_intent')
    builder.add_conditional_edges(
        'analyze_intent',
        route_after_analyze_intent,
        {'call_kb_search': 'call_kb_search', 'generate_direct': 'generate_direct'},
    )
    builder.add_edge('call_kb_search', 'generate_with_context')
    builder.add_edge('generate_with_context', END)
    builder.add_edge('generate_direct', END)
    return builder
