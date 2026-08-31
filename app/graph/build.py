from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.clients.polza_final_response import FinalResponseGenerator
from app.core.settings import GraphContextSettings, McpSettings
from app.graph.edges import route_after_analyze_intent
from app.graph.nodes.analyze_intent import create_analyze_intent_node
from app.graph.nodes.call_consultation_email import (
    create_call_consultation_email_node,
)
from app.graph.nodes.call_kb_search import create_call_kb_search_node
from app.graph.nodes.generate_direct import create_generate_direct_node
from app.graph.nodes.generate_with_context import create_generate_with_context_node
from app.graph.state import AgentState


def build_graph(
    chat_model: ChatOpenAI,
    final_response_generator: FinalResponseGenerator,
    kb_search_tool: BaseTool,
    consultation_email_tool: BaseTool,
    mcp_settings: McpSettings,
    context_settings: GraphContextSettings,
) -> StateGraph:
    """Собирает граф агента (`AGENT_VERA_ARCHITECTURE.md`,
    раздел "Граф агента Веры"):

    ```
    START -> analyze_intent -> [call_kb_search -> generate_with_context] -> END
                             -> [call_consultation_email -> generate_direct]
                             -> [generate_direct]                        -> END
    ```

    Компилируется **без** checkpointer'а — Redis checkpointer подключается
    отдельно в Этапе 5 (`graph.compile(checkpointer=...)`), чтобы граф
    оставался тестируемым независимо от Redis.
    """
    builder = StateGraph(AgentState)
    builder.add_node(
        'analyze_intent',
        create_analyze_intent_node(
            chat_model,
            kb_search_tool,
            consultation_email_tool,
            context_settings,
        ),
    )
    builder.add_node('call_kb_search', create_call_kb_search_node(kb_search_tool, mcp_settings))
    builder.add_node(
        'call_consultation_email',
        create_call_consultation_email_node(
            consultation_email_tool,
            mcp_settings,
        ),
    )
    builder.add_node(
        'generate_with_context',
        create_generate_with_context_node(final_response_generator, context_settings),
    )
    builder.add_node(
        'generate_direct',
        create_generate_direct_node(final_response_generator, context_settings),
    )

    builder.add_edge(START, 'analyze_intent')
    builder.add_conditional_edges(
        'analyze_intent',
        route_after_analyze_intent,
        {
            'call_kb_search': 'call_kb_search',
            'call_consultation_email': 'call_consultation_email',
            'generate_direct': 'generate_direct',
        },
    )
    builder.add_edge('call_kb_search', 'generate_with_context')
    builder.add_edge('call_consultation_email', 'generate_direct')
    builder.add_edge('generate_with_context', END)
    builder.add_edge('generate_direct', END)
    return builder
