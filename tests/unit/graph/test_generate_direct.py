from langchain_core.messages import HumanMessage

from app.graph.nodes.generate_direct import create_generate_direct_node
from tests.unit.graph._mock_llm import chat_model_with_handler, stream_response


def _state():
    return {
        'session_id': 's',
        'user_id': None,
        'messages': [HumanMessage(content='Привет!')],
        'retrieved_chunks': [],
        'tool_calls': [],
        'search_unavailable': False,
    }


async def test_generate_direct_returns_accumulated_streamed_answer():
    chat_model = chat_model_with_handler(lambda request: stream_response(['Здравствуйте', '! Чем могу помочь?']))
    node = create_generate_direct_node(chat_model)

    result = await node(_state())

    assert result['messages'][0].content == 'Здравствуйте! Чем могу помочь?'
