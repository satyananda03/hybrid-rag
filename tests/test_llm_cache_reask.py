"""cached_chain's corrective retry: no database, network or model."""

import asyncio

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from hybrid_rag.core.llm_cache import cached_chain


class Answer(BaseModel):
    text: str


def _flaky_llm(replies: list):
    """A fake structured model: pops one reply per call, records the messages it got."""
    calls: list = []

    async def call(messages):
        calls.append(messages)
        return Answer.model_validate(replies.pop(0))

    return RunnableLambda(call), calls


def _chain(llm, reask_invalid):
    prompt = ChatPromptTemplate.from_messages([("human", "{q}")])
    return cached_chain(prompt, llm, cache_type="test", model_id="m", output_model=Answer,
                        reask_invalid=reask_invalid)


def test_invalid_reply_is_asked_again_with_its_error():
    llm, calls = _flaky_llm([{"wrong": "x"}, {"text": "ok"}])
    assert asyncio.run(_chain(llm, True).ainvoke({"q": "hi"})) == Answer(text="ok")
    assert len(calls) == 2 and len(calls[1]) == len(calls[0]) + 1
    feedback = calls[1][-1]
    assert isinstance(feedback, HumanMessage) and "text" in feedback.content  # names the missing field


def test_second_invalid_reply_propagates():
    llm, _ = _flaky_llm([{"wrong": 1}, {"wrong": 2}])
    with pytest.raises(Exception):
        asyncio.run(_chain(llm, True).ainvoke({"q": "hi"}))


def test_without_reask_the_error_propagates_at_once():
    llm, calls = _flaky_llm([{"wrong": "x"}, {"text": "ok"}])
    with pytest.raises(Exception):
        asyncio.run(_chain(llm, False).ainvoke({"q": "hi"}))
    assert len(calls) == 1
