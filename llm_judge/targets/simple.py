"""A single-call LLM target: static system prompt + optional retrieved context.

This is the most common starting point for local benchmarking - especially for
RAG, where you want to grade the *generation* given some context. It records a
single step so the trace format is identical to a multi-step agent.
"""

from __future__ import annotations

from typing import Optional

from ..config import Config, default_config
from ..dataset import TestCase
from ..models import LLMClient, build_client
from ..schema import Message, ModelSpec, Role, Step, Trace, Usage
from .base import Target

DEFAULT_SYSTEM_PROMPT = "You are a helpful, accurate assistant."

# How retrieved context is rendered into the prompt. Overridable.
DEFAULT_CONTEXT_TEMPLATE = (
    "Use the following context to answer the question.\n\n"
    "<context>\n{context}\n</context>\n\n"
    "Question: {question}"
)


class SimpleTarget(Target):
    def __init__(
        self,
        model: ModelSpec,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        context_template: str = DEFAULT_CONTEXT_TEMPLATE,
        client: Optional[LLMClient] = None,
        config: Optional[Config] = None,
        name: str = "simple",
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.context_template = context_template
        self.config = config or default_config
        self.client = client or build_client(model, self.config)
        self.name = name

    def run(self, case: TestCase) -> Trace:
        # Compose the user message, folding in any pre-supplied context.
        question = case.input if isinstance(case.input, str) else ""
        if case.context:
            context_text = "\n\n".join(
                f"[{c.source or i}] {c.content}"
                for i, c in enumerate(case.context)
            )
            user_content = self.context_template.format(
                context=context_text, question=question
            )
        else:
            user_content = question

        input_messages: list[Message] = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
        ]
        if isinstance(case.input, list):
            input_messages.extend(case.input)
        else:
            input_messages.append(Message(role=Role.USER, content=user_content))

        response = self.client.complete(input_messages)

        step = Step(
            index=0,
            name="generate",
            model=self.model,
            static_prompt=self.system_prompt,
            retrieved_context=list(case.context),
            input_messages=input_messages,
            output_messages=[
                Message(role=Role.ASSISTANT, content=response.text)
            ],
            usage=response.usage,
        )

        return Trace(
            model=self.model,
            case_id=case.id,
            static_prompts={"system": self.system_prompt},
            initial_context=input_messages,
            steps=[step],
            final_output=response.text,
            usage=response.usage,
        )
