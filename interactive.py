"""Interactive Q&A session for the Shopping Advisor.

Uses the OpenAI SDK pointed at Together.ai — no CrewAI overhead.
"""

import datetime
import json
import os
import re
import uuid
from pathlib import Path

import yaml
from openai import OpenAI

from tools import WebPageReaderTool
from crewai_tools import (
    SerperDevTool
)

_CONFIG_FILE = Path(__file__).parent / "config" / "interactive.yaml"
_MODELS_CFG = yaml.safe_load(
    (Path(__file__).parent / "config" / "models.yaml").read_text(encoding="utf-8")
)


def _load_config() -> dict:
    return yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8"))


class SessionMemory:
    """Persists a single shopping session (query, report, Q&A) to a JSON file."""

    FILE = Path(__file__).parent / "shopping_memory.json"

    @classmethod
    def load(cls) -> dict | None:
        """Return the saved session dict, or None if no memory file exists."""
        if cls.FILE.exists():
            try:
                return json.loads(cls.FILE.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    @classmethod
    def save(cls, query: str, report: str, qa_pairs: list[dict]) -> None:
        cls.FILE.write_text(
            json.dumps(
                {
                    "query": query,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "report": report[:6000],
                    "qa": qa_pairs,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def print_result(result: str) -> None:
    """Render a Markdown string to the terminal (falls back to plain print)."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown

        Console().print(Markdown(result))
    except ImportError:
        print(result)


class InteractiveSession:
    """Post-report Q&A session backed by a direct Together.ai tool-calling loop.

    Maintains a single ``messages`` list across all questions so the model has
    genuine conversation history rather than a reconstructed text block.

    Usage::

        session = InteractiveSession(query, report)
        session.run()

        # Or resume from saved memory:
        memory = InteractiveSession.load_memory()
        session = InteractiveSession(memory["query"], memory["report"], memory["qa"])
        session.run()
    """

    def __init__(
        self, query: str, report: str, qa_pairs: list[dict] | None = None
    ) -> None:
        self.query = query
        self.report = report
        self.qa_pairs: list[dict] = list(qa_pairs or [])

    # ------------------------------------------------------------------ context

    def _build_context(self) -> str:
        ctx = f"## Original Shopping Query\n{self.query}\n\n"
        ctx += f"## Shopping Research Report\n{self.report[:20000]}"
        if len(self.report) > 20000:
            ctx += "\n...[report truncated for brevity]"
        return ctx

    # ------------------------------------------------------------------ tools

    @staticmethod
    def _make_tool_schemas(tools: list) -> list[dict]:
        """Convert BaseTool instances to OpenAI-format schemas for Together.ai."""
        schemas = []
        for tool in tools:
            schema_fn = getattr(tool.args_schema, "model_json_schema", None) or \
                        getattr(tool.args_schema, "schema", None)
            params = schema_fn() if schema_fn else {"type": "object", "properties": {}}
            params.pop("title", None)
            params.pop("description", None)
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params,
                },
            })
        return schemas

    @staticmethod
    def _dispatch_tool(name: str, args: dict, registry: dict) -> str:
        """Call the named tool's ``_run()`` method. Returns an error string on failure."""
        tool = registry.get(name)
        if tool is None:
            return f"[Tool Error] Unknown tool: {name!r}"
        try:
            return str(tool._run(**args))
        except Exception as exc:
            return f"[Tool Error] {name!r} raised {type(exc).__name__}: {exc}"

    @staticmethod
    def _parse_native_tool_calls(content: str) -> list[dict]:
        """Parse DeepSeek's native tool-call token format from message content.

        Together.ai sometimes returns native model tokens in ``content`` instead
        of populating the structured ``tool_calls`` field. The token pattern is::

            <｜tool▁calls▁begin｜>
            <｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{json_args}<｜tool▁call▁end｜>
            <｜tool▁calls▁end｜>

        Returns a list of plain dicts in the same shape as a serialised
        ``tool_calls`` entry so the dispatch loop can handle both formats
        uniformly.
        """
        if "<｜tool▁calls▁begin｜>" not in content:
            return []
        calls = []
        pattern = r"<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>"
        for m in re.finditer(pattern, content, re.DOTALL):
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "function": {
                    "name": m.group(1).strip(),
                    "arguments": m.group(2).strip(),
                },
            })
        return calls

    # ------------------------------------------------------------------ API loop

    def _chat_with_tools(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        tool_registry: dict,
        max_rounds: int = 8,
    ) -> tuple[str, dict]:
        """Run the Together.ai tool-calling loop via the OpenAI SDK.

        Sends ``messages`` to the API, dispatches any tool calls returned by
        the model, appends results, and repeats until a plain text answer is
        produced or ``max_rounds`` is reached.

        Returns ``(answer_text, usage_dict)``.
        """
        client = OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
        )
        _llm_cfg = _MODELS_CFG["interactive_llm"]
        model = _llm_cfg["model"]
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for _ in range(max_rounds):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                temperature=_llm_cfg["temperature"],
                max_tokens=_llm_cfg["max_tokens"],
                timeout=_llm_cfg["timeout"],
            )

            u = response.usage
            if u:
                usage["prompt_tokens"] += u.prompt_tokens or 0
                usage["completion_tokens"] += u.completion_tokens or 0
                usage["total_tokens"] += u.total_tokens or 0

            message = response.choices[0].message

            # Normalise tool calls: structured OpenAI format takes priority;
            # fall back to parsing DeepSeek's native token format from content.
            if message.tool_calls:
                tool_call_list = [
                    {
                        "id": tc.id,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
                content = message.content or ""
            else:
                tool_call_list = self._parse_native_tool_calls(message.content or "")
                # Strip the raw tokens from content so they don't pollute history
                content = "" if tool_call_list else (message.content or "")

            # Serialize to a plain dict — the messages list must stay JSON-serialisable
            msg_dict: dict = {"role": "assistant", "content": content}
            if tool_call_list:
                msg_dict["tool_calls"] = [
                    {"id": tc["id"], "type": "function", "function": tc["function"]}
                    for tc in tool_call_list
                ]
            messages.append(msg_dict)

            if tool_call_list:
                for tc in tool_call_list:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError):
                        fn_args = {}
                    print(f"  [Tool] {fn_name}({fn_args})")
                    result = self._dispatch_tool(fn_name, fn_args, tool_registry)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                continue

            return content.strip(), usage

        return (messages[-1].get("content") or "").strip(), usage

    def _ask(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        tool_registry: dict,
        question: str,
    ) -> tuple[str, dict]:
        """Append ``question`` to ``messages`` and run the tool-call loop."""
        messages.append({"role": "user", "content": question})
        return self._chat_with_tools(messages, tool_schemas, tool_registry)

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        """Start the interactive Q&A loop."""
        search_tool = SerperDevTool()
        webreader_tool = WebPageReaderTool() 
        
        tools_list = [
            search_tool,
            webreader_tool,
        ]
        tool_schemas = self._make_tool_schemas(tools_list)
        # Primary registry keyed by exact tool name ("Expert Review Search", …)
        # Plus snake_case aliases for DeepSeek's native tool-call format
        # (e.g. "expert_review_search") so both formats resolve correctly.
        tool_registry = {t.name: t for t in tools_list}
        tool_registry.update({t.name.lower().replace(" ", "_"): t for t in tools_list})

        system_prompt = _load_config()["interactive_advisor"]["system_prompt"]
        messages: list[dict] = [
            {
                "role": "system",
                "content": system_prompt + "\n" + self._build_context(),
            }
        ]
        # Replay prior Q&A as genuine conversation turns
        for pair in self.qa_pairs:
            messages.append({"role": "user", "content": pair["q"]})
            messages.append({"role": "assistant", "content": pair["a"]})

        session_prompt = 0
        session_completion = 0

        print()
        print("=" * 60)
        print("INTERACTIVE MODE — ask follow-up questions")
        print("Type 'exit' or press Ctrl+C to quit.")
        print("=" * 60)

        while True:
            print()
            try:
                question = input("Your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive mode.")
                break

            if not question or question.lower() in ("exit", "quit", "q"):
                break

            try:
                answer, usage = self._ask(messages, tool_schemas, tool_registry, question)
            except Exception as exc:
                print(f"[Error] {exc}")
                continue

            session_prompt += usage.get("prompt_tokens", 0)
            session_completion += usage.get("completion_tokens", 0)
            print(
                f"  [Tokens: {usage.get('prompt_tokens', 0):,}p"
                f" / {usage.get('completion_tokens', 0):,}c]"
            )

            print()
            print_result(answer)

            self.qa_pairs.append({"q": question, "a": answer})
            SessionMemory.save(self.query, self.report, self.qa_pairs)
            print(f"\n[Session saved — {len(self.qa_pairs)} Q&A pair(s) in memory]")

        if session_prompt:
            print()
            print("=" * 60)
            print(
                f"SESSION TOKENS: {session_prompt:,}p / {session_completion:,}c"
                f"  (total {session_prompt + session_completion:,})"
            )
            print("=" * 60)
