"""CLI entry point for the Demoshop product assistant.

A conversational agent is run in a read-eval loop. Retrieval is delegated to
``src.tools.retrieval`` via a registered tool; the agent decides when to query
the knowledge base.
"""

from pathlib import Path

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.config import settings
from src.tools import retrieval as retrieval_tool


def _load_system_prompt(filepath: str | Path) -> str:
    """Load agent instructions from disk.

    Raises:
        FileNotFoundError: When the prompt file does not exist.
    """
    prompt_path = Path(filepath)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def agent_load() -> Agent:
    """Construct an agent with the set chat model and retrieval tool."""
    model = OpenAIChatModel(
        settings.llm_model,
        provider=OpenAIProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        ),
    )

    agent = Agent(
        model,
        instructions=_load_system_prompt(settings.prompt_path),
    )

    @agent.tool_plain
    def retrieve_context(query: str) -> str:
        """Search the knowledge base for product, vendor, or platform information."""
        # Implementation is kept in the retrieval module to separate agent wiring
        # from embedding, search, and reranking logic.
        return retrieval_tool.retrieve_context(query)

    return agent


def main() -> None:
    """Run the interactive CLI until exit, quit, or an unrecoverable error."""

    print("Agentic RAG — Product Assistant")
    print(f"Collection: {settings.qdrant_collection}")
    print("Type 'exit' or 'quit' to stop.\n")

    agent: Agent = agent_load()
    history = []

    while True:
        try:
            question = input("[You] ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                print("[Assistant] Goodbye!")
                break

            # Message history is passed across turns so follow-up questions retain context.
            result: AgentRunResult = agent.run_sync(question, message_history=history)
            
            print(f"\n[Assistant] {result.output}\n")

            history = result.all_messages()

        except KeyboardInterrupt:
            print("\n[Assistant] Goodbye.")
            break
        except Exception as e:
            print(f"\n[Assistant] Error: {e}")
            break


if __name__ == "__main__":
    main()
