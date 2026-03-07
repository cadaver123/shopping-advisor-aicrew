"""
Shopping Advisor crew — assembled with the @CrewBase pattern.

Current pipeline:
  Step 1  product_scout  → shopping search sorted by rating, return top-rated products

Agent roles/goals/backstories live in  config/agents.yaml
Task descriptions/expected outputs live in config/tasks.yaml
Tool wiring and execution details stay here in Python.
"""

import os
import yaml
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, before_kickoff, crew, task
from crewai_tools import SerperDevTool

from tools import ShoppingSearchTool, WebPageReaderTool

_MODELS_CFG = yaml.safe_load(
    (Path(__file__).parent / "config" / "models.yaml").read_text(encoding="utf-8")
)


def _make_llm(cfg_key: str) -> LLM:
    cfg = _MODELS_CFG[cfg_key]
    return LLM(
        model=f"together_ai/{cfg['model']}",
        api_key=os.environ.get("TOGETHER_API_KEY"),
        temperature=cfg["temperature"],
    )


@CrewBase
class ShoppingAdvisorCrew:
    """Shopping advisor powered by Together.ai."""
    search_tool = SerperDevTool()
    shopping_search_tool = ShoppingSearchTool()
    webreader_tool = WebPageReaderTool()
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ------------------------------------------------------------------
    # Pre-kickoff: run the shopping search before any LLM is involved
    # ------------------------------------------------------------------

    @before_kickoff
    def fetch_products(self, inputs: dict) -> dict:
        inputs["shopping_results"] = self.shopping_search_tool._run(inputs["query"])
        return inputs

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    @agent
    def product_scout(self) -> Agent:
        """Step 1 — shopping search sorted by rating, return top-rated products."""
        return Agent(
            config=self.agents_config["product_scout"],
            tools=[],
            llm=_make_llm("muscle_llm"),
            memory=False,
            allow_delegation=False,
            max_iter=2,
            max_execution_time=60,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    @task
    def product_discovery_task(self) -> Task:
        return Task(
            config=self.tasks_config["product_discovery_task"],
            agent=self.product_scout(),
        )

    # ------------------------------------------------------------------
    # Crew
    # ------------------------------------------------------------------

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,
        )
