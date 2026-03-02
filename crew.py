"""
Shopping Advisor crew — assembled with the @CrewBase pattern.

Agent roles/goals/backstories live in  config/agents.yaml
Task descriptions/expected outputs live in config/tasks.yaml
Tool wiring and execution details stay here in Python.
"""

import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task

from tools import (
    AllegroSearchTool,
    AliExpressSearchTool,
    ReviewSearchTool,
    RedditSearchTool,
    WebPageReaderTool,
)


@CrewBase
class ShoppingAdvisorCrew:
    """Multi-agent shopping advisor powered by Together.ai / DeepSeek-V3."""

    # Paths are resolved relative to this file by @CrewBase
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    @property
    def llm(self) -> LLM:
        model = os.environ.get("TOGETHER_MODEL", "deepseek-ai/DeepSeek-V3")
        return LLM(
            model=f"together_ai/{model}",
            api_key=os.environ.get("TOGETHER_API_KEY"),
            temperature=0.3,
            max_tokens=4096,
        )

    # ------------------------------------------------------------------
    # Agents  (config= pulls role/goal/backstory from agents.yaml)
    # ------------------------------------------------------------------

    @agent
    def review_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["review_researcher"],
            tools=[ReviewSearchTool(), WebPageReaderTool()],
            llm=self.llm,
            verbose=True,
            memory=False,
        )

    @agent
    def reddit_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["reddit_researcher"],
            tools=[RedditSearchTool(), WebPageReaderTool()],
            llm=self.llm,
            verbose=True,
            memory=False,
        )

    @agent
    def allegro_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["allegro_researcher"],
            tools=[AllegroSearchTool(), WebPageReaderTool()],
            llm=self.llm,
            verbose=True,
            memory=False,
        )

    @agent
    def aliexpress_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["aliexpress_researcher"],
            tools=[AliExpressSearchTool(), WebPageReaderTool()],
            llm=self.llm,
            verbose=True,
            memory=False,
        )

    @agent
    def shopping_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["shopping_analyst"],
            tools=[],
            llm=self.llm,
            verbose=True,
            memory=False,
        )

    @agent
    def shopping_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config["shopping_advisor"],
            tools=[],
            llm=self.llm,
            verbose=True,
            memory=False,
        )

    # ------------------------------------------------------------------
    # Tasks  (config= pulls description/expected_output from tasks.yaml)
    # Execution details (async, context) stay here in Python.
    # ------------------------------------------------------------------

    @task
    def review_task(self) -> Task:
        return Task(
            config=self.tasks_config["review_task"],
            async_execution=True,
        )

    @task
    def reddit_task(self) -> Task:
        return Task(
            config=self.tasks_config["reddit_task"],
            async_execution=True,
        )

    @task
    def allegro_task(self) -> Task:
        return Task(
            config=self.tasks_config["allegro_task"],
            async_execution=True,
        )

    @task
    def aliexpress_task(self) -> Task:
        return Task(
            config=self.tasks_config["aliexpress_task"],
            async_execution=True,
        )

    @task
    def analysis_task(self) -> Task:
        # Waits for all four parallel research tasks
        return Task(
            config=self.tasks_config["analysis_task"],
            context=[
                self.review_task(),
                self.reddit_task(),
                self.allegro_task(),
                self.aliexpress_task(),
            ],
        )

    @task
    def final_report_task(self) -> Task:
        return Task(
            config=self.tasks_config["final_report_task"],
            context=[self.analysis_task()],
        )

    # ------------------------------------------------------------------
    # Crew
    # ------------------------------------------------------------------

    @crew
    def crew(self) -> Crew:
        # self.agents and self.tasks are auto-populated by @CrewBase
        # from all @agent / @task decorated methods above, in definition order.
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            output_log_file="crew_run.log",
        )
