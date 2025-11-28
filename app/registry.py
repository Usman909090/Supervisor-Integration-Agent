"""
Agent registry: the supervisor uses this to (a) brief the planner on available
capabilities and (b) look up connection details when calling a worker.
"""
from __future__ import annotations

from typing import List

from .models import AgentMetadata


def load_registry() -> List[AgentMetadata]:
    """Return the known worker agents. Replace endpoints/commands with real ones."""
    return [
        AgentMetadata(
            name="progress_accountability_agent",
            description="Tracks goals, tasks, and progress to provide accountability insights.",
            intents=["progress.track"],
            type="http",
            endpoint="https://example.com/progress/handle",
            healthcheck="https://example.com/progress/health",
        ),
        AgentMetadata(
            name="email_priority_agent",
            description="Analyzes incoming emails and assigns priority.",
            intents=["email.prioritize"],
            type="http",
            endpoint="https://example.com/email/handle",
            healthcheck="https://example.com/email/health",
        ),
        AgentMetadata(
            name="document_summarizer_agent",
            description="Summarizes documents or text into concise summaries.",
            intents=["summary.create"],
            type="http",
            endpoint="https://example.com/summarizer/handle",
            healthcheck="https://example.com/summarizer/health",
        ),
       AgentMetadata(
            name="meeting_followup_agent",
            description="ONLY for meeting transcripts: extracts action items (with assignees and deadlines), generates meeting summaries, identifies follow-up tasks. Use ONLY when query mentions meetings, transcripts, standups, action items, or follow-ups.",
            intents=["meeting.followup", "meeting.process", "meeting.analyze", "action_items.extract"],
            type="http",
            endpoint="https://meeting-minutes-backend-spm-production.up.railway.app/agents/supervisor/meeting-followup",
            healthcheck="https://meeting-minutes-backend-spm-production.up.railway.app/agents/supervisor/health",
            timeout_ms=30000,
        ),
        AgentMetadata(
            name="onboarding_buddy_agent",
            description="Guides new employees through onboarding milestones.",
            intents=["onboarding.guide"],
            type="http",
            endpoint="https://example.com/onboarding/handle",
            healthcheck="https://example.com/onboarding/health",
        ),
        AgentMetadata(
            name="KnowledgeBaseBuilderAgent",
            description="Creates and manages tasks from plain English input. Uses LLM to parse task information (task_id, task_name, task_description, task_deadline) from any input format and stores tasks in MongoDB with default status 'todo'.",
            intents=["create_task"],
            type="http",
            endpoint="http://vps.zaim-abbasi.tech/knowledge-builder/message",
            healthcheck="http://vps.zaim-abbasi.tech/knowledge-builder/health",
            timeout_ms=30000,
        ),
        AgentMetadata(
            name="task_dependency_agent",
            description="Analyzes dependencies between tasks in a project.",
            intents=["tasks.dependencies"],
            type="http",
            endpoint="https://example.com/tasks/handle",
            healthcheck="https://example.com/tasks/health",
        ),
        AgentMetadata(
            name="deadline_guardian_agent",
            description="Monitors deadlines, detects risks, and alerts when deadlines are at risk.",
            intents=["deadline.monitor"],
            type="http",
            endpoint="https://example.com/deadline/handle",
            healthcheck="https://example.com/deadline/health",
        ),
    ]


def find_agent_by_name(name: str, registry: List[AgentMetadata]) -> AgentMetadata:
    for agent in registry:
        if agent.name == name:
            return agent
    raise KeyError(f"Agent {name} not found in registry")
