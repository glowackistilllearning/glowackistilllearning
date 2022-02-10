"""
ML Pipeline Orchestrator — lightweight DAG-based orchestration for
ML training and inference pipelines with retry logic and observability.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
import logging
import time

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    name: str
    fn: Callable
    dependencies: list[str] = field(default_factory=list)
    retries: int = 1
    timeout_s: Optional[float] = None
    skip_on_failure: bool = False

    status: StepStatus = StepStatus.PENDING
    result: object = None
    error: Optional[Exception] = None
    duration_s: float = 0.0


@dataclass
class PipelineRun:
    pipeline_name: str
    steps: list[PipelineStep]
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_duration_s(self) -> float:
        return round(self.end_time - self.start_time, 3)

    @property
    def success(self) -> bool:
        return all(
            s.status in (StepStatus.SUCCESS, StepStatus.SKIPPED)
            for s in self.steps
        )


class MLPipelineOrchestrator:
    """
    Executes ML pipeline steps in dependency order with retry, timeout,
    and structured logging. Designed for training and batch inference workflows.
    """

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self._steps: dict[str, PipelineStep] = {}

    def register(self, step: PipelineStep) -> MLPipelineOrchestrator:
        self._steps[step.name] = step
        return self

    def _resolve_order(self) -> list[PipelineStep]:
        visited: set[str] = set()
        order: list[PipelineStep] = []

        def visit(name: str):
            if name in visited:
                return
            step = self._steps[name]
            for dep in step.dependencies:
                visit(dep)
            visited.add(name)
            order.append(step)

        for name in self._steps:
            visit(name)
        return order

    def _run_step(self, step: PipelineStep, context: dict) -> None:
        step.status = StepStatus.RUNNING
        for attempt in range(1, step.retries + 1):
            try:
                start = time.perf_counter()
                step.result = step.fn(context)
                step.duration_s = round(time.perf_counter() - start, 3)
                step.status = StepStatus.SUCCESS
                logger.info("[%s] SUCCESS (%.2fs, attempt %d)", step.name, step.duration_s, attempt)
                return
            except Exception as exc:
                step.error = exc
                logger.warning("[%s] attempt %d failed: %s", step.name, attempt, exc)
        if step.skip_on_failure:
            step.status = StepStatus.SKIPPED
            logger.warning("[%s] SKIPPED after %d retries", step.name, step.retries)
        else:
            step.status = StepStatus.FAILED
            raise RuntimeError(f"Step '{step.name}' failed after {step.retries} retries") from step.error

    def run(self, initial_context: Optional[dict] = None) -> PipelineRun:
        context = initial_context or {}
        ordered = self._resolve_order()
        run = PipelineRun(pipeline_name=self.pipeline_name, steps=ordered, start_time=time.time())
        for step in ordered:
            dep_failed = any(
                self._steps[d].status == StepStatus.FAILED
                for d in step.dependencies
                if d in self._steps
            )
            if dep_failed and not step.skip_on_failure:
                step.status = StepStatus.SKIPPED
                continue
            self._run_step(step, context)
            if step.result is not None:
                context[step.name] = step.result
        run.end_time = time.time()
        return run
