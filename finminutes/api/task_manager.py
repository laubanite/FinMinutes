import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from finminutes.core.config_manager import ConfigManager
from finminutes.core.pipeline import Pipeline, PipelineResult


class TaskStatus:
    def __init__(
        self,
        task_id: str,
        status: str = "pending",
        progress: str = "",
        created_at: str = "",
        completed_at: str | None = None,
        markdown: str = "",
        summary: str = "",
        error: str | None = None,
    ):
        self.task_id = task_id
        self.status = status
        self.progress = progress
        self.created_at = created_at
        self.completed_at = completed_at
        self.markdown = markdown
        self.summary = summary
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "has_result": bool(self.markdown),
            "summary": self.summary,
            "error": self.error,
        }


class TaskManager:
    def __init__(self, config: ConfigManager | None = None):
        self._config = config or ConfigManager()
        self._tasks: dict[str, TaskStatus] = {}
        self._lock = threading.Lock()

    def create_task(
        self,
        transcript: str,
        background_path: str = "",
        glossary_tag: str = "",
        template_name: str = "",
        mode: str = "full",
    ) -> str:
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        status = TaskStatus(
            task_id=task_id,
            status="pending",
            progress="任务已创建",
            created_at=now,
        )
        with self._lock:
            self._tasks[task_id] = status

        t = threading.Thread(
            target=self._run,
            args=(task_id, transcript, background_path, glossary_tag, template_name, mode),
            daemon=True,
        )
        t.start()
        return task_id

    def get_task(self, task_id: str) -> TaskStatus | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[TaskStatus]:
        with self._lock:
            return list(self._tasks.values())

    def _run(
        self,
        task_id: str,
        transcript: str,
        background_path: str,
        glossary_tag: str,
        template_name: str,
        mode: str,
    ):
        self._update(task_id, status="running", progress="正在预处理...")
        pipeline = Pipeline(self._config)
        # 注意：pipeline.run 不接受 template_name（成品稿模板属于 render 步骤，非流水线概念）
        result = pipeline.run(
            transcript=transcript,
            background_path=background_path,
            glossary_tag=glossary_tag,
            mode=mode,
        )

        now = datetime.now(timezone.utc).isoformat()
        if result.success:
            self._update(
                task_id,
                status="completed",
                progress="处理完成",
                completed_at=now,
                markdown=result.markdown_output,
                summary=result.summary,
            )
        else:
            error_msg = "; ".join(result.errors)
            self._update(
                task_id,
                status="failed",
                progress="处理出错",
                completed_at=now,
                error=error_msg,
                markdown=result.markdown_output,
                summary=result.summary,
            )

    def _update(self, task_id: str, **kwargs):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                for k, v in kwargs.items():
                    setattr(task, k, v)
