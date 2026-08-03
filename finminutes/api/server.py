from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from finminutes import __version__
from finminutes.core.config_manager import ConfigManager
from finminutes.api.task_manager import TaskManager
from finminutes.core.glossary_loader import GlossaryLoader
from finminutes.core.template_loader import TemplateLoader

_manager: TaskManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager
    _manager = TaskManager(ConfigManager())
    yield


app = FastAPI(title="FinMinutes API", version=__version__, lifespan=lifespan)


class TaskRequest(BaseModel):
    transcript: str
    background_path: str = ""
    glossary_tag: str = ""
    template_name: str = ""
    mode: str = "full"


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/api/v1/templates")
def list_templates():
    loader = TemplateLoader()
    return {"templates": loader.list_templates()}


@app.get("/api/v1/glossaries")
def list_glossaries():
    loader = GlossaryLoader()
    return {"glossaries": loader.list_tags()}


@app.post("/api/v1/tasks")
def create_task(req: TaskRequest):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="transcript cannot be empty")
    if req.mode not in ("fast", "standard", "full"):
        raise HTTPException(status_code=400, detail=f"invalid mode: {req.mode}")
    task_id = _manager.create_task(
        transcript=req.transcript,
        background_path=req.background_path,
        glossary_tag=req.glossary_tag,
        template_name=req.template_name,
        mode=req.mode,
    )
    return {"task_id": task_id, "status": "pending"}


@app.get("/api/v1/tasks/{task_id}")
def get_task(task_id: str):
    task = _manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task.to_dict()


@app.get("/api/v1/tasks")
def list_tasks():
    return {"tasks": [t.to_dict() for t in _manager.list_tasks()]}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
