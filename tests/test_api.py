import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from finminutes.api.server import app, _manager
from finminutes.api.task_manager import TaskManager, TaskStatus


@pytest.fixture(autouse=True)
def reset_manager():
    old = _manager
    yield
    import finminutes.api.server as srv
    srv._manager = old


@pytest.fixture
def mock_task_manager():
    mgr = MagicMock(spec=TaskManager)
    import finminutes.api.server as srv
    srv._manager = mgr
    return mgr


class TestTaskStatus:
    def test_defaults(self):
        ts = TaskStatus(task_id="abc")
        assert ts.task_id == "abc"
        assert ts.status == "pending"
        assert ts.progress == ""
        assert ts.markdown == ""
        assert ts.summary == ""
        assert ts.error is None

    def test_to_dict_pending(self):
        ts = TaskStatus(task_id="abc", status="running", progress="working")
        d = ts.to_dict()
        assert d["task_id"] == "abc"
        assert d["status"] == "running"
        assert d["progress"] == "working"
        assert d["has_result"] is False
        assert d["error"] is None

    def test_to_dict_completed(self):
        ts = TaskStatus(task_id="abc", status="completed", markdown="# result", summary="done")
        d = ts.to_dict()
        assert d["has_result"] is True
        assert d["summary"] == "done"

    def test_to_dict_failed(self):
        ts = TaskStatus(task_id="abc", status="failed", error="something broke")
        d = ts.to_dict()
        assert d["error"] == "something broke"


class TestTaskManager:
    def test_create_task_returns_id(self):
        mgr = TaskManager()
        task_id = mgr.create_task("test transcript", mode="fast")
        assert task_id is not None
        assert isinstance(task_id, str)
        assert len(task_id) > 10

    def test_get_task_created(self):
        mgr = TaskManager()
        task_id = mgr.create_task("hello", mode="fast")
        task = mgr.get_task(task_id)
        assert task is not None
        assert task.status in ("pending", "running", "completed")

    def test_get_task_not_found(self):
        mgr = TaskManager()
        assert mgr.get_task("nonexistent") is None

    def test_list_tasks(self):
        mgr = TaskManager()
        mgr.create_task("task1", mode="fast")
        mgr.create_task("task2", mode="fast")
        tasks = mgr.list_tasks()
        assert len(tasks) == 2

    def test_task_completes_in_fast_mode(self):
        mgr = TaskManager()
        task_id = mgr.create_task("test transcript", mode="fast")
        time.sleep(0.3)
        task = mgr.get_task(task_id)
        assert task.status == "completed"
        assert task.markdown != ""

    def test_task_has_summary_after_completion(self):
        mgr = TaskManager()
        task_id = mgr.create_task("test content", mode="fast")
        time.sleep(0.3)
        task = mgr.get_task(task_id)
        assert task.summary != ""

    def test_task_status_update(self):
        mgr = TaskManager()
        task_id = mgr.create_task("test", mode="fast")
        ts = mgr.get_task(task_id)
        mgr._update(task_id, status="running", progress="processing")
        ts2 = mgr.get_task(task_id)
        assert ts2.status == "running" or ts.status == "running"


class TestAPIHealth:
    def test_health(self, mock_task_manager):
        client = TestClient(app)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestAPITasks:
    def test_create_task(self, mock_task_manager):
        mock_task_manager.create_task.return_value = "fake-task-id-123"
        client = TestClient(app)
        resp = client.post("/api/v1/tasks", json={"transcript": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "fake-task-id-123"
        assert data["status"] == "pending"

    def test_create_task_empty_transcript(self, mock_task_manager):
        client = TestClient(app)
        resp = client.post("/api/v1/tasks", json={"transcript": ""})
        assert resp.status_code == 400
        assert "cannot be empty" in resp.json()["detail"]

    def test_create_task_invalid_mode(self, mock_task_manager):
        client = TestClient(app)
        resp = client.post("/api/v1/tasks", json={"transcript": "test", "mode": "invalid"})
        assert resp.status_code == 400
        assert "invalid mode" in resp.json()["detail"]

    def test_get_task(self, mock_task_manager):
        mock_task_manager.get_task.return_value = TaskStatus(
            task_id="t1", status="completed", markdown="# done",
        )
        client = TestClient(app)
        resp = client.get("/api/v1/tasks/t1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "t1"
        assert data["status"] == "completed"

    def test_get_task_not_found(self, mock_task_manager):
        mock_task_manager.get_task.return_value = None
        client = TestClient(app)
        resp = client.get("/api/v1/tasks/nonexistent")
        assert resp.status_code == 404

    def test_list_tasks(self, mock_task_manager):
        mock_task_manager.list_tasks.return_value = [
            TaskStatus(task_id="t1", status="completed"),
            TaskStatus(task_id="t2", status="running"),
        ]
        client = TestClient(app)
        resp = client.get("/api/v1/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tasks"]) == 2


class TestAPITemplates:
    def test_list_templates(self, mock_task_manager):
        client = TestClient(app)
        resp = client.get("/api/v1/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        assert isinstance(data["templates"], list)

    def test_list_glossaries(self, mock_task_manager):
        client = TestClient(app)
        resp = client.get("/api/v1/glossaries")
        assert resp.status_code == 200
        data = resp.json()
        assert "glossaries" in data
        assert isinstance(data["glossaries"], list)
