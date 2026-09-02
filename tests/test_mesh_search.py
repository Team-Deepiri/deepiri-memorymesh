"""Tests for cross-project Memory Mesh search and project listing."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepiri_memorymesh.config import Settings
from deepiri_memorymesh.models import MemoryRecord
from deepiri_memorymesh.sync_service import MemoryMesh


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "test.db",
        embedding_backend="fallback",
        providers=[],
    )


def _seed_project(mesh: MemoryMesh, project: str, provider: str, messages: list[str]) -> None:
    mesh.init()
    for i, content in enumerate(messages):
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider=provider,
                    project=project,
                    conversation_id=f"{project}-{provider}-conv",
                    role="user" if i % 2 == 0 else "assistant",
                    content=content,
                )
            ]
        )


class TestListAllProjects:
    def test_empty_db(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        mesh.init()
        assert mesh.store.list_all_projects() == []

    def test_multiple_projects(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "alpha", "claude", ["hello world"])
        _seed_project(mesh, "beta", "cursor", ["goodbye world"])
        projects = mesh.store.list_all_projects()
        assert "alpha" in projects
        assert "beta" in projects


class TestListProjects:
    def test_empty(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        mesh.init()
        assert mesh.list_projects() == []

    def test_returns_projects(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "proj-a", "claude", ["test message"])
        projects = mesh.list_projects()
        assert "proj-a" in projects


class TestMeshSearch:
    def test_empty_db(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        mesh.init()
        result = mesh.mesh_search("hello", top_k=5)
        assert result.rows == []
        assert result.report.total_eligible_embeddings == 0

    def test_cross_project_results(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "proj-x", "claude", [
            "Python is a programming language",
            "Functions are reusable code blocks",
        ])
        _seed_project(mesh, "proj-y", "cursor", [
            "JavaScript runs in the browser",
            "Python has dynamic typing",
        ])
        mesh.embed_project("proj-x")
        mesh.embed_project("proj-y")
        result = mesh.mesh_search("Python programming", top_k=5)
        assert len(result.rows) > 0
        projects_found = {r.get("project") for r in result.rows}
        assert len(projects_found) >= 1

    def test_scoped_to_project(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "only-here", "claude", ["unique quantum computing topic"])
        _seed_project(mesh, "other", "cursor", ["different topic entirely"])
        mesh.embed_project("only-here")
        mesh.embed_project("other")
        result = mesh.mesh_search("quantum computing", project="only-here", top_k=5)
        for row in result.rows:
            assert row.get("project") == "only-here"

    def test_provider_filter(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "p", "claude", ["test message one"])
        _seed_project(mesh, "p", "cursor", ["test message two"])
        mesh.embed_project("p")
        result = mesh.mesh_search("test message", provider="claude", top_k=5)
        for row in result.rows:
            assert row.get("provider") == "claude"

    def test_strategy_exact(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "p", "claude", ["test message"])
        mesh.embed_project("p")
        result = mesh.mesh_search("test", strategy="exact", top_k=5)
        assert result.report.strategy_used == "exact"

    def test_strategy_indexed(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "p", "claude", ["test message"])
        mesh.embed_project("p")
        result = mesh.mesh_search("test", strategy="indexed", top_k=5)
        assert result.report.strategy_used == "indexed"

    def test_invalid_strategy_raises(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        mesh.init()
        with pytest.raises(ValueError, match="Unsupported retrieval strategy"):
            mesh.mesh_search("test", strategy="bogus")

    def test_list_embeddings_cross_project(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "a", "claude", ["msg a"])
        _seed_project(mesh, "b", "cursor", ["msg b"])
        mesh.embed_project("a")
        mesh.embed_project("b")
        rows = mesh.store.list_embeddings_cross_project()
        assert len(rows) == 2
        projects = {r.get("project") for r in rows}
        assert projects == {"a", "b"}

    def test_count_embeddings_cross_project(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "a", "claude", ["msg one"])
        _seed_project(mesh, "b", "cursor", ["msg two"])
        mesh.embed_project("a")
        mesh.embed_project("b")
        assert mesh.store.count_embeddings_cross_project() == 2

    def test_count_embeddings_cross_project_by_provider(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "a", "claude", ["msg one"])
        _seed_project(mesh, "a", "cursor", ["msg two"])
        mesh.embed_project("a")
        assert mesh.store.count_embeddings_cross_project(provider="claude") == 1
        assert mesh.store.count_embeddings_cross_project(provider="cursor") == 1


class TestMeshSearchHttpApi:
    def test_http_mesh_projects_and_search(self, tmp_path: Path) -> None:
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib import error as urlerror
        from urllib import request as urlrequest
        from deepiri_memorymesh.supervised_service import SupervisedService

        settings = _make_settings(tmp_path)
        mesh = MemoryMesh(settings)
        _seed_project(mesh, "proj-1", "claude", ["Hello world search test"])
        mesh.embed_project("proj-1")

        probe = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        port = probe.server_address[1]
        probe.server_close()

        svc = SupervisedService(
            host="127.0.0.1", port=port, settings=settings, auth_mode="off"
        )
        assert svc.start() == "started"
        base_url = f"http://127.0.0.1:{port}"

        try:
            # Test GET /mesh/projects
            req = urlrequest.Request(base_url + "/mesh/projects", method="GET")
            with urlrequest.urlopen(req, timeout=5) as resp:
                status = resp.status
                body = json.loads(resp.read().decode("utf-8"))
            assert status == 200
            assert body["ok"] is True
            assert len(body["projects"]) == 1
            assert body["projects"][0]["project"] == "proj-1"

            # Test POST /mesh/search valid query
            post_data = json.dumps({"q": "Hello", "top_k": 5}).encode("utf-8")
            req = urlrequest.Request(
                base_url + "/mesh/search",
                data=post_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=5) as resp:
                status = resp.status
                body = json.loads(resp.read().decode("utf-8"))
            assert status == 200
            assert body["ok"] is True
            assert len(body["results"]) >= 1

            # Test POST /mesh/search missing query
            post_data = json.dumps({"q": ""}).encode("utf-8")
            req = urlrequest.Request(
                base_url + "/mesh/search",
                data=post_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urlerror.HTTPError) as exc_info:
                urlrequest.urlopen(req, timeout=5)
            assert exc_info.value.code == 400
        finally:
            svc.shutdown()


class TestMeshSearchCli:
    def test_cli_mesh_projects_and_search(self, tmp_path: Path) -> None:
        from unittest import mock
        from typer.testing import CliRunner
        from deepiri_memorymesh.cli import app

        settings = _make_settings(tmp_path)
        mesh = MemoryMesh(settings)
        _seed_project(mesh, "my-proj", "claude", ["Testing CLI mesh search capabilities"])
        mesh.embed_project("my-proj")

        runner = CliRunner()
        with mock.patch("deepiri_memorymesh.cli._mesh", return_value=mesh):
            # Test mesh projects
            result = runner.invoke(app, ["mesh", "projects"])
            assert result.exit_code == 0
            assert "my-proj" in result.output

            # Test mesh search
            result = runner.invoke(app, ["mesh", "search", "--q", "CLI"])
            assert result.exit_code == 0
            assert "my-proj" in result.output
            assert "Testing CLI" in result.output

            # Test mesh search no results on empty mesh
            empty_mesh = MemoryMesh(Settings(db_path=tmp_path / "empty.db", embedding_backend="fallback"))
            empty_mesh.init()
            with mock.patch("deepiri_memorymesh.cli._mesh", return_value=empty_mesh):
                result = runner.invoke(app, ["mesh", "projects"])
                assert result.exit_code == 0
                assert "No projects found" in result.output

                result = runner.invoke(app, ["mesh", "search", "--q", "nonexistent"])
                assert result.exit_code == 0
                assert "No results found" in result.output

