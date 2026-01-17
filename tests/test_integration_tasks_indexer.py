from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import threading

from src.api.tasks_service import AnalysisTaskService
from src.config import ArchitextSettings
from src.indexer import create_index_from_paths


def test_analysis_task_service_runs_structure(temp_repo_path, tmp_path):
    settings = ArchitextSettings(storage_path=str(tmp_path / "storage"))
    executor = ThreadPoolExecutor(max_workers=1)

    service = AnalysisTaskService(
        task_store={},
        task_store_path=tmp_path / "task_store.json",
        lock=threading.Lock(),
        executor=executor,
        storage_roots=[tmp_path],
        source_roots=[Path(temp_repo_path).resolve()],
        base_settings=settings,
    )

    payload = SimpleNamespace(source=temp_repo_path, output_format="json", depth="shallow")
    result = service.run_analysis_task("analyze-structure", payload)

    assert result["format"] == "json"
    assert "summary" in result
    assert "tree" in result

    executor.shutdown(wait=True)


def test_create_index_from_paths_batches(mocker):
    class DummyIndex:
        def __init__(self):
            self.inserted = []

        def insert_documents(self, batch):
            self.inserted.append(batch)

    dummy_index = DummyIndex()

    mocker.patch("src.indexer._build_vector_store", return_value=mocker.Mock())
    mocker.patch("src.indexer.StorageContext.from_defaults", return_value=mocker.Mock())
    mocker.patch("src.indexer.VectorStoreIndex.from_documents", return_value=dummy_index)

    batches = [[mocker.Mock()], [mocker.Mock(), mocker.Mock()]]
    mocker.patch("src.indexer.iter_document_batches", return_value=iter(batches))

    settings = ArchitextSettings()
    index = create_index_from_paths(["dummy.py"], storage_path="./storage", settings=settings)

    assert index is dummy_index
    assert dummy_index.inserted == [batches[1]]
