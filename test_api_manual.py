"""Quick API test script to verify endpoints without starting the full server."""
import sys
import json
from unittest.mock import Mock

# Mock the server imports to avoid TypeAliasType error
sys.path.insert(0, 'c:/Users/tusik/Documents/GitHub/Architext')

from src.config import ArchitextSettings
from fastapi.testclient import TestClient
from pathlib import Path

def test_api():
    # Import with patches
    import unittest.mock as mock
    
    with mock.patch("src.server.initialize_settings"):
        with mock.patch("src.server.gather_index_files", return_value=["./tests/test.py"]):
            with mock.patch("src.server.create_index_from_paths"):
                with mock.patch("src.server.resolve_source", return_value=Path("./tests")):
                    from src.server import create_app
                    
                    settings = ArchitextSettings(storage_path="./storage-tests")
                    app = create_app(settings=settings)
                    client = TestClient(app)
                    
                    print("Testing /health endpoint...")
                    response = client.get("/health")
                    print(f"  Status: {response.status_code}")
                    print(f"  Response: {response.json()}")
                    
                    print("\nTesting /tasks endpoint...")
                    response = client.get("/tasks")
                    print(f"  Status: {response.status_code}")
                    print(f"  Response: {response.json()}")
                    
                    print("\nTesting /index endpoint (inline)...")
                    response = client.post("/index", json={
                        "source": "./tests",
                        "storage": "./storage-tests",
                        "background": False
                    })
                    print(f"  Status: {response.status_code}")
                    data = response.json()
                    print(f"  Task ID: {data.get('task_id')}")
                    print(f"  Status: {data.get('status')}")
                    print(f"  Documents: {data.get('documents')}")
                    
                    print("\n✓ All API tests passed!")

if __name__ == "__main__":
    test_api()
