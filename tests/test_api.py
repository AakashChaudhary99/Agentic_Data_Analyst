import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Verifies that the /health endpoint returns the correct status and metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["message"] == "API is healthy"
    assert "environment" in data


def test_legacy_health_check():
    """Verifies that the root legacy health check returns backwards-compatible dictionary."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Message": "API is healthy"}


def test_processing_invalid_extension():
    """Verifies that non-CSV file uploads are rejected with 400 Bad Request."""
    files = {"file": ("test.txt", "some,text,file\n", "text/plain")}
    data = {"instruction": "Get average age"}
    
    response = client.post("/processing", files=files, data=data)
    assert response.status_code == 400
    assert "Only CSV files are supported" in response.json()["detail"]


@patch("app.routes.analysis.workflow.invoke")
def test_processing_text_response(mock_invoke):
    """Verifies text analysis request matches structure and returns plain text."""
    mock_invoke.return_value = {
        "output_mode": "text",
        "text_output": "The highest earning department is Engineering."
    }

    csv_data = "name,salary,department\nAlice,100000,Engineering\nBob,80000,Sales\n"
    files = {"file": ("data.csv", csv_data, "text/csv")}
    data = {"instruction": "Find department with highest salary"}

    response = client.post("/processing", files=files, data=data)
    assert response.status_code == 200
    assert response.text == '"The highest earning department is Engineering."'


@patch("app.routes.analysis.workflow.invoke")
def test_processing_csv_response(mock_invoke):
    """Verifies CSV analysis request returns transformed dataset as file attachment."""
    mock_invoke.return_value = {
        "output_mode": "csv",
        "transformation_plan": ["df = df[df['salary'] > 90000]"],
        "instruction": "Filter salaries > 90000"
    }

    csv_data = "name,salary,department\nAlice,100000,Engineering\nBob,80000,Sales\n"
    files = {"file": ("data.csv", csv_data, "text/csv")}
    data = {"instruction": "Filter salaries > 90000"}

    response = client.post("/processing", files=files, data=data)
    
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment; filename=transformed_data.csv" in response.headers["content-disposition"]
    
    content = response.text
    assert "name,salary,department" in content
    assert "Alice,100000,Engineering" in content
    assert "Bob,80000,Sales" not in content
