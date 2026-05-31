import pytest
import pandas as pd
from app.services.executor import run_pandas_code, apply_transformation_plan

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "salary": [50000, 60000, 70000]
    })


def test_run_pandas_code_success(sample_df):
    """Verifies that running a valid code step correctly transforms the dataframe."""
    code_step = "df = df[df['age'] > 28]"
    result_df = run_pandas_code(sample_df, code_step)
    
    assert len(result_df) == 2
    assert "Alice" not in result_df["name"].values


def test_run_pandas_code_missing_df(sample_df):
    """Verifies that code that does not assign to 'df' raises a ValueError."""
    code_step = "x = 42"
    with pytest.raises(ValueError) as excinfo:
        run_pandas_code(sample_df, code_step)
    assert "did not yield" in str(excinfo.value)


def test_run_pandas_code_invalid_type(sample_df):
    """Verifies that code modifying 'df' to a non-DataFrame type raises a TypeError."""
    code_step = "df = 'not a dataframe'"
    with pytest.raises(TypeError) as excinfo:
        run_pandas_code(sample_df, code_step)
    assert "Expected resulting 'df' to be a pandas DataFrame" in str(excinfo.value)


def test_apply_transformation_plan_with_correction(sample_df):
    """Verifies that when a step fails, the executor triggers the correction callback and runs successfully."""
    plan = [
        "df = df[df['salary'] > 55000]",
        "df = df[df['non_existent_column'] == 1]",  # This will fail
        "df = df.sort_values(by='age')"
    ]
    
    # Mock self correction callback to return a fixed step
    def mock_fix_callback(failed_step: str, instruction: str, exception: Exception) -> str:
        assert "non_existent_column" in failed_step
        return "df = df[df['age'] > 32]" # Replace with a valid step
        
    result_df = apply_transformation_plan(
        sample_df,
        plan,
        "Get employees earning more than 55000 and older than 32, sorted by age",
        mock_fix_callback
    )
    
    # Alice (age 25, sal 50k) filtered out by step 1
    # Bob (age 30, sal 60k) filtered out by the fixed step (age > 32)
    # Charlie (age 35, sal 70k) remains
    assert len(result_df) == 1
    assert result_df.iloc[0]["name"] == "Charlie"
