import logging
import pandas as pd
import numpy as np
from typing import List, Callable, Optional
from app.utils.security import is_code_safe

logger = logging.getLogger(__name__)


def run_pandas_code(df: pd.DataFrame, code_step: str) -> pd.DataFrame:
    """
    Executes a single pandas transformation step in a highly restricted 
    execution scope to prevent code injection.
    
    Args:
        df: The pandas DataFrame to transform.
        code_step: A string containing a single python assignment, e.g. "df = df[df['age'] > 30]"
        
    Returns:
        The transformed pandas DataFrame.
    """
    clean_step = code_step.strip()
    if not clean_step:
        return df

    # Run security checks
    is_safe, error_msg = is_code_safe(clean_step)
    if not is_safe:
        raise ValueError(f"Security validation failed: {error_msg}")

    # Set up restricted local and global variables
    local_vars = {}
    allowed_globals = {
        "df": df,
        "pd": pd,
        "np": np,
        "abs": abs,
        "round": round,
        "sum": sum,
        "min": min,
        "max": max,
        "len": len,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "__builtins__": {}  # Clear default builtins like open, eval, exec
    }

    # Execute code in sandbox
    try:
        exec(clean_step, allowed_globals, local_vars)
    except Exception as e:
        logger.error(f"Error during sandbox code execution of '{clean_step}': {e}")
        raise e

    # Post-execution sanity checks
    if "df" not in local_vars:
        raise ValueError("Code execution did not yield the required 'df' dataframe.")
        
    result_df = local_vars["df"]
    if not isinstance(result_df, pd.DataFrame):
        raise TypeError(f"Expected resulting 'df' to be a pandas DataFrame, got {type(result_df).__name__}")

    return result_df


def apply_transformation_plan(
    df: pd.DataFrame,
    plan: List[str],
    instruction: str,
    fix_step_callback: Optional[Callable[[str, str, Exception], str]] = None
) -> pd.DataFrame:
    """
    Executes a series of pandas transformation steps sequentially on a DataFrame.
    Supports optional self-correction using a callback if execution fails.
    
    Args:
        df: Input pandas DataFrame.
        plan: List of python pandas steps.
        instruction: Original user instruction (used for correction context).
        fix_step_callback: Optional function that takes (failed_step, instruction, exception) 
                           and returns a corrected code step.
                           
    Returns:
        Transformed pandas DataFrame.
    """
    current_df = df.copy()

    for idx, step in enumerate(plan):
        step = step.strip()
        if not step:
            continue
            
        logger.info(f"Executing step {idx + 1}/{len(plan)}: {step}")
        
        try:
            current_df = run_pandas_code(current_df, step)
        except Exception as e:
            logger.warning(f"Step {idx + 1} execution failed: {e}")
            if fix_step_callback:
                logger.info("Attempting LLM self-correction...")
                try:
                    fixed_step = fix_step_callback(step, instruction, e)
                    logger.info(f"LLM returned corrected step: {fixed_step}")
                    current_df = run_pandas_code(current_df, fixed_step)
                except Exception as fix_err:
                    logger.error(f"LLM self-correction failed: {fix_err}")
                    raise RuntimeError(
                        f"Execution failed on step: '{step}'. "
                        f"Correction failed with error: {fix_err}"
                    ) from e
            else:
                raise e

    return current_df
