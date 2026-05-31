import ast
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# List of builtins that present security risks and are forbidden in executed scripts
FORBIDDEN_BUILTINS = {
    "open", "eval", "exec", "compile", "__import__", "globals", "locals", 
    "vars", "getattr", "setattr", "delattr", "input", "breakpoint", "builtins",
    "exit", "quit"
}

# Standard system modules/names that should never be accessed
FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "shutil", "socket", "requests", "urllib",
    "ctypes", "platform", "pty", "pathlib", "builtins"
}

# Attributes (e.g. methods or properties) that are blocked for safety (prevent writing/reading disk files)
FORBIDDEN_ATTRIBUTES = {
    "to_csv", "to_excel", "to_json", "to_sql", "to_feather", "to_parquet",
    "to_stata", "to_pickle", "to_markdown", "to_html", "to_xml", "to_latex",
    "to_gbq", "to_hdf", "to_clipboard", "read_csv", "read_excel", "read_json",
    "read_sql", "read_feather", "read_parquet", "read_html", "read_xml"
}

# Raw string patterns that we also check for defense-in-depth
FORBIDDEN_SUBSTRINGS = [
    "import", "os.", "sys.", "subprocess", "open(", "__", "eval(", "exec(",
    "remove(", "unlink("
]

class SafeASTVisitor(ast.NodeVisitor):
    """
    AST Visitor that walks the parsed tree and raises StopIteration 
    if any unsafe node type, builtin function, or attribute is encountered.
    """
    def __init__(self) -> None:
        self.is_safe = True
        self.violation = ""

    def fail(self, node: ast.AST, msg: str) -> None:
        self.is_safe = False
        self.violation = f"Line {node.lineno}: {msg}"
        # Stop iteration immediately to abort further traversal
        raise StopIteration(self.violation)

    def visit_Import(self, node: ast.Import) -> None:
        self.fail(node, "Imports are strictly forbidden.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.fail(node, "Import From statements are strictly forbidden.")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.fail(node, "Defining functions is not allowed.")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.fail(node, "Defining async functions is not allowed.")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.fail(node, "Defining classes is not allowed.")

    def visit_Delete(self, node: ast.Delete) -> None:
        self.fail(node, "Delete statements are not allowed.")

    def visit_Try(self, node: ast.Try) -> None:
        self.fail(node, "Try/except structures are not allowed.")

    def visit_With(self, node: ast.With) -> None:
        self.fail(node, "With/context blocks are not allowed.")

    def visit_Raise(self, node: ast.Raise) -> None:
        self.fail(node, "Raise statements are not allowed.")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_BUILTINS:
            self.fail(node, f"Access to forbidden builtin: '{node.id}'")
        if node.id in FORBIDDEN_NAMES:
            self.fail(node, f"Access to forbidden system name: '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.fail(node, f"Access to dunder attribute '{node.attr}' is forbidden.")
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.fail(node, f"Access to forbidden attribute: '{node.attr}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check if the function being called is in the forbidden list
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_BUILTINS:
                self.fail(node, f"Calling forbidden builtin function: '{node.func.id}'")
        self.generic_visit(node)


def is_code_safe(code: str) -> Tuple[bool, str]:
    """
    Validates a Python code snippet for security violations using both
    lexical checks (substring matching) and AST analysis.
    
    Returns:
        (is_safe, error_message)
    """
    clean_code = code.strip()
    if not clean_code:
        return True, ""

    # 1. Substring Defense-In-Depth Check
    for pattern in FORBIDDEN_SUBSTRINGS:
        if pattern in clean_code:
            return False, f"Unsafe step pattern detected lexically: '{pattern}'"

    # 2. AST Parsing and Verification
    try:
        tree = ast.parse(clean_code)
    except SyntaxError as e:
        logger.warning(f"Syntax error parsing LLM code: {e}")
        return False, f"Syntax Error: {e.msg} at line {e.lineno}"

    visitor = SafeASTVisitor()
    try:
        visitor.visit(tree)
    except StopIteration as e:
        logger.warning(f"AST Safety violation detected: {e}")
        return False, str(e)
    except Exception as e:
        logger.error(f"Unexpected AST validation error: {e}")
        return False, f"AST verification failed: {e}"

    return visitor.is_safe, ""
