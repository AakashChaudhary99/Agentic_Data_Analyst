from app.utils.security import is_code_safe

def test_safe_code_steps():
    """Verifies that standard, harmless pandas expressions pass security validation."""
    safe_steps = [
        "df = df[df['age'] > 30]",
        "df = df.groupby('department')['salary'].mean().reset_index()",
        "df['total'] = df['price'] * df['quantity']",
        "df = df.sort_values(by='score', ascending=False)",
        "df = df.dropna(subset=['email'])",
        "df = df.rename(columns={'old_col': 'new_col'})"
    ]
    for step in safe_steps:
        is_safe, error = is_code_safe(step)
        assert is_safe, f"Expected step to be safe: '{step}'. Error: {error}"


def test_unsafe_lexical_patterns():
    """Verifies that forbidden substrings are blocked immediately."""
    unsafe_steps = [
        "import os",
        "from sys import exit",
        "df.to_csv('out.csv')",
        "f = open('secrets.txt')",
        "os.system('rm -rf /')",
        "df['leak'] = eval('1 + 2')",
        "df['sub'] = subprocess.run(['ls'])"
    ]
    for step in unsafe_steps:
        is_safe, error = is_code_safe(step)
        assert not is_safe, f"Expected step to be rejected: '{step}'"
        assert "pattern" in error.lower() or "forbidden" in error.lower() or "syntax" in error.lower()


def test_unsafe_ast_patterns():
    """Verifies that obfuscated security bypasses are blocked by AST analysis."""
    evasive_steps = [
        "getattr(df, '__class__')",
        "df.attr = df.__globals__",
        "df = df.apply(lambda x: globals())",
        "my_open = builtins.open",
        "exec('import os')",
        "class MaliciousClass:\n    pass",
        "def helper():\n    return 42",
        "delete df",
        "try:\n    x = 1\nexcept:\n    pass"
    ]
    for step in evasive_steps:
        is_safe, error = is_code_safe(step)
        assert not is_safe, f"Expected step to be rejected by AST: '{step}'"
        assert len(error) > 0
