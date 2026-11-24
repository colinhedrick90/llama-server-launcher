"""Static checks on launcher_app.py that need neither Tk nor a display."""
import ast
import unittest
from pathlib import Path

APP_SOURCE = Path(__file__).resolve().parent.parent / "launcher_app.py"


def _late_exception_reads(tree: ast.AST) -> list[str]:
    """Find lambdas / nested functions that read an `except ... as name` variable.

    Python unbinds that name when the except block ends. A callback handed to
    `root.after(...)` or a thread runs later, so reading the name there raises
    NameError instead of showing the error. Binding the value first is fine:
    default arguments (`lambda text=str(exc): ...`) are evaluated immediately.
    """
    problems = []
    for handler in ast.walk(tree):
        if not isinstance(handler, ast.ExceptHandler) or not handler.name:
            continue
        for statement in handler.body:
            for node in ast.walk(statement):
                if not isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                bodies = [node.body] if isinstance(node, ast.Lambda) else node.body
                for body in bodies:
                    for inner in ast.walk(body):
                        if (
                            isinstance(inner, ast.Name)
                            and inner.id == handler.name
                            and isinstance(inner.ctx, ast.Load)
                        ):
                            problems.append(f"line {inner.lineno}: deferred read of '{handler.name}'")
    return problems


class DeferredExceptionTests(unittest.TestCase):
    def test_checker_catches_the_bug_and_accepts_the_fix(self):
        broken = "try:\n    pass\nexcept OSError as exc:\n    after(lambda: show(str(exc)))\n"
        fixed = "try:\n    pass\nexcept OSError as exc:\n    after(lambda text=str(exc): show(text))\n"
        self.assertEqual(1, len(_late_exception_reads(ast.parse(broken))))
        self.assertEqual([], _late_exception_reads(ast.parse(fixed)))

    def test_no_callback_reads_an_exception_after_its_except_block(self):
        tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
        self.assertEqual([], _late_exception_reads(tree))


class FileDialogFilterTests(unittest.TestCase):
    def test_no_hardcoded_star_dot_star_in_filedialog_filetypes(self):
        """Linux Tk treats '*.*' literally and hides extensionless executables."""
        tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
        bad = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "filetypes":
                    continue
                for inner in ast.walk(kw.value):
                    if isinstance(inner, ast.Constant) and inner.value == "*.*":
                        bad.append(f"line {inner.lineno}: hardcoded '*.*' in filetypes")
        self.assertEqual([], bad)


if __name__ == "__main__":
    unittest.main()
