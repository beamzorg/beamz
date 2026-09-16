"""
Executes user-typed code against a shared namespace, REPL-style: if the
input is a single expression its value is echoed (like the Python
interactive shell, and like Lumerical's script prompt), otherwise it runs
as a statement. Kept deliberately separate from any Qt widget so it can be
unit-tested without a QApplication.
"""
from __future__ import annotations

import ast
import contextlib
import io
import traceback


def run_snippet(code: str, namespace: dict) -> tuple[str, bool]:
    """Run `code` in `namespace` (mutated in place, like any exec target).

    Returns (output_text, is_error). `output_text` combines anything the
    snippet printed with the repr of its trailing expression's value, if
    any — mirroring what typing the same line into `python` interactively
    would show.

    Classifies expression-vs-statement via `ast.parse` *before* running
    anything, rather than the more obvious "try eval, except SyntaxError:
    fall back to exec". The naive version actually executes nothing on the
    failed eval attempt, but a genuine runtime error from the exec fallback
    still ends up chained onto that discarded SyntaxError in the traceback
    (Python auto-chains via `__context__`), producing a confusing wall of
    two unrelated errors for something as ordinary as a FrozenInstanceError
    from a typo'd mutation. Parsing first means only the branch that
    actually runs can ever raise.
    """
    try:
        is_expression = True
        try:
            tree = ast.parse(code, mode="eval")
        except SyntaxError:
            is_expression = False

        buf = io.StringIO()
        if is_expression:
            compiled = compile(tree, "<console>", "eval")
            with contextlib.redirect_stdout(buf):
                result = eval(compiled, namespace)
            if result is not None:
                buf.write(repr(result))
        else:
            compiled = compile(code, "<console>", "exec")
            with contextlib.redirect_stdout(buf):
                exec(compiled, namespace)
    except Exception:
        return traceback.format_exc(), True
    return buf.getvalue(), False
