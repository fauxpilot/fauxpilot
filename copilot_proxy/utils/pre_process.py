import ast
from typing import List


def extract_functions_names(code: str) -> List[str]:
    nodes = ast.parse(code)
    functions_names = []
    for node in ast.walk(nodes):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            functions_names.append(node.name)
    return functions_names
