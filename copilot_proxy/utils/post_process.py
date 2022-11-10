import ast


def clean_test_code(test_code: str) -> str:
    while len(test_code) > 0:
        try:
            ast.parse(test_code)
            break
        except SyntaxError:
            test_code = '\n'.join(test_code.split('\n')[:-1])
            print("Removing the last line of the test code")
    return test_code
