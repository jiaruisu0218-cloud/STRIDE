"""Optional Numba JIT wrapper for evolved functions (requires ``numba``)."""
import ast


def add_numba_decorator(
        program: str,
        function_to_evolve: str,
) -> str:
    """
    Accelerates code evaluation by adding @numba.jit() decorator to the target function.

    Note: Not all NumPy functions are compatible with Numba acceleration.

    Example:
    Input:  def func(a: np.ndarray): return a * 2
    Output: @numba.jit()
            def func(a: np.ndarray): return a * 2
    """
    tree = ast.parse(program)

    numba_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'numba' for alias in node.names):
            numba_imported = True
            break

    if not numba_imported:
        import_node = ast.Import(names=[ast.alias(name='numba', asname=None)])
        tree.body.insert(0, import_node)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_to_evolve:
            decorator = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='numba', ctx=ast.Load()),
                    attr='jit', 
                    ctx=ast.Load()
                ),
                args=[],  
                keywords=[ast.keyword(arg='nopython', value=ast.NameConstant(value=True))]  
            )
            node.decorator_list.append(decorator)

    modified_program = ast.unparse(tree)
    return modified_program


import textwrap

if __name__ == '__main__':
    code = '''
        import numpy as np
        import numba

        def func1():
            return 3

        def func():
            return 5
    '''
    res = add_numba_decorator(textwrap.dedent(code), 'func')
    print(res)
