# calculator module
import ast
import operator

# Supported operators
operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def eval_expr(expr):
    """
    Safely evaluate a math expression string using AST parsing.
    Supports +, -, *, /, ** and unary minus.
    """
    def _eval(node):
        if isinstance(node, ast.Num):  # <number>
            return node.n
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](left, right)
            else:
                raise TypeError(f"Unsupported operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):  # unary - <operand>
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](operand)
            else:
                raise TypeError(f"Unsupported unary operator: {op_type}")
        else:
            raise TypeError(f"Unsupported expression: {node}")

    try:
        parsed = ast.parse(expr, mode='eval').body
        return _eval(parsed)
    except Exception as e:
        return f"Error evaluating expression: {e}"

if __name__ == "__main__":
    test_exprs = [
        "2 + 3 * 4",
        "-5 + 6**2",
        "10 / 0",
        "2 ** 8",
        "100 - (20 + 30)",
        "invalid + 1",
    ]
    for expr in test_exprs:
        print(f"{expr} = {eval_expr(expr)}")
