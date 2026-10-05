import ast, operator as op, math
OPS={ast.Add:op.add,ast.Sub:op.sub,ast.Mult:op.mul,ast.Div:op.truediv,ast.Pow:op.pow,ast.Mod:op.mod,ast.USub:op.neg}
FUNCS={k:getattr(math,k) for k in ['sqrt','sin','cos','tan','log','log10','exp','fabs']}
def calculate(expr:str):
    def ev(n):
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)): return n.value
        if isinstance(n,ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.left),ev(n.right))
        if isinstance(n,ast.UnaryOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.operand))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in FUNCS: return FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError('unsupported expression')
    return ev(ast.parse(expr,mode='eval').body)
