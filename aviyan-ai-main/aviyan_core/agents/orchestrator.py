from aviyan_core.tools.calculator import calculate
from aviyan_core.tools.python_exec import run_python
class Orchestrator:
    def plan(self,task):
        return ['understand_request','select_tools','execute','verify','respond']
    def execute_tool(self,name,args):
        if name=='calculator': return calculate(args['expression'])
        if name=='python': return run_python(args['code'])
        raise ValueError('unknown tool')
