import subprocess,sys,tempfile,os

def run_python(code:str, timeout:int=10):
    with tempfile.TemporaryDirectory(prefix='aviyan_') as d:
        p=subprocess.run([sys.executable,'-I','-c',code],cwd=d,text=True,capture_output=True,timeout=timeout)
        return {'returncode':p.returncode,'stdout':p.stdout[-12000:],'stderr':p.stderr[-12000:]}
