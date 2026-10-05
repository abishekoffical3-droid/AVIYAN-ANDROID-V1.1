import os,subprocess,shutil,uuid
class DeployManager:
    def __init__(self,root='deployments'): self.root=root; os.makedirs(root,exist_ok=True)
    def deploy(self,source,command=None):
        did=str(uuid.uuid4()); dest=os.path.join(self.root,did); shutil.copytree(source,dest,dirs_exist_ok=True)
        result={'id':did,'path':dest,'status':'uploaded'}
        if command:
            p=subprocess.run(command,shell=True,cwd=dest,text=True,capture_output=True,timeout=300); result.update(status='ready' if p.returncode==0 else 'failed',stdout=p.stdout[-5000:],stderr=p.stderr[-5000:])
        return result
