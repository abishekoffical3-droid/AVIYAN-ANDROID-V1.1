import re,os,json
class SimpleRAG:
    def __init__(self,path='data/rag_store.jsonl'):
        self.path=path; os.makedirs(os.path.dirname(path) or '.',exist_ok=True)
    def add(self,text,source='unknown'):
        with open(self.path,'a',encoding='utf8') as f: f.write(json.dumps({'text':text,'source':source},ensure_ascii=False)+'\n')
    def search(self,query,limit=5):
        if not os.path.exists(self.path): return []
        terms=set(re.findall(r'\w+',query.lower())); out=[]
        for line in open(self.path,encoding='utf8'):
            x=json.loads(line); score=len(terms & set(re.findall(r'\w+',x['text'].lower())))
            if score: out.append((score,x))
        return [x for _,x in sorted(out,reverse=True)[:limit]]
