import sqlite3,time,os
class MemoryStore:
    def __init__(self,path='data/aviyan_memory.db'):
        os.makedirs(os.path.dirname(path) or '.',exist_ok=True); self.db=sqlite3.connect(path,check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, user_id TEXT, text TEXT, created REAL)'); self.db.commit()
    def add(self,user_id,text):
        self.db.execute('INSERT INTO memories(user_id,text,created) VALUES(?,?,?)',(user_id,text,time.time())); self.db.commit()
    def search(self,user_id,query,limit=8):
        terms=[x.lower() for x in query.split() if len(x)>2]
        rows=self.db.execute('SELECT text FROM memories WHERE user_id=? ORDER BY created DESC LIMIT 500',(user_id,)).fetchall()
        scored=sorted(((sum(t in r[0].lower() for t in terms),r[0]) for r in rows),reverse=True)
        return [x[1] for x in scored[:limit] if x[0]>0]
