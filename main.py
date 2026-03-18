from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
 
CASE_BASE=[
    {"problem":"laptop not turning on battery dead","solution":"Replace battery or charge for 30 min","category":"hardware"},
      {"problem":"wifi not connecting network issue","solution":"Reset network settings, restart router","category":"network"},
    {"problem":"blue screen of death BSOD crash","solution":"Check RAM, run sfc /scannow, update drivers","category":"OS"},
    {"problem":"slow computer performance high CPU","solution":"Check Task Manager, clean startup, add RAM","category":"performance"},
    {"problem":"printer not found not detected","solution":"Reinstall printer drivers, check USB cable","category":"hardware"},
    {"problem":"email not sending SMTP error","solution":"Check SMTP settings, verify credentials","category":"software"},
    {"problem":"application crashing on startup","solution":"Reinstall app, check compatibility mode","category":"software"},
]
 
class CBRSystem:
    def __init__(self, case_base):
        self.cases=case_base
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2))
        problems=[c['problem'] for c in case_base]
        self.X=self.vectorizer.fit_transform(problems)
 
    def retrieve(self, query, top_k=3):
        q_vec=self.vectorizer.transform([query])
        sims=cosine_similarity(q_vec, self.X)[0]
        top=np.argsort(sims)[::-1][:top_k]
        return [(self.cases[i], sims[i]) for i in top]
 
    def resolve(self, query):
        results=self.retrieve(query,1)
        if results and results[0][1]>0.1:
            return results[0][0]['solution']
        return "Escalate to human support"
 
cbr=CBRSystem(CASE_BASE)
queries=["my laptop won't start","internet keeps dropping","computer running very slow"]
for q in queries:
    print(f"Query: '{q}'"); print(f"  → {cbr.resolve(q)}
")
