import pandas as pd
import os
import shutil
df = pd.read_csv('labels_train.csv', header=None, names=['Column1', 'Column2'])
data = df.values.tolist()


estados = set({})
d = {}
a=[]
for i in data: 
    estados.add(i[1])
    a.append([])
for e in estados: 
    d[e]=[]
del d['Label']
for i in data:
    if i[1]!='Label':
        d[i[1]].append(i[0])
for e in estados: 
    if not os.path.exists(f"./{e}"):
        os.mkdir(e)
        
for k in d.keys():
    for f in d[k]:
        src = f"./train/{f}"
        dest = f"./{k}/"
        shutil.copy(src, dest)
