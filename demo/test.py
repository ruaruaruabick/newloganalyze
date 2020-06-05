from polyglot.mapping import Embedding,CaseExpander, DigitExpander
import pandas as pd
import csv
def renodotentity(lylist):
    temp = []
    for i in lylist:
        if '.' not in i:
            temp.append(i)
    return temp
embeddings = Embedding.load("C:/Users/99349/AppData/Roaming/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2")
embeddings.apply_expansion(CaseExpander)
embeddings.apply_expansion(DigitExpander)
neighbors = embeddings.nearest_neighbors("green")
print(neighbors)
print(embeddings.distances("green", neighbors))
eanda = pd.read_csv("IPLoM_result/entityandattribute.csv")
#把所有实体提取出来
entities = eanda["entity"].tolist()
attdic = {}
#字典：属性-实体
for index, row in eanda.iterrows():
    if row[1] in attdic.keys() and row[0] not in attdic[row[1]]:
        attdic[row[1]].append(row[0])
    else:
        attdic[row[1]] = []
        attdic[row[1]].append(row[0])
#计算实体之间的相似度并写入csv
with open('IPLoM_result/likely.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow((["entity","entitylike","similarity"]))
    for i in attdic.keys():
        attdic[i] = renodotentity(attdic[i])
        for j in attdic[i]:
            try:
                neighbors = embeddings.nearest_neighbors(j)
                for k in neighbors:
                    if k in attdic[i]:
                        writer.writerow(([j, k, embeddings.distances(j, [k])]))
#                        print(k)
#                        print(embeddings.distances(j, [k]))
#                print(embeddings.nearest_neighbors(j))
#                print(embeddings.distances(j,neighbors))
            except Exception:
                pass