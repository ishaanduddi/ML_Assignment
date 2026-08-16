import math
class0 = [[1, 2],[2, 3],[3, 1],[2, 1],[3, 2]]
class1 = [[7, 8],[8, 9],[9, 7],[8, 7],[9, 8]]
target_point = [5, 5]

def euclidean(a,b):
    euc=0
    for i,j in zip(a,b):
                euc+=(i-j)**2
    euc=math.sqrt(euc)
    return euc

for i in class0:
    euc = euclidean(target_point,i)
    print(target_point,"to",i, "=", euc)
for i in class1:
    euc = euclidean(target_point,i)
    print(target_point,"to",i, "=", euc)
distances=[]    
