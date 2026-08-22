import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

class0=[[2,55],[3,60],[1,50],[4,58],[2,62],[5,65],[3,52],[1,48],[4,61],[2,57],[6,68],[3,59],[5,55],[2,64],[4,54],[1,45],[5,63],[3,56],[6,60],[2,51],[4,66],[3,62],[5,59],[1,53],[6,64],[2,58],[4,57],[3,54],[5,61],[2,56],[6,67],[4,63],[1,49],[3,61],[5,64],[2,53],[4,60],[6,62],[3,58],[5,57],[1,52],[4,65],[2,60],[5,62],[3,55],[6,66],[2,54],[4,59],[5,60],[3,57]]

class1=[[5,64],[6,66],[7,70],[8,76],[5,68],[7,74],[8,78],[6,72],[9,82],[5,70],[7,77],[8,80],[10,88],[6,69],[9,84],[5,67],[7,73],[8,75],[10,90],[6,71],[9,80],[7,79],[8,82],[5,65],[10,86],[6,74],[9,78],[7,76],[8,84],[5,69],[10,92],[6,70],[9,86],[7,72],[8,79],[5,66],[10,89],[6,73],[9,83],[7,75],[8,81],[5,68],[10,91],[6,76],[9,85],[7,78],[8,77],[5,71],[10,87],[6,72]]

x=class0+class1
y=[0]*len(class0)+[1]*len(class1)
targetpoint=[6,68]

# A1

def euclidean(a,b):
    euc=0
    for i,j in zip(a,b):
        euc+=(i-j)**2
    euc=math.sqrt(euc)
    return euc

def bubblesort(distances):
    for i in range(len(distances)):
        for j in range(len(distances)-i-1):
            if distances[j][0]>distances[j+1][0]:
                distances[j],distances[j+1]=distances[j+1],distances[j]
    return distances

def selectionsort(distances):
    for i in range(len(distances)):
        minindex=i
        for j in range(i+1,len(distances)):
            if distances[j][0]<distances[minindex][0]:
                minindex=j
        distances[i],distances[minindex]=distances[minindex],distances[i]
    return distances

def insertionsort(distances):
    for i in range(1,len(distances)):
        key=distances[i]
        j=i-1
        while j>=0 and distances[j][0]>key[0]:
            distances[j+1]=distances[j]
            j-=1
        distances[j+1]=key
    return distances

def getneighbours(xtrain,ytrain,target,k,sorttype):
    distances=[]
    for i in range(len(xtrain)):
        distance=euclidean(target,xtrain[i])
        distances.append([distance,ytrain[i]])
    if sorttype=="bubble":
        distances=bubblesort(distances)
    elif sorttype=="selection":
        distances=selectionsort(distances)
    else:
        distances=insertionsort(distances)
    neighbours=[]
    for i in range(k):
        neighbours.append(distances[i])
    return neighbours

def majorityclass(neighbours):
    class0count=0
    class1count=0
    for i in neighbours:
        if i[1]==0:
            class0count+=1
        else:
            class1count+=1
    if class0count>class1count:
        return 0
    else:
        return 1

for i in class0:
    euc=euclidean(targetpoint,i)
    print(targetpoint,"to",i,"=",euc)

for i in class1:
    euc=euclidean(targetpoint,i)
    print(targetpoint,"to",i,"=",euc)

distances=[]

for i in class0:
    euc=euclidean(targetpoint,i)
    distances.append([euc,0])

for i in class1:
    euc=euclidean(targetpoint,i)
    distances.append([euc,1])

print("\nDistances:")

for i in distances:
    print(i)

distances=bubblesort(distances)

print("\nSorted distances:")

for i in distances:
    print(i)

k=3
neighbours=getneighbours(x,y,targetpoint,k,"bubble")

print("\nK nearest neighbours:")

for i in neighbours:
    print(i)

result=majorityclass(neighbours)
print("Predicted class =",result)

# A2

def weightedclass(neighbours):
    class0weight=0
    class1weight=0
    for i in neighbours:
        distance=i[0]
        label=i[1]
        if distance==0:
            return label
        weight=1/distance
        if label==0:
            class0weight+=weight
        else:
            class1weight+=weight
    if class0weight>class1weight:
        return 0
    else:
        return 1

weightedresult=weightedclass(neighbours)
class0weight=0
class1weight=0

for i in neighbours:
    if i[0]==0:
        if i[1]==0:
            class0weight=float("inf")
        else:
            class1weight=float("i   nf")
    elif i[1]==0:
        class0weight+=1/i[0]
    else:
        class1weight+=1/i[0]

print("\nA2")
print("Class 0 weight =",class0weight)
print("Class 1 weight =",class1weight)
print("Weighted predicted class =",weightedresult)

# A3

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)

print("\nA3")
print("X train =",xtrain)
print("X test =",xtest)
print("Y train =",ytrain)
print("Y test =",ytest)

# A4

neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(xtrain,ytrain)

print("\nA4")
print("KNN classifier trained")

# A5

accuracy=neigh.score(xtest,ytest)

print("\nA5")
print("Accuracy =",accuracy)

# A6

prediction=neigh.predict(xtest)

print("\nA6")
print("Predictions:")

for i in prediction:
    print(i)

# A7

def fit(xtrain,ytrain):
    return xtrain,ytrain

def predict(model,target,k):
    xtrain,ytrain=model
    neighbours=getneighbours(xtrain,ytrain,target,k,"bubble")
    return majorityclass(neighbours)

def score(model,xtest,ytest,k):
    correct=0
    for i in range(len(xtest)):
        if predict(model,xtest[i],k)==ytest[i]:
            correct+=1
    return correct/len(xtest)

model=fit(xtrain,ytrain)
result=predict(model,xtest[0],3)
accuracy=score(model,xtest,ytest,3)

print("\nA7")
print("Prediction =",result)
print("My KNN accuracy =",accuracy)

# A8

kvalues=[]
myaccuracy=[]
sklearnaccuracy=[]

for k in range(1,6):
    neigh=KNeighborsClassifier(n_neighbors=k)
    neigh.fit(xtrain,ytrain)
    acc1=neigh.score(xtest,ytest)
    acc2=score(model,xtest,ytest,k)
    kvalues.append(k)
    sklearnaccuracy.append(acc1)
    myaccuracy.append(acc2)

print("\nA8")
print("K values =",kvalues)
print("My KNN accuracy =",myaccuracy)
print("Sklearn KNN accuracy =",sklearnaccuracy)

plt.plot(kvalues,myaccuracy,marker="o")
plt.plot(kvalues,sklearnaccuracy,marker="*")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("My KNN vs Sklearn KNN")
plt.grid()
plt.show()

# A9

def weightedpredict(model,target,k):
    xtrain,ytrain=model
    neighbours=getneighbours(xtrain,ytrain,target,k,"bubble")
    return weightedclass(neighbours)

def weightedscore(model,xtest,ytest,k):
    correct=0
    for i in range(len(xtest)):
        if weightedpredict(model,xtest[i],k)==ytest[i]:
            correct+=1
    return correct/len(xtest)

weightedkvalues=[]
myweightedaccuracy=[]
sklearnweightedaccuracy=[]

for k in range(1,6):
    neigh=KNeighborsClassifier(n_neighbors=k,weights="distance")
    neigh.fit(xtrain,ytrain)
    acc1=neigh.score(xtest,ytest)
    acc2=weightedscore(model,xtest,ytest,k)
    weightedkvalues.append(k)
    sklearnweightedaccuracy.append(acc1)
    myweightedaccuracy.append(acc2)

print("\nA9")
print("K values =",weightedkvalues)
print("My weighted KNN accuracy =",myweightedaccuracy)
print("Sklearn weighted KNN accuracy =",sklearnweightedaccuracy)

plt.plot(weightedkvalues,myweightedaccuracy,marker="o")
plt.plot(weightedkvalues,sklearnweightedaccuracy,marker="*")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("My Weighted KNN vs Sklearn Weighted KNN")
plt.grid()
plt.show()