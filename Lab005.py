import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
class0=[[2,55],[3,60],[1,50],[4,58],[2,62],[5,65],[3,52],[1,48],[4,61],[2,57],[6,68],[3,59],[5,55],[2,64],[4,54],[1,45],[5,63],[3,56],[6,60],[2,51],[4,66],[3,62],[5,59],[1,53],[6,64],[2,58],[4,57],[3,54],[5,61],[2,56],[6,67],[4,63],[1,49],[3,61],[5,64],[2,53],[4,60],[6,62],[3,58],[5,57],[1,52],[4,65],[2,60],[5,62],[3,55],[6,66],[2,54],[4,59],[5,60],[3,57]]
class1=[[5,64],[6,66],[7,70],[8,76],[5,68],[7,74],[8,78],[6,72],[9,82],[5,70],[7,77],[8,80],[10,88],[6,69],[9,84],[5,67],[7,73],[8,75],[10,90],[6,71],[9,80],[7,79],[8,82],[5,65],[10,86],[6,74],[9,78],[7,76],[8,84],[5,69],[10,92],[6,70],[9,86],[7,72],[8,79],[5,66],[10,89],[6,73],[9,83],[7,75],[8,81],[5,68],[10,91],[6,76],[9,85],[7,78],[8,77],[5,71],[10,87],[6,72]]
X=class0+class1
y=[0]*len(class0)+[1]*len(class1)
target_point=[6,68]

# A1

def euclidean(a,b):
    euc=0
    for i,j in zip(a,b):
        euc+=(i-j)**2
    euc=math.sqrt(euc)
    return euc

def bubble_sort(distances):
    for i in range(len(distances)):
        for j in range(len(distances)-i-1):
            if distances[j][0]>distances[j+1][0]:
                distances[j],distances[j+1]=distances[j+1],distances[j]
    return distances

def selection_sort(distances):
    for i in range(len(distances)):
        min_index=i
        for j in range(i+1,len(distances)):
            if distances[j][0]<distances[min_index][0]:
                min_index=j
        distances[i],distances[min_index]=distances[min_index],distances[i]
    return distances

def insertion_sort(distances):
    for i in range(1,len(distances)):
        key=distances[i]
        j=i-1
        while j>=0 and distances[j][0]>key[0]:
            distances[j+1]=distances[j]
            j-=1
        distances[j+1]=key
    return distances

def get_neighbours(X_train,y_train,target,k,sort_type):
    distances=[]
    for i in range(len(X_train)):
        distance=euclidean(target,X_train[i])
        distances.append([distance,y_train[i]])
    if sort_type=="bubble":
        distances=bubble_sort(distances)
    elif sort_type=="selection":
        distances=selection_sort(distances)
    else:
        distances=insertion_sort(distances)
    neighbours=[]
    for i in range(k):
        neighbours.append(distances[i])
    return neighbours

def majority_class(neighbours):
    class0_count=0
    class1_count=0
    for i in neighbours:
        if i[1]==0:
            class0_count+=1
        else:
            class1_count+=1
    if class0_count>class1_count:
        return 0
    else:
        return 1

for i in class0:
    euc=euclidean(target_point,i)
    print(target_point,"to",i,"=",euc)

for i in class1:
    euc=euclidean(target_point,i)
    print(target_point,"to",i,"=",euc)
distances=[]

for i in class0:
    euc=euclidean(target_point,i)
    distances.append([euc,0])

for i in class1:
    euc=euclidean(target_point,i)
    distances.append([euc,1])
print("\nDistances:")

for i in distances:
    print(i)
distances=bubble_sort(distances)
print("\nSorted distances:")

for i in distances:
    print(i)
k=3
neighbours=get_neighbours(X,y,target_point,k,"bubble")
print("\nK nearest neighbours:")

for i in neighbours:
    print(i)
result=majority_class(neighbours)
print("Predicted class =",result)

# A2

def weighted_class(neighbours):
    class0_weight=0
    class1_weight=0
    for i in neighbours:
        distance=i[0]
        label=i[1]
        if distance==0:
            return label
        weight=1/distance
        if label==0:
            class0_weight+=weight
        else:
            class1_weight+=weight
    if class0_weight>class1_weight:
        return 0
    else:
        return 1

weighted_result=weighted_class(neighbours)
class0_weight=0
class1_weight=0

for i in neighbours:
    if i[0]==0:
        if i[1]==0:
            class0_weight=float("inf")
        else:
            class1_weight=float("inf")
    elif i[1]==0:
        class0_weight+=1/i[0]
    else:
        class1_weight+=1/i[0]

print("\nA2")
print("Class 0 weight =",class0_weight)
print("Class 1 weight =",class1_weight)
print("Weighted predicted class =",weighted_result)

# A3

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
print("\nA3")
print("X_train =",X_train)
print("X_test =",X_test)
print("y_train =",y_train)
print("y_test =",y_test)

# A4
neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
print("\nA4")
print("KNN classifier trained")

# A5

accuracy=neigh.score(X_test,y_test)
print("\nA5")
print("Accuracy =",accuracy)

# A6

prediction=neigh.predict(X_test)
print("\nA6")
print("Predictions:")
for i in prediction:
    print(i)

# A7

def fit(X_train,y_train):
    return X_train,y_train

def predict(model,target,k):
    X_train,y_train=model
    neighbours=get_neighbours(X_train,y_train,target,k,"bubble")
    return majority_class(neighbours)

def score(model,X_test,y_test,k):
    correct=0
    for i in range(len(X_test)):
        if predict(model,X_test[i],k)==y_test[i]:
            correct+=1
    return correct/len(X_test)

model=fit(X_train,y_train)
result=predict(model,X_test[0],3)
accuracy=score(model,X_test,y_test,3)
print("\nA7")
print("Prediction =",result)
print("My KNN accuracy =",accuracy)

# A8

k_values=[]
my_accuracy=[]
sklearn_accuracy=[]

for k in range(1,6):
    neigh=KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train,y_train)
    acc1=neigh.score(X_test,y_test)
    acc2=score(model,X_test,y_test,k)
    k_values.append(k)
    sklearn_accuracy.append(acc1)
    my_accuracy.append(acc2)

print("\nA8")
print("K values =",k_values)
print("My KNN accuracy =",my_accuracy)
print("Sklearn KNN accuracy =",sklearn_accuracy)
plt.plot(k_values,my_accuracy,marker="o")
plt.plot(k_values,sklearn_accuracy,marker="*")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("My KNN vs Sklearn KNN")
plt.grid()
plt.show()

# A9

def weighted_predict(model,target,k):
    X_train,y_train=model
    neighbours=get_neighbours(X_train,y_train,target,k,"bubble")
    return weighted_class(neighbours)

def weighted_score(model,X_test,y_test,k):
    correct=0
    for i in range(len(X_test)):
        if weighted_predict(model,X_test[i],k)==y_test[i]:
            correct+=1
    return correct/len(X_test)

weighted_k_values=[]
my_weighted_accuracy=[]
sklearn_weighted_accuracy=[]

for k in range(1,6):
    neigh=KNeighborsClassifier(n_neighbors=k,weights="distance")
    neigh.fit(X_train,y_train)
    acc1=neigh.score(X_test,y_test)
    acc2=weighted_score(model,X_test,y_test,k)
    weighted_k_values.append(k)
    sklearn_weighted_accuracy.append(acc1)
    my_weighted_accuracy.append(acc2)

print("\nA9")
print("K values =",weighted_k_values)
print("My weighted KNN accuracy =",my_weighted_accuracy)
print("Sklearn weighted KNN accuracy =",sklearn_weighted_accuracy)
plt.plot(weighted_k_values,my_weighted_accuracy,marker="o")
plt.plot(weighted_k_values,sklearn_weighted_accuracy,marker="*")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("My Weighted KNN vs Sklearn Weighted KNN")
plt.grid()
plt.show()