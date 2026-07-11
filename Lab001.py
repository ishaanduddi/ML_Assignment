#SET A
##Q1
print("Q1")
def sum(list,target):
    sum=0
    for i in range(len(list)):
        for j in range(i+1,len(list)):
            if list[i]+list[j]==target:
                sum+=1
    return sum
    
list=[2,7,4,1,3,6]
target=10
print("The total count is :-",sum(list,target))

##Q2
print("Q2")
def max(list):
    max=list[0]
    for i in range(len(list)):
        if list[i]>max:
            max=list[i]
    return max

def min(list):
    min=list[0]
    for i in range(len(list)):
        if min>list[i]:
            min=list[i]
    return min

def ranger(list):
    if (len(list)<3):
        return "Range determination not possible"
    maximum=max(list)
    minimum=min(list)
    ranges=maximum-minimum
    return ranges
    
list1=[5,3,8,1,0,4]
print("range is :- ",ranger(list1))

##Q3
print("Q3")
def matrixmultiplication(matrix,m,n,M):
    if m!=n:
        return"not a square matrix"
    result=matrix

    for times in range(M-1):
        temp=[[0 for i in range(m)] for j in range(m)]
        for i in range(m):
            for j in range(n):
                for k in range(m):
                    temp[i][j]+=result[i][k]*matrix[k][j]
        result=temp
    return result

matrix=[[1,2,3],[4,5,6],[7,8,9]]
m=len(matrix)
n=len(matrix[0])
print("Answer is :- ",matrixmultiplication(matrix,m,n,5))

##Q4
print("Q4")
word="hippopotamus"
def lettercount(word):
    letter=[]
    maxcount=0
    maxletter=""
    for ch in word:
        if ch not in letter:
            letter.append(ch)
    for ch in letter:
        count=0
        if ch in letter:
            for a in word:
                if a==ch:
                    count+=1
            if count>maxcount:
                maxcount=count
                maxletter=ch
        
            print(ch,count)
    print("max letter and count is")
    print(maxletter,maxcount)

lettercount(word)

##Q5
print("Q5")
import random
def numberlist():
    numbers=[]
    for i in range(25):
        numbers.append(random.randint(1,10))
    return numbers

def mean(numbers):
    length=len(numbers)
    sum=0
    for i in range(len(numbers)):
        sum+=numbers[i]
    avg=sum/length
    return avg

def median(numbers):
    numbers.sort()
    n=len(numbers)
    if n%2!=0:
        return numbers[n//2]
    else:
        return numbers[(n//2)-1],numbers[(n//2)]
    
def mode(numbers):
    max_count = 0
    mode_value = None

    for num in numbers:
        count = 0
        for x in numbers:
            if x == num:
                count += 1

        if count > max_count:
            max_count = count
            mode_value = num

    return mode_value, max_count
number=numberlist()
print("numbers",number)
print("mode",mode(number))
print("mean",mean(number))
print("median",median(number))
