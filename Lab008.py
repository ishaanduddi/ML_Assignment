import math
import matplotlib.pyplot as plt

def summation(x,w):
    s=0
    for i in range(len(x)):
        s=s+(x[i]*w[i])
    return s

def step(x):
    if x>=0:
        return 1
    else:
        return 0

def bipolar_step(x):
    if x>=0:
        return 1
    else:
        return -1

def sigmoid(x):
    value=1/(1+math.exp(-x))
    return value

def tanh(x):
    return math.tanh(x)

def relu(x):
    if x>0:
        return x
    else:
        return 0

def leaky_relu(x):
    if x>0:
        return x
    else:
        return 0.01*x

def comparator(o,t):
    error=t-o
    return error

print("A1\n")

x=[2,1]
w=[0.5,0.3]

s=summation(x,w)

print("summation :",s)
print("step :",step(s))
print("bipolar step :",bipolar_step(s))
print("sigmoid :",sigmoid(s))
print("tanH :",tanh(s))
print("reLU :",relu(s))
print("leaky relu :",leaky_relu(s))
print("error :",comparator(1,0))

print("\n")

def step(net):
    if net>=0:
        return 1
    else:
        return 0

print("A2\n")

inputs=[[0,0],[0,1],[1,0],[1,1]]
targets=[0,0,0,1]

w0=10
w1=0.2
w2=-0.75
lr=0.05
epoch=0
sse_list=[]

while epoch<1000:
    sse=0

    for i in range(len(inputs)):
        net=w0+(inputs[i][0]*w1)+(inputs[i][1]*w2)
        output=step(net)
        error=targets[i]-output
        sse=sse+(error*error)

        w0=w0+(lr*error)
        w1=w1+(lr*error*inputs[i][0])
        w2=w2+(lr*error*inputs[i][1])

    sse_list.append(sse)
    epoch=epoch+1

    if sse<=0.002:
        break

print("epochs :",epoch)
print("w0 :",w0)
print("w1 :",w1)
print("w2 :",w2)

plt.plot(range(1,epoch+1),sse_list)
plt.xlabel("epoch")
plt.ylabel("sse")
plt.show()

print("\n")

print("A3\n")

def train(act_func,targets):
    inputs=[[0,0],[0,1],[1,0],[1,1]]

    w0=10
    w1=0.2
    w2=-0.75
    lr=0.05
    epoch=0

    while epoch<1000:
        sse=0

        for i in range(len(inputs)):
            net=w0+(inputs[i][0]*w1)+(inputs[i][1]*w2)
            output=act_func(net)
            error=targets[i]-output
            sse=sse+(error*error)

            w0=w0+(lr*error)
            w1=w1+(lr*error*inputs[i][0])
            w2=w2+(lr*error*inputs[i][1])

        epoch=epoch+1

        if sse<=0.002:
            break

    return epoch

bipolar_epochs=train(bipolar_step,[-1,-1,-1,1])
sigmoid_epochs=train(sigmoid,[0,0,0,1])
relu_epochs=train(relu,[0,0,0,1])

print("bipolar step epochs :",bipolar_epochs)
print("sigmoid epochs :",sigmoid_epochs)
print("relu epochs :",relu_epochs)

print("\n")