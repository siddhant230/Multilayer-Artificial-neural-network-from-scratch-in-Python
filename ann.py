import numpy as np
class network:
    def __init__(self,l):
        self.l=l
        self.lr=0.1
        self.we,self.bias=[],[]
        '''assigning weights'''
        for i in range(len(l)-1):
            w=np.random.normal(0.0,pow(l[i+1],-0.5),(l[i+1],l[i]))
            self.we.append(w)
        '''assigmning biases'''
        for i in range(len(l)-1):
            b=np.random.normal(0.0,pow(l[i+1],-0.5),(l[i+1],1))
            self.bias.append(b)
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def dsigmoid(self,x):
        return x*(1-x)
        
    def feedforward(self,inp):
        inp=np.array(inp)
        val_arr=[]
        x=np.reshape(inp,(self.l[0],1))
        print(x)
        for i in range(len(self.we)):
            val_arr.append(x)
            z=np.dot(self.we[i],x)+self.bias[i]
            a=self.sigmoid(z)
            x=a
        val_arr.append(x)
        return val_arr[-1]
    
    def backprop(self,inp,target,it):
        target=np.array(target)
        inp=np.array(inp)
        val_arr=[]
        x=np.reshape(inp,(self.l[0],1))
        for i in range(len(self.we)):
            val_arr.append(x)
            z=np.dot(self.we[i],x)+self.bias[i]
            a=self.sigmoid(z)
            x=a
        val_arr.append(x)
        got=val_arr[-1]
        
        '''computing error and feeding it back to circuit'''
        error=target-got
        if it%500==0:
            print('error=>      ',error)
        for i in range(len(self.we)-1,-1,-1):
            grad_w=self.lr*np.dot((error*self.dsigmoid(val_arr[i+1])),np.transpose(val_arr[i]))
            self.we[i]+=grad_w
            
            grad_b=self.lr*error*self.dsigmoid(val_arr[i+1])
            self.bias[i]+=grad_b
            
            error=np.dot(np.transpose(self.we[i]),error)
        
def XOR():
    inp=[]
    tar=[]
    import random
    for i in range(2):
        r=random.choice('10')
        inp.append(int(r))
    if inp[0]==inp[1]:
        tar.append(0)
    else:
        tar.append(1)
    return inp,tar

l=[2]
hid_arr=[]
hid=int(input('enter number of hidden layer=>'))
for i in range(hid):
  hid_arr.append(int(input('nodes in layer=>')))
l.extend(hid_arr)
l.append(1)
obj=network(l)
epochs=int(input('how many times u want the iteration=>'))
for i in range(epochs):
    inp,tar=XOR()
    obj.backprop(inp,tar,i)
v=[]
for i in range(2):
    v.append(int(input('val=>')))
val=obj.feedforward(v)
print(val)