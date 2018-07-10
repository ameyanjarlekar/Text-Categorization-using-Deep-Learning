# 20Newsgroupsprac

#part 1


import numpy as np
#import matplotlib.pyplot as plt
import math

from sklearn.datasets import fetch_20newsgroups_vectorized

newsgroups_train = fetch_20newsgroups_vectorized(subset="train")
newsgroups_test = fetch_20newsgroups_vectorized(subset="test")
# print(newsgroups_train.data[1])
Xtrain = newsgroups_train.data
Ytrain = newsgroups_train.target
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(150,5))
clf.fit(Xtrain,Ytrain)
print(clf.predict(newsgroups_test.data[355]))
print(newsgroups_test.target[355])
from sklearn.metrics import accuracy_score
print(accuracy_score(newsgroups_test.target,clf.predict(newsgroups_test.data)))


#part 2

import numpy as np
import math


from sklearn.datasets import fetch_20newsgroups
cat = ['alt.atheism',
 'rec.autos',
 'sci.electronics',
'talk.politics.guns',
'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset="train",categories = cat )
newsgroups_test = fetch_20newsgroups(subset="test", categories = cat )
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
theta_1 = 2*np.random.rand(20,5001)-1
theta_2 = 2*np.random.rand(20,20)-1
theta_3 = 2*np.random.rand(20,20)-1
theta_4 = 2*np.random.rand(10,20)-1
theta_6 = 2*np.random.rand(10,10)-1
theta_5 = 2*np.random.rand(5,10)-1
vectorizer = TfidfVectorizer(max_features=20000)

m = 0
n = 0
rs = 6  
vectors3 = vectorizer.fit_transform(newsgroups_train.data)
#print(vectors2.shape)
vectors2 = vectors3.toarray()
vectors = np.random.rand(math.floor(2810*0.1*rs),5001)
while m < math.floor(2810*0.1*rs) :
    while n < 5001 :
        if n == 0:
           vectors[m][n] = 1
        else :
            vectors[m][n] = vectors2[m][n-1]
        n = n+1
    m = m+1
    n =0
m = 0
n = 0

testvectors3 = vectorizer.fit_transform(newsgroups_test.data)
testvectors2 = testvectors3.toarray()
testvectors = np.random.rand(math.floor(18700*0.1*rs),5001)
m = 0
n = 0
while m < math.floor(1870*0.1*rs) :
    while n < 5001 :
        if n == 0:
            testvectors[m][n] = 1
        else :
            testvectors[m][n] = testvectors2[m][n-1]
        n = n+1
    m = m+1
    n =0
m = 0
n = 0
tracker = 0 
while tracker < 5000 :
    
    cost2 = 0
    cost_final = 0

    v = np.array(vectors)-0.01
    g1 =  1/(1 + np.exp(- np.dot(theta_1,v.T) ))
    g2 =  1/(1 + np.exp(- np.dot(theta_2,g1) ))
    g3 =  1/(1 + np.exp(- np.dot(theta_3,g2) ))
    g4 =  1/(1 + np.exp(- np.dot(theta_4,g3) ))
    g6 =  1/(1 + np.exp(- np.dot(theta_6,g4) ))
    g5 =  1/(1 + np.exp(-np.dot(theta_5,g6)))

    cost1 = 0
    def find(a):
        out = [0]*5
        v = 0        
        while v <5:
            if a == v:
                out[v]=1
                v = 5
            v = v+1                        
        return out 


    p = 0
    q = 0
    while p < math.floor(2810*0.1*rs):
        value = find(newsgroups_train.target[p])
        
        while q < 5:
            
            
           
            #cost1 = cost1 + value[q]*math.log(g5[q][p]) + (1-value[q])*math.log(1-g5[q][p])
            k = (value[q]-g5[q][p])*(value[q]-g5[q][p])

            cost1 = cost1 + k
            q = q+1
            
        
        p = p+1 
        
        q = 0
    
    l = 0

    cost = cost1/(2810*0.1*rs*2) 
    
    r = 0
    final = []
    while r<math.floor(2810*0.1*rs):
        final = final + [find(newsgroups_train.target[r])]
        r = r+1
    
    ans = np.array(final)

    d5 = ( g5 - ans.T )*g5*(1-g5)    
    D5 = np.dot(d5,g6.T)
    
    d6 = np.dot(theta_5.T,d5)*g6*(1-g6)  
    D6 = np.dot(d6,g4.T)
    
    d4 = np.dot(theta_6.T,d6)*g4*(1-g4)  
    D4 = np.dot(d4,g3.T)
    
    d3 = np.dot(theta_4.T,d4)*g3*(1-g3)  
    D3 = np.dot(d3,g2.T)
    
    d2 = np.dot(theta_3.T,d3)*g2*(1-g2)  
    D2 = np.dot(d2,g1.T)
    
    d1 = np.dot(theta_2.T,d2)*g1*(1-g1)  
    D1 = np.dot(d1,v)

    l1 = 15
    l2 = 15
    l3 = 15
    l4 = 15
    l6 = 15
    l5 = 15
    l = 0.5

    theta_1 = theta_1*(1-l/2810*0.1*rs) - (l1/2810*0.1*rs)*D1
    theta_2 = theta_2*(1-l/2810*0.1*rs) - (l2/2810*0.1*rs)*D2
    theta_3 = theta_3*(1-l/2810*0.1*rs) - (l3/2810*0.1*rs)*D3
    theta_4 = theta_4*(1-l/2810*0.1*rs) - (l4/2810*0.1*rs)*D4
    theta_6 = theta_6*(1-l/2810*0.1*rs) - (l6/2810*0.1*rs)*D6
    theta_5 = theta_5*(1-l/2810*0.1*rs) - (l5/2810*0.1*rs)*D5        
    
    accuracy = np.random.rand(math.floor(2810*0.1*rs))
    k = 0
    a = -1000
    p = 0
    while k < math.floor(2810*0.1*rs) :        
        while p<5:
            a = max(a,g5[p][k])                            
            p = p+1
        p = 0
        while p <5:
            if a == g5[p][k]:
                accuracy[k]=p               
                p = 5
            p = p+1
        p = 0
        a = -1000
        k = k+1
   
    tracker = tracker + 1
    if tracker % (100 ) == 0:
        print(tracker)
        print("cost")
        print(cost)
        print("train_accuracy")
        print(accuracy_score(accuracy,newsgroups_train.target[:math.floor(2810*0.1*rs)]))
        cost_final = 0
        cost2 = 0
        p = 0
        q = 0
        while p < math.floor(1870*0.1*rs):
            value = find(newsgroups_test.target[p])        
            while q < 5:            
                k = (value[q]-g5[q][p])*(value[q]-g5[q][p])          
                cost2 = cost2 + k
                q = q+1
                    
            p = p+1         
            q = 0
        vf = np.array(testvectors)-0.01
        g1f = 1/(1 + np.exp(- np.dot(theta_1,vf.T) ))
        g2f = 1/(1 + np.exp(- np.dot(theta_2,g1f) ))
        g3f = 1/(1 + np.exp(- np.dot(theta_3,g2f) ))
        g4f = 1/(1 + np.exp(- np.dot(theta_4,g3f) ))
        g6f = 1/(1 + np.exp(- np.dot(theta_6,g4f) ))
        g5f =  1/(1 + np.exp(-np.dot(theta_5,g6f)))
        measure = np.random.rand(math.floor(1870*0.1*rs))
        b = 0
        z = 0
        m = 0
        while b < math.floor(1870*0.1*rs) :
            while m <5:
                z = max(z,g5f[m][b])
                m = m+1
            m = 0
            while m<5:        
                if z == g5f[m][b]:                          
                    measure[b] = m
                    m =5
                m = m+1
            z = 0
            b = b+1
            m = 0
        cost_final = cost2/(1870*0.1*rs*2)
        print("testcost")
        print(cost_final)
        print("accuracy")
        print(accuracy_score(measure,newsgroups_test.target[:math.floor(1870*0.1*rs)]))
  
