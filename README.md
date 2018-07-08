# 20Newsgroupsprac
import numpy as np
import math


from sklearn.datasets import fetch_20newsgroups
cat = ['alt.atheism',
 'rec.autos',
 'sci.electronics',
'talk.politics.guns',
'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset="train",categories = cat)
newsgroups_test = fetch_20newsgroups(subset="test", categories = cat)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
vectorizer = TfidfVectorizer(max_features=29000)

# max is 90000
# 11314 and 7532
vectors = vectorizer.fit_transform(newsgroups_train.data)
#vectors3 = vectors4.toarray()
#brown = [[1]]*2810
#vectors2 = vectors3.tolist()
#vectors4 = np.array(vectors2)
#print(vectors4.shape)
#vectors1 = vectors4.tolist()

#vectors = [[0]]*2810
#bit = 0
#while bit < 2810:
#    vectors[bit] = brown[bit] + vectors3[bit].tolist()
#    bit = bit +1
#print(vectors[5].shape)
#print(vectors3[5].shape)
#vectors = np.array(vectors2)    
# subtract by average divide by variance
testvectors = vectorizer.fit_transform(newsgroups_test.data)
#testvectors3 = testvectors4.toarray()
#testvectors1 = testvectors4.tolist()
#white = [[1]]*1870
#testvectors = [[0]]*1870
#bi =0
#while bi < 1870:
#    testvectors[bi] = white[bi] + testvectors3[bi]
#    bi = bi +1
#testvectors = np.array(testvectors2)

#initialisation -1 to 1 
theta_1 = 2*np.random.rand(20,29001)-1
theta_2 = 2*np.random.rand(20,20)-1
theta_3 = 2*np.random.rand(20,20)-1
theta_4 = 2*np.random.rand(20,20)-1
theta_6 = 2*np.random.rand(20,20)-1
theta_5 = 2*np.random.rand(5,20)-1
tracker = 0 

#9920553846

#total examples = 2840
while tracker < 10000 :
   
    
    #print(theta_1.shape)
    #print(v.shape)

    v = np.array(vectors)
    print(v.shape)
    g1 =  1/(1 + np.exp(- np.dot(theta_1,v.T) ))
    g2 =  1/(1 + np.exp(- np.dot(theta_2,g1) ))
    g3 =  1/(1 + np.exp(- np.dot(theta_3,g2) ))
    g4 =  1/(1 + np.exp(- np.dot(theta_4,g3) ))
    g6 =  1/(1 + np.exp(- np.dot(theta_6,g4) ))
    g5 =  1/(1 + np.exp(-np.dot(theta_5,g6)))

        
    #print("forward prop done")
    
    #print(g5.shape)
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
    while p < 2810:
        value = find(newsgroups_train.target[p])
        
        while q < 5:
            
            
           
            #cost1 = cost1 + value[q]*math.log(g5[q][p]) + (1-value[q])*math.log(1-g5[q][p])
            k = (value[q]-g5[q][p])*(value[q]-g5[q][p])
            #if p == 7:
             #   print(k)
            
            cost1 = cost1 + k
            q = q+1
            
        
        p = p+1 
        
        q = 0
    
    l = 0

    cost = cost1/(2810*2) 
    #print(cost)
    
    #print("cost found out")
    
    r = 0
    final = []
    while r<2810:
        final = final + [find(newsgroups_train.target[r])]
        r = r+1
    
    ans = np.array(final)
    #print(ans.shape)
    
    #print("hurdle")

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

    #print("derivative found")
    
    l1 = 20
    l2 = 20
    l3 = 20
    l4 = 20
    l6 = 20
    l5 = 20

    theta_1 = theta_1 - (l1/2810)*D1
    theta_2 = theta_2 - (l2/2810)*D2
    theta_3 = theta_3 - (l3/2810)*D3
    theta_4 = theta_4 - (l4/2810)*D4
    theta_6 = theta_6 - (l6/2810)*D6
    theta_5 = theta_5 - (l5/2810)*D5        
    
    accuracy = np.random.rand(2810)
    k = 0
    a = -1000
    p = 0
    while k < 2810 :        
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
    if tracker%200 == 0:
        print(tracker)
        print(accuracy_score(accuracy,newsgroups_train.target))
        
        vf = np.array(testvectors)
        g1f = 1/(1 + np.exp(- np.dot(theta_1,vf.T) ))
        g2f = 1/(1 + np.exp(- np.dot(theta_2,g1f) ))
        g3f = 1/(1 + np.exp(- np.dot(theta_3,g2f) ))
        g4f = 1/(1 + np.exp(- np.dot(theta_4,g3f) ))
        g6f = 1/(1 + np.exp(- np.dot(theta_6,g4f) ))
        g5f =  1/(1 + np.exp(-np.dot(theta_5,g6f)))
        measure = np.random.rand(1870)
        b = 0
        z = 0
        m = 0
        while b < 1870 :
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
        print(accuracy_score(measure,newsgroups_test.target))
    

print("end")
