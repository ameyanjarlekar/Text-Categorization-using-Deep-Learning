# 20Newsgroupsprac
import numpy as np
import math
from sklearn.datasets import fetch_20newsgroups
cat = ['alt.atheism',
 'comp.graphics',
 'misc.forsale',
 'rec.autos',
 'rec.sport.baseball']
newsgroups_train = fetch_20newsgroups(subset="train",categories=cat)
newsgroups_test = fetch_20newsgroups(subset="test",categories=cat)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
vectors = vectorizer.fit_transform(newsgroups_train.data)
testvectors = vectorizer.fit_transform(newsgroups_test.data)
#print(vectors.shape)
#theta_1 = np.ones((5,1000))*0.01
theta_1 = np.random.rand(10,1000)*0.01
#theta_2 = np.random.rand(5,5)*0.01
#theta_3 = np.random.rand(5,5)*0.01
#theta_4 = np.random.rand(5,5)*0.01
theta_5 = np.random.rand(5,10)*0.01
theta_01 = np.random.rand(10,2840)*0.01
#theta_02 = np.random.rand(5,2840)*0.01
#theta_03 = np.random.rand(5,2840)*0.01
#theta_04 = np.random.rand(5,2840)*0.01
theta_05 = np.random.rand(5,2840)*0.01
tracker = 0
#total examples = 2840
while tracker < 100000 :
   
    
    #print(theta_1.shape)
    #print(v.shape)

    v = vectors.toarray()

    a1 = np.dot(theta_1,v.T) + theta_01
    #print(a1.shape)
    i = 0
    j = 0
    g1 = np.random.rand(10,2840)
    g1d = np.random.rand(10,2840)
    while j < 2840 :
        while i < 10 :
            g1[i][j] = 1/(1 + math.exp(-a1[i][j]))
            g1d[i][j] = g1[i][j]*(1-g1[i][j])
            i = i+1
        
        j = j + 1
        i = 0
    
    #a2 = np.dot(theta_2,g1)+ theta_02
    #print(a2.shape)
    #i = 0
    #j = 0
    #g2 = np.random.rand(5,2840)
    #g2d = np.random.rand(5,2840)
    #while j < 2840 :
    #    while i < 5 :
    #        g2[i][j] = 1/(1 + math.exp(-a2[i][j]))
    #        g2d[i][j] = g2[i][j]*(1-g2[i][j])
    #        i = i+1
        
    #    j = j+1
    #    i = 0
    
    #a3 = np.dot(theta_3,g2)+theta_03
    #print(a3.shape)
    #i = 0
    #j = 0
    #g3 = np.random.rand(5,2840)
    #g3d = np.random.rand(5,2840)
    #while j < 2840 :
    #    while i < 5 :
    #        g3[i][j] = 1/(1 + math.exp(-a3[i][j]))
    #        g3d[i][j] = g3[i][j]*(1-g3[i][j])
    #        i = i+1
        
    #    j = j+1
    #    i = 0
    
    #a4 = np.dot(theta_4,g3)+theta_04
    #print(a4.shape)
    #i = 0
    #j = 0
    #g4 = np.random.rand(5,2840)
    #g4d = np.random.rand(5,2840)
    #while j < 2840 :       
    #    while i < 5 :
    #        g4[i][j] = 1/(1 + math.exp(-a4[i][j]))
    #        g4d[i][j] = g4[i][j]*(1-g4[i][j])
    #        i = i+1
            
    #    j = j+1
    #    i = 0
        
    a5 = np.dot(theta_5,g1)+theta_05
    i = 0
    j = 0
    g5 = np.random.rand(5,2840)
    g5d = np.random.rand(5,2840)
    while j < 2840 :
        while i < 5 :
            g5[i][j] = 1/(1 + math.exp(-a5[i][j]))
            g5d[i][j] = g5[i][j]*(1-g5[i][j])
            i = i+1
        
        j = j+1
        i = 0
        
    #print("forward prop done")
    
    #print(g5.shape)
    cost1 = 0
    def find(a):
        if a == 0:
            out = [1,0,0,0,0]
        else:
            if a == 1:
                out = [0,1,0,0,0]
            else:
                if a == 2:
                    out = [0,0,1,0,0]
                else:
                    if a == 3:
                        out = [0,0,0,1,0]
                    else:
                        if a == 4:
                            out = [0,0,0,0,1]
                        
        return out 


    p = 0
    q = 0
    while p < 2840:
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
    
    l = 0.05
    cost2 = 0
    square1 = 0 
    square2 = 0
    square3 = 0
    square4 = 0
    square5 = 0
    y = 0
    z = 0
    while y < 1000:
        while z < 10 :
            square1 = square1 + (theta_1[z][y])*(theta_1[z][y])
            z = z+1
        
    
        y = y+1
        z = 0
    
    y = 0
    z = 0

    #while y < 5:
    #    while z < 5 :
    #        square2 = square2 + (theta_2[z][y])*(theta_2[z][y])
    #        z = z+1
    
    #    y = y+1
    #    z = 0       
        
    #y = 0
    #z = 0        
        
    #while y < 5:
    #    while z < 5 :
    #        square3 = square3 + (theta_3[z][y])*(theta_3[z][y])
    #        z = z+1
    
    #    y = y+1
    #    z = 0       
        
    #y = 0
    #z = 0   

    #while y < 5:
    #    while z < 5 :
    #        square4 = square4 + (theta_4[z][y])*(theta_4[z][y])
    #        z = z+1
    
    #    y = y+1
    #    z = 0       
        
    #y = 0
    #z = 0   

    while y < 10:
        while z < 5 :
            square5 = square5 + (theta_5[z][y])*(theta_5[z][y])
            z = z+1
    
        y = y+1
        z = 0       
        
    y = 0
    z = 0       

    cost2 = square1 + square2 + square3 + square4 + square5
    cost = cost1/(2840*2) + cost2*l/(2840*2)
    print(cost)
    
    #print("cost found out")
    
    r = 0
    final = []
    while r<2840:
        final = final + [find(newsgroups_train.target[r])]
        r = r+1
    
    ans = np.array(final)
    #print(ans.shape)
    
    #print("hurdle")

    d5 = np.random.rand(5,2840)
    p = 0
    q = 0
    while p < 2840:    
        while q < 5 :
            middle5 =( g5 - ans.T )
            d5[q][p] = middle5[q][p]*g5d[q][p]
            q=q+1
    
        p = p+1
        q = 0
    
    p=0
    q=0
    #print(d5.T[3])
    #print(middle5.T[3])
    #print(g5d.T[3])
    
    D5 = np.dot(d5,g1.T)
    #print(d5.shape)
    #print(g4d.shape)
    #print("task1")

    #d4 = np.random.rand(5,2840)
    #p = 0
    #q = 0
    #while p < 2840:
    #    while q < 5 :
    #        middle4 = np.dot(theta_5.T,d5)
            
    #        d4[q][p] = (middle4[q][p])*g4d[q][p]
           
    #        q=q+1
    
    #    p = p+1
    #    q = 0
    
    #p=0
    #q=0
    
    #print("take1")
    
    #D4 = np.dot(d4,g3.T)
    
    #print("task2")


    #d3 = np.random.rand(5,2840)
    #p = 0
    #q = 0
    #while p < 2840:    
    #    while q < 5 :  
    #        middle3 = np.dot(theta_4.T,d4)
    #        d3[q][p] = (middle3[q][p])*g3d[q][p]
    #        q=q+1
    
    #    p = p+1
    #    q = 0
    
    #p=0
    #q=0
    
    #D3 = np.dot(d3,g2.T)
    
    #print("task3")

    #d2 = np.random.rand(5,2840)
    #p = 0
    #q = 0
    #while p < 2840:    
    #    while q < 5 :        
    #        middle2 =np.dot(theta_3.T,d3)
    #        d2[q][p] = (middle2[q][p])*g2d[q][p]
    #        q=q+1
    
    #    p = p+1
    #    q = 0
    
    #p=0
    #q=0
    
    #D2 = np.dot(d2,g1.T)
    
    #print("task4")


    d1 = np.random.rand(10,2840)
    p = 0
    q = 0
    while p < 2840:
        while q < 10 :
            middle1 =np.dot(theta_5.T,d5)
            d1[q][p] = (middle1[q][p])*g1d[q][p]
            q = q+1
       
        p = p+1
        q = 0
    
    p=0
    q=0
    
    D1 = np.dot(d1,v)

    #print("derivative found")
    
    l1 = 0.75
    l2 = 0.75
    l3 = 0.75
    l4 = 0.75
    l5 = 0.75
    l01 = 0.5
    l02 = 0.5
    l03 = 0.5
    l04 = 0.5
    l05 = 0.5
    #print(D5[1][1]/2840)
    #print(D4[1][1]/2840)
    #print(D3[1][1]/2840)
    #print(D2[1][1]/2840)
    #print(D1[1][1]/2840)
    theta_1 = (1-l/28400)*theta_1 - (l1/28400)*D1
    
    #theta_2 = (1-l/28400)*theta_2 - (l2/28400)*D2 
    #theta_3 = (1-l/28400)*theta_3 - (l3/28400)*D3 
    #theta_4 = (1-l/28400)*theta_4 - (l4/28400)*D4 
    theta_5 = (1-l/28400)*theta_5 - (l5/28400)*D5 

    theta_01 = theta_01 - (l01/28400)*d1
    #theta_02 = theta_02 - (l02/28400)*d2
    #theta_03 = theta_03 - (l03/28400)*d3
    #theta_04 = theta_04 - (l04/28400)*d4
    theta_05 = theta_05 - (l05/28400)*d5
    
    print(tracker )

    tracker = tracker + 1      
print("test")
    
