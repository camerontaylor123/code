#!/usr/bin/env python
# coding: utf-8

# In[2]:


import msprime
import matplotlib.pyplot as plt
from IPython.display import display, SVG
import numpy as np
from PIL import Image
from scipy import stats
import tskit
import random


# In[3]:


# generates a demography and tree sequences, outputs genotype matrices
def tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate,
                  mut_rate, B_sample, C_sample, mig_rate):
        # create a big population model and then split it 
    demography = msprime.Demography()
    demography.add_population(name = "A", initial_size = initial_A)
    demography.add_population(name = "B", initial_size = initial_B)
    demography.add_population(name = "C", initial_size = initial_C)
    demography.add_population_split(time = time_bottleneck, derived=["B", "C"], 
                                    ancestral = "A")
    demography.set_migration_rate('C', 'B', mig_rate)
    
    ts1 = msprime.sim_ancestry(samples = {"B":B_sample, "C":C_sample}, 
                               recombination_rate= recom_rate, sequence_length = seq_len,
                               demography=demography)
    mts1=msprime.mutate(ts1,rate=mut_rate)
    X = mts1.genotype_matrix().transpose()

    XAJ = X[B_sample*2:,:]
    Xgen = X[:B_sample*2,:]
    
    return X, Xgen, XAJ, mts1


#find frequency of each SNP
def find_freq(X): 
    N , L = np.shape(X)
    f = [0 for i in range(L)]
    for i in range(L):
        f[i] = np.sum(X[:,i])/N
    return f
    
# creates vector of Bs based on chosen popuolation(Xgen here), selection coefficient and inclusion probability     
def generate_B(Xgen, s, inclusion_prob, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    N, L = np.shape(Xgen)
    f = find_freq(Xgen)
    B = np.zeros(L)
    
    for i in range (L):
        if f[i]==0 or f[i]==1:
            sd = 0
        else:
            sd= np.power((f[i]*(1-f[i])), s)
        B[i] = np.random.normal(loc=0.0, scale =sd**2)
        if random.random()  >= inclusion_prob :
            B[i] = 0
    #if seed is not None:
        #np.random.seed()
        #random.seed()
    return B



# input the genotype matrices, B and heritability and it generates phenotypes for the two populations
def phenotype(Xgen, XAJ, B, h2):
    #find variation in genotype variance of large population
    N, L = np.shape(Xgen)
    Y0gen = np.zeros(N)
    for i in range(N-1):
        Y0gen[i] = np.dot(B,Xgen[i,:])
    VarY0gen = np.var(Y0gen)
        
    # use chosen heritability to find variance of phenotype
    if VarY0gen == 0:
        VarE  = 0.0001
    else:
        VarE = VarY0gen*(1-h2)/h2  
    
    E = np.zeros(N)
    for i in range (N):
        E[i] = np.random.normal(loc = 0, scale = (VarE)**(0.5))
    
    
    Ygen = np.zeros(N)
    for i in range (N):
        Ygen[i] = Y0gen[i] + E[i]
 

    Y0AJ = np.zeros(N)
    YAJ = np.zeros(N)
    for i in range (N):
        Y0AJ[i] = np.dot(B,XAJ[i,:])
        YAJ[i] = Y0AJ[i] + E[i]
    
    return YAJ, Ygen, Y0AJ, Y0gen

# input genotype matrices, effect sizes and chosen heritability in the general population and output heritability in both populations 
def heritability(Xgen, XAJ, B, h2):
    N, L = np.shape(Xgen)
    Y0gen = np.zeros(N)
    for i in range(N-1):
        Y0gen[i] = np.dot(B,Xgen[i,:])
    VarY0gen = np.var(Y0gen)
        
    # use chosen heritability to find variance of phenotype
    if VarY0gen == 0:
        VarE  = 0.0001
    else:
        VarE = VarY0gen*(1-h2)/h2  
    
    # find variation in genotype for AJ
    Y0AJ = np.zeros(N)
    for i in range (N):
        Y0AJ[i] = np.dot(B,XAJ[i,:])
    VarY0AJ = np.var(Y0AJ)
    
    # use previously found Ve to find heritability for AJ
    h2AJ = VarY0AJ/(VarE+VarY0AJ)
    h2gen = h2
    return h2AJ, h2gen


# input demography things and output heritability 
# input demography things and output heritability 
def demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate, seed=None):
    X, Xgen, XAJ, mts1 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, 
                                 seq_len,recom_rate, mut_rate, B_sample, C_sample, mig_rate)
    B = generate_B(Xgen, s, inclusion_prob, seed=None)
    h2AJ, h2gen = heritability(Xgen, XAJ, abs(B), h2)
    return h2AJ, h2gen



def minor_allele_freq(f):
    fnew=[0 for i in range(len(f))]
    for i in range (len(f)):
        if f[i]<0.5:
            fnew[i] = f[i]
        else:
            fnew[i] = 1-f[i]
    return fnew


def inference_power (f, B):
    power = [0 for i in range(len(f))]
    for i in range(len(f)):
        power[i] = f[i]*(1-f[i])*(np.power(B[i], 2))
    total_power = sum(power)
    return total_power


def show_corr_mat(X):
    plt.matshow(abs(np.corrcoef(X.transpose())), cmap = 'Reds')
    #plt.colorbar()
    plt.show()


# In[4]:


initial_A = 10_000
initial_B = 1_000
initial_C = 10_000
seq_len = 10000
recom_rate = 0.0001
mut_rate = 0.000001
B_sample = 1000
C_sample = 1000
s=-0.5
h2 = 0.5 #chosen heritability in general population
inclusion_prob = 1
time_bottleneck = 200
mig_rate = 0


# In[17]:


# sequence length 10000 recom rate 0.0001

n =500
heritable = [0 for i in range (n)]
for i in range (n):
    heritable[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]
mean = np.mean(heritable)
variance = np.var(heritable)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution")
plt.legend()
plt.show()


# In[6]:


# sequence length 1000000 recom 0.00001

n =1000
heritable = [0 for i in range (n)]
for i in range (n):
    heritable[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]
mean = np.mean(heritable)
variance = np.var(heritable)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution")
plt.legend()
plt.show()


# In[10]:


heritable1 = []
for i in range (len(heritable)):
    if heritable[i] != 0:
        heritable1.append(heritable[i])


# In[12]:



mean = np.mean(heritable1)
variance = np.var(heritable1)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable1, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution")
plt.legend()
plt.show()


# In[ ]:


# sequence length 10000 recom 0.0001

n =1000
heritable = [0 for i in range (n)]
for i in range (n):
    heritable[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]
mean = np.mean(heritable)
variance = np.var(heritable)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution")
plt.legend()
plt.show()


# In[57]:


n =1000
heritable = [0 for i in range (n)]
for i in range (n):
    heritable[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]


# In[87]:


# base case recom 0.00001 genome len 10000

mean = np.mean(heritable)
variance = np.var(heritable)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution")
plt.legend()
plt.show()


# In[89]:


mig_rate1 = 0.01
n =500
heritable1 = [0 for i in range (n)]
for i in range (n):
    heritable1[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate1)[0]


# In[90]:


mean = np.mean(heritable1)
variance = np.var(heritable1)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable1, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution migration = 0.01")
plt.legend()
plt.show()


# In[91]:


mig_rate2 = 0.1
n =500
heritable2 = [0 for i in range (n)]
for i in range (n):
    heritable2[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate2)[0]


# In[92]:


mean = np.mean(heritable2)
variance = np.var(heritable2)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable2, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution migration = 0.1")
plt.legend()
plt.show()


# In[93]:


mig_rate3 = 0.2
n =500
heritable3 = [0 for i in range (n)]
for i in range (n):
    heritable3[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate3)[0]


# In[94]:


mean = np.mean(heritable3)
variance = np.var(heritable3)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable3, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution migration = 0.2")
plt.legend()
plt.show()


# In[95]:


mig_rate4 = 0.25
n =1000
heritable4 = [0 for i in range (n)]
for i in range (n):
    heritable4[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate4)[0]


# In[105]:


mean = np.mean(heritable4)
variance = np.var(heritable4)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable, bins=25, alpha=0.7, label = "no migration")
plt.hist(heritable4, bins=25, alpha = 0.7, label = '0.25 migration')
plt.xlim(xmin=0, xmax = 1)
#plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution with different levels of migration")
plt.legend()
plt.show()


# In[106]:


#20 generations ago
n =1000
heritable5 = [0 for i in range (n)]
for i in range (n):
    heritable5[i] = demography_heritability (initial_A, initial_B, initial_C, 20, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]


# In[107]:


plt.figure(figsize=(10,6))
plt.hist(heritable5, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution 20 generations ago)
plt.legend()
plt.show()


# In[15]:


n = 30
means = [0 for i in range (20)]
mig_rate = np.linspace(0,1, num= 20)
for j in range (20):
    heritable6 = [0 for i in range (n)]
    for i in range (n):
        heritable6[i] = demography_heritability (initial_A, initial_B, initial_C, 20, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate[j])[0]
        
    means[j] = np.mean(heritable6)


# In[17]:


plt.scatter(mig_rate, means)


# In[25]:


n = 200


seq_len = 100
recom_rate = 0.001


her1 = [0 for i in range (n)]
for i in range (n):
    her1[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                                    h2, inclusion_prob, mig_rate)[0]
mean1 = np.mean(her1)
variance1 = np.var(her1)


seq_len = 1000
recom_rate = 0.0001
her2 = [0 for i in range (n)]
for i in range (n):
    her2[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                                       h2, inclusion_prob, mig_rate)[0]

    
mean2 = np.mean(her2)
variance2 = np.var(her2)

seq_len = 10000
recom_rate = 0.00001

seq_len = 100000
recom_rate = 0.000001

her3 = [0 for i in range (n)]
for i in range (n):
    her3[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s,    
                                    h2, inclusion_prob, mig_rate)[0]
mean3 = np.mean(her3)
variance3 = np.var(her3)


seq_len = 100000
recom_rate = 0.000001


her4 = [0 for i in range (n)]
for i in range (n):
    her4[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s,    
                                    h2, inclusion_prob, mig_rate)[0]

mean4 = np.mean(her4)
variance4 = np.var(her4)


# In[26]:


fig, axs = plt.subplots(2, 2, figsize = (17,8))

axs[0,0].set_title('seq_len = 100 recom_rate = 0.001')
axs[0,0].hist(her1, bins =25)

axs[0,1].set_title('seq_len = 1000 recom_rate = 0.0001')
axs[0,1].hist(her2, bins =25)

axs[1,0].set_title('seq_len = 10000 recom_rate = 0.00001')
axs[1,0].hist(her3, bins =25)

axs[1,1].set_title('seq_len = 100000 recom_rate = 0.000001')
axs[1,1].hist(her4, bins =25)


# In[27]:


print(mean1, mean2, mean3, mean4)


# In[28]:


print(variance1, variance2, variance3, variance4)


# In[15]:


n = 500


seq_len = 100
recom_rate = 0.001


her1 = [0 for i in range (n)]
for i in range (n):
    her1[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                                    h2, inclusion_prob, mig_rate)[0]
mean1 = np.mean(her1)
variance1 = np.var(her1)


seq_len = 10000
recom_rate = 0.00001
her2 = [0 for i in range (n)]
for i in range (n):
    her2[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                                       h2, inclusion_prob, mig_rate)[0]

mean2 = np.mean(her2)
variance2 = np.var(her2)
    


# In[135]:


print(mean1, mean2, variance1, variance2)


# In[16]:


fig, axs = plt.subplots(1, 2, figsize = (17,5))

axs[0].set_title('Sequence Length 100, Recom Rate 0.001')
axs[0].hist(her1, bins =25)
axs[0].set_xlabel('Heriability')
axs[0].set_ylabel('Frequency')

axs[1].set_title('Sequence Length 10000, Recom Rate 0.00001')
axs[1].hist(her2, bins =25)
axs[1].set_xlabel('Heriability')
axs[1].set_ylabel('Frequency')
axs[1].set_xlim(0,1)


plt.show()


# In[104]:



seq_len = 100
recom_rate = 0.001
X1, Xgen1, XAJ1, mts11 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen1 = minor_allele_freq(find_freq(Xgen1))
fAJ1 = minor_allele_freq(find_freq(XAJ1))
B1 = generate_B(Xgen1, s, inclusion_prob)


X2, Xgen2, XAJ2, mts12 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen2 = minor_allele_freq(find_freq(Xgen2))
fAJ2 = minor_allele_freq(find_freq(XAJ2))
B2 = generate_B(Xgen2, s, inclusion_prob)

X3, Xgen3, XAJ3, mts13 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen3 = minor_allele_freq(find_freq(Xgen3))
fAJ3 = minor_allele_freq(find_freq(XAJ3))
B3 = generate_B(Xgen3, s, inclusion_prob)


# In[124]:


X2, Xgen2, XAJ2, mts12 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen2 = minor_allele_freq(find_freq(Xgen2))
fAJ2 = minor_allele_freq(find_freq(XAJ2))
B2 = generate_B(Xgen2, s, inclusion_prob)


# In[126]:


fig, axs = plt.subplots(1, 3, figsize = (17,5))
plt.suptitle("SNP frequency and effect size in sequence Length 100, recom rate 0.001")

axs[0].scatter(fAJ1, abs(B1), label ='Bottleneck Population')
axs[0].scatter(fgen1, abs(B1), label = 'General Population')
axs[0].legend()
axs[0].set_ylabel('Effect Size, Beta')
axs[0].set_xlabel('Minor Allele Frequency')


axs[1].scatter(fAJ2, abs(B2), label ='Bottleneck Population')
axs[1].scatter(fgen2, abs(B2), label = 'General Population')
axs[1].legend()
axs[1].set_ylabel('Effect Size, Beta')
axs[1].set_xlabel('Minor Allele Frequency')

axs[2].scatter(fAJ3, abs(B3), label ='Bottleneck Population')
axs[2].scatter(fgen3, abs(B3), label = 'General Population')
axs[2].legend()
axs[2].set_ylabel('Effect Size, Beta')
axs[2].set_xlabel('Minor Allele Frequency')
plt.show()


# In[133]:


seq_len = 10000
recom_rate = 0.00001
X2, Xgen2, XAJ2, mts12 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen2 = minor_allele_freq(find_freq(Xgen2))
fAJ2 = minor_allele_freq(find_freq(XAJ2))
B2 = generate_B(Xgen2, s, inclusion_prob)


# In[134]:


plt.figure(figsize = (8,6))
plt.scatter(fAJ2, abs(B2), label ='Bottleneck Population')
plt.scatter(fgen2, abs(B2), label = 'General Population')
plt.legend()
plt.ylabel('Effect Size, Beta')
plt.xlabel('Minor Allele Frequency')
plt.title("SNP frequency and effect size in sequence Length 1000, recom rate 0.0001")
plt.show()


# In[128]:



seq_len = 10000
recom_rate = 0.00001
X1, Xgen1, XAJ1, mts11 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen1 = minor_allele_freq(find_freq(Xgen1))
fAJ1 = minor_allele_freq(find_freq(XAJ1))
B1 = generate_B(Xgen1, s, inclusion_prob)


X2, Xgen2, XAJ2, mts12 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen2 = minor_allele_freq(find_freq(Xgen2))
fAJ2 = minor_allele_freq(find_freq(XAJ2))
B2 = generate_B(Xgen2, s, inclusion_prob)

X3, Xgen3, XAJ3, mts13 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen3 = minor_allele_freq(find_freq(Xgen3))
fAJ3 = minor_allele_freq(find_freq(XAJ3))
B3 = generate_B(Xgen3, s, inclusion_prob)


# In[132]:


fig, axs = plt.subplots(1, 3, figsize = (17,5))
plt.suptitle("SNP frequency and effect size in sequence Length 10000, recom rate 0.00001")

axs[0].scatter(fAJ1, abs(B1), label ='Bottleneck Population')
axs[0].scatter(fgen1, abs(B1), label = 'General Population')
axs[0].legend()
axs[0].set_ylabel('Effect Size, Beta')
axs[0].set_xlabel('Minor Allele Frequency')


axs[1].scatter(fAJ2, abs(B2), label ='Bottleneck Population')
axs[1].scatter(fgen2, abs(B2), label = 'General Population')
axs[1].legend()
axs[1].set_ylabel('Effect Size, Beta')
axs[1].set_xlabel('Minor Allele Frequency')

axs[2].scatter(fAJ3, abs(B3), label ='Bottleneck Population')
axs[2].scatter(fgen3, abs(B3), label = 'General Population')
axs[2].legend()
axs[2].set_ylabel('Effect Size, Beta')
axs[2].set_xlabel('Minor Allele Frequency')
plt.show()


# In[7]:


n = 500


seq_len = 10000
recom_rate = 0.00001
mut_rate = 0.000001

her1 = [0 for i in range (n)]
for i in range (n):
    her1[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                                    h2, inclusion_prob, mig_rate)[0]
mean1 = np.mean(her1)
variance1 = np.var(her1)


seq_len = 10000
recom_rate = 0.000000001
mut_rate = 0.000001
her2 = [0 for i in range (n)]
for i in range (n):
    her2[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                                       h2, inclusion_prob, mig_rate)[0]

mean2 = np.mean(her2)
variance2 = np.var(her2)
    


# In[11]:


fig, axs = plt.subplots(1, 2, figsize = (17,5))

axs[0].set_title('Seq Len=10000, Recom Rate=0.00001, Mut Rate=0.000001')
axs[0].hist(her1, bins =25)
axs[0].set_xlabel('Heriability')
axs[0].set_ylabel('Frequency')

axs[1].set_title('Seq Len=10000, Recom Rate=0.000000001, Mut Rate= 0.000001')
axs[1].hist(her2, bins =25)
axs[1].set_xlabel('Heriability')
axs[1].set_ylabel('Frequency')
axs[1].set_xlim(0,1)


plt.show()


# In[9]:


print(mean1, mean2)


# In[10]:


print(variance1, variance2)


# In[26]:


seq_len = 10000
recom_rate = 0.00001
mut_rate = 0.000001
X1, Xgen1, XAJ1, mts11 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen1 = minor_allele_freq(find_freq(Xgen1))
fAJ1 = minor_allele_freq(find_freq(XAJ1))
B1 = generate_B(Xgen1, s, inclusion_prob)


X2, Xgen2, XAJ2, mts12 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen2 = minor_allele_freq(find_freq(Xgen2))
fAJ2 = minor_allele_freq(find_freq(XAJ2))
B2 = generate_B(Xgen2, s, inclusion_prob)

X3, Xgen3, XAJ3, mts13 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen3 = minor_allele_freq(find_freq(Xgen3))
fAJ3 = minor_allele_freq(find_freq(XAJ3))
B3 = generate_B(Xgen3, s, inclusion_prob)


# In[29]:


X2, Xgen2, XAJ2, mts12 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen2 = minor_allele_freq(find_freq(Xgen2))
fAJ2 = minor_allele_freq(find_freq(XAJ2))
B2 = generate_B(Xgen2, s, inclusion_prob)


# In[30]:


fig, axs = plt.subplots(1, 3, figsize = (17,5))
plt.suptitle("SNP frequency and effect size in sequence Length 10000, recom rate 0.00001, mut rate 0.000001")

axs[0].scatter(fAJ1, abs(B1), label ='Bottleneck Population')
axs[0].scatter(fgen1, abs(B1), label = 'General Population')
axs[0].legend()
axs[0].set_ylabel('Effect Size, Beta')
axs[0].set_xlabel('Minor Allele Frequency')


axs[1].scatter(fAJ2, abs(B2), label ='Bottleneck Population')
axs[1].scatter(fgen2, abs(B2), label = 'General Population')
axs[1].legend()
axs[1].set_ylabel('Effect Size, Beta')
axs[1].set_xlabel('Minor Allele Frequency')

axs[2].scatter(fAJ3, abs(B3), label ='Bottleneck Population')
axs[2].scatter(fgen3, abs(B3), label = 'General Population')
axs[2].legend()
axs[2].set_ylabel('Effect Size, Beta')
axs[2].set_xlabel('Minor Allele Frequency')
plt.show()


# In[86]:


seq_len = 10000
recom_rate = 0.000000001
mut_rate = 0.000001
X1, Xgen1, XAJ1, mts11 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen1 = minor_allele_freq(find_freq(Xgen1))
fAJ1 = minor_allele_freq(find_freq(XAJ1))
B1 = generate_B(Xgen1, s, inclusion_prob)


X2, Xgen2, XAJ2, mts12 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen2 = minor_allele_freq(find_freq(Xgen2))
fAJ2 = minor_allele_freq(find_freq(XAJ2))
B2 = generate_B(Xgen2, s, inclusion_prob)

X3, Xgen3, XAJ3, mts13 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen3 = minor_allele_freq(find_freq(Xgen3))
fAJ3 = minor_allele_freq(find_freq(XAJ3))
B3 = generate_B(Xgen3, s, inclusion_prob)


# In[93]:



X2, Xgen2, XAJ2, mts12 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen2 = minor_allele_freq(find_freq(Xgen2))
fAJ2 = minor_allele_freq(find_freq(XAJ2))
B2 = generate_B(Xgen2, s, inclusion_prob)


# In[94]:


fig, axs = plt.subplots(1, 3, figsize = (17,5))
plt.suptitle("SNP frequency and effect size in sequence Length 10000, recom rate 0.000000001, mut rate = 0.000001")

axs[0].scatter(fAJ1, abs(B1), label ='Bottleneck Population')
axs[0].scatter(fgen1, abs(B1), label = 'General Population')
axs[0].legend()
axs[0].set_ylabel('Effect Size, Beta')
axs[0].set_xlabel('Minor Allele Frequency')


axs[1].scatter(fAJ2, abs(B2), label ='Bottleneck Population')
axs[1].scatter(fgen2, abs(B2), label = 'General Population')
axs[1].legend()
axs[1].set_ylabel('Effect Size, Beta')
axs[1].set_xlabel('Minor Allele Frequency')

axs[2].scatter(fAJ3, abs(B3), label ='Bottleneck Population')
axs[2].scatter(fgen3, abs(B3), label = 'General Population')
axs[2].legend()
axs[2].set_ylabel('Effect Size, Beta')
axs[2].set_xlabel('Minor Allele Frequency')
plt.show()


# In[81]:


seq_len = 1000
recom_rate = 0.001

X1, Xgen1, XAJ1, mts11 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen1 = minor_allele_freq(find_freq(Xgen1))
fAJ1 = minor_allele_freq(find_freq(XAJ1))
B1 = generate_B(Xgen1, s, inclusion_prob)


X2, Xgen2, XAJ2, mts12 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen2 = minor_allele_freq(find_freq(Xgen2))
fAJ2 = minor_allele_freq(find_freq(XAJ2))
B2 = generate_B(Xgen2, s, inclusion_prob)

X3, Xgen3, XAJ3, mts13 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0)
fgen3 = minor_allele_freq(find_freq(Xgen3))
fAJ3 = minor_allele_freq(find_freq(XAJ3))
B3 = generate_B(Xgen3, s, inclusion_prob)


# In[146]:


fig, axs = plt.subplots(1, 3, figsize = (17,5))
plt.suptitle("SNP frequency and effect size in sequence Length 1000, recom rate 0.001")

axs[0].scatter(fAJ1, abs(B1), label ='Bottleneck Population')
axs[0].scatter(fgen1, abs(B1), label = 'General Population')
axs[0].legend()
axs[0].set_ylabel('Effect Size, Beta')
axs[0].set_xlabel('Minor Allele Frequency')


axs[1].scatter(fAJ2, abs(B2), label ='Bottleneck Population')
axs[1].scatter(fgen2, abs(B2), label = 'General Population')
axs[1].legend()
axs[1].set_ylabel('Effect Size, Beta')
axs[1].set_xlabel('Minor Allele Frequency')

axs[2].scatter(fAJ3, abs(B3), label ='Bottleneck Population')
axs[2].scatter(fgen3, abs(B3), label = 'General Population')
axs[2].legend()
axs[2].set_ylabel('Effect Size, Beta')
axs[2].set_xlabel('Minor Allele Frequency')
plt.show()


# In[5]:


seq_len = 10000
recom_rate = 0.00001
mut_rate = 0.000001


n =100
heritable = [0 for i in range (n)]
for i in range (n):
    heritable[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]
mean = np.mean(heritable)
variance = np.var(heritable)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution")
plt.legend()
plt.show()


# In[10]:


seq_len = 1000
recom_rate = 0.0001
mut_rate = 0.00001


n =500
heritable2 = [0 for i in range (n)]
for i in range (n):
    heritable2[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]
mean = np.mean(heritable2)
variance = np.var(heritable2)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable2, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution")
plt.legend()
plt.show()


# In[7]:


seq_len = 100
recom_rate = 0.001
mut_rate = 0.0001


n =100
heritable3 = [0 for i in range (n)]
for i in range (n):
    heritable3[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]
mean = np.mean(heritable3)
variance = np.var(heritable3)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable3, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution")
plt.legend()
plt.show()


# In[9]:


seq_len = 10
recom_rate = 0.01
mut_rate = 0.001


n =100
heritable4 = [0 for i in range (n)]
for i in range (n):
    heritable4[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]
mean = np.mean(heritable4)
variance = np.var(heritable4)
print("mean = ", mean, "variance =", variance,)

plt.figure(figsize=(10,6))
plt.hist(heritable4, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability")
plt.ylabel("Frequency")
plt.title("Heritability distribution")
plt.legend()
plt.show()


# In[11]:


n =500

seq_len = 10000
recom_rate = 0.00001
mut_rate = 0.000001
heritable1 = [0 for i in range (n)]
for i in range (n):
    heritable1[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]



    
seq_len = 10
recom_rate = 0.01
mut_rate = 0.001    
heritable4 = [0 for i in range (n)]
for i in range (n):
    heritable4[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]


# In[16]:


fig, axs = plt.subplots(1, 2, figsize = (17,5))

axs[0].set_title('Sequence Length 10000, Recom Rate 0.00001, Mut Rate 0.000001')
axs[0].hist(heritable1, bins =25)
axs[0].set_xlabel('Heriability')
axs[0].set_ylabel('Frequency')
axs[0].set_xlim(0,1)
axs[0].axvline(np.mean(heritable1), color='r', linestyle='dashed', linewidth=2, label = "mean")

axs[1].set_title('Sequence Length 10, Recom Rate 0.01, Mut Rate 0.001')
axs[1].hist(heritable4, bins =25)
axs[1].set_xlabel('Heriability')
axs[1].set_ylabel('Frequency')
axs[1].set_xlim(0,1)
axs[1].axvline(np.mean(heritable4), color='r', linestyle='dashed', linewidth=2, label = 'mean', )

plt.show()


# In[17]:


np.mean(heritable1)


# In[18]:


np.mean(heritable4)


# In[19]:


np.var(heritable1)


# In[20]:


np.var(heritable4)


# In[21]:


seq_len = 100
recom_rate = 0.001
mut_rate = 0.0001    
heritable3 = [0 for i in range (n)]
for i in range (n):
    heritable3[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]


# In[22]:


seq_len = 1000
recom_rate = 0.0001
mut_rate = 0.00001    
heritable2 = [0 for i in range (n)]
for i in range (n):
    heritable2[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate)[0]


# In[25]:


fig, axs = plt.subplots(2, 2, figsize = (17,10))

axs[0,0].set_title('Seq Len = 10000, Recom Rate = 0.00001, Mut Rate = 0.000001')
axs[0,0].hist(heritable1, bins =25)
axs[0,0].set_xlabel('Heriability')
axs[0,0].set_ylabel('Frequency')
axs[0,0].set_xlim(0,1)
axs[0,0].axvline(np.mean(heritable1), color='r', linestyle='dashed', linewidth=2, label = "mean")

axs[0,1].set_title('Seq Len = 1000, Recom Rate = 0.0001, Mut Rate = 0.00001')
axs[0,1].hist(heritable2, bins =25)
axs[0,1].set_xlabel('Heriability')
axs[0,1].set_ylabel('Frequency')
axs[0,1].set_xlim(0,1)
axs[0,1].axvline(np.mean(heritable2), color='r', linestyle='dashed', linewidth=2, label = "mean")

axs[1,0].set_title('Seq Len = 100, Recom Rate = 0.001, Mut Rate = 0.0001')
axs[1,0].hist(heritable3, bins =25)
axs[1,0].set_xlabel('Heriability')
axs[1,0].set_ylabel('Frequency')
axs[1,0].set_xlim(0,1)
axs[1,0].axvline(np.mean(heritable3), color='r', linestyle='dashed', linewidth=2, label = "mean")


axs[1,1].set_title('Seq Len = 10, Recom Rate = 0.01, Mut Rate = 0.001')
axs[1,1].hist(heritable4, bins =25)
axs[1,1].set_xlabel('Heriability')
axs[1,1].set_ylabel('Frequency')
axs[1,1].set_xlim(0,1)
axs[1,1].axvline(np.mean(heritable4), color='r', linestyle='dashed', linewidth=2, label = "mean")


# In[ ]:




