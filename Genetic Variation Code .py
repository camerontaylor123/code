#!/usr/bin/env python
# coding: utf-8

# In[120]:


import msprime
import matplotlib.pyplot as plt
from IPython.display import display, SVG
import numpy as np
from PIL import Image
from scipy import stats
import tskit
import random
import pandas as pd


# In[121]:


def store_data(name, data):
    path = f"Data//{name}.csv"
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, header=False)
    return None


# In[122]:


def get_data(name):
    path = f"Data//{name}.csv"
    df = pd.read_csv(path, header=None)
    return np.array(df)


# In[123]:


# generates a demography and tree sequences, outputs genotype matrices and tree sequence data 
# ancestral population A, Bottleneck population B, General population C
def tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate,
                  mut_rate, B_sample, C_sample, mig_rate, growth_rate, seed_ts=None):
        # create a big population model and then split it 
    demography = msprime.Demography()
    demography.add_population(name = "A", initial_size = initial_A)
    demography.add_population(name = "B", initial_size = initial_B, growth_rate=growth_rate)
    demography.add_population(name = "C", initial_size = initial_C)
    demography.add_population_split(time = time_bottleneck, derived=["B", "C"], 
                                    ancestral = "A")
    demography.set_migration_rate('C', 'B', mig_rate)
    
    ts1 = msprime.sim_ancestry(samples = {"B":B_sample, "C":C_sample}, 
                               recombination_rate= recom_rate, sequence_length = seq_len,
                               demography=demography, random_seed= seed_ts)
    mts1=msprime.mutate(ts1,rate=mut_rate, random_seed=seed_ts)
    X = mts1.genotype_matrix().transpose()

    XB = X[B_sample*2:,:]
    XC = X[:B_sample*2,:]
    
    return X, XB, XC, mts1


#find frequency of each SNP in genotype matrix 
def find_freq(X): 
    N , L = np.shape(X)
    f = [0 for i in range(L)]
    for i in range(L):
        f[i] = np.sum(X[:,i])/N
    return f
    
# creates vector of Bs based on chosen popuolation(XC here), selection coefficient and inclusion probability     
def generate_B(XC, s, inclusion_prob, seed_B=None):
    if seed_B is not None:
        np.random.seed(seed_B)
        random.seed(seed_B)
    N, L = np.shape(XC)
    f = find_freq(XC)
    B = np.zeros(L)
    
    for i in range (L):
        if f[i]==0 or f[i]==1:
            sd = 0
        else:
            sd= np.power((f[i]*(1-f[i])), s)
        B[i] = np.random.normal(loc=0.0, scale =sd**2)
        if random.random()  >= inclusion_prob :
            B[i] = 0
    return B



# input the genotype matrices, B and heritability and it generates phenotypes for the two populations
def phenotype(XB, XC, B, h2):
    #find variance Vg of population C
    N, L = np.shape(XC)
    Y0C = np.zeros(N)
    for i in range(N-1):
        Y0C[i] = np.dot(B,XC[i,:])
    VarY0C = np.var(Y0C)
        
    # use chosen heritability to find environmental noise E
    if VarY0C == 0:
        VarE  = 0.0001
    else:
        VarE = VarY0gen*(1-h2)/h2  
    
    E = np.zeros(N)
    for i in range (N):
        E[i] = np.random.normal(loc = 0, scale = (VarE)**(0.5))
    
    # create phenotype vector for C
    YC = np.zeros(N)
    for i in range (N):
        YC[i] = Y0C[i] + E[i]
 
    # create phenotyoe vector for B
    Y0B = np.zeros(N)
    YB = np.zeros(N)
    for i in range (N):
        Y0B[i] = np.dot(B,XB[i,:])
        YB[i] = Y0B[i] + E[i]
    
    return YB, YC, Y0B, Y0C

# input genotype matrices, effect sizes and chosen heritability in the general population and output heritability in both populations 
def heritability(XB, XC, B, h2):
    N, L = np.shape(XC)
    Y0C = np.zeros(N)
    for i in range(N-1):
        Y0C[i] = np.dot(B,XC[i,:])
    VarY0C = np.var(Y0C)
        
    # use chosen heritability to find variance of phenotype
    if VarY0C == 0:
        VarE  = 0.0001
    else:
        VarE = VarY0C*(1-h2)/h2  
    
    # find variation in genotype for AJ
    Y0B = np.zeros(N)
    for i in range (N):
        Y0B[i] = np.dot(B,XB[i,:])
    VarY0B = np.var(Y0B)
    
    # use previously found Ve to find heritability for AJ
    h2B = VarY0B/(VarE+VarY0B)
    h2C = h2
    return h2B, h2C


# input demography things and output heritability 
def demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate,growth_rate, seed_ts=None, seed_B=None):
    X, XB, XC, mts1 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, 
                                 seq_len,recom_rate, mut_rate, B_sample, C_sample, mig_rate, growth_rate, seed_ts)
    B = generate_B(XC, s, inclusion_prob, seed_B=None)
    h2B, h2C = heritability(XB, XC, abs(B), h2)
    return h2B, h2C



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
    plt.show()


# # Base Case Parameters

# In[180]:


initial_A = 10_000
initial_B = 1_000
initial_C = 10_000
seq_len = 10000
recom_rate = 0.00001
mut_rate = 0.000001
B_sample = 1000
C_sample = 1000
s=-0.5
h2 = 0.5 #chosen heritability in general population
inclusion_prob = 1
time_bottleneck = 200
mig_rate = 0
growth_rate = 0


# # Plot 1 -PCA correct parameters

# In[6]:


X, Xgen, XAJ, mts1 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,mig_rate, growth_rate=0, seed_ts = 2)
U, D , V =np.linalg.svd(X, full_matrices = True)


# In[7]:


plt.figure(figsize=(12,8))
length = np.shape(U)[0]
plt.scatter(U[int(length/2):,0],U[int(length/2):,1], c = "blue", label="Bottleneck Population")
plt.scatter(U[:int(length/2),0],U[:int(length/2),1], c="red", label = 'General Population')
plt.legend(fontsize=14)


# # Plot 2 - PCA incorrect parameters
# 

# In[8]:


seq_len1 = 10000
recom_rate1 = 0.0000001

X, Xgen, XAJ, mts1 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len1,recom_rate1, mut_rate, B_sample, C_sample,mig_rate, growth_rate=0, seed_ts = 2)
U, D , V =np.linalg.svd(X, full_matrices = True)
plt.figure(figsize=(12,8))
length = np.shape(U)[0]
plt.scatter(U[int(length/2):,0],U[int(length/2):,1], c = "blue", label="Bottleneck Population")
plt.scatter(U[:int(length/2),0],U[:int(length/2),1], c="red", label = 'General Population')
plt.legend(fontsize=14)


# # Plot 3 LD 

# In[9]:


B_sample1 = 100
C_sample1 = 100

X, Xgen, XAJ, mts1 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample1, C_sample1,mig_rate, growth_rate=0, seed_ts = 2)

fig, axs = plt.subplots(1, 2, figsize = (15,8))

st = fig.suptitle("Linkage Disequilibrium General Population v Bottleneck Population", fontsize="x-large")

axs[0].set_title("Main Population")
axs[0].matshow(abs(np.corrcoef(Xgen.transpose())), cmap = 'Reds')


axs[1].set_title("Bottleneck Population")
axs[1].matshow(abs(np.corrcoef(XAJ.transpose())), cmap = 'Reds')

plt.show()


# # Plot 4 - Effect size plot

# In[154]:


X, Xgen, XAJ, mts1 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,mig_rate,growth_rate, seed_ts =2)
B = generate_B(Xgen, s, inclusion_prob,0)
fgen = minor_allele_freq(find_freq(Xgen))
fAJ = minor_allele_freq(find_freq(XAJ))


# In[155]:


plt.figure(figsize=(10,8))
plt.scatter(fAJ, abs(B), label = "Bottleneck Population")
plt.scatter(fgen, abs(B), label = "General Population")
plt.xlabel("Minor Allele Frequency", fontsize=14)
plt.ylabel("Effect Size - Beta", fontsize = 14)
plt.legend(fontsize=14)
plt.show()


# # Plot 5 - Base Heritability

# In[162]:


n =1000
heritable = [0 for i in range (n)]
for i in range (n):
    heritable[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate, growth_rate)[0]
        
mean = np.mean(heritable)
variance = np.var(heritable)
print("mean = ", mean, "variance =", variance,)


# In[165]:


store_data("Plot5_heritablevalues", heritable)


# In[164]:


heritable = get_data("Plot5_heritablevalues")
plt.figure(figsize=(10,6))
plt.hist(heritable, bins=25)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.axvline(0.5, color = 'orange', linewidth =5, label = "Heritability in the General Population")
plt.xlabel("Heritability",fontsize=14)
plt.ylabel("Frequency",fontsize=14)
plt.title("Heritability distribution")
plt.legend(fontsize=14)
plt.show()


# # Plot 6 - bottleneck size - heritability

# In[170]:


N=50
samples = 10
bottleneck_size = np.linspace(500,10000, num=N)
h2_values = np.zeros((N, samples))

for j in range (samples):
    for i in range (N):
        h2_values[i,j] = demography_heritability(initial_A, bottleneck_size[i], initial_C, time_bottleneck, seq_len, recom_rate, mut_rate, B_sample, C_sample, s, h2, inclusion_prob, mig_rate, growth_rate)[0]

sd = [0 for i in range(N)]
for i in range (N):
    sd[i] = np.std(h2_values[i,:])
sd = np.array(sd)
    
hvals = [0 for i in range (N)]
for i in range (N):
    hvals[i] = np.mean(h2_values[i,:])
hvals = np.array(hvals)


# In[16]:


store_data("Plot6_heritablematrix", h2_values)


# In[172]:


h2_values = get_data("Plot6_heritablematrix")
sd = [0 for i in range(N)]
for i in range (N):
    sd[i] = np.std(h2_values[i,:])
sd = np.array(sd)
    
hvals = [0 for i in range (N)]
for i in range (N):
    hvals[i] = np.mean(h2_values[i,:])
hvals = np.array(hvals)

plt.figure(figsize=(12,8))
plt.xticks(np.linspace(500,10000, 20))
plt.plot(bottleneck_size, hvals)
plt.axhline(0.5, color='r', linestyle='dashed', linewidth=2, label = 'Heritability in general population')
plt.fill_between(bottleneck_size, hvals-sd, hvals+sd,
    alpha=0.5)
plt.ylabel("Heritability",fontsize=14)
plt.xlabel("Size of the Initial Bottleneck Population",fontsize=14)
plt.legend(fontsize=12)
plt.show()


# # Plot 7 - bottleneck size heritbaility distribution

# In[18]:


size1 = 500
n =500
heritable1 = [0 for i in range (n)]
for i in range (n):
    heritable1[i] = demography_heritability (initial_A, size1, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, h2, inclusion_prob, mig_rate, growth_rate)[0]

size2 = 8000    
heritable2 = [0 for i in range (n)]
for i in range (n):
    heritable2[i] = demography_heritability (initial_A, size2, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate, growth_rate)[0]    
    


# In[19]:


store_data("Plot7_size1", heritable1)
store_data("Plot7_size2", heritable2)


# In[168]:


heritable1 = get_data("Plot7_size1")
heritable2 = get_data("Plot7_size2")
plt.figure(figsize=(10,6))
plt.hist(heritable1, bins=25, alpha=0.7, label = "Initial Bottleneck Population Size 500 ")
plt.hist(heritable2, bins=25, alpha = 0.7, label = 'Initial Bottleneck Population Size 8000')
plt.xlim(xmin=0, xmax = 1)
plt.axvline(0.5, color = 'black', linewidth =5, label = "Heritability in the General Population")
plt.xlabel("Heritability",fontsize=14)
plt.ylabel("Frequency",fontsize=14)
plt.title("Heritability distribution with Different Bottleneck Sizes")
plt.legend(fontsize=12)
plt.show()


# # Plot 8 - time of bottleneck - heritability 

# In[68]:


N=50
samples = 15
bottleneck_time = np.linspace(50,500, num=N)
h2_values1 = np.zeros((N, samples))

for j in range (samples):
    for i in range (N):
        h2_values1[i,j] = demography_heritability(initial_A, initial_B, initial_C, bottleneck_time[i], seq_len, recom_rate, mut_rate, B_sample, C_sample, s, h2, inclusion_prob, mig_rate, growth_rate)[0]
        
        
hvals1 = [0 for i in range (N)]
for i in range (N):
    hvals1[i] = np.mean(h2_values1[i,:])
    
hvals1 = np.array(hvals1)

sd = [0 for i in range(N)]
for i in range (N):
    sd[i] = np.std(h2_values1[i,:])
sd = np.array(sd)
    


# In[71]:


store_data("Plot8_heritablematrix", h2_values1)


# In[69]:


plt.figure(figsize=(12,8))
#plt.xticks(np.linspace(500,10000, 20))
#plt.axhline(0.5, color='r', linestyle='dashed', linewidth=2, label = '0.5')
plt.fill_between(bottleneck_time, hvals1-sd, hvals1+sd,
    alpha=0.5)
plt.ylabel("Heritability",fontsize=14)
plt.xlabel("Time of the population split (generations ago)",fontsize=14)
plt.plot(bottleneck_time, hvals1)


# # Plot 9 - time of bottleneck heritbaility distribution

# In[43]:


time1 = 50
n =500
heritable1 = [0 for i in range (n)]
for i in range (n):
    heritable1[i] = demography_heritability (initial_A, initial_B, initial_C, time1, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, h2, inclusion_prob, mig_rate, growth_rate)[0]

time2 = 500    
heritable2 = [0 for i in range (n)]
for i in range (n):
    heritable2[i] = demography_heritability (initial_A, initial_B, initial_C, time2, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate, growth_rate)[0]    


# In[44]:


store_data("Plot9_time1", heritable1)
store_data("Plot9_time2", heritable2)


# In[173]:


heritable1 = get_data("Plot9_time1")
heritable2 = get_data("Plot9_time2")

plt.figure(figsize=(10,6))
plt.hist(heritable2, bins=25, alpha = 0.7, label = '500 generations ago')
plt.hist(heritable1, bins=25, alpha=0.7, label = "50 generations ago")

plt.xlim(xmin=0, xmax = 1)
#plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability",fontsize=14)
plt.ylabel("Frequency",fontsize=14)
plt.title("Heritability distribution with differing population split times")
plt.axvline(0.5, color = 'black', linewidth =5, label = "Heritability in the General Population")
plt.legend(fontsize=14)
plt.show()


# # Plot 10 - bottleneck time variance plot

# In[74]:


N=20
samples = 100
bottleneck_time = np.linspace(50,500, num=N)
h2_values1 = np.zeros((N, samples))

for j in range (samples):
    for i in range (N):
        h2_values1[i,j] = demography_heritability(initial_A, initial_B, initial_C, bottleneck_time[i], seq_len, recom_rate, mut_rate, B_sample, C_sample, s, h2, inclusion_prob, mig_rate, growth_rate)[0]
        

hvals1 = [0 for i in range (N)]
for i in range (N):
    hvals1[i] = np.var(h2_values1[i,:])
    


# In[76]:


plt.figure(figsize=(12,8))
plt.ylabel("Variance in heritability",fontsize=14)
plt.xlabel("Time of the population split (generations ago)",fontsize=14)
plt.plot(bottleneck_time, hvals1)


# # Plot 11 - continent island plot

# In[29]:


generation = np.linspace(0,500,200)
pc = 0

p1= [1 for i in range(len(generation))]            
m1 = 0.01
for i in range(len(generation)-1):
    p1[i+1] = (1-m1)*p1[i] + m1*pc

p2= [1 for i in range(len(generation))]            
m2 = 0.05
for i in range(len(generation)-1):
    p2[i+1] = (1-m2)*p2[i] + m2*pc

p3= [1 for i in range(len(generation))]            
m3 = 0.1
for i in range(len(generation)-1):
    p3[i+1] = (1-m3)*p3[i] + m3*pc       


# In[30]:


plt.figure(figsize=(12,8))


plt.plot(generation, p1, label ='m=0.01')
plt.plot(generation, p2, label = 'm=0.05')
plt.plot(generation, p3, label = 'm=0.1')
plt.xlabel('Generations',fontsize=14)
plt.ylabel('Allele Frequency in Island Population, p',fontsize=14)
plt.legend(fontsize=14)
plt.show()


# # Plot 12 - effect size migration plots 

# In[31]:


seed_B = 6
seed_ts = 2
# no migration 

X1, Xgen1, XAJ1, mts11 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0, growth_rate, seed_ts)
fgen1 = minor_allele_freq(find_freq(Xgen1))
fAJ1 = minor_allele_freq(find_freq(XAJ1))
B1 = generate_B(Xgen1, s, inclusion_prob,seed_B)


# migration 0.01

X2, Xgen2, XAJ2, mts12 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0.01, growth_rate, seed_ts)
fgen2 = minor_allele_freq(find_freq(Xgen2))
fAJ2 = minor_allele_freq(find_freq(XAJ2))
B2 = generate_B(Xgen2, s, inclusion_prob,seed_B)


# migration 0.02

X3, Xgen3, XAJ3, mts13 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0.02,growth_rate, seed_ts)
fgen3 = minor_allele_freq(find_freq(Xgen3))
fAJ3 = minor_allele_freq(find_freq(XAJ3))
B3 = generate_B(Xgen3, s, inclusion_prob,seed_B)

# migration 0.05

X4, Xgen4, XAJ4, mts14 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0.05,growth_rate, seed_ts)
fgen4 = minor_allele_freq(find_freq(Xgen4))
fAJ4 = minor_allele_freq(find_freq(XAJ4))
B4 = generate_B(Xgen4, s, inclusion_prob,seed_B)


# migration 0.1

X5, Xgen5, XAJ5, mts15 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0.1, growth_rate, seed_ts)
fgen5 = minor_allele_freq(find_freq(Xgen5))
fAJ5 = minor_allele_freq(find_freq(XAJ5))
B5 = generate_B(Xgen5, s, inclusion_prob,seed_B)

# migration 0.2

X6, Xgen6, XAJ6, mts16 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,0.2, growth_rate, seed_ts)
fgen6 = minor_allele_freq(find_freq(Xgen6))
fAJ6 = minor_allele_freq(find_freq(XAJ6))
B6 = generate_B(Xgen6, s, inclusion_prob,seed_B)


# In[32]:


fig, axs = plt.subplots(2, 3, figsize = (17,8))

#st = fig.suptitle("Distribution of the Trait Over Different Proportions of Null SNPs", fontsize="x-large")

axs[0,0].set_title("Migration Rate 0")
axs[0,0].scatter(fAJ1, abs(B1), label = 'Bottleneck Population')
axs[0,0].scatter(fgen1, abs(B1), label='General Population')
#axs[0,0].set_xlabel("Allele Frequency in the Population")
axs[0,0].set_ylabel("Effect Size - Beta")
axs[0,0].legend()

axs[0,1].set_title("Migration Rate 0.01")
axs[0,1].scatter(fAJ2, abs(B2), label = 'Bottleneck Population')
axs[0,1].scatter(fgen2, abs(B2), label='General Population')
#axs[0,1].set_xlabel("Allele Frequency in the Population")
axs[0,1].set_ylabel("Effect Size - Beta")
axs[0,1].legend()

axs[0,2].set_title("Migration Rate 0.02")
axs[0,2].scatter(fAJ3, abs(B3), label = 'Bottleneck Population')
axs[0,2].scatter(fgen3, abs(B3), label='General Population')
#axs[0,2].set_xlabel("Allele Frequency in the Population")
axs[0,2].set_ylabel("Effect Size - Beta")
axs[0,2].legend()


axs[1,0].set_title("Migration Rate 0.05")
axs[1,0].scatter(fAJ4, abs(B4), label = 'Bottleneck Population')
axs[1,0].scatter(fgen4, abs(B4), label='General Population')
axs[1,0].set_xlabel("Allele Frequency in the Population")
axs[1,0].set_ylabel("Effect Size - Beta")
axs[1,0].legend()

axs[1,1].set_title("Migration Rate 0.1")
axs[1,1].scatter(fAJ5, abs(B5), label = 'Bottleneck Population')
axs[1,1].scatter(fgen5, abs(B5), label='General Population')
axs[1,1].set_xlabel("Allele Frequency in the Population")
axs[1,1].set_ylabel("Effect Size - Beta")
axs[1,1].legend()

axs[1,2].set_title("Migration Rate 0.2")
axs[1,2].scatter(fAJ6, abs(B6), label = 'Bottleneck Population')
axs[1,2].scatter(fgen6, abs(B6), label='General Population')
axs[1,2].set_xlabel("Allele Frequency in the Population")
axs[1,2].set_ylabel("Effect Size - Beta")
axs[1,2].legend()
plt.show()


# # Plot 13 - 3D heritability migration time plot 

# In[77]:


N=10
samples = 10
bottleneck_time = np.linspace(20, 400, num=N)
mig_rate1 = np.linspace(0,1,num=N)
h2_values = np.zeros((N,N,samples))

for j in range (samples):
    for i in range (N):
        for k in range(N):
            h2_values[i,k,j] = demography_heritability(initial_A, initial_B, initial_C, bottleneck_time[k], seq_len, recom_rate, mut_rate, B_sample, C_sample, s, h2, inclusion_prob, mig_rate1[i], growth_rate)[0]

hvals = np.zeros((N,N))
for i in range (N):
    for k in range(N):
        hvals[i,k] = np.mean(h2_values[i,k,:])
        


# In[78]:


store_data("Plot13_hertiablematrix", hvals)


# In[79]:


plt.rcParams["figure.figsize"] = [9, 9]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
y = bottleneck_time
x = mig_rate1
X, Y = np.meshgrid(x, y)
Z = hvals.transpose()
plt.ylabel("Time of Bottleneck(generations)",fontsize=14)
plt.xlabel("Migration Rate", fontsize=14)
#plt.label("Heritability")

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="plasma", linewidth=0, antialiased=False)

plt.show()


# # Plot 14 - Slices of heritability migration time plot

# In[82]:


plt.figure(figsize=(10,7))
plt.plot(mig_rate1, hvals[:,0], label=("20 generations ago"))
#plt.plot(mig_rate, hvals[:,3], label=("115 generations ago"))
plt.plot(mig_rate1, hvals[:,4], label=("230 generations ago"))
plt.plot(mig_rate1, hvals[:,9], label=('400 generations ago'))
plt.xlabel('Migration Rate',fontsize=14)
plt.ylabel("Heritability",fontsize=14)
plt.title('Effects on Heritability over Different Migration Rates, for Differing times of Bottleneck')
plt.legend(fontsize=14)


# # Plot 15 - heritability growth plot 

# In[192]:


N = 60
samples = 10
N_present = np.linspace(1000,70000, num = N)


def find_alpha(N_present, N_past, time):
    return(-np.log(N_past/N_present)/time)

alpha = [find_alpha(N_present[i],1000,200) for i in range (N)]


h2_values = np.zeros((N, samples))

for j in range (samples):
    for i in range (N):
        h2_values[i,j] = demography_heritability(initial_A, N_present[i], initial_C, time_bottleneck, seq_len, recom_rate, mut_rate, B_sample, C_sample, s, h2, inclusion_prob, mig_rate, alpha[i])[0]

sd = [0 for i in range(N)]
for i in range (N):
    sd[i] = np.std(h2_values[i,:])
sd = np.array(sd)
    
hvals = [0 for i in range (N)]
for i in range (N):
    hvals[i] = np.mean(h2_values[i,:])
hvals = np.array(hvals)


# In[191]:


store_data("Plot15_heritabilitymatrix", h2_values)


# In[193]:


plt.figure(figsize=(10,8))
plt.plot(N_present, hvals)
plt.xlabel('Current Population Size, N_current',fontsize=14)
plt.ylabel('Heritability',fontsize=14)
plt.title('Impact of Population Growth on Heritability')
plt.fill_between(N_present, hvals-sd, hvals+sd,
    alpha=0.5)
plt.show()


# # Plot 16- heritability growth distribution

# In[211]:


N_present1 = 1000
alpha1 = find_alpha(N_present1, 1000, 200)

n =500
heritable1 = [0 for i in range (n)]
for i in range (n):
    heritable1[i] = demography_heritability (initial_A, N_present1, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate, alpha1)[0]

N_present2 = 10000
alpha2 = find_alpha(N_present2, 1000, 200)
heritable2 = [0 for i in range (n)]
for i in range (n):
    heritable2[i] = demography_heritability (initial_A, N_present2, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate, alpha2)[0]
    
N_present3 = 30000
alpha3 = find_alpha(N_present3, 1000, 200)
heritable3 = [0 for i in range (n)]
for i in range (n):
    heritable3[i] = demography_heritability (initial_A, N_present3, initial_C, time_bottleneck, 
                             seq_len, recom_rate, mut_rate, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate, alpha3)[0]


# In[212]:


store_data("Plot16_nogrowth", heritable1)
store_data("Plot16_growth", heritable2)
store_data("Plot16_biggrowth", heritable3)


# In[213]:


plt.figure(figsize=(10,6))
plt.hist(heritable1, bins=25, alpha=0.7, label = "N_current = 1000")
plt.hist(heritable2, bins=25, alpha = 0.7, label = 'N_current = 10000')
plt.hist(heritable3, bins=25, alpha=0.7, label = "N_current = 30000")
plt.xlim(xmin=0, xmax = 1)
#plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label = 'mean')
plt.xlabel("Heritability",fontsize=14)
plt.ylabel("Frequency",fontsize=14)
plt.title("Heritability distribution with different levels of growth")
plt.axvline(0.5, color = 'black', linewidth =5, label = "Heritability in the General Population")
plt.legend(fontsize=10)
plt.show()


# # Plot 17 - Growth Linkage disequilibrium

# In[103]:


B_sample1 = 100
C_sample1 = 100

X1, Xgen1, XAJ1, mts11 = tree_sequence(initial_A, 10000, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample1, C_sample1,mig_rate, 0 )

X1, Xgen2, XAJ2, mts11 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample1, C_sample1,mig_rate, find_alpha(10000,1000, 200))

X1, Xgen3, XAJ3, mts11 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample1, C_sample1,mig_rate,0)


# In[104]:


store_data("Plot17_1", XAJ1)
store_data("Plot17_2", XAJ2)
store_data("Plot17_3", XAJ3)


# In[105]:


fig, axs = plt.subplots(1, 3, figsize=(20,6.5))

st = fig.suptitle("Population Growth Impact on Linkage Disequilibrium", fontsize="x-large")

axs[0].set_title("No Bottleneck and No Growth")
axs[0].matshow(abs(np.corrcoef(XAJ1.transpose())), cmap = 'Reds')


axs[2].set_title("Bottleneck and Growth")
axs[2].matshow(abs(np.corrcoef(XAJ2.transpose())), cmap = 'Reds')



axs[1].set_title("Bottleneck and No Growth")
axs[1].matshow(abs(np.corrcoef(XAJ3.transpose())), cmap = 'Reds')
plt.show()


# # Plot 18 - 4x4 genetic architectures 

# In[108]:


n =500

seq_len1 = 10000
recom_rate1 = 0.00001
mut_rate1 = 0.000001
heritable1 = [0 for i in range (n)]
for i in range (n):
    heritable1[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len1, recom_rate1, mut_rate1, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate, growth_rate)[0]

    


seq_len2 = 1000
recom_rate2 = 0.0001
mut_rate2 = 0.00001    
heritable2 = [0 for i in range (n)]
for i in range (n):
    heritable2[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len2, recom_rate2, mut_rate2, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate, growth_rate)[0]
    


seq_len3 = 100
recom_rate3 = 0.001
mut_rate3 = 0.0001    
heritable3 = [0 for i in range (n)]
for i in range (n):
    heritable3[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len3, recom_rate3, mut_rate3, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate, growth_rate)[0]
    
seq_len4 = 10
recom_rate4 = 0.01
mut_rate4 = 0.001    
heritable4 = [0 for i in range (n)]
for i in range (n):
    heritable4[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len4, recom_rate4, mut_rate4, B_sample, C_sample, s, 
                             h2, inclusion_prob, mig_rate, growth_rate)[0]


# In[109]:


store_data("Plot18_1_hertibalevalues", heritable1)
store_data("Plot18_2_hertibalevalues", heritable2)
store_data("Plot18_3_hertibalevalues", heritable3)
store_data("Plot18_4_hertibalevalues", heritable4)


# In[110]:


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


# # Plot 19 - different genetic architectures 

# In[111]:


n = 500


seq_len5 = 10000
recom_rate5 = 0.00001
mut_rate5 = 0.000001

her1 = [0 for i in range (n)]
for i in range (n):
    her1[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len5, recom_rate5, mut_rate5, B_sample, C_sample, s, 
                                    h2, inclusion_prob, mig_rate, growth_rate)[0]

seq_len6 = 10000
recom_rate6 = 0.000000001
mut_rate6 = 0.000001
her2 = [0 for i in range (n)]
for i in range (n):
    her2[i] = demography_heritability (initial_A, initial_B, initial_C, time_bottleneck, 
                             seq_len6, recom_rate6, mut_rate6, B_sample, C_sample, s, 
                                       h2, inclusion_prob, mig_rate, growth_rate)[0]
    


# In[112]:


store_data("Plot19_1_hertibalevalues", her1)
store_data("Plot19_2_hertibalevalues", her2)


# In[214]:


her1 = get_data("Plot19_1_hertibalevalues")
her2 = get_data("Plot19_2_hertibalevalues")

fig, axs = plt.subplots(1, 2, figsize = (17,5))

axs[0].set_title('Seq Len=10000, Recom Rate=0.00001, Mut Rate=0.000001')
axs[0].hist(her1, bins =25)
axs[0].set_xlabel('Heriability')
axs[0].set_ylabel('Frequency')
axs[0].set_xlim(0,1)

axs[1].set_title('Seq Len=10000, Recom Rate=0.000000001, Mut Rate= 0.000001')
axs[1].hist(her2, bins =25)
axs[1].set_xlabel('Heriability', fontsize=12)
axs[1].set_ylabel('Frequency', fontsize=12)
axs[1].set_xlim(0,1)


plt.show()


# # Plot 20 - effect size localised genetic architecture 

# In[134]:


seq_len = 10000
recom_rate = 0.000000001
mut_rate = 0.000001
X1, Xgen1, XAJ1, mts11 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,mig_rate, growth_rate)
fgen1 = minor_allele_freq(find_freq(Xgen1))
fAJ1 = minor_allele_freq(find_freq(XAJ1))
B1 = generate_B(Xgen1, s, inclusion_prob)


X2, Xgen2, XAJ2, mts12 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample, mig_rate, growth_rate)
fgen2 = minor_allele_freq(find_freq(Xgen2))
fAJ2 = minor_allele_freq(find_freq(XAJ2))
B2 = generate_B(Xgen2, s, inclusion_prob)

X3, Xgen3, XAJ3, mts13 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,mig_rate, growth_rate)
fgen3 = minor_allele_freq(find_freq(Xgen3))
fAJ3 = minor_allele_freq(find_freq(XAJ3))
B3 = generate_B(Xgen3, s, inclusion_prob)


# In[148]:


store_data("Plot20_1_fAJ", fAJ1)
store_data("Plot20_1_fgen", fgen1)
store_data("Plot20_1_B", B1)
store_data("Plot20_2_fAJ", fAJ2)
store_data("Plot20_2_fgen", fgen2)
store_data("Plot20_2_B", B2)
store_data("Plot20_3_fAJ", fAJ3)
store_data("Plot20_3_fgen", fgen3)
store_data("Plot20_3_B", B3)


# In[147]:


fig, axs = plt.subplots(1, 3, figsize = (17,5))
plt.suptitle("SNP frequency and effect size in sequence Length 10000, recom rate 0.000000001, mut rate = 0.000001")

axs[0].scatter(fAJ1, abs(B1), label ='Bottleneck Population')
axs[0].scatter(fgen1, abs(B1), label = 'General Population')
axs[0].legend()
axs[0].set_ylabel('Effect Size, Beta',fontsize=12)
axs[0].set_xlabel('Minor Allele Frequency',fontsize=12)


axs[1].scatter(fAJ2, abs(B2), label ='Bottleneck Population')
axs[1].scatter(fgen2, abs(B2), label = 'General Population')
axs[1].legend()
axs[1].set_ylabel('Effect Size, Beta',fontsize=12)
axs[1].set_xlabel('Minor Allele Frequency',fontsize=12)

axs[2].scatter(fAJ3, abs(B3), label ='Bottleneck Population')
axs[2].scatter(fgen3, abs(B3), label = 'General Population')
axs[2].legend()
axs[2].set_ylabel('Effect Size, Beta',fontsize=12)
axs[2].set_xlabel('Minor Allele Frequency',fontsize=12)
plt.show()


# # Plot 21 - effect size genome wide architecture

# In[130]:


seq_len = 10000
recom_rate = 0.00001
mut_rate = 0.000001
X1, Xgen1, XAJ1, mts11 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,mig_rate, growth_rate)
fgen1 = minor_allele_freq(find_freq(Xgen1))
fAJ1 = minor_allele_freq(find_freq(XAJ1))
B1 = generate_B(Xgen1, s, inclusion_prob)


X2, Xgen2, XAJ2, mts12 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,mig_rate, growth_rate)
fgen2 = minor_allele_freq(find_freq(Xgen2))
fAJ2 = minor_allele_freq(find_freq(XAJ2))
B2 = generate_B(Xgen2, s, inclusion_prob)

X3, Xgen3, XAJ3, mts13 = tree_sequence(initial_A, initial_B, initial_C, time_bottleneck, seq_len,recom_rate, mut_rate, B_sample, C_sample,mig_rate, growth_rate)
fgen3 = minor_allele_freq(find_freq(Xgen3))
fAJ3 = minor_allele_freq(find_freq(XAJ3))
B3 = generate_B(Xgen3, s, inclusion_prob)


# In[118]:


store_data("Plot21_1_fAJ", fAJ1)
store_data("Plot21_1_fgen", fgen1)
store_data("Plot21_1_B", B1)
store_data("Plot21_2_fAJ", fAJ2)
store_data("Plot21_2_fgen", fgen2)
store_data("Plot21_2_B", B2)
store_data("Plot21_3_fAJ", fAJ3)
store_data("Plot21_3_fgen", fgen3)
store_data("Plot21_3_B", B3)


# In[131]:


fig, axs = plt.subplots(1, 3, figsize = (17,5))
plt.suptitle("SNP frequency and effect size in sequence Length 10000, recom rate 0.00001, mut rate 0.000001")

axs[0].scatter(fAJ1, abs(B1), label ='Bottleneck Population')
axs[0].scatter(fgen1, abs(B1), label = 'General Population')
axs[0].legend()
axs[0].set_ylabel('Effect Size, Beta',fontsize=12)
axs[0].set_xlabel('Minor Allele Frequency',fontsize=12)


axs[1].scatter(fAJ2, abs(B2), label ='Bottleneck Population')
axs[1].scatter(fgen2, abs(B2), label = 'General Population')
axs[1].legend()
axs[1].set_ylabel('Effect Size, Beta',fontsize=12)
axs[1].set_xlabel('Minor Allele Frequency',fontsize=12)

axs[2].scatter(fAJ3, abs(B3), label ='Bottleneck Population')
axs[2].scatter(fgen3, abs(B3), label = 'General Population')
axs[2].legend()
axs[2].set_ylabel('Effect Size, Beta',fontsize=12)
axs[2].set_xlabel('Minor Allele Frequency',fontsize=12)
plt.show()


# In[ ]:





# In[ ]:




