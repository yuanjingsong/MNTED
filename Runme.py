import scipy.io as sio
from MNTED import MNTED
import time


# dataset_name = "aminer"
# dataset_name = "blog_perturb"
dataset_name = "congress"
# dataset_name = "Disney"
#dataset_name = "flickr"
# dataset_name = "wiki"

file_name_dic = {'aminer': 'aminer/aminer_duicheng',
                 "blog_perturb": 'blog_perturb/blog_perturb',
                 "congress": 'congress/congress_',
                 "Disney": 'Disney/disney_disturb_',
                 "flickr": 'flickr/flickr_disturb_',
                 "wiki": 'wiki/wiki_disturb_'}

'''################# Load dataset #################'''
file_name=file_name_dic[dataset_name]
print("Dataset:",dataset_name)
if dataset_name=='Disney':
    AttributeMatrixFileName = 'best_X'
    NetworkMatrixFileName = 'best_A'
else:
    AttributeMatrixFileName='X'
    NetworkMatrixFileName='A'
lambd = 10**-0.6  # the regularization parameter
rho = 5  # the penalty parameter
# mat_contents = sio.loadmat('Flickr.mat')
# lambd = 0.0425  # the regularization parameter
# rho = 4  # the penalty parameter

File_name=file_name+'1'+'.mat'
mat_contents = sio.loadmat(File_name)
i=1
G=[]
A=[]
while mat_contents is not None:
    i = i+1
    Network=mat_contents[NetworkMatrixFileName]
    Attribute=mat_contents[AttributeMatrixFileName]
    # print('the data type of ndarray:',Network.dtype)
    if 'float' not in str(Network.dtype)  or  'double' not in str(Network.dtype)  : Network=Network.astype(float)
    if 'float' not in str(Attribute.dtype)  or  'double' not in str(Attribute.dtype)  : Attribute=Attribute.astype(float)
    G.append(Network)#（5196,5196）
    A.append(Attribute)#（5196,8189）
    # Label = mat_contents["Label"]
    del mat_contents
    n = G[0].shape[0]
    if n>=1000:
        d = 100# the dimension of the embedding representation
    else:
        d=int(n/10)

    # CombG.append(G[Group1+Group2, :][:, Group1+Group2]) #shape：（5196,5196）把原来的G的顺序打乱
    # CombA.append(A[Group1+Group2, :])#shape：（5196,8189）把原来的A的顺序打乱
    try:
        mat_contents = sio.loadmat(file_name+str(i)+'.mat')
    except (FileNotFoundError):
        break








'''################# Multilayer Network Tax Evasion Detection #################'''
print("Multilayer Network Tax Evasion Detection (MNTED)")
# use use_method to control the new or origin method
# if use_method != "new" then algorithm will use the origin method

use_method = "new"
#use_method = "origin"

print("use " + use_method + " method")
start_time = time.time()
V_MNTED = MNTED(G, A, d, lambd=lambd, rho=rho, method=use_method).train()
V_MNTED_time = time.time() - start_time
print("time elapsed: {:.2f}s".format(V_MNTED_time))

'''################# MNTED for a Pure Network #################'''
print("MNTED for a pure network:")
start_time = time.time()
V_Net = MNTED(G, G, d, lambd=lambd, rho=rho, method=use_method).train()
V_Net_time=time.time() - start_time
print("time elapsed: {:.2f}s".format(V_Net_time))

'''################# Save the embedding result into files #################'''
for g in range(len(V_MNTED)):
    print("g is " + str(g))
    print(V_MNTED[g])
    sio.savemat("result/"+file_name+str(g)+'_Embedding.mat', {"V_MNTED": V_MNTED[g], "V_Net": V_Net[g]})
print("Embedding.mat printed")

print("data file path is result/" + file_name + "_" + "embedding_time_result.txt")
data = open("result/"+file_name+"_"+"embedding_time_result"+".txt", 'a+')
print("Dataset:", dataset_name, file=data)
print("Method:" + use_method, file=data)
print("time length:", len(V_MNTED), file=data)
print("Time spent on MNTED: %.12f s" % V_MNTED_time, file=data)
print("Time spent on NET: %.12f s" % V_Net_time, file=data)
data.close()