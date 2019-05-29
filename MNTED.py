import numpy as np
import torch
from torch.autograd import Variable
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from math import ceil
import time

class MNTED:
    """
    when u use the extra para u should explicit declare the para name and value
    for example :
        MNTED(MultiNet, MultiAttri, d, n=10, lambda = 10**-6)

    :param  MultiNet: a set of the weighted adjacency matrices（a list of ndarray(n*n)）
    :param  MultiAttri: a set of the weighted adjacency matrices （a list of ndarray(n*m)）
    :param  d: the dimension of the embedding representation
    :param  n: the num of nodes per layer
    :param  m: the dimension of the attribute
    :param  k: the num of layers in total
    :param  lambd: the regularization parameter
    :param  rho: the penalty parameter
    :param  maxiter: the maximum number of iteration
    :param  'Att': refers to conduct Initialization from the SVD of Attri
    :param  splitnum: the number of pieces we split the SA for limited cache
    :param  window_len: the length of the moving window
    :param method: choose new / origin method to train
    :return the comprehensive embedding representation H(a list containing all the time stamps' representation)
    """
    def __init__(self, MultiNet, MultiAttri, d, **kwargs):
        """

        :param MultiNet: a list  of network matrices with shape of (n,n)
        :param MultiAttri: a list of attribute matrices with shape of (n,m)
        :param d: the dimension of the embedding representation
        :param varargs: 0：lambd，1：rho，2：maxiter，3：Att(use "Att" or "Net" for H's init), 4:splitnum
        :param self.H: a list of representation of layers (len: self.k)
        :param self.Z: a copy of self.H (len: self.k)
        :param self.U: a list of dual variable(len: self.k)
        :param self.V: window's representation (len: 1)
        :returns initialization of multiple core variable

        """
        self.window_len=8
        self.maxiter = 2  # Max num of iteration
        self.lambd = 0.05  # Initial regularization parameter
        self.rho = 5  # Initial penalty parameter
        self.k = len(MultiNet)#the num of layers of Multilayer Network
        print('k:', self.k)
        self.d = d
        self.worknum = 3
        # number of pieces we split the SA for limited cache
        splitnum = 1
        # n = Total num of nodes, m = attribute category num
        [self.n, m] = MultiAttri[0].shape
        print('MultiNet.shape:', MultiNet[0].shape)
        Nets = []
        Attris = []
        for Net in MultiNet:
            Net = sparse.lil_matrix(Net)
            Net.setdiag(np.zeros(self.n))
            Net = csc_matrix(Net)  # 在用python进行科学运算时，常常需要把一个稀疏的np.array压缩
            #Net = torch.from_numpy(Net)
            Nets.append(Net)
        # Nets=[csc_matrix(sparse.lil_matrix(Net).setdiag(np.zeros(self.n))) for Net in MultiNet]
        for Attri in MultiAttri:
            Attri = csc_matrix(Attri)
            #Atrri = torch.from_numpy(Atrri)
            Attris.append(Attri)
        # Attris=[csc_matrix(Attri) for Attri in MultiAttri]
        self.H = []

        if len(kwargs) >= 4 and "Att" in kwargs:
            # 将属性矩阵A打乱成n*m（或n*10d）的新矩阵再进行svd分解后的酉矩阵，规格为n*d，作为H的初始值
            for Atrri in Attris:
                sum_col = np.arange(m)# [0,1,...,m-1]
                np.random.shuffle(sum_col)# 打乱
                self.H.append(svds(Atrri[:, sum_col[0:min(10 * d, m)]], d)[0])
        else:
            # 将拓扑矩阵打乱成n*n或（n*10d）的新矩阵（按照纵向求和的从大到小排列），再进行svd分解后的酉矩阵，规格为n*d，作为H的初始值
            for Net in Nets:
                sum_col = Net.sum(0)
                H_initial = svds(Net[:, sorted(range(0, self.n), key=lambda r: sum_col[0, r], reverse=True)[0:min(10 * d, self.n)]], d)[0]
                self.H.append(H_initial)

        #
        # if len(varargs) >= 4 and varargs[3] == 'Att':
        #     # 将属性矩阵A打乱成n*m（或n*10d）的新矩阵再进行svd分解后的酉矩阵，规格为n*d，作为H的初始值
        #     for Atrri in Attris:
        #         sum_col = np.arange(m)# [0,1,...,m-1]
        #         np.random.shuffle(sum_col)# 打乱
        #         self.H.append(svds(Atrri[:, sum_col[0:min(10 * d, m)]], d)[0])
        # else:
        #     # 将拓扑矩阵打乱成n*n或（n*10d）的新矩阵（按照纵向求和的从大到小排列），再进行svd分解后的酉矩阵，规格为n*d，作为H的初始值
        #     for Net in Nets:
        #         sum_col = Net.sum(0)#将Net沿纵方向向下加，成为一个长度为n的向量
        #         H_initial=svds(Net[:, sorted(range(0,self.n), key=lambda r: sum_col[0, r], reverse=True)[0:min(10 * d, self.n)]], d)[0]
        #         self.H.append(H_initial)
        #         # svds(Net[:, sorted(range(self.n), key=lambda r: sum_col[0, r], reverse=True)[0:min(10 * d, self.n)]], d)[0]

        if len(kwargs) > 0:
            self.lambd = kwargs["lambd"]
            self.rho = kwargs["rho"]
            if len(kwargs) >= 3:
                if "maxiter" in kwargs:
                    self.maxiter = kwargs["maxiter"]
            if len(kwargs) >= 5:
                if "splitum" in kwargs:
                    splitnum = kwargs["splitnum"]

        # if len(varargs) > 0:
        #     self.lambd = varargs[0]
        #     self.rho = varargs[1]
        #     if len(varargs) >= 3:
        #         self.maxiter = varargs[2]
        #         if len(varargs) >= 5:
        #             splitnum = varargs[4]

        # Treat at least（most？？） each 7575 nodes as a block。即将n个节点分成splitnum个block，1个block最多有7575个节点
        self.block = min(int(ceil(float(self.n) / splitnum)), 7575)
        # 重新计算split_num
        self.split_num = int(ceil(float(self.n) / self.block))
        # inf will be ignored,即不管异常与否，都会进行下面的计算
        with np.errstate(divide='ignore'):
            #计算属性矩阵
            self.Attri = [Attri.transpose() * sparse.diags(np.ravel(np.power(Attri.power(2).sum(1), -0.5))) for Attri in Attris]
            # self.Attri = torch.from_numpy(self.Attri)
        self.Z = self.H.copy()
        # Index for affinity matrix sa
        self.affi = -1
        # U是k*n*d的全0矩阵
        # self.U = [np.zeros((self.n, d)) for i in range(self.k)]
        self.U = np.zeros(self.n * d * self.k).reshape((self.k, self.n, d))

        # V的初始值和H的第一个图的初始值一样
        self.V = self.H[0]
        # 将每一列的非零元素的坐标分开，得到A list of sub-arrays.
        self.nexidx = [np.split(Net.indices, Net.indptr[1:-1]) for Net in Nets]
        # self.nexidx = torch.from_numpy(self.nexidx)
        # 将每一列的非零数据分开，得到A list of sub-arrays
        self.Net = [np.split(Net.data, Net.indptr[1:-1]) for Net in Nets]
        # self.Net = torch.from_numpy(self.Net)

        self.methd = kwargs["method"]



    '''################# Update functions #################'''
    def updateH (self, k):
        xtx = np.dot(self.Z[k].transpose(), self.Z[k]) * 2 + (2 + self.rho) * np.eye(self.d)
        # Split nodes into different Blocks
        for blocki in range(self.split_num):
            # Index for splitting blocks
            index_block = self.block * blocki
            next_index = index_block + min(self.n - index_block, self.block)
            if self.affi != blocki:
                self.sa = self.Attri[k][:, range(index_block, index_block + min(self.n - index_block, self.block))].transpose() * self.Attri[k]
                self.affi = blocki
            sums = self.sa.dot(self.Z[k]) * 2
            for i in range(index_block, next_index):
                neighbor = self.Z[k][self.nexidx[k][i], :]  # the set of adjacent nodes of node i
                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - self.H[k][i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index
                    if np.any(nzidx):
                        normi_j = (self.lambd * self.Net[k][i][nzidx]) / normi_j[nzidx]
                        self.H[k][i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(self.d),
                              sums[i - index_block, :] + (neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) -(2-self.rho)*self.Z[k][i,:]+4*self.V[i,:]-self.rho*self.U[k][i,:])
                    else:
                        # if not_equal_index  is False
                        self.H[k][i, :] = np.linalg.solve(xtx,
                              sums[i - index_block, :] - (2 - self.rho) * self.Z[k][i, :] + 4*self.V[i, :] - self.rho*self.U[k][i, :])

    def updateZ (self, k):
        xtx = np.dot(self.H[k].transpose(), self.H[k]) * 2 + (2+self.rho) * np.eye(self.d)
        for blocki in range(self.split_num):  # Split nodes into different Blocks
            index_block = self.block * blocki  # Index for splitting blocks
            if self.affi != blocki:
                self.sa = self.Attri[k][:, range(index_block, index_block + min(self.n - index_block, self.block))].transpose() * self.Attri[k]
                self.affi = blocki
            sums = self.sa.dot(self.H[k]) * 2
            for i in range(index_block, index_block + min(self.n - index_block, self.block)):
                neighbor = self.H[k][self.nexidx[k][i], :]  # the set of adjacent nodes of node i
                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - self.Z[k][i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index
                    if np.any(nzidx):
                        normi_j = (self.lambd * self.Net[k][i][nzidx]) / normi_j[nzidx]
                        self.Z[k][i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(self.d), sums[i - index_block, :] + (
                                    neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0)
                                    - (2-self.rho)*self.H[k][i,:]+4*self.V[i,:]-self.rho*self.U[k][i,:])
                    else:
                        self.Z[k][i, :] = np.linalg.solve(xtx, sums[i - index_block, :]
                                    - (2-self.rho)*self.H[k][i,:]+4*self.V[i,:]-self.rho*self.U[k][i,:])

    def updateV(self, k):
        self.V = 1/2 * (self.H[k] + self.Z[k])

    def init_Tensor(self):
        self.H = np.array(self.H)
        self.Z = np.array(self.Z)
        self.H = Variable(torch.Tensor(self.H), requires_grad=True)
        self.Z = Variable(torch.Tensor(self.Z), requires_grad=True)
        # make self.V float 32
        self.V = torch.tensor(self.V).float()
        self.U = torch.tensor(self.U).float()

    def Update(self, ith_layer, Net):
        # self.H = Variable(torch.Tensor(self.H), requires_grad = True)
        for block_i in range(self.split_num):
            index_block = self.block * block_i
            next_index = index_block + min(self.n - index_block, self.block)
            if self.affi != block_i:
                self.sa = self.Attri[ith_layer][:, range(index_block, next_index)].transpose() * self.Attri[ith_layer]
                # convert self.sa to float
                self.sa = self.sa.toarray().astype(np.float64)
                self.sa = Variable(torch.Tensor(self.sa))

                #self.sa = self.Attri[ith_layer][: range(index_block, index_block + min(self.n - index_block, self.block))].transpose() * self.Attri[ith_layer]
                self.affi = block_i
            for i in range(index_block, next_index):
                # self.Z[ith_layer] 可能出现 nan
                # self.H[ith_layer][i, :] 可能出现nan

                # sa = 100 * 100
                # self.H = 8 * 100 * 10
                # self.Z = 8 * 100 * 10
                # term1 = -2 * (self.sa - self.H[ith_layer][i, :].dot(self.Z[ith_layer].transpose())) * self.Z[ith_layer]

                # term1 = -2 * (self.sa - self.H[ith_layer][i].dot(self.Z[ith_layer].transpose())) * self.Z[index_block][i]

                # print("before term1's shape is " + str(term1.shape))
                # term1 = term1[i - index_block, :]

                # print("term1 's shape is " + str(term1.shape))
                #local_H = Variable(H[ith_layer][i, :], requires_grad=True)
                # np 版本
                #term1 = -2 * (self.sa - self.H[ith_layer][i, :].dot(self.Z[ith_layer].transpose())) * self.Z[ith_layer]

                # tensor 版本
                resize_H = self.H[ith_layer][i, :].reshape((1, len(self.H[ith_layer][i, :])))
                #transponse_Z = Variable(torch.tensor(self.Z[ith_layer].transpose(0, 1)))
                transponse_Z = self.Z[ith_layer].transpose(0, 1)

                #resize_H = self.H[ith_layer][i, :].resize_((1, len(self.H[ith_layer][i, :])))
                #print(resize_H.size())

                #term1 = -2 * (self.sa.double() - resize_H.mm(transponse_Z)).mm(Variable(torch.tensor(self.Z[ith_layer])))
                #term1 = -2 * (self.sa - resize_H.mm(self.Z[ith_layer].transpose(0, 1))) * self.Z[ith_layer]

                print(resize_H.size())
                print(transponse_Z.size())
                term1 = -2 * (self.sa[i - index_block, :] - resize_H.mm(transponse_Z)) .mm(self.Z[ith_layer])
                #term1 = -2 * (self.sa - resize_H.mm(transponse_Z)) * self.Z[ith_layer]


                #term1 = term1[i - index_block, :]
                # term = 1 * 10

                print("i is " + str(i))
                print("before backward")
                print(self.H[ith_layer][i, :])
                #term1 = Variable(torch.Tensor(term1), requires_grad=True)

                #term3 = 2 * (self.H[ith_layer][i, :] + self.Z[ith_layer][i, :] - 2 * self.V[i, :])
                #term3 = 2 * (self.H[ith_layer] + self.Z[ith_layer] - 2 * self.V)
                term3 = 2 * (self.H[ith_layer] + self.Z[ith_layer] - 2 * self.V)
                print(term3.size())

                term3 = term3[i, :]

                # print("term3 's shape is " + str(term3.shape))
                #term3 = Variable(torch.Tensor(term3), requires_grad=True)
                #term3 = term3.double()

                #term4 = self.rho * (self.H[ith_layer][i, :] - self.V[i][i, :] + self.U[ith_layer][i, :])

                term4 = self.rho * (self.H[ith_layer][i, :] - self.V[i] + self.U[ith_layer][i, :])

                #term4 = self.rho * (self.H[ith_layer] - self.V + self.U[ith_layer])
                # print("term4's shape is " + str(term4.shape))
                #term4 = Variable(torch.Tensor(term4), requires_grad=True)
                #term4 = Variable(torch.Tensor(term4), requires_grad = True)
                #term4 = term4.double()

                # neighbor the set of adjacent nodes of node i
                neighbor = self.Z[ith_layer][self.nexidx[ith_layer][i], :]
                if len(neighbor) == 0:
                    # 此时 不能直接求导
                    # 用 numpy 计算
                    """
                    neighbor = self.Z[ith_layer][self.nexidx[ith_layer][i], :].detach().numpy()
                    numpy_H = self.H[ith_layer][i, :].detach().numpy()
                    normi_j = np.linalg.norm(neighbor - numpy_H, axis=1)
                    neighbor = torch.Tensor(neighbor)
                    normi_j = (self.lambd * self.Net[ith_layer][i]) / normi_j
                    normi_j = torch.Tensor(normi_j)
                    """
                    term2 = torch.zeros([1, term4.size()[0]])
                else:
                    # 用 pytorch 计算
                    normi_j = torch.norm(neighbor - self.H[ith_layer][i, :], p=1)
                    normi_j = (self.lambd * (torch.Tensor(np.array(self.Net[ith_layer][i])))) / normi_j
                    term2 = ((self.H[ith_layer][i, :] - neighbor) * normi_j.reshape((-1, 1))).sum(0)

                #term2 = (self.H[ith_layer] - neighbor) * normi_j.reshape((-1, 1)).sum(0)


                #print(neighbor.shape)
                #print((self.H[ith_layer][i, :] - neighbor).shape)

                #print("term2 's shape is " + str(term2.shape))
                #term2 = Variable(torch.Tensor(term2), requires_grad=True)
                #term2 = term2.double()
                # print(term1.size())
                # print(term2.size())
                # print(term3.size())
                # print(term4.size())

                """
                print(term1.dtype)
                print(term2.dtype)
                print(term3.dtype)
                print(term4.dtype)
                """

                final = term1 + term2 + term3 + term4
                #final.backward(torch.ones_like(final))
                #final.backward(torch.Tensor(self.H[ith_layer][i, :]))

                if Net == "H":
                    final.backward(self.H[ith_layer][i, :].resize(1, 10), retain_graph = True)
                    print("after backward")
                    with torch.no_grad():
                        self.H[ith_layer][i, :] = self.H.grad[ith_layer][i, :]
                    print(self.H[ith_layer][i, :])
                elif Net == "Z":
                    final.backward(self.Z[ith_layer][i, :].resize(1, 10), retain_graph = True)
                    #print("after backward")
                    with torch.no_grad():
                        self.Z[ith_layer][i, :] = self.Z.grad[ith_layer][i, :]
                    #print(self.Z[ith_layer][i, :])
                #self.H[ith_layer][i, :] = term1.grad + term2.grad + term3.grad + term4.grad
                #self.H[ith_layer][i, :] = final.grad

    # term1,term2,term3,term4是对应的式子的四项
    def UpdateGraph(self, ith_layer, net_class):
        for block_i in range(self.split_num):
            index_block = self.block * block_i
            next_block = index_block + min(self.n - index_block, self.block)
            if self.affi != block_i:
                self.sa = self.Attri[index_block] [:, range(index_block, next_block)].transpose()*self.Attri[ith_layer]
                self.sa = self.sa.toarray()
                #self.sa = Variable(torch.tensor(self.sa))

                # make self.sa float
                self.sa = torch.tensor(self.sa).float()
                self.affi = block_i

            # 对于一个二维网络而言的 term3 term4
            term3_Net = 2 * (self.H[ith_layer] + self.Z[ith_layer] - 2 * self.V)
            term4_Net = self.rho * (self.H[ith_layer] - self.V + self.U[ith_layer])

            for i in range(index_block, next_block) :
                resize_H = self.H[ith_layer][i, :].reshape((1, len(self.H[ith_layer][i, :])))
                transponse_Z = self.Z[ith_layer].transpose(0, 1)

                term1 = -2 * (self.sa[i - index_block, :] - resize_H.mm(transponse_Z)).mm(self.Z[ith_layer])
                term3 = term3_Net[i, :]
                term4 = term4_Net[i, :]
                neighbor = self.Z[ith_layer][self.nexidx[ith_layer][i], :]

                if len(neighbor) == 0:
                    term2 = torch.zeros([1, term4.size()[0]])
                else:
                    normi_j = torch.norm(neighbor - self.H[ith_layer][i, :], p=1)
                    normi_j = (self.lambd * (torch.tensor(np.array(self.Net[ith_layer][i])).float())) / normi_j
                    term2 = ((self.H[ith_layer][i, :] - neighbor) * normi_j.reshape((-1, 1))).sum(0)

                final = term1 + term2 + term3 + term4
                if net_class == "H":
                    final.backward(self.H[ith_layer][i, :].resize(1, 10), retain_graph=True)
                    with torch.no_grad():
                        self.H[ith_layer][i, :] = self.H.grad[ith_layer][i, :]
                elif net_class == "Z":
                    final.backward(self.Z[ith_layer][i, :].resize(1, 10), retain_graph=True)
                    with torch.no_grad() :
                        self.Z[ith_layer][i, :] = self.Z.grad[ith_layer][i, :]

    def UpdateH(self, ith_layer):
        self.UpdateGraph(ith_layer, "H")
        #self.Update(ith_layer, "H")

    def UpdateZ(self, ith_layer):
        self.UpdateGraph(ith_layer, "Z")
        #self.Update(ith_layer, "Z")

    def train(self):
        V_list = list()
        '''################# Iterations #################'''

        """
        def update_with_range_layers(start, end):
            # end is not included
            if start == end:
                for itr in range(self.maxiter - 1):
                    self.updateH(start)
                    self.updateZ(start)
                    self.updateV(start)
                    self.U[start] = self.U[start] + self.H[start] - self.Z[start]
            else:
                for i in np.arange(start, end+1):
                    for itr in range(self.maxiter - 1):
                        self.updateH(i)
                        self.updateZ(i)
                        self.updateV(i)
                        self.U[i] = self.U[i] + self.H[i] - self.Z[i]
        """

        def update_with_range_layers(start, end):
            for i in range(start, end+1):
                for itr in range(self.maxiter - 1):
                    #self.UpdateH(i)

                    if self.methd == "new":
                        self.UpdateH(i)
                        self.UpdateZ(i)
                    else:
                        self.updateH(i)
                        self.updateZ(i)

                    #self.updateH(i)
                    self.updateV(i)
                    self.U[i] = self.U[i] = self.H[i] - self.Z[i]

        """
        for i in range(self.k):
            update_with_range_layers(max(0, i - self.window_len+1), i)
            V_list.append(self.V)
        """

        self.init_Tensor()
        for i in range(self.k):
            if i < self.window_len-1:
                update_with_range_layers(0, i)
                V_list.append(self.V.detach().numpy())
            else:
                update_with_range_layers(i-self.window_len+1, i)
                V_list.append(self.V.detach().numpy())

        # for i in range(self.window_len):
        #     for itr in range(self.maxiter - 1):
        #         self.updateH(i)
        #         self.updateZ(i)
        #         self.updateV(i)
        #         self.U[i]=self.U[i] + self.H[i] - self.Z[i]
        #print(V_list)
        return V_list
