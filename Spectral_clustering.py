import numpy as np
from .kmeans import k_means

class spectral_clustering():
    def __init__(self, k = 5, max_iter = 2000):
        self.k = k
        self.max_iter = max_iter
    
    def remove_isolated_nodes(self, nodes, edges):
        s= set()
        nodes_remained = []
        for a, b in edges:
            s.add(a)
            s.add(b)
        # idx = 0
        for i, label in nodes:
            if i in s:
                nodes_remained.append([i, label])
                # idx += 1
        nodes = np.array(nodes_remained)
        return nodes
        
    def create_adj_matrix(self, nodes, edges):
        num_nodes = nodes.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i, j in edges:
            x = np.argwhere(nodes[:, 0] == i)[0][0]
            y = np.argwhere(nodes[:, 0] == j)[0][0]
            adj_matrix[x][y] += 1
            adj_matrix[y][x] += 1
        return adj_matrix
    
    def create_graph_laplacian(self, adj_matrix):
#         from collections import Counter
        D = np.diag(np.sum(adj_matrix, axis=1))
        L = D - adj_matrix
#         for i in range(L.shape[0]):
#             print(Counter(L[i, :]))
        # normalize Laplacian matrix
        sqrt_D = np.diag(1.0 / (np.sum(adj_matrix, axis=1) **0.5))
        return np.dot(sqrt_D, L).dot(sqrt_D)
        # return L
    
    def smallest_eigval_counts(self, eigval):
        count = 0
        for i in eigval:
            if i == min(eigval):
                count += 1
        return count
    
    def count_majority_label(self, labels, nodes):
        l = dict()
        r = dict()
        mismatch_num = 0
        for i in range(self.k):
            idx = np.argwhere(labels == i)
            data_labels = nodes[idx, 1].flatten()
            label_counts = np.bincount(data_labels)
            print(label_counts)
            major_label = np.argmax(label_counts)
            num_cluster = np.sum(label_counts)
            mismatch_rate = (num_cluster - np.max(label_counts)) / num_cluster
            mismatch_num += num_cluster - np.max(label_counts)
            l[i] = major_label
            r[i] = mismatch_rate
        return l, r, mismatch_num
    
    def forward(self, nodes, edges):
        nodes = self.remove_isolated_nodes(nodes, edges)
        adj_matrix = self.create_adj_matrix(nodes, edges)
        L = self.create_graph_laplacian(adj_matrix)
        val, vec = np.linalg.eig(L)
        val = np.real(val)
        vec = np.real(vec)
        # print(min(val))
        # count = self.smallest_eigval_counts(val)
        # print(count)
        # idx_smallest_eigval = np.argsort(val)[:count]
        idx_smallest_eigval = np.argsort(val)[:self.k]
        # print(idx_smallest_eigval)
        Z = vec[:, idx_smallest_eigval]
        # print(np.sum(Z[1, :]))
        # eigenvector normalization
        rows_norm = np.linalg.norm(Z, axis=1)
        Z = (Z.T / rows_norm).T
        model = k_means(self.k, self.max_iter)
        _, labels = model.forward(Z)
        major_labels, mismatch_rate, mismatch_num = self.count_majority_label(labels, nodes)
        
        print('Total mismatch num is {}'.format(mismatch_num))
        
        return major_labels, mismatch_rate, mismatch_num

if __name__ == '__main__':
    # read nodes txt files
    nodes = []
    with open('data/nodes.txt') as f:
        data = f.readlines()
        for line in data:
            line = line.strip('\n')
            nodes.append(line.split()[0:3:2])
    nodes = np.array(nodes).astype(np.int32)
    # print(nodes.shape)
    # (1490, 2)

    # read edges txt files
    edges = []
    with open('data/edges.txt') as f:
        data = f.readlines()
        for line in data:
            line = line.strip('\n')
            edges.append(line.split())
    edges = np.array(edges).astype(np.int32)
    # print(edges.shape)
    # (19090, 2)

    sc = spectral_clustering(5, 2000)
    labels, rate, mis_num = sc.forward(nodes, edges)
