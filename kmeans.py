import random
import numpy as np
import time
from PIL import Image


class k_medoids():
    def __init__(self, k = 5, max_iter = 2000):
        self.k = k
        self.max_iter = max_iter
    
    # calculate distance of
    # 2 vectors or (1 vector and many vectors)
    # with vector broadcast feature
    def distance(self, x, y):
        return np.sum(np.square(x - y), axis=-1)
    
    # initialize k cluster centers
    def random_initialize_center(self, pixels):
        num_samples, dim = pixels.shape
        centroids = np.zeros((self.k, dim))
        index = np.array(random.sample(range(0, num_samples), self.k))
        for i in range(self.k):
            centroids[i] = pixels[index[i]]
        return centroids
    
    # assign one sample to nearest centroids
    def cluster_assignment(self, x, centroids):
        distances = self.distance(x, centroids)
        return np.argmin(distances)
    
    # output: a nx1 vector with labels in [0, k-1]
    def update_cluster_labels(self, pixels, centroids):
        num_samples = pixels.shape[0]
        labels = np.zeros((num_samples, 1))
        for i in range(num_samples):
            labels[i] = self.cluster_assignment(pixels[i], centroids)
        return labels
    
    # update representative for each cluster
    def update_centroids(self, pixels, centroids, labels):
        original_loss = 0
        new_loss = 0
        stop_flag = 0
        for i in range(self.k):
            pixels_i = pixels[labels[:, 0]==i]
            original_loss += np.sum(self.distance(centroids[i], pixels_i))
            center = np.sum(pixels_i, axis=0) / pixels_i.shape[0]
            nearest_index = np.argmin(self.distance(center, pixels_i))
            centroids[i] = pixels_i[nearest_index]
            new_loss += np.sum(self.distance(centroids[i], pixels_i))
        if original_loss == new_loss:
            stop_flag = 1
        return new_loss, stop_flag
    
    def forward(self, pixels):
        if pixels.shape[0] <= self.k:
            print("The number of k is too big!")
            return
        
        random.seed(555)
        
        tic = time.time()
        
        centroids = self.random_initialize_center(pixels)
        count = 0
        for _ in range(self.max_iter):
            cluster_labels = self.update_cluster_labels(pixels, centroids)
            loss, stop_flag = self.update_centroids(pixels, centroids, cluster_labels)
            print(loss)
            count += 1
            if stop_flag == 1:
                break

        print('iter time: {}'.format(count))
        print('time: {}'.format(time.time() - tic))
        
        cluster_labels = cluster_labels.astype(int).flatten()
        return centroids, cluster_labels

class k_means():
    def __init__(self, k = 5, max_iter = 2000):
        self.k = k
        self.max_iter = max_iter
    
    # calculate distance of
    # 2 vectors or (1 vector and many vectors)
    # with vector broadcast feature
    def distance(self, x, y):
        return np.sum(np.square(x - y), axis=-1)
    
    # initialize k cluster centers
    def random_initialize_center(self, pixels):
        num_samples, dim = pixels.shape
        centroids = np.zeros((self.k, dim))
        index = np.array(random.sample(range(0, num_samples), self.k))
        for i in range(self.k):
            centroids[i] = pixels[index[i]]
        return centroids
    
    # assign one sample to nearest centroids
    def cluster_assignment(self, x, centroids):
        distances = self.distance(x, centroids)
        return np.argmin(distances)
    
    # output: a nx1 vector with labels in [0, k-1]
    def update_cluster_labels(self, pixels, centroids):
        num_samples = pixels.shape[0]
        labels = np.zeros((num_samples, 1))
        for i in range(num_samples):
            labels[i] = self.cluster_assignment(pixels[i], centroids)
        return labels
    
    # update the center for each cluster
    def update_centroids(self, pixels, centroids, labels):
        original_loss = 0
        new_loss = 0
        stop_flag = 0
        for i in range(self.k):
            pixels_i = pixels[labels[:, 0] == i]
            original_loss += np.sum(self.distance(centroids[i], pixels_i))
            centroids[i] = np.sum(pixels_i, axis=0) / pixels_i.shape[0]
            new_loss += np.sum(self.distance(centroids[i], pixels_i))
        if original_loss == new_loss:
            stop_flag = 1
        return new_loss, stop_flag
    
    def forward(self, pixels):
        if pixels.shape[0] <= self.k:
            print("The number of k is too big!")
            return
        
        random.seed(555)
        
        tic = time.time()
        
        centroids = self.random_initialize_center(pixels)
        cluster_labels = np.zeros((pixels.shape[0], 1))
        count = 0
        for _ in range(self.max_iter):
            cluster_labels = self.update_cluster_labels(pixels, centroids)
            loss, stop_flag = self.update_centroids(pixels, centroids, cluster_labels)
            print(loss)
            count += 1
            if stop_flag == 1:
                break

        print('iter time: {}'.format(count))
        print('time: {}'.format(time.time() - tic))
        
        cluster_labels = cluster_labels.astype(int).flatten()
        return centroids, cluster_labels

if __name__ == '__main__':
    # read bmp image file
    I = Image.open('./data/beach.bmp')
    I_array = np.array(I)

    # read jpeg image file
    I = Image.open('./data/pic1.jpg')
    I_array = np.array(I)

    # conver input image dimension shape
    I_array = np.reshape(I_array, (I_array.shape[0]*I_array.shape[1], 3))
    # print(I_array.shape) 
    # (76800, 3)

    model1 = k_medoids(5)
    centroids1, labels1 = model1.forward(I_array)
    model2 = k_means(3)
    centroids2, labels2 = model2.forward(I_array)

    print(centroids2.shape)
    print(labels2.shape)
