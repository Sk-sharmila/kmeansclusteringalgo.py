#libraries
import numpy as np

class matrix:
    def __init__(self, file, array_2d=[]):
        self.array_2d = array_2d
        self.file = file  # it will return matrix with the data in csv file
        self.array_2d = self.load_from_csv(file)

    def return_array_2d(self):  # it is a helper function to return the data matrix
        return self.array_2d

    def load_from_csv(self, file):  # it loads the csv file
        text = open(file, 'r')
        for i in text:
            x = i.split(',')
            self.array_2d.append([float(i) for i in x])
        return self.array_2d

    def standardise(self, array_2d):  # it standardise the matrix according to the formula in the appendix
        standardise_matrix = array_2d
        rows = len(self.array_2d)
        col = len(self.array_2d[0])
        avg, max_col, min_col = [], [], []
        for i in range(col):
            max_, min_, sum_ = 0, 100000, 0
            for j in range(rows):
                sum_ += array_2d[j][i]
                if min_ > array_2d[j][i]:
                    min_ = array_2d[j][i]
                if max_ < array_2d[j][i]:
                    max_ = array_2d[j][i]
            avg.append(sum_ / rows)
            max_col.append(max_)
            min_col.append(min_)
        for i in range(rows):
            for j in range(col):
                standardise_matrix[i][j] = (array_2d[i][j] - avg[j]) / (max_col[j] - min_col[j])
        return standardise_matrix

    def get_count_frequency(self):  # it will count the no of repeating elements in the matrix
        if len(self.array_2d[0]) == 1:
            dict_ = {}
            for i in array_2d:
                if array_2d[i][0] not in dict_:
                    dict_[array_2d[i][0]] = 0
                dict_[array_2d[i][0]] += 1
            return dict_

    def get_distance(self, other_matrix, weights, beta):
        other_matrix, weights = np.array(other_matrix), np.array(weights)
        if other_matrix.shape[0] <= 356 and weights.shape[0] < 6:
            euclidean_distances = []
            for i in range(other_matrix.shape[0]):
                for j in range(weights.shape[0]):
                    dist = np.linalg.norm(other_matrix[i] - weights[
                        j])  # euclidean distances between two first element of one matrix with rest of other matrices
                    euclidean_distances.append(dist)
            # print(euclidean_distances)
            euclidean_distances = np.array(euclidean_distances)
            return euclidean_distances





def get_initial_weights(m):
    # target is to generate the random numbers who sum to make 1 and the value should be ranging from 0 to 1
    div, i = 1 / m, 0
    res = []
    rem = 1 - (div * m)
    while i < m:
        if i == m - 1:
            div = div + rem
        res.append(div)
        i += 1
    return res

    def get_groups(object_, K, beta):
        array_2d = object_.return_array_2d()  # it will return an array containing the data in the csv file

    S = []
    for i in range(len(array_2d)):  # creating an array with the rows in the array_2d matrix
        S.append([0])
    standardise_matrix = m.standardise(array_2d)
    centroids = get_centroids(standardise_matrix, S, K)
    eucli_dist = object_.get_distance(standardise_matrix, centroids, beta)

    for i in range(len(centroids)):
        nearest, index = 1, 0
        for j in range(len(centroids)):
            for k in range(len(centroids[0])):
                if nearest > centroids[j][k]:
                    nearest = centroids[j][k]
                    index = j
        S[j][0] = j
    for i in range(len(standardise_matrix[0])):  # generating new weigths of the matrix which sum to 1
        weights = get_new_weights(standardise_matrix, centroids, S, K, beta)

    return object_


def create_centroids(array_2d, K):
    centroids = []
    D = random.randint(K, size=(K))  # generatin some random k value and storing them in the list
    for i in D:
        centroids.append(array_2d[i])  # copying  the random rows from the data matrix to the centroids
    return centroids


def get_centroids(standardise_matrix, s, K):
    # making centroids
    res = create_centroids(standardise_matrix, K)
    rows = len(standardise_matrix)
    col = len(standardise_matrix[0])
    mean_ = []  # calculating mean for each column
    for j in range(col):
        sum_ = 0
        for i in range(rows):
            sum_ += standardise_matrix[i][j]
        mean_.append(sum_ / rows)
    for i in range(K):
        for j in range(len(res[i])):
            if res[i][j] == mean_[j] and s[i][j] == K:  # checking the centroids with equal column mean and the s matrix element with the value of k
                res[i][j] = K  # updatin the the s matrix each value with the if the above the condition is true
    return res


def get_new_weights(standardise_matrix, centroids, S, K, beta):
    new_weights = []
    for i in range(K):
        cal = 1
        for j in range(len(standardise_matrix)):
            if S[i] == K:  # implementing the calculating weights formula when the we find the element which is equal to k
                u = 1
                cal = cal * u(standardise_matrix[i][j] - centroids[K][j]) ** 2
            else:
                u = 0
                w = 0
    if cal == 0:
        w = 0
        return w
    else:
        w = 1
        for i in range(1, len(
                standardise_matrix[0])):  # implementing the formula when the value of cal is not equal to zero
            w = w + (cal / i) ** (1 / (beta - 1))
        return w


def run_test():
    m = matrix('data_cluster.csv')
    for k in range(2, 5):
        for beta in range(11, 25):
            m = get_groups(m, k, beta / 10)
            print(str(k) + '-' + str(beta) + '=' + str(m.get_count_frequency()))