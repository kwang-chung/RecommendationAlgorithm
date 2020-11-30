import pandas as pd
import numpy as np
from numpy.linalg import norm
import copy
import sys
import os.path
import math
import random
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds



def cos_sim(x,y):
    return np.dot(x,y)/(norm(x) * norm(y))

def cos_sim_acc(training_data, input_data, num_recommendation, training_set = 0):
    training_data = copy.copy(training_data)
    training_data = training_data.reset_index(drop=True)
    
    training_data = training_data.drop(['channel_title'], axis = 1)
    training_decision = training_data['decision']
    training_data = training_data.drop(['decision'], axis = 1)

    input_data = input_data.drop('decision')
    input_data = input_data.drop('channel_title')


    similarity = np.zeros(len(training_data))


    for i in range(len(training_data)):
        similarity[i] = cos_sim(training_data.loc[i], input_data)
    training_data['cos_sim'] = similarity
    training_data['decision'] = training_decision
    training_data = training_data.sort_values(by=['cos_sim'], axis = 0, ascending = False).reset_index(drop=True)
    if training_set == 1:
        training_data = training_data.drop(0).reset_index(drop=True)
    s = 0
    count = 0
    output = np.zeros(5)
    
    for i in range(num_recommendation):
        s += training_data.loc[i, 'decision']
        if (i == 0) or (i == 2) or (i == 4) or (i == 6) or (i == 9):
                if (s >= (num_recommendation/2)):
                    output[count] = 1
                else:
                    output[count] = 0
                count += 1
    return output
    

def item_based_collaborative(training_data, test_data):

    x_val = [1, 3, 5, 7, 10]
    training_count = np.zeros(len(x_val))
    test_count = np.zeros(len(x_val))
    training_acc = np.zeros(len(x_val))
    test_acc = np.zeros(len(x_val))

        
    for i in range(len(training_data)):
        test = copy.copy(training_data)
        temp = cos_sim_acc(test, training_data.loc[i], 10, training_set = 1)
        for j in range(len(x_val)):
            if temp[j] == training_data.loc[i, 'decision']:
                training_count[j] += 1
                
    for i in range(len(test_data)):
        temp = cos_sim_acc(test_data, test_data.loc[i], 10)
        for j in range(len(x_val)):
            if temp[j] == test_data.loc[i, 'decision']:
                test_count[j] += 1
    
    print("Item based Collaborative Filtering - cosine similarity")
    for i in range(len(x_val)):
        training_acc[i] = round(training_count[i] / len(training_data),2)
        test_acc[i] = round(test_count[i] / len(test_data), 2)
        
        print("When the number of recommendation is " + str(x_val[i]) + ", ")
        print("the accuracy of training data is " + str(training_acc[i]))
        print("the accuracy of test data is " + str(test_acc[i]))
        
    #graph
    
    plt.plot(x_val, training_acc, label="training data")
    plt.plot(x_val, test_acc, label="test data")
    plt.xlabel('The number of recommendation')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Item based Collaborative Filtering - cosine similarity')
    plt.legend()
    plt.show()            

def diff(x, y):
    add = 0
    for i in range(len(x)):
        add += abs(x[i] - y[i])
    return add        
        
def match(training_data, col_name, input_data, num_recommendation):
    matched = 0
    output = np.zeros(5)
    count = 0
    
    training_data = training_data.reset_index(drop=True)
    
    training_data = training_data.drop(['channel_title'], axis = 1)
    training_decision = training_data['decision']
    training_data = training_data.drop(['decision'], axis = 1)
    
    input_data = input_data.drop('decision')
    input_data = input_data.drop('channel_title')

    dis = np.zeros(len(training_data))
    for i in range(len(training_data)):
        dis[i] = diff(training_data.loc[i, :], input_data)
    

    training_data['distance'] = dis
    training_data['decision'] = training_decision
    training_data = training_data.sort_values(by=['distance'], axis = 0, ascending = True).reset_index(drop=True)
    

    matched = 0
    output = np.zeros(5)
    count = 0
    for i in range(num_recommendation):

        matched += training_data.loc[i, 'decision']
        if (i == 0) or (i == 2) or (i == 4) or (i == 6) or (i == 9):
            if (matched >= (num_recommendation/2)):
                output[count] = 1
            else:
                output[count] = 0
            count += 1
    return output

    
def latent_factor(training_data, test_data):
    
    # SVD
    training_decision = training_data['decision']
    training_temp = training_data.drop(['decision'], axis = 1)
    col_name = training_temp.columns
    training_temp = training_temp.transpose()
    
    
    matrix = np.zeros([len(training_temp.columns), 5])
    
    # make matrix base one the rating
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = training_temp[i][j+1]
    
    
    
    # # mean of rating
    rating_mean = np.mean(matrix, axis=1)
    # # each rating of title - mean of rating
    matrix_rating_mean = matrix - rating_mean.reshape(-1,1)
    
    
    U, s, Vt = svds(matrix_rating_mean, k = 4)
    
    #recontruct SVD
    Sigma = np.diag(s)
    svd_matrix_rating_mean = np.dot(np.dot(U,Sigma), Vt) + rating_mean.reshape(-1,1)
    
    after_svd_data = pd.DataFrame(columns = col_name)
    after_svd_data['channel_title'] = training_data['channel_title']
    col_name = col_name[1:]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            after_svd_data.at[i, col_name[j]] = matrix[i][j]
    after_svd_data['decision'] = training_decision
    
    
    
    #num_recommendation
    x_val = [1, 3, 5, 7, 10]
    training_count = np.zeros(len(x_val))
    test_count = np.zeros(len(x_val))
    training_acc = np.zeros(len(x_val))
    test_acc = np.zeros(len(x_val))
    
    for i in range(len(training_data)):
        # print(i)
        my_decision = match(after_svd_data, col_name, training_data.loc[i,:], 10)
        for j in range(len(x_val)):
            if my_decision[j] == training_data.loc[i,'decision']:
                training_count[j] += 1
    
    for i in range(len(test_data)):
        my_decision = match(after_svd_data, col_name, test_data.loc[i,:], 10)
        for j in range(len(x_val)):
            if my_decision[j] == test_data.loc[i,'decision']:
                test_count[j] += 1
    print("Latent Factor Collaborative Filtering - SVD")
    for i in range(len(x_val)):
        training_acc[i] = round(training_count[i] / len(training_data),2)
        test_acc[i] = round(test_count[i] / len(test_data), 2)
        
        print("When the number of recommendation is " + str(x_val[i]) + ", ")
        print("the accuracy of training data is " + str(training_acc[i]))
        print("the accuracy of test data is " + str(test_acc[i]))
                                                                     
                                                                        
    #graph
    
    plt.plot(x_val, training_acc, label="training data")
    plt.plot(x_val, test_acc, label="test data")
    plt.xlabel('The number of recommendation')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Latent Factor Collaborative Filtering - SVD')
    plt.legend()
    plt.show()


# Read into pandas dataframe
data = pd.read_csv("USvideos.csv") 

#  Drop data 
data = data.drop([ 'video_id', 'trending_date', 'title', 'publish_time', 'description', 'thumbnail_link'], axis = 1)

# Reorganize tags by have or have not
for i, row in data.iterrows():
    tag_val = 1
    if row['tags'] == "[none]":
        tag_val = 0
    data.at[i,'tags'] = tag_val
temp = ['comments_disabled', 'ratings_disabled', 'video_error_or_removed']
data = data.drop(temp, axis = 1)

# Cut the views column into 2
bin_result = pd.cut(data['views'], [0, 682000, 225000000], labels=[0, 1])
data['views'] = bin_result.tolist()
data = data.rename(columns={'views': 'decision'})


max_likes = max(data['likes'])
# Split likes, dislikes, and comment count into bins by quantiles
data['likes'] = pd.qcut(data['likes'], 10, labels = False)


data['dislikes'] = pd.qcut(data['dislikes'], 10, labels = [9,8,7,6,5,4,3,2,1,0])
data['comment_count'] = pd.qcut(data['comment_count'], 10, labels = False)

max_category = max(data['category_id'])
num_cagegory = np.zeros(max_category + 1)
for i in range(max_category + 1):
    num = len(data[data['category_id'] == i])
    num_cagegory[i] = num
    
for i, row in data.iterrows():
    data.at[i,'category_id'] = num_cagegory[row['category_id']]
data['category_id'] = pd.qcut(data['category_id'], 4, labels = False)



test_data = data.sample(random_state = 69, frac = 0.2).reset_index(drop=True)
training_data = data.drop(test_data.index).reset_index(drop=True)

# item_based_collaborative filtering - cosine similarity

item_based_collaborative(training_data, test_data)

# Latent_factor collaborative filtering - SVD

latent_factor(training_data, test_data)





