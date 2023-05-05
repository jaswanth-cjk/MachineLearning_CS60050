import numpy as np
import pandas as pd

def train(data):
    '''
    Input :
        data : The actual training data
    Output:
        pos_list : The list of dictionary of probabilities of occuring of all unique values in a column given that the Response is 1 (P(Xi|1))
        neg_list : The list of dictionary of probabilities of occuring of all unique values in a column given that the Response is 0 (P(Xi|0))
        val_list : The list of dictionary of probabilities of occuring of all unique values in a column (P(Xi))
    '''
    # Dummy value initialisation to compensate for not taking first column containing ID
    pos_list = [{}]
    neg_list = [{}]
    val_list = [{}]
    # Looping through the remaining columns except the result column
    for index in range(1, data.shape[1] - 1):
        # Dictionary initialisation
        pos_dict = {}
        neg_dict = {}
        val_dict = {}
        # Looping through the unique values in each column
        for val in data.iloc[:, index].unique():
            # Calculating probabilities as required
            pos_dict[val] = len(data[(data.iloc[:, index] == val) & (data['Response'] == 1)]) / len(data[data['Response'] == 1])
            neg_dict[val] = len(data[(data.iloc[:, index] == val) & (data['Response'] == 0)]) / len(data[data['Response'] == 0])
            val_dict[val] = len(data[data.iloc[:, index] == val]) / len(data)
        # Appending the dictionaries to their respective lists
        pos_list.append(pos_dict)
        neg_list.append(neg_dict)
        val_list.append(val_dict)
    return pos_list, neg_list, val_list

def test(data, pos_dict, neg_dict, val_dict, pos_prob_tot):
    '''
    Input :
        data : The actual testing data
        pos_dict : The list containing the dictionaries of probabilities of each unique value in a column given that the Response is 1 (P(Xi|1))
        neg_dict : The list containing the dictionaries of probabilities of each unique value in a column given that the Response is 0 (P(Xi|0))
        val_dict : The list containing the dictionaries of probabilities of each unique value in a column (P(Xi))
        pos_prob_tot : The probability of the Response being 1 (P(1))
    Output:
        accuracy : The calculated accuracy of the testing data on the trained model
    '''
    success_cnt = 0
    tot_cnt = 0
    # Initialisation for P(y)
    pos_prob = pos_prob_tot # P(1)
    neg_prob = 1 - pos_prob_tot # P(0)
    # Looping through all data for testing
    for i in range(len(data)):
        # Actual Response value
        actual_val = data['Response'].iloc[i]
        # Predicted Response value
        test_val = 0
        # Looping through all columns
        for index in range(1, data.shape[1] - 1):
            if data.iloc[i, index] in val_dict[index]:
                # Multiplying by P(Xi|y) / P(Xi)
                pos_prob *= pos_dict[index][data.iat[i, index]] / val_dict[index][data.iat[i, index]]
                neg_prob *= neg_dict[index][data.iat[i, index]] / val_dict[index][data.iat[i, index]]
        if pos_prob > neg_prob:
            test_val = 1
        else:
            test_val = 0
        if actual_val == test_val:
            success_cnt = success_cnt + 1
        tot_cnt = tot_cnt + 1
    accuracy = success_cnt / tot_cnt
    return accuracy

def train_laplace(data):
    '''
    Input :
        data : The actual training data
    Output:
        pos_list : The list of dictionary of probabilities of occuring of all unique values in a column given that the Response is 1 (P(Xi|1))
        neg_list : The list of dictionary of probabilities of occuring of all unique values in a column given that the Response is 0 (P(Xi|0))
        val_list : The list of dictionary of probabilities of occuring of all unique values in a column (P(Xi))
    '''
    # Dummy value initialisation to compensate for not taking first column containing ID
    pos_list = [{}]
    neg_list = [{}]
    val_list = [{}]
    # Looping through the remaining columns except the result column
    for index in range(1, data.shape[1] - 1):
        # Dictionary initialisation
        pos_dict = {}
        neg_dict = {}
        val_dict = {}
        unique_val = len(data.iloc[:, index].unique())
        # Looping through the unique values in each column
        for val in data.iloc[:, index].unique():
            # Calculating probabilities as required for Laplacian correction
            pos_dict[val] = (len(data[(data.iloc[:, index] == val) & (data['Response'] == 1)]) + 1) / (len(data[data['Response'] == 1]) + unique_val)
            neg_dict[val] = (len(data[(data.iloc[:, index] == val) & (data['Response'] == 0)]) + 1) / (len(data[data['Response'] == 0]) + unique_val)
            val_dict[val] = len(data[data.iloc[:, index] == val]) / len(data)
        # Appending the dictionaries to their respective lists
        pos_list.append(pos_dict)
        neg_list.append(neg_dict)
        val_list.append(val_dict)
    return pos_list, neg_list, val_list

# The actual implementation starts here

# Reading data from the given dataset
df = pd.read_csv("Dataset_C.csv")
output = open('Q2_output.txt', 'w')

# Conversion of continuous data to discrete data of ~10 unique values and categorical encoding of object types
df.loc[df['Gender'] == 'Male', 'Gender'] = 0
df.loc[df['Gender'] == 'Female', 'Gender'] = 1
df['Age'] //= 10
df['Region_Code'] //= 10
df.loc[df['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0
df.loc[df['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1
df.loc[df['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2
df.loc[df['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0
df.loc[df['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1
df.loc[df['Annual_Premium'] < 110000, 'Annual_Premium'] //= 10000
df.loc[df['Annual_Premium'] >= 110000, 'Annual_Premium'] = 11
df['Policy_Sales_Channel'] //= 20
df['Vintage'] //= 30

# Finding the outliers as per the given conditions
df["Outlier_Count"] = np.zeros(len(df), dtype = int).tolist()
for i in range(1, df.shape[1] - 1):
    mean = df.iloc[:, i].mean() # Mean calculation
    sd = df.iloc[:, i].std() # Standard Deviation calculation
    df.loc[df.iloc[:, i] > (mean + 3 * sd), "Outlier_Count"] += 1

max_outlier = max(df["Outlier_Count"])
df = df.drop(df[df["Outlier_Count"] == max_outlier].index)
del df["Outlier_Count"]

# Data for training and testing 80-20 split
train_data = df.sample(frac= 0.8)
test_data = df.drop(train_data.index)

# Split for 10-fold validation
data_buckets = np.array_split(train_data, 10)

# For picking best model trained on 90% training set and tested on validation set of 10%
pos_dict_best = []
neg_dict_best = []
val_dict_best = []
pos_prob_best = 0
best_accuracy = 0
for i in range(10):
    print("Iteration : ", i + 1, file = output)
    # Data segregation into training and validation sets
    test_data_bucket = data_buckets[i]
    train_data_bucket = train_data.drop(test_data_bucket.index)

    cur_pos_dict, cur_neg_dict, cur_val_dict = train(train_data_bucket)
    pos_prob_cur = len(train_data_bucket[train_data_bucket['Response'] == 1]) / len(train_data_bucket)
    accuracy = test(test_data_bucket, cur_pos_dict, cur_neg_dict, cur_val_dict, pos_prob_cur)
    print("The accuracy obtained after testing on the validation set is ", accuracy, file = output)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        pos_dict_best = cur_pos_dict
        neg_dict_best = cur_neg_dict
        val_dict_best = cur_val_dict
        pos_prob_best = pos_prob_cur

# Final accuracy on testing set
accuracy = test(test_data, pos_dict_best, neg_dict_best, val_dict_best, pos_prob_best)
print("\nThe final accuracy obtained after testing on the best trained model is ", accuracy, file = output)

laplace_pos_dict, laplace_neg_dict, laplace_val_dict = train_laplace(train_data)
pos_prob_best = len(train_data[train_data['Response'] == 1]) / len(train_data)
laplace_accuracy = test(test_data, laplace_pos_dict, laplace_neg_dict, laplace_val_dict, pos_prob_best)
print("\nThe accuracy obtained after testing on the model trained using Laplacian correction is ", laplace_accuracy, file = output)

output.close()