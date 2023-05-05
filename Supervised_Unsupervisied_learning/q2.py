import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import random

# Data Import
df = pd.read_csv('lung-cancer.csv', header=None)
output = open('Q2_output.txt', 'w')

# Replacing the unknown values by the mode value of that column
for index in range(1, df.shape[1]):
    df.iloc[df.iloc[:, index] == '?', index] = df.iloc[:, index].mode()[0]
    df.iloc[:, index] = pd.to_numeric(df.iloc[:, index])    # For converting any stray string data to numeric

def scaler(X):
    # Scaler function
    for index in range(X.shape[1]):
        mean_val = X.iloc[:, index].mean()
        std_val = X.iloc[:, index].std()
        X.iloc[:, index] = (X.iloc[:, index] - mean_val) / std_val
    return X

def sampler(X, frac):
    # Sampler function
    row_index = random.sample(range(X.shape[0]),int(frac * X.shape[0]))
    return X.iloc[row_index, :]

X = df.iloc[:, 1:]              # Excluding the first column (class label)
actual_val = df.iloc[:, 0]
X = scaler(X)
train_data = sampler(X, 0.8)   # Datapoints for train data
test_data = X.drop(train_data.index)   # Datapoints for test data
train_index = train_data.index.to_list()    # List of indexes used in train data
train_val = actual_val.iloc[train_index]    # List of results for train data
test_val = actual_val.drop(train_val.index) # List of results for test data

# Categorical encoding was not necessary since no string or coded data was involved

# Applying binary SVM classifier using linear kernel
linear_svc = SVC(kernel='linear')
linear_svc.fit(train_data, train_val)
linear_pred_val = linear_svc.predict(test_data)
linear_accuracy = (test_val.values == linear_pred_val).sum() / test_val.size
print("The accuracy obtained for binary SVM classifier using linear kernel : ", linear_accuracy, file = output)

# Applying binary SVM classifier using quadratic kernel
quadratic_svc = SVC(kernel='poly', degree=2)
quadratic_svc.fit(train_data, train_val)
quadratic_pred_val = quadratic_svc.predict(test_data)
quadratic_accuracy = (test_val.values == quadratic_pred_val).sum() / test_val.size
print("The accuracy obtained for binary SVM classifier using quadratic kernel : ", quadratic_accuracy, file = output)

# Applying binary SVM classifier using radial basis function kernel
rbf_svc = SVC(kernel='rbf')
rbf_svc.fit(train_data, train_val)
rbf_pred_val = rbf_svc.predict(test_data)
rbf_accuracy = (test_val.values == rbf_pred_val).sum() / test_val.size
print("The accuracy obtained for binary SVM classifier using radial basis function kernel : ", rbf_accuracy, file = output)

# Applying MLP Classifier using the parameters : stochastic gradient descent optimiser, learning rate as 0.001 and batch size of 32
# Parameters for first classifier : 1 hidden layer with 16 nodes
p1_MLP = MLPClassifier(hidden_layer_sizes=(16,), solver='sgd', learning_rate='constant',learning_rate_init=0.001, batch_size=32)
p1_MLP.fit(train_data, train_val)
p1_MLP_pred_val = p1_MLP.predict(test_data)
p1_MLP_accuracy = (test_val.values == p1_MLP_pred_val).sum() / test_val.size
print("\nThe accuracy obtained on the first MLP Classifier : ", p1_MLP_accuracy, file = output)
# Parameters for second classifier : 2 hidden layers with 256 and 16 nodes respectively
p2_MLP = MLPClassifier(hidden_layer_sizes=(256,16), solver='sgd', learning_rate='constant',learning_rate_init=0.001, batch_size=32)
p2_MLP.fit(train_data, train_val)
p2_MLP_pred_val = p2_MLP.predict(test_data)
p2_MLP_accuracy = (test_val.values == p2_MLP_pred_val).sum() / test_val.size
print("The accuracy obtained on the second MLP Classifier : ", p2_MLP_accuracy, file = output)

best_model = p1_MLP if p1_MLP_accuracy > p2_MLP_accuracy else p2_MLP    # Selecting the MLPClassifier with best accuracy
best_model_accuracy = []
learning_rate_list = []
learning_rate = 0.1
print("\nThe accuracy obtained for the respective learning rates on the best model are :", file = output)
for i in range(5):
    cur_model = best_model.set_params(learning_rate_init=learning_rate) # Changing the learning rate
    cur_model.fit(train_data, train_val)
    cur_model_pred_val = cur_model.predict(test_data)
    cur_model_accuracy = (test_val.values == cur_model_pred_val).sum() / test_val.size
    best_model_accuracy.append(cur_model_accuracy)
    learning_rate_list.append(learning_rate)
    print(learning_rate, " : ", cur_model_accuracy, file = output)
    learning_rate /= 10

# Plot of learning rate vs best model accuracy
plt.xscale("log")
plt.plot(learning_rate_list, best_model_accuracy)
plt.xlabel('Learning rate used')
plt.ylabel('Accuracy obtained')
plt.title('Learning Rate vs Accuracy')
plt.savefig('Q2_plot.png')

# Forward Feature Selection
def forward_feature_selection(train_data, test_data, train_val, test_val):
    col_select = {}
    test_df = pd.DataFrame()    # Empty dataframe
    train_df = pd.DataFrame()   # Empty dataframe
    col_train_data = {}
    col_test_data = {}
    best_acc = -1
    for i in range(train_data.shape[1]):
        # Storing column values as a list in dictionary
        col_train_data[i] = train_data[train_data.columns[i]].values.tolist()
        col_test_data[i] = test_data[test_data.columns[i]].values.tolist()
    for i in range(train_data.shape[1]):
        best_val = -1
        for j in range(train_data.shape[1]):
            if j in col_select:         # If the feature has already been selected
                continue
            train_df.insert(0, str(j), col_train_data[j])   # Add feature data
            test_df.insert(0, str(j), col_test_data[j])     # Add feature data
            best_model.fit(train_df, train_val)
            best_model_pred_val = best_model.predict(test_df)
            best_model_accuracy = (test_val.values == best_model_pred_val).sum() / test_val.size
            if best_model_accuracy >= best_acc:         # Select feature with best accuracy
                best_acc = best_model_accuracy
                best_val = j
            test_df.drop(test_df.columns[[0]], axis=1, inplace=True)    # Remove feature data
            train_df.drop(train_df.columns[[0]], axis=1, inplace=True)  # Remove feature data
        if best_val == -1:      # Case of no improvement in accuracy
            break
        col_select[best_val] = 1        # Setting the column as selected
        train_df.insert(0, str(best_val), col_train_data[best_val])     # Add feature data permanently
        test_df.insert(0, str(best_val), col_test_data[best_val])       # Add feature data permanently
    index_list = [int(i) for i in train_df.columns]
    index_list.sort()
    return index_list

index_list = forward_feature_selection(train_data, test_data, train_val, test_val)
print("\nThe list of column indexes selected after applying Forward Feature Selection :\n", index_list, file = output)

# Ensemble learning with candidate models as quadratic SVM, radial basis function SVM, best MLPClassifier
ensemble_pred_val = []
best_model_pred_val = p1_MLP_pred_val if p1_MLP_accuracy > p2_MLP_accuracy else p2_MLP_pred_val # Prediction values for best MLPClassifier
for i in range(test_data.shape[0]):
    pred_val = [quadratic_pred_val[i], rbf_pred_val[i], best_model_pred_val[i]]
    ensemble_pred_val.append(max(set(pred_val), key=pred_val.count))    # Prediction values set as mode of prediction values of candidate models
final_accuracy = (test_val.values == np.array(ensemble_pred_val)).sum() / test_val.size
print("\nThe final accuracy obtained after using ensemble learning with max voting technique is : ", final_accuracy, file = output)

output.close()