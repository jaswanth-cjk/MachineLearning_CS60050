import pandas as pd
import matplotlib.pyplot as plt
import math

class Node:
    def __init__(self, isLeaf, value, best_val = -1):
        if isLeaf == False:
            self.index = value # The value which is taken for splitting
            self.isLeaf = False
            self.best_val = best_val # The max probability value at that node before the splitting
            self.child = []
        else:
            self.isLeaf = True
            self.best_val = value # The value of the leaf node

# Function for calculating the Information Gain base on the split on given data and column index
def IG(data, index):
    weighted_entropy = 0
    tot_cnt = len(data)
    for val in data.iloc[:, index].unique():
        data_val = data[data.iloc[:, index] == val]
        cnt_val = len(data_val)
        pos_cnt = len(data_val[data_val['Response'] == 1])
        neg_cnt = cnt_val - pos_cnt
        if pos_cnt == 0 or neg_cnt == 0:
            pos_entropy = 0
        else:
            pos_entropy = math.log2(pos_cnt / cnt_val) * pos_cnt / cnt_val + math.log2(neg_cnt / cnt_val) * neg_cnt / cnt_val
        weighted_entropy += pos_entropy * cnt_val / tot_cnt
    cnt_val = len(data)
    pos_cnt = len(data[data['Response'] == 1])
    neg_cnt = cnt_val - pos_cnt
    if pos_cnt == 0 or neg_cnt == 0:
        pos_entropy = 0
    else:
        pos_entropy = math.log2(pos_cnt / cnt_val) * pos_cnt / cnt_val + math.log2(neg_cnt / cnt_val) * neg_cnt / cnt_val
    weighted_entropy -= pos_entropy * cnt_val / tot_cnt
    return weighted_entropy

# ID3 Function to build tree using the best Information Gain (recursive if it is a non-leaf node)
def ID3(data, cur_val, cur_index, attr, parent):
    cnt_val = len(data)
    pos_cnt = len(data[data['Response'] == 1])
    neg_cnt = cnt_val - pos_cnt
    if pos_cnt == 0:
        curr_node = Node(True, 0)
    elif neg_cnt == 0:
        curr_node = Node(True, 1)
    elif len(attr) == 0:
        if pos_cnt > neg_cnt:
            curr_node = Node(True, 1)
        else:
            curr_node = Node(True, 0)
    else:
        best_IG = -1
        best_index = -1
        for index in attr:
            cur_IG = IG(data, index)
            if cur_IG > best_IG:
                best_IG = cur_IG
                best_index = index
        attr.remove(best_index)
        best_val = -1
        if pos_cnt > neg_cnt:
            best_val = 1
        else:
            best_val = 0
        curr_node = Node(False, best_index, best_val)
        for val in data.iloc[:, best_index].unique():
            data_val = data[data.iloc[:, best_index] == val]
            ID3(data_val, val, best_index, attr, curr_node)
        attr.append(best_index)
    parent.child.append((curr_node, cur_val, cur_index))

# Function for testing accuracy
def check(data, parent):
    if parent.isLeaf == True:
        return parent.best_val
    for child in parent.child:
        if child[1] == data[child[2]]:
            return check(data, child[0])
    return parent.best_val

# Function for testing accuracy
def test(data, parent):
    tot_cnt = 0
    success_cnt = 0
    for i in range(len(data)):
        row_data = data.values[i]
        value = check(row_data, parent)
        if value == row_data[-1]:
            success_cnt = success_cnt + 1
        tot_cnt = tot_cnt + 1
    accuracy = success_cnt / tot_cnt
    return accuracy

def DFS_depth(cur_node):        # DFS for finding the depth of the tree
    if cur_node.isLeaf == True:
        return 0
    mx = 0
    for child in cur_node.child:
        mx = max(mx, DFS_depth(child[0]))
    return 1 + mx

def DFS_pruning(parent, data, head):      # Reduced error pruning
    for child in parent.child:
        if child[0].isLeaf == False:
            DFS_pruning(child[0], data, head)
    cur_accuracy = test(data, head)
    child_list = parent.child
    parent.child = []
    parent.isLeaf = True
    par_accuracy = test(data, head)
    if cur_accuracy > par_accuracy:
        parent.child = child_list
        parent.isLeaf = False

# Function for testing accuracy at a particular depth
def depth_acc_check(data, parent, cur_depth, given_depth):
    if cur_depth == given_depth:
        return parent.best_val
    if parent.isLeaf == True:
        return parent.best_val
    for child in parent.child:
        if child[1] == data[child[2]]:
            return depth_acc_check(data, child[0], cur_depth + 1, given_depth)
    return parent.best_val

# Function for testing accuracy at a particular depth
def depth_acc_test(data, parent, cur_depth, given_depth):
    tot_cnt = 0
    success_cnt = 0
    for i in range(len(data)):
        row_data = data.values[i]
        value = depth_acc_check(row_data, parent, cur_depth, given_depth)
        if value == row_data[-1]:
            success_cnt = success_cnt + 1
        tot_cnt = tot_cnt + 1
    accuracy = success_cnt / tot_cnt
    return accuracy

# Function for pruning the tree
def print_tree(head, df, output, level = 0):
    if head.isLeaf == False:
        print("    " * level, "Split : ", df.columns[head.index], ", Best value : ", head.best_val, sep='', file = output)
        for child in head.child:
            print_tree(child[0], df, output, level + 1)
    else:
        print("    " * level, "Best value : ", head.best_val, sep='', file = output)

# The actual implementation starts here

# Reading data from the given dataset
df = pd.read_csv('Dataset_C.csv')
output = open('Q1_output.txt', 'w')

# Conversion of continuous data to discrete data of 2 unique values for quick tree building, pruning and testing against huge dataset
df['Age'] //= 50
df['Region_Code'] //= 30
df.loc[df['Annual_Premium'] < 60000, 'Annual_Premium'] = 0
df.loc[df['Annual_Premium'] >= 60000, 'Annual_Premium'] = 1
df['Policy_Sales_Channel'] //= 100
df['Vintage'] //= 150

# attr contains the parameters which are still available for pruning
attr = []
for i in range (1, df.shape[1] - 1):
    attr.append(i)

best_accuracy = 0
best_depth = -1
best_head = None
best_validate_data = None
best_test_data = None
for count in range(10):
    # Splitting the data into 80%-20% for training data and testing data respectively
    print("Iteration : ", count + 1)
    train_data = df.sample(frac = 0.8)
    test_data = df.drop(train_data.index)
    validate_data = train_data.sample(frac = 0.1)
    train_data = train_data.drop(validate_data.index)

    head = Node(False, -1)
    ID3(train_data, -1, -1, attr, head)
    accuracy = test(test_data, head.child[0][0])
    depth = DFS_depth(head.child[0][0])

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth
        best_head = head
        best_validate_data = validate_data
        best_test_data = test_data

print("The best test accuracy obtained so far : ", best_accuracy, file = output)
print("The depth of the tree is : ", best_depth, file = output)

# Reduced error pruning
DFS_pruning(best_head.child[0][0], best_validate_data, best_head.child[0][0])
best_depth = DFS_depth(best_head.child[0][0])

# Processing for accuracy vs depth plot
acc_data = []
depth_data = []
for i in range(best_depth + 1):
    acc_data.append(depth_acc_test(best_test_data, best_head.child[0][0], 0, i))
    depth_data.append(i)

# Plot
plt.plot(depth_data, acc_data, marker = 'o')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.title('Variation of test accuracy with varying depth')
plt.savefig('Q1_plot.png')

# Printing the tree
print_tree(best_head.child[0][0], df, output)
output.close()