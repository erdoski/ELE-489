# İmport neccesary libraries
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load the dataset without headers
df = pd.read_csv('wine/wine.data', header=None)
columns = [
    'Class',
    'Alcohol',
    'Malic_acid',
    'Ash',
    'Alcalinity_of_ash',
    'Magnesium',
    'Total_phenols',
    'Flavanoids',
    'Nonflavanoid_phenols',
    'Proanthocyanins',
    'Color_intensity',
    'Hue',
    'OD280/OD315_of_diluted_wines',
    'Proline'
]
# Add column labels
df.columns = columns


# For the number of the classes find the number of train datas (%80) for each class
class_counts = []
for i in range(1,4):
    count = (df.iloc[:,0] == i).sum()
    class_counts.append(count)

class_filtered = [int(0.8 * count) for count in class_counts]

# Seperate teh data into train and test
row1 = df[0:class_filtered[0]]
row2 = df[class_counts[0]:class_counts[0]+class_filtered[1]]
row3 = df[class_counts[0]+class_counts[1]:class_counts[0]+class_counts[1]+class_filtered[2]]

train_datas = pd.concat([row1,row2,row3], ignore_index= True)

r1 = df[class_filtered[0]:class_counts[0]]
r2 = df[class_counts[0]+class_filtered[1]:class_counts[0]+class_counts[1]]
r3 = df[class_counts[0] + class_counts[1]+ class_filtered[2]:]

test_datas = pd.concat([r1,r2,r3],ignore_index= True)

# Remove the class label
train_features = train_datas.iloc[:,1:]
test_features = test_datas.iloc[:,1:]

# Normalize the test and trian datas according to the mean and std of train datas
train_mean = train_features.mean()
train_std = train_features.std()

normalized_train_data  = (train_features - train_mean)/train_std
normalized_test_data = (test_features - train_mean)/train_std

# Add class label to the train data
class_label = np.array(train_datas.iloc[:,0])
class_label = class_label.reshape(-1, 1)

labeled_norm_train = np.hstack((class_label,np.array(normalized_train_data)))

# Select k
k = 7

#Initlize the estimated class vector which contains test datas which is labeled by KNN
estimated_clas = np.zeros((normalized_test_data.shape[0],1))

for i in range(normalized_test_data.shape[0]):
    # Select a row from test data
    test = normalized_test_data.iloc[i,:]
    
    #Calculate Euclidean distance
    cost = (normalized_train_data - test)**2
    distances = np.sqrt(np.sum(cost,axis=1))  
    
    # Sort the distances ascending order
    sorted_dist = np.argsort(distances)
    
    # Select the first k indices for classification
    closest_k_indices = np.array(sorted_dist.iloc[0:k])
    
    # Find the class of corresponding k minumum distances from the train data
    closest_class_label = labeled_norm_train[closest_k_indices,0]
    
    # Find which class number is more frequent
    appended_clas = mode(closest_class_label).mode
    
    # Assing the class to test data
    estimated_clas[i,0] = appended_clas

# Select the original class of the test data
original_class = (np.array(test_datas.iloc[:,0])).reshape(-1,1)

# Print the confusion matrix and classifiaction report
conf_matrix = confusion_matrix(original_class, estimated_clas)
print(conf_matrix)
print(classification_report(original_class, estimated_clas))




