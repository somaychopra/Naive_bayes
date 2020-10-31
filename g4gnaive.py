# Importing library 
import math 
import random 
import csv 
import sklearn
import sklearn.decomposition
from sklearn.decomposition import PCA
import numpy as np

epsilon = 1e-6
# the categorical class names are changed to numberic data 
# eg: yes and no encoded to 1 and 0 
def encode_class(mydata): 
	classes = [] 
	for i in range(len(mydata)): 
		if mydata[i][-1] not in classes: 
			classes.append(mydata[i][-1])
	# print(len(classes)) 
	for i in range(len(classes)): 
		for j in range(len(mydata)): 
			if mydata[j][-1] == classes[i]:
				mydata[j][-1] = i 
	return mydata			 
			

# Splitting the data 
def splitting(mydata, ratio): 
	train_num = int(len(mydata) * ratio) 
	train = [] 
	# initally testset will have all the dataset 
	test = list(mydata) 
	while len(train) < train_num: 
		# index generated randomly from range 0 
		# to length of testset 
		index = random.randrange(len(test)) 
		# from testset, pop data rows and put it in train 
		train.append(test.pop(index)) 
	return train, test 


# Group the data rows under each class yes or 
# no in dictionary eg: dict[yes] and dict[no] 
def groupUnderClass(mydata): 
	dict = {} 
	for i in range(len(mydata)): 
		if (mydata[i][-1] not in dict): 
			dict[mydata[i][-1]] = [] 
		dict[mydata[i][-1]].append(mydata[i]) 
	return dict


# Calculating Mean 
def mean(numbers): 
	return sum(numbers) / float(len(numbers)) 

# Calculating Standard Deviation 
def std_dev(numbers): 
	avg = mean(numbers) 
	variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1) 
	return math.sqrt(variance) 

def MeanAndStdDev(mydata): 
	info = [(mean(attribute), std_dev(attribute)) for attribute in zip(*mydata)] 
	# eg: list = [ [a, b, c], [m, n, o], [x, y, z]] 
	# here mean of 1st attribute =(a + m+x), mean of 2nd attribute = (b + n+y)/3 
	# delete summaries of last class 
	del info[-1] 
	return info 

# find Mean and Standard Deviation under each class 
def MeanAndStdDevForClass(mydata): 
	info = {} 
	dict = groupUnderClass(mydata) 
	for classValue, instances in dict.items():
		info[classValue] = MeanAndStdDev(instances)
	return info 


# Calculate Gaussian Probability Density Function 
def calculateGaussianProbability(x, mean, stdev): 
	expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2) + epsilon))) 
	return (1 / (math.sqrt(2 * math.pi) * stdev + epsilon)) * expo 

def calculateGaussianLogProbability(data, mu, sigma) :
	# print(2*math.pi*(sigma**2))
	ll = math.log(2*math.pi*(sigma**2) + epsilon)/2 + ((data-mu)**2)/(2 * (sigma**2) + epsilon)
	return -ll

# Calculate Class Probabilities 
def calculateClassLogProbabilities(info, logBerProb, test): 
	probabilities = {}
	for classValue, classSummaries in info.items(): 
		probabilities[classValue] = 0
		for i in range(len(classSummaries)): 
			mean, std_dev = classSummaries[i]
			x = test[i]
			probabilities[classValue] += calculateGaussianLogProbability(x, mean, std_dev)
		probabilities[classValue] += logBerProb[int(classValue)]
	# print(probabilities)
	return probabilities 


# Make prediction - highest probability is the prediction 
def predict(info, logBerProb, test): 
	probabilities = calculateClassLogProbabilities(info, logBerProb, test) 
	bestLabel, bestProb = None, -float('inf')
	for classValue, probability in probabilities.items(): 
		if bestLabel is None or probability > bestProb: 
			bestProb = probability 
			bestLabel = classValue
	return bestLabel 


# returns predictions for a set of examples 
def getPredictions(info, logBerProb, test): 
	predictions = []
	for i in range(len(test)): 
		result = predict(info, logBerProb, test[i]) 
		predictions.append(result) 
	return predictions 

# Accuracy score 
def accuracy_rate(test, predictions): 
	correct = 0
	for i in range(len(test)): 
		if test[i][-1] == predictions[i]: 
			correct += 1
	return (correct / float(len(test))) * 100.0

def normalize_data(train_data, test_data) :
	train_data_np = np.array(train_data)
	test_data_np = np.array(test_data)
	train_X_np = train_data_np[:,:-1]  
	train_Y_np = train_data_np[:,-1].reshape((train_data_np.shape[0],1))
	test_X_np = test_data_np[:,:-1]
	test_Y_np = test_data_np[:,-1].reshape((test_data_np.shape[0],1))
	
	mean = np.mean(train_X_np, axis=0)
	std = np.std(train_X_np, axis=0) + epsilon
	train_X_np = (train_X_np - mean)/std
	# train_X_np /= std
	test_X_np = (test_X_np - mean)/std
	# test_X_np /= std
	train_data = np.concatenate((train_X_np, train_Y_np), axis=1).tolist()
	test_data = np.concatenate((test_X_np, test_Y_np), axis=1).tolist()

	return train_data, test_data

def normalize_data_2(data) :
	data_np = np.array(data)
	X_np = data_np[:,:-1]  
	Y_np = data_np[:,-1].reshape((data_np.shape[0],1))
	
	mean = (np.mean(X_np, axis=0))
	std = (np.std(X_np, axis=0) + epsilon)
	X_np -= mean
	X_np /= std
	data = np.concatenate((X_np, Y_np), axis=1).tolist()

	return data

def augment_data(mydata) :
	mydata_np = np.array(mydata)
	for i in range(mydata_np.shape[0]) :
		if mydata_np[i,-1] == 1 : 
			mydata_np = np.concatenate((mydata_np, mydata_np[i,:].reshape(1,-1)), axis=0)
	mydata = mydata_np.tolist()
	return mydata

def apply_pca(train_data, test_data) :
	train_data_np = np.array(train_data)
	test_data_np = np.array(test_data)

	train_X_np = train_data_np[:,:-1]  
	train_Y_np = train_data_np[:,-1].reshape((train_data_np.shape[0],1))
	test_X_np = test_data_np[:,:-1]
	test_Y_np = test_data_np[:,-1].reshape((test_data_np.shape[0],1))

	std = (np.std(train_X_np, axis=0) + epsilon)
	# train_X_np /= std
	# test_X_np /= std

	# pca = PCA()
	# pca = PCA(n_components=0.95, svd_solver='full')
	pca = PCA(n_components=5)
	train_X_np_pca = pca.fit_transform(train_X_np)
	test_X_np_pca = pca.transform(test_X_np)
	print(test_X_np_pca.shape)
	print(train_X_np_pca.shape)

	train_data = np.concatenate((train_X_np_pca, train_Y_np), axis=1).tolist()
	test_data = np.concatenate((test_X_np_pca, test_Y_np), axis=1).tolist()

	return train_data, test_data


def berProb(train_data) :
	train_data_np = np.array(train_data)
	train_data_np = train_data_np[:,-1]
	# print(train_data_np.shape[0])
	# print(np.sum(train_data_np, axis=0))
	p = np.sum(train_data_np, axis=0)/train_data_np.shape[0]
	logBerProb = []
	logBerProb.append(np.log(1-p))
	logBerProb.append(np.log(p))
	return logBerProb

def k_fold_cv(train_data):
	k = 1
	print("Performing 5 fold cross validation : ")
	kfold = KFold(5)
	for train, valid in kfold.split(train_data):
		i_fold_train = []
		i_fold_valid = []
		for i in train:
			i_fold_train.append(train_data[i])
		for i in valid:
			i_fold_valid.append(train_data[i])
		#i_fold_train = train_data[train]
		#i_fold_valid = train_data[valid]
		info = MeanAndStdDevForClass(i_fold_train) 
		predictions = getPredictions(info, i_fold_valid) 
		accuracy = accuracy_rate(i_fold_valid, predictions) 
		print("Accuracy on validation set on",k,"th fold is: ", accuracy)
		print(predictions)
		k = k + 1 
		
# driver code 
# add the data path in your system 
filename = r'after_prepoc.csv'


# load the file and store it in mydata list 
mydata = csv.reader(open(filename, "rt")) 
mydata = list(mydata)
mydata = encode_class(mydata) 
for i in range(len(mydata)): 
	mydata[i] = [float(x) for x in mydata[i]] 
# mydata = normalize_data_2(mydata)

	
# 80% of data is training data and 20% is test data used for testing
ratio = 0.8
# mydata = augment_data(mydata)

train_data, test_data = splitting(mydata, ratio) 
# train_data = augment_data(train_data)

train_data, test_data = normalize_data(train_data, test_data)
train_data, test_data = apply_pca(train_data, test_data)


print('Total number of examples are: ', len(mydata)) 
print('Out of these, training examples are: ', len(train_data)) 
print("Test examples are: ", len(test_data)) 


print(len(train_data[0]))
print(len(test_data[0]))
# train_data = np.concatenate((train_X_np, train_Y_np), axis=1).tolist()
# test_data = np.concatenate((test_X_np, test_Y_np), axis=1).tolist()


# prepare model 
info = MeanAndStdDevForClass(train_data)
logBerProb = berProb(train_data)

# test model 
predictions = getPredictions(info, logBerProb, test_data)
print(predictions)
accuracy = accuracy_rate(test_data, predictions) 
print("Accuracy of your model is: ", accuracy) 
