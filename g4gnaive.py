# Importing library 
import math 
import random 
import csv 
import sklearn
import sklearn.decomposition
from sklearn.decomposition import PCA
import numpy as np
import sklearn
from sklearn.model_selection import KFold 
from sklearn import preprocessing
from scipy.stats import norm
import matplotlib.pyplot as plt


epsilon = 1e-12
# the categorical class names are changed to numberic data 
# eg: yes and no encoded to 1 and 0
def plot(x, y, save_path, title):
	# Plot dataset
    plt.figure()
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('1st Feature')
    plt.ylabel('2nd Feature')
    plt.legend(['No death', 'Death'])
    plt.title(title)
    # plt.show()
    plt.savefig(save_path)
    plt.close()

def plot_hist(x, save_path, title) :
	arr = np.arange(len(x))
	plt.bar(arr, height=x)
	xtic = []
	for i in range(1,len(x)+1) : 
		xtic.append('{}'.format(i))
	plt.xticks(arr, xtic)
	plt.title(title)
	plt.savefig(save_path)
	plt.close()


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
	# if sigma == 0 :
		# return 0
	ll = math.log(2*math.pi*(sigma**2) + epsilon)/2 + ((data-mu)**2)/(2 * (sigma**2) + epsilon)
	return -ll

# Calculate Class Probabilities 
def calculateClassLogProbabilities(info, logBerProb, test): 
	probabilities = {}
	to_Remove_Ft_list = []
	for classValue, classSummaries in info.items(): 
		probabilities[classValue] = 0
		for i in range(len(classSummaries)): 
			mean, std_dev = classSummaries[i]
			if(std_dev == 0) :
				to_Remove_Ft_list.append(i)
				# print("Std_Dev = 0; classValue : {}; feature number : {}".format(classValue, i))
			x = test[i]
			probabilities[classValue] += calculateGaussianLogProbability(x, mean, std_dev)
		probabilities[classValue] += logBerProb[int(classValue)]
	to_Remove_Ft_list.sort()
	theSet= set(to_Remove_Ft_list)
	to_Remove_Ft_list = list(theSet)
	# print(len(to_Remove_Ft_list))
	# exit()
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
	
	normalizer = preprocessing.Normalizer()
	train_X_np = normalizer.fit_transform(train_X_np)
	test_X_np = normalizer.transform(test_X_np)
	# mean = np.mean(train_X_np, axis=0)
	# std = np.std(train_X_np, axis=0) + epsilon
	# train_X_np = (train_X_np - mean)/std
	# test_X_np = (test_X_np - mean)/std
	train_data = np.concatenate((train_X_np, train_Y_np), axis=1).tolist()
	test_data = np.concatenate((test_X_np, test_Y_np), axis=1).tolist()

	return train_data, test_data

def normalize_data_full_data(data) :
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

def apply_pca(train_data, test_data, count) :
	train_data_np = np.array(train_data)
	test_data_np = np.array(test_data)

	train_X_np = train_data_np[:,:-1]  
	train_Y_np = train_data_np[:,-1].reshape((train_data_np.shape[0],1))
	test_X_np = test_data_np[:,:-1]
	test_Y_np = test_data_np[:,-1].reshape((test_data_np.shape[0],1))


	# pca = PCA()
	pca = PCA(n_components=0.95)
	# pca = PCA(n_components=5)
	train_X_np_pca = pca.fit_transform(train_X_np)
	test_X_np_pca = pca.transform(test_X_np)
	# print(test_X_np_pca.shape)
	# print(train_X_np_pca.shape)
	plot(x=test_X_np_pca[:,:2],y=test_Y_np.reshape(-1),save_path="Projected Features {}.".format(count),title="Projected Features in the first two dimension.")
	train_data = np.concatenate((train_X_np_pca, train_Y_np), axis=1).tolist()
	test_data = np.concatenate((test_X_np_pca, test_Y_np), axis=1).tolist()

	pca_plot = PCA()
	pca_plot.fit_transform(train_X_np)
	# plot_hist(pca_plot.explained_variance_ratio_, save_path="PCA Graph Explained Variance Ratio", title="Explained Variance Ratio")
	plot_hist(pca.explained_variance_ratio_, save_path="PCA Graph Explained Variance Ratio of Selected Components {}".format(count), title="Explained Variance Ratio of Selected Components")

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

def InbuiltGaussianNb(train_data, test_data) : 
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()
	train_data_np = np.array(train_data)
	test_data_np = np.array(test_data)

	train_X_np = train_data_np[:,:-1]  
	train_Y_np = train_data_np[:,-1]
	test_X_np = test_data_np[:,:-1]
	test_Y_np = test_data_np[:,-1]
	clf.fit(train_X_np , train_Y_np)
	
	test_y_predicted = clf.predict(test_X_np)
	tf = [test_y_predicted==test_Y_np]
	tf = np.array(tf[0])
	accuracy = np.sum(tf)/tf.shape[0]

	train_y_predicted = clf.predict(train_X_np)
	tf = [train_y_predicted==train_Y_np]
	tf = np.array(tf[0])
	accuracy_train = np.sum(tf)/tf.shape[0]

	return accuracy, accuracy_train


def k_fold_cv(train_data, pca=False, use_inbuilt_nb=False):
	k = 1
	print("Performing 5 fold cross validation : ")
	kfold = KFold(5, shuffle=True)
	avg_accuracy = 0
	avg_accuracy_train = 0
	count = 0
	for train, valid in kfold.split(train_data):
		count += 1
		i_fold_train = []
		i_fold_valid = []
		for i in train:
			i_fold_train.append(train_data[i])
		for i in valid:
			i_fold_valid.append(train_data[i])
		# i_fold_train = train_data[train]
		# i_fold_valid = train_data[valid]
		i_fold_train, i_fold_valid = normalize_data(i_fold_train, i_fold_valid)
		if pca :
			i_fold_train, i_fold_valid = apply_pca(i_fold_train, i_fold_valid, count)
			print("Number of features selected : {}".format(len(i_fold_train[0])-1))

		if use_inbuilt_nb : 
			accuracy, accuracy_train = InbuiltGaussianNb(i_fold_train, i_fold_valid)
		else :
			info = MeanAndStdDevForClass(i_fold_train)
			logBerProb = berProb(i_fold_train)
			predictions_train = getPredictions(info, logBerProb, i_fold_train) 
			predictions = getPredictions(info, logBerProb, i_fold_valid) 
			accuracy_train = accuracy_rate(i_fold_train, predictions_train) 
			accuracy = accuracy_rate(i_fold_valid, predictions) 
		avg_accuracy += accuracy
		avg_accuracy_train += accuracy_train 
		print("Accuracy on validation set on",k,"th fold is: ",accuracy)
		# print("Accuracy on training set on",k,"th fold is: ",accuracy_train)
		# print(predictions)
		k = k + 1
	avg_accuracy /= 5
	avg_accuracy_train /= 5
	print("Average Accuracy on validation set on : ",avg_accuracy)
	print("Average Accuracy on training set on : ",avg_accuracy_train)

		
# driver code 
# add the data path in your system 
filename = r'after_prepoc.csv'
use_inbuilt_nb = False

# load the file and store it in mydata list 
mydata = csv.reader(open(filename, "rt")) 
mydata = list(mydata)
mydata = encode_class(mydata) 
for i in range(len(mydata)): 
	mydata[i] = [float(x) for x in mydata[i]] 
# mydata = normalize_data_2(mydata)

	

k_fold_cv(mydata, pca=False, use_inbuilt_nb=use_inbuilt_nb)
k_fold_cv(mydata, pca=True, use_inbuilt_nb=use_inbuilt_nb)
exit()