# Importing library 
import math 
import random 
import csv 
import copy
import sklearn
import sklearn.decomposition
from sklearn.decomposition import PCA
import numpy as np
import sklearn
from sklearn.model_selection import KFold 
from sklearn import preprocessing
from scipy.stats import norm
import matplotlib.pyplot as plt
import argparse


epsilon = 1e-12
parser = argparse.ArgumentParser(description='All arguments for the program.')
parser.add_argument('--use_inbuilt_nb', type=None, default=False, help='To use inbuilt naive bayes function.')
args = parser.parse_args()
use_inbuilt_nb = args.use_inbuilt_nb


def normalize_data(train_data, val_data, test_data) :
	train_data_np = np.array(train_data)
	test_data_np = np.array(test_data)
	val_data_np = np.array(val_data)
	train_X_np = train_data_np[:,:-1]  
	train_Y_np = train_data_np[:,-1].reshape((train_data_np.shape[0],1))
	test_X_np = test_data_np[:,:-1]
	test_Y_np = test_data_np[:,-1].reshape((test_data_np.shape[0],1))
	val_X_np = val_data_np[:,:-1]
	val_Y_np = val_data_np[:,-1].reshape((val_data_np.shape[0],1))
	normalizer = preprocessing.Normalizer()
	train_X_np = normalizer.fit_transform(train_X_np)
	test_X_np = normalizer.transform(test_X_np)
	val_X_np = normalizer.transform(val_X_np)
	train_data = np.concatenate((train_X_np, train_Y_np), axis=1).tolist()
	test_data = np.concatenate((test_X_np, test_Y_np), axis=1).tolist()
	val_data = np.concatenate((val_X_np, val_Y_np), axis=1).tolist()
	return train_data, val_data, test_data

def augment_data(mydata) :
	mydata_np = np.array(mydata)
	for i in range(mydata_np.shape[0]) :
		if mydata_np[i,-1] == 1 : 
			mydata_np = np.concatenate((mydata_np, mydata_np[i,:].reshape(1,-1)), axis=0)
	mydata = mydata_np.tolist()
	return mydata

def plot(x, y, save_path, title):
	# Plot dataset
    plt.figure()
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.xlabel('1st Feature')
    plt.ylabel('2nd Feature')
    plt.legend(['No death', 'Death'])
    plt.title(title)
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
#Encoding the raw data
def encode_class(mydata): 
	classes = [] 
	for i in range(len(mydata)): 
		if mydata[i][-1] not in classes: 
			classes.append(mydata[i][-1]) 
	for i in range(len(classes)): 
		for j in range(len(mydata)): 
			if mydata[j][-1] == classes[i]:
				mydata[j][-1] = i 
	return mydata			 
			
# Splitting the data 
def splitting(mydata, ratio): 
	train_num = int(len(mydata) * ratio) 
	train = [] 
	test = list(mydata) 
	while len(train) < train_num:  
		index = random.randrange(len(test)) 
		train.append(test.pop(index)) 
	return train, test 

# Group the data rows under each class 1 or 0
def groupUnderClass(mydata): 
	dict = {} 
	for i in range(len(mydata)): 
		if (mydata[i][-1] not in dict): 
			dict[mydata[i][-1]] = [] 
		dict[mydata[i][-1]].append(mydata[i]) 
	return dict

#Mean Calculator
def mean(numbers): 
	return sum(numbers) / float(len(numbers)) 

#Standard Deviation calc.
def std_dev(numbers): 
	avg = mean(numbers) 
	variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1) 
	return math.sqrt(variance) 

def MeanAndStdDev(mydata): 
	info = [(mean(attribute), std_dev(attribute)) for attribute in zip(*mydata)] 
	del info[-1] 
	return info 

# Find Mean and Standard Deviation under each class 
def MeanAndStdDevForClass(mydata): 
	info = {} 
	dict = groupUnderClass(mydata) 
	for classValue, instances in dict.items():
		info[classValue] = MeanAndStdDev(instances)
	return info 

# Calculate Gaussian Log Probability Density Function  
def calculateGaussianLogProbability(data, mu, sigma) :
	if sigma == 0 :
		sigma = epsilon
	ll = math.log(2*math.pi*(sigma**2))/2 + ((data-mu)**2)/(2 * (sigma**2) )
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
			x = test[i]
			probabilities[classValue] += calculateGaussianLogProbability(x, mean, std_dev)
		probabilities[classValue] += logBerProb[int(classValue)]
	to_Remove_Ft_list.sort()
	theSet= set(to_Remove_Ft_list)
	to_Remove_Ft_list = list(theSet)
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

def accuracy_rate(test, predictions): 
	correct = 0
	for i in range(len(test)): 
		if test[i][-1] == predictions[i]: 
			correct += 1
	return (correct / float(len(test))) * 100.0

def apply_pca(train_data, val_data, test_data, count) :
	train_data_np = np.array(train_data)
	val_data_np = np.array(val_data)
	test_data_np = np.array(test_data)

	train_X_np = train_data_np[:,:-1]  
	train_Y_np = train_data_np[:,-1].reshape((train_data_np.shape[0],1))
	val_X_np = val_data_np[:,:-1]  
	val_Y_np = val_data_np[:,-1].reshape((val_data_np.shape[0],1))
	test_X_np = test_data_np[:,:-1]
	test_Y_np = test_data_np[:,-1].reshape((test_data_np.shape[0],1))

	pca = PCA(n_components=0.95)
	train_X_np_pca = pca.fit_transform(train_X_np)
	test_X_np_pca = pca.transform(test_X_np)
	val_X_np_pca = pca.transform(val_X_np)
	
	plot(x=test_X_np_pca[:,:2],y=test_Y_np.reshape(-1),save_path="Projected Features {}.".format(count),title="Projected Features in the first two dimension.")
	train_data = np.concatenate((train_X_np_pca, train_Y_np), axis=1).tolist()
	test_data = np.concatenate((test_X_np_pca, test_Y_np), axis=1).tolist()
	val_data = np.concatenate((val_X_np_pca, val_Y_np), axis=1).tolist()

	plot_hist(pca.explained_variance_ratio_, save_path="PCA Graph Explained Variance Ratio of Selected Components {}".format(count), title="Explained Variance Ratio of Selected Components")

	return train_data, val_data, test_data


def berProb(train_data) :
	train_data_np = np.array(train_data)
	train_data_np = train_data_np[:,-1]
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


def k_fold_cv(train_data=None, pca=False, use_inbuilt_nb=False, test_data=None):
	k = 1
	print("Performing 5 fold cross validation : ")
	kfold = KFold(5, shuffle=True)
	avg_accuracy = 0
	avg_accuracy_train = 0
	count = 0
	test_orig = copy.deepcopy(test_data)
	for train, valid in kfold.split(train_data):
		test_data = copy.deepcopy(test_orig)
		count += 1
		i_fold_train = []
		i_fold_valid = []
		for i in train:
			i_fold_train.append(train_data[i])
		for i in valid:
			i_fold_valid.append(train_data[i])
		i_fold_train, i_fold_valid, test_data = normalize_data(train_data=i_fold_train,val_data=i_fold_valid,test_data=test_data)
		if pca :
			i_fold_train, i_fold_valid, test_data = apply_pca(train_data=i_fold_train, val_data=i_fold_valid, test_data=test_data, count=count)
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
		k = k + 1
	avg_accuracy /= 5
	avg_accuracy_train /= 5
	print("Average Accuracy on validation set : ",avg_accuracy)
	print("Average Accuracy on training set : ",avg_accuracy_train)
	if not use_inbuilt_nb :
		predictions_test = getPredictions(info, logBerProb, test_data)
		accuracy_test = accuracy_rate(test_data, predictions_test)
		print("Accuracy on test set : ",accuracy_test)

def k_fold_cv_without_print(train_data, test_data):
	k = 1
	kfold = KFold(5, shuffle=True)
	avg_accuracy = 0
	avg_accuracy_train = 0
	for train, valid in kfold.split(train_data):
		i_fold_train = []
		i_fold_valid = []
		for i in train:
			i_fold_train.append(train_data[i])
		for i in valid:
			i_fold_valid.append(train_data[i])
		i_fold_train, i_fold_valid, test_data = normalize_data(i_fold_train, i_fold_valid, test_data)
		info = MeanAndStdDevForClass(i_fold_train)
		logBerProb = berProb(i_fold_train) 
		predictions = getPredictions(info, logBerProb, i_fold_valid) 
		accuracy = accuracy_rate(i_fold_valid, predictions) 
		avg_accuracy += accuracy
		k = k + 1
	avg_accuracy /= 5
	return avg_accuracy
		
def seq_back_sel(train_data, test_data):
	test_dup = copy.deepcopy(test_data)
	original_avg_acc = k_fold_cv_without_print(train_data, test_dup)
	initial_total = len(train_data[0])
	print("Initial Features:",initial_total)
	index_list = []
	for i in range(0,len(train_data[0])):
		index_list.append(i) 
	print("Before Sequential Backward Selection accuracy : ",original_avg_acc)
	print("Performing Sequential Backward Selection....")
	print("=======This may take some time due to iteration on large number of features(199), number of times=======")
	while(len(train_data[0])>1):
		feature_index_to_delete = -1
		max_acc = original_avg_acc
		for ind in range(0,len(train_data[0])):
			train_after_del = copy.deepcopy(train_data)
			for row in train_after_del:
				del row[ind]
			del_acc = k_fold_cv_without_print(train_data=train_after_del, test_data=test_dup)
			if del_acc>max_acc:
				max_acc = del_acc
				feature_index_to_delete = ind
		if max_acc>original_avg_acc:
			original_avg_acc = max_acc
			print("Feature Index removed : ",feature_index_to_delete)
			print("New Accuracy:",original_avg_acc)
			del index_list[feature_index_to_delete]
			for row in train_data:
				del row[feature_index_to_delete]
			for row in test_data:
				del row[feature_index_to_delete]
			for row in test_dup:
				del row[feature_index_to_delete]
		else:
			break
	print("Final accuracy after Sequential Backward Selection: ",original_avg_acc)
	print("Number of features left:",len(train_data[0]),", Number of features removed:",initial_total-len(train_data[0]))
	print("Index number of features left:",index_list)
	k_fold_cv(train_data=train_data, test_data=test_data)

def outlier_del(data):
	print("Initializing sample removal....")
	print("Initial sample size:",len(data))
	to_return = []
	feature_mean_std = MeanAndStdDev(data)
	row_num = 0
	zeros = 0
	ones = 0
	twos = 0
	for sample in data:
		row_num = row_num + 1
		outliers = 0
		for i in range(0,len(sample)-1):
			m,v = feature_mean_std[i]
			if(m+v*3 < sample[i]):
				outliers = outliers + 1
		if(outliers==0):
			zeros = zeros+1
			to_return.append(sample)
		if(outliers==1):
			ones = ones+1
			to_return.append(sample)
		if(outliers==2):
			twos = twos+1
		row_num = row_num + 1
	print("Sample with Zero outlier:",zeros,"Sample with One outlier:",ones,"Sample with Twos outliers:",twos)
	print("After sample removal training size:",len(to_return),", Samples removed:",len(data)-len(to_return))
	return to_return

def main() : 
	filename = r'after_prepoc.csv'
	
	mydata = csv.reader(open(filename, "rt")) 
	mydata = list(mydata)
	mydata = encode_class(mydata) 
	for i in range(len(mydata)): 
		mydata[i] = [float(x) for x in mydata[i]] 

	ratio = 0.8
	train_data, test_data = splitting(mydata, ratio) 
		
	print(" ========= Part 1 =========")
	k_fold_cv(train_data=train_data , pca=False, use_inbuilt_nb=use_inbuilt_nb, test_data=test_data)
	print("\n\n ========= Part 2 =========")
	k_fold_cv(train_data=train_data, pca=True, use_inbuilt_nb=use_inbuilt_nb, test_data=test_data)	
	print("\n\n ========= Part 3 =========")
	after_samp_del = outlier_del(train_data)
	seq_back_sel(after_samp_del,test_data)

if __name__ == "__main__":
	main()
