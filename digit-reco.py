
import csv
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA




class DigitRecognizer(object):

	def __init__(self, datapath = 'train.csv', N = 3000):

		train_set = pd.read_csv(datapath, nrows= N)
		number_of_samples = len(train_set)
		pixels_col = [col for col in train_set.columns if col != 'label']
		self.submission_samples= pd.read_csv('test.csv')
		self.df = train_set.reindex(np.random.permutation(train_set.index))     
		self.target = self.df['label']
		self.samples = self.df[pixels_col] 
		train_length = int(number_of_samples * 0.8)
		self.train_target = self.target.iloc[:train_length].values
		self.train_samples = self.samples.iloc[:train_length,:].values
		self.test_target = self.target.iloc[train_length:].values
		self.test_samples = self.samples.iloc[int(train_length):,].values



	def data_PCA(self):

		n_components = 25
		pca = PCA(n_components=n_components,whiten=True).fit(self.train_samples)
		self.train_samples_pca = pca.transform(self.train_samples)
		self.test_samples_pca = pca.transform(self.test_samples)
		self.submission_samples_pca = pca.transform(self.submission_samples)
		return self.train_samples_pca, self.test_samples_pca, self.submission_samples_pca



	def grid_search(self):

		param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
		'C': [1, 10, 100, 1000]},
		{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
		svm_grid = GridSearchCV(svm.SVC(), param_grid, cv=2)
		X_train_pca = self.data_PCA()[0]
		svm_fit = svm_grid.fit(X_train_pca, self.train_target)
		best_params = svm_grid.best_params_
		self.best_kernel =  best_params['kernel']
		self.best_C = best_params['C']
		self.best_gamma = best_params['gamma']
		return self.best_kernel, self.best_C, self.best_gamma



	def svm_classification(self):
		X_train_pca = self.data_PCA()[0]
		svm_kernel, svm_c, svm_gamma = self.grid_search()
		self.clf_SVM = svm.SVC(kernel = svm_kernel , C = svm_c , gamma = svm_gamma)
		self.clf_SVM.fit(X_train_pca, self.train_target)
		joblib.dump(self.clf_SVM, 'test/SVM_linear.pkl')
		


	def submission_data(self):
		submission_X_pca = self.data_PCA()[2]
		predicted_digits = self.clf_SVM.predict(submission_X_pca)
		columns = ['ImageId','Label']
		sub_df = pd.DataFrame(columns=columns)
		sub_df['Label'] = predicted_digits
		sub_df['ImageId'] = sub_df.index + 1
		sub_file = sub_df.to_csv('submission.csv', index=False)
		submission_set = pd.read_csv('submission.csv' , nrows= 10)
		print(submission_set)



	def load_clf(self):
		
		test_X_pca = self.data_PCA()[1]
		clf = joblib.load('test/SVM_linear.pkl')
		classifier_score = clf.score(test_X_pca, self.test_target)
		print(classifier_score)




data = DigitRecognizer()
data.svm_classification()

data.load_clf()
data.submission_data()






   






