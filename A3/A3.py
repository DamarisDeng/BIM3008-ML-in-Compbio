import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2

class Classifier(object):
    def __init__(self, reads, label, genes):
        self.reads = reads
        self.label = label
        self.genes = genes
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.svm = SVC() # will be updated after the grid search
        self.rf = RandomForestClassifier()
        self.knn = KNeighborsClassifier()

    def preprossessing(self):
        '''This function calls the preprocess_label and preprocess_reads functions and merge two dataframes
        '''
        self.preprocess_reads()
        self.preprocess_label()
        data = pd.merge(self.reads, self.label, on='sample_ID', how='inner') # merge two dataframes
        newname = data.columns[1:-1].str.split('.').str[0]
        data.rename(columns=dict(zip(data.columns[1:-1], newname)), inplace=True)
        print('The shape of dataframe after merging is: ', data.shape)
        self.data = data
        print('-'*14, 'Finish preprocessing', '-'*14, '\n')
    
    def preprocess_label(self):
        '''This function merges some substages into one stage, drop the samples with no diagnosis, 
        and drop the samples with "not reported" diagnosis
        '''
        print('Processing labels')
        label = self.label
        label.loc[label['tumor_stage.diagnoses'] == 'stage ia', 'tumor_stage.diagnoses'] = 'stage i'
        label.loc[label['tumor_stage.diagnoses'] == 'stage ib', 'tumor_stage.diagnoses'] = 'stage i'
        label.loc[label['tumor_stage.diagnoses'] == 'stage iia', 'tumor_stage.diagnoses'] = 'stage ii'
        label.loc[label['tumor_stage.diagnoses'] == 'stage iib', 'tumor_stage.diagnoses'] = 'stage ii'
        label.loc[label['tumor_stage.diagnoses'] == 'stage iiia', 'tumor_stage.diagnoses'] = 'stage iii'
        label.loc[label['tumor_stage.diagnoses'] == 'stage iiib', 'tumor_stage.diagnoses'] = 'stage iii'
        label.loc[label['tumor_stage.diagnoses'] == 'stage iiic', 'tumor_stage.diagnoses'] = 'stage iii'
        label = label.loc[:, ['submitter_id.samples', 'tumor_stage.diagnoses']].rename(columns={'submitter_id.samples':'sample_ID', 'tumor_stage.diagnoses': 'diagnosis'})
        label.dropna(inplace=True) # drop samples with no diagnosis
        label = label.query('diagnosis != "not reported"') # exclude samples with 'not reported' diagnoses
        self.label = label
        print('After processing, the shape of the label dataframe is:', label.shape)
        print('Summary of labels:\n', label.diagnosis.value_counts())
        print('Finish processing labels')
        return label
    
    def preprocess_reads(self):
        '''This function transposes the reads dataframe, rename the first column into sample_ID to prepare for merging, duplicates and NA are also dropped'''
        print('Processing reads')
        print('This procedure may take 30 seconds to 1 minute.')
        reads = self.reads.set_index('Ensembl_ID').T
        reads.reset_index(inplace=True)
        reads.dropna(inplace=True)
        reads.drop_duplicates(inplace=True)
        reads.rename(columns={'index':'sample_ID'}, inplace=True)
        print('After processing, the shape of the reads dataframe is:', reads.shape)
        print('Finish processing reads')
        self.reads = reads

    def train_test_split(self):
        '''First convert the label to binary numerical values, then split the data into training and testing sets;
        A label dictionary is created to map the numerical values back to the original labels
        '''
        print('Splitting data')
        y = self.data['diagnosis']
        X = self.data.drop(['sample_ID', 'diagnosis'], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)
        print('The shape of training set is: ', self.X_train.shape)
        print('The shape of testing set is: ', self.X_test.shape)
        print('Finish splitting data')

    def train_test(self):
        '''This function calls three models (SVM, RF, KNN) to train the data'''
        print('SVM, RF, and KNN will be used to train the data.')
        self.train_svm()
        self.train_rf()
        self.train_knn()
        print('\n')
        
    def train_svm(self):
        '''This function trains the SVM model'''
        print('Training SVM...')
        self.svm.fit(self.X_train, self.y_train)
        ypred_svm = self.svm.predict(self.X_test)
        print('The accuracy of SVM is: ', accuracy_score(self.y_test, ypred_svm))
        print('The confusion matrix of SVM is: \n', confusion_matrix(self.y_test, ypred_svm))
        print('\n')
    
    def train_rf(self):
        '''This function trains the RF model'''
        print('Training RF...')
        self.rf.fit(self.X_train, self.y_train)
        ypred_rf = self.rf.predict(self.X_test)
        print('The accuracy of RF is: ', accuracy_score(self.y_test, ypred_rf))
        print('The confusion matrix of RF is: \n', confusion_matrix(self.y_test, ypred_rf))
        print('\n')

    def train_knn(self):
        '''This function trains the KNN model'''
        print('Training KNN...')
        self.knn.fit(self.X_train, self.y_train)
        ypred_knn = self.knn.predict(self.X_test)
        print('The accuracy of KNN is: ', accuracy_score(self.y_test, ypred_knn))
        print('The confusion matrix of KNN is: \n', confusion_matrix(self.y_test, ypred_knn))
        print('\n')

    def hyperparameter_tuning(self):
        '''GridSearchCV is used to find the best parameters for the selected model, this process may take a long while to finish.'''
        print('-'*10, 'Hyperparameter tuning', '-'*10)
        svm_params = {
            'kernel':['linear', 'sigmoid', 'poly', 'rbf'],
            'gamma':['auto', 'scale'],
        }
        grid1 = GridSearchCV(SVC(), svm_params, cv=3, verbose=3)
        grid1.fit(self.X_train, self.y_train)
        print('The best parameters for SVM are:', grid1.best_params_)
        self.svm = grid1.best_estimator_ # save the best svm model
        print('SVM has been updated.')
        print('Evaluate the performance on test set...')
        ypred1 = grid1.predict(self.X_test)
        print('Accuracy:', accuracy_score(self.y_test, ypred1))
        print('Confusion matrix: \n', confusion_matrix(self.y_test, ypred1))

        rf_params = {
            'bootstrap': [True, False],
            'max_depth': [5, 50, 100, None],
            'max_features': [20, 50, 80],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [10, 100, 200]
        }
        grid2 = GridSearchCV(RandomForestClassifier(), rf_params, cv=3, verbose=3, n_jobs=-1)
        grid2.fit(self.X_train, self.y_train)
        print('The best parameters for RF are:', grid2.best_params_)
        self.rf = grid2.best_estimator_ # save the best rf model
        print('RF has been updated.')
        print('Evaluate the performance on test set')
        ypred2 = grid2.predict(self.X_test)
        print('Accuracy:', accuracy_score(self.y_test, ypred2))
        print('Confusion matrix: \n', confusion_matrix(self.y_test, ypred2))

        knn_params = {'n_neighbors':[5, 10, 20, 50, 100]}
        grid3 = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3, verbose=3, n_jobs=-1)
        grid3.fit(self.X_train, self.y_train)
        print('The best parameters for KNN are:', grid3.best_params_)
        self.knn = grid3.best_estimator_ # save the best knn model
        print('KNN has been updated.')
        print('Evaluate the performance on test set')
        ypred3 = grid3.predict(self.X_test)
        print('Accuracy:', accuracy_score(self.y_test, ypred3))
        print('Confusion matrix: \n', confusion_matrix(self.y_test, ypred3))

    def feature_selection(self):
        '''Six methods are called by this method. The first three are three ways to select features, the last three are three ways to evaluate the importance of features.'''
        self.feature_selection_variance()
        self.feature_selection_chi2()
        self.feature_selection_prior(self.genes)
        self.training1()
        self.training2()
        self.training3()

    def feature_selection_variance(self):
        '''This function uses variance threshold (0.5) to select features'''
        selector = VarianceThreshold(threshold=0.5)
        selector.fit(self.X_train)
        self.X_train1 = selector.transform(self.X_train)
        self.X_test1 = selector.transform(self.X_test)
        print(f'Feature selection: [{self.X_test1.shape[1]}] features with variance > 0.5 are selected.')
        
    def feature_selection_chi2(self):
        '''This function selects genes based on chi2 test'''
        selector = SelectKBest(chi2, k=566)
        selector.fit(self.X_train, self.y_train)
        self.X_train2 = selector.transform(self.X_train)
        self.X_test2 = selector.transform(self.X_test)
        print(f'Feature selection: [{self.X_test2.shape[1]}] features with p-value < 0.01 are selected.')

    def feature_selection_prior(self, genes):
        '''This function selects genes that are mentioned in the literature'''
        ID = genes['ID']
        self.X_train3 = self.X_train.loc[:, ID]
        self.X_test3 = self.X_test.loc[:, ID]
        print(f'Feature selection: [{self.X_test3.shape[1]}] features are selected based on literature.')

    def training1(self):
        '''This function trains the models using the first set of selected features'''
        print('Performance of different models using the first set of selected features:')
        self.svm.fit(self.X_train1, self.y_train)
        ypred1 = self.svm.predict(self.X_test1)
        print('The accuracy of SVM is: ', accuracy_score(self.y_test, ypred1))
        print('The confusion matrix of SVM is: \n', confusion_matrix(self.y_test, ypred1))
        self.rf.fit(self.X_train1, self.y_train)
        ypred2 = self.rf.predict(self.X_test1)
        print('The accuracy of RF is: ', accuracy_score(self.y_test, ypred2))
        print('The confusion matrix of RF is: \n', confusion_matrix(self.y_test, ypred2))
        self.knn.fit(self.X_train1, self.y_train)
        ypred3 = self.knn.predict(self.X_test1)
        print('The accuracy of KNN is: ', accuracy_score(self.y_test, ypred3))
        print('The confusion matrix of KNN is: \n', confusion_matrix(self.y_test, ypred3))
        print('\n')

    def training2(self):
        '''This function trains the models using the second set of selected features'''
        print('Performance of different models using the second set of selected features:')
        self.svm.fit(self.X_train2, self.y_train)
        ypred1 = self.svm.predict(self.X_test2)
        print('The accuracy of SVM is: ', accuracy_score(self.y_test, ypred1))
        print('The confusion matrix of SVM is: \n', confusion_matrix(self.y_test, ypred1))
        self.rf.fit(self.X_train2, self.y_train)
        ypred2 = self.rf.predict(self.X_test2)
        print('The accuracy of RF is: ', accuracy_score(self.y_test, ypred2))
        print('The confusion matrix of RF is: \n', confusion_matrix(self.y_test, ypred2))
        self.knn.fit(self.X_train2, self.y_train)
        ypred3 = self.knn.predict(self.X_test2)
        print('The accuracy of KNN is: ', accuracy_score(self.y_test, ypred3))
        print('The confusion matrix of KNN is: \n', confusion_matrix(self.y_test, ypred3))
        print('\n')
    
    def training3(self):
        '''This function trains the models using the third set of selected features'''
        print('Performance of different models using the third set of selected features:')
        self.svm.fit(self.X_train3, self.y_train)
        ypred1 = self.svm.predict(self.X_test3)
        print('The accuracy of SVM is: ', accuracy_score(self.y_test, ypred1))
        print('The confusion matrix of SVM is: \n', confusion_matrix(self.y_test, ypred1))
        self.rf.fit(self.X_train3, self.y_train)
        ypred2 = self.rf.predict(self.X_test3)
        print('The accuracy of RF is: ', accuracy_score(self.y_test, ypred2))
        print('The confusion matrix of RF is: \n', confusion_matrix(self.y_test, ypred2))
        self.knn.fit(self.X_train3, self.y_train)
        ypred3 = self.knn.predict(self.X_test3)
        print('The accuracy of KNN is: ', accuracy_score(self.y_test, ypred3))
        print('The confusion matrix of KNN is: \n', confusion_matrix(self.y_test, ypred3))
        print('\n')

if __name__ == '__main__':
    print('Start loading data...')
    genes = pd.read_csv('genes.txt', sep='\t', header=0)
    genes = genes.drop(['Unnamed: 2'], axis=1).rename(columns={'Gene Symbol': 'symbol', 'Ensembl Gene ID': 'ID'})
    reads = pd.read_csv('TCGA-BRCA.htseq_fpkm.tsv', sep='\t', header=0)
    label = pd.read_csv('TCGA-BRCA.GDC_phenotype.tsv', sep='\t', header=0)

    C = Classifier(reads, label, genes)
    C.preprossessing()
    C.train_test_split()
    C.train_test()
    C.hyperparameter_tuning()
    C.feature_selection()
    print('Finished.')