import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition.pca import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def load_data():
    filename[0] = 'C01_actiLife_processed.csv'
    filename[1] = 'C02_actiLife_processed.csv'
    filename[2] = 'C03_actiLife_processed.csv'
    filename[3] = 'C04_actiLife_processed.csv'
    filename[4] = 'C05_actiLife_processed.csv'
    filename[5] = 'C06_actiLife_processed.csv'
    filename[6] = 'C07_actiLife_processed.csv'
    filename[7] = 'C08_actiLife_processed.csv'
    filename[8] = 'C09_actiLife_processed.csv'
    filename[9] = 'C10_actiLife_processed.csv'
    filename[10] = 'S11_actiLife_processed.csv'
    filename[11] = 'S12_actiLife_processed.csv'
    filename[12] = 'S13_actiLife_processed.csv'
    filename[13] = 'S14_actiLife_processed.csv'
    filename[14] = 'S15_actiLife_processed.csv'
    filename[15] = 'S16_actiLife_processed.csv'
    filename[16] = 'S17_actiLife_processed.csv'
    filename[17] = 'S18_actiLife_processed.csv'
    filename[18] = 'S19_actiLife_processed.csv'
    filename[19] = 'S20_actiLife_processed.csv'
    filename[20] = 'Intersubject_normal_processed.csv'
    filename[21] = 'Intersubject_stroke_processed.csv'
    
    filename2[0] = '2_C01_actiLife_processed.csv'
    filename2[1] = '2_C02_actiLife_processed.csv'
    filename2[2] = '2_C03_actiLife_processed.csv'
    filename2[3] = '2_C04_actiLife_processed.csv'
    filename2[4] = '2_C05_actiLife_processed.csv'
    filename2[5] = '2_C06_actiLife_processed.csv'
    filename2[6] = '2_C07_actiLife_processed.csv'
    filename2[7] = '2_C08_actiLife_processed.csv'
    filename2[8] = '2_C09_actiLife_processed.csv'
    filename2[9] = '2_C10_actiLife_processed.csv'
    filename2[10] = '2_S11_actiLife_processed.csv'
    filename2[11] = '2_S12_actiLife_processed.csv'
    filename2[12] = '2_S13_actiLife_processed.csv'
    filename2[13] = '2_S14_actiLife_processed.csv'
    filename2[14] = '2_S15_actiLife_processed.csv'
    filename2[15] = '2_S16_actiLife_processed.csv'
    filename2[16] = '2_S17_actiLife_processed.csv'
    filename2[17] = '2_S18_actiLife_processed.csv'
    filename2[18] = '2_S19_actiLife_processed.csv'
    filename2[19] = '2_S20_actiLife_processed.csv'

def algorithm_default(p, k_knn, d_rf, gam_svm, c_svm, status, pca, check):
    df = pd.read_csv(filename[p])

    Label = df['Label'].values

    df.drop(['Label'], axis=1, inplace=True)
    df.drop(['Prob'], axis=1, inplace=True)

    data = df.values
    
    global data2
    kf = KFold(n_splits=10, shuffle=status, random_state=None).split(data)
    for train_index, test_index in kf:
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = Label[train_index], Label[test_index]
        
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        data2_scaled = scaler.transform(data2)
        #print(data2_scaled)

        knn_score, rf_score, clf_score, rbf_score, rf_accuracy = machine_learning(data2_scaled, label2, x_train_scaled, x_test, y_train, y_test, k_knn, d_rf, gam_svm, c_svm)
        if knn_score > 0: knn_acc.append(knn_score)
        if rf_score > 0: rf_acc.append(rf_score)
        if clf_score > 0: clf_acc.append(clf_score)
        if rbf_score >0: rbf_acc.append(rbf_score)
        if rf_accuracy > 0: avg_rf_accuracy.append(rf_accuracy)

        #print('knn: ',np.mean(knn_acc)*100,'rf: ',np.mean(rf_acc)*100,'clf: ',np.mean(clf_acc)*100,'rbf: ',np.mean(rbf_acc)*100)
        print('knn: ',np.std(knn_acc)*100,'rf: ',np.std(rf_acc)*100,'clf: ',np.std(clf_acc)*100,'rbf: ',np.std(rbf_acc)*100)

    inter_acc_calc(knn_acc, rf_acc, clf_acc, rbf_acc, avg_rf_accuracy)

def machine_learning(data2_scaled, label2, x_train_scaled, x_test, y_train, y_test, k_knn, d_rf, gam_svm, c_svm):
    global c, max_rf, max_k
    global bool
    
    knn_score = 0

    if  (c == u+1) and (bool == True):
        bool = False

        knn = KNeighborsClassifier(n_neighbors=k_knn)
        knn.fit(x_train_scaled, y_train)
        y_pred = knn.predict(data2_scaled)
        knn_score = accuracy_score(label2, y_pred)
       
        bool = True
    
    rf_acc = 0
    if  (c == u+1) and (bool == True):
        bool = False

        #param = {
        #    'bootstrap': [True],
        #    'n_estimators': [100],
        #    'class_weight': ['balanced', None]
        #}

        #grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param)

        #grid_search.fit(x_train_scaled, y_train)
        #best_params = grid_search.best_params_
        
        #print(best_params)
        #count2 = 0
        #for i in range(len(label2)):
        #    if label2[i] == 2:
        #        count2 += 1
        #count2 = count2/len(label2)
        #print('Count label: ',count2*100)

        #rf = RandomForestClassifier(bootstrap=True, n_estimators = best_params["n_estimators"], class_weight=best_params["class_weight"])
        
        rf = RandomForestClassifier(bootstrap=True, n_estimators = 100, class_weight = "balanced")
        rf.fit(x_train_scaled, y_train)
        y_pred = rf.predict(data2_scaled)
        rf_acc = accuracy_score(label2, y_pred)

        max_rf = rf_acc

        count = 0
        for i in range(len(y_pred)):
            if y_pred[i] == 2:
                count += 1
        rf_count_temp = count/len(y_pred)

        print('Amount of Average Functional Movements: ', round((np.mean(rf_count_temp) * 100), 2), '%')

        print('Random Forest Accuracy For Each Fold: ', round((np.mean(rf_acc) * 100), 2), '%')

        bool = True

    clf_score = 0
    
    if  (c == u+1) and (bool == True):
        bool = False
        
        clf = LinearSVC(random_state=0)
        clf.fit(x_train_scaled, y_train)
        y_pred = clf.predict(data2_scaled)
        clf_score = accuracy_score(label2, y_pred)

        bool = True

    rbf_score = 0
    
    if  (c == u+1) and (bool == True):
        bool = False

        rbf = SVC(kernel='rbf', gamma=gam_svm, C=c_svm)
        rbf.fit(x_train_scaled, y_train)
        y_pred = rbf.predict(data2_scaled)
        rbf_score = accuracy_score(label2, y_pred)
            
    return knn_score, max_rf, clf_score, rbf_score, rf_acc

def inter_acc_calc(knn_acc, rf_acc, clf_acc, rbf_acc, avg_rf_accuracy):
    print('K-nearest neighbor avg acc: ', round((np.mean(knn_acc) * 100),4), '%')
    print('K-nearest neighbor std: ', round(np.std(knn_acc),6), '\n')
    
    print('Linear SVM classfier avg acc: ', round((np.mean(clf_acc) * 100),4), '%')
    print('Linear SVM classfier std: ', round(np.std(clf_acc),6), '\n')

    print('RBF SVM classfier avg acc: ', round((np.mean(rbf_acc) * 100),4), '%')
    print('RBF SVM classfier std: ', round(np.std(rbf_acc),6), '\n')
    
    avg_knn_accuracy.append(np.mean(knn_acc))
    avg_clf_accuracy.append(np.mean(clf_acc))
    avg_rbf_accuracy.append(np.mean(rbf_acc))

def avg_acc_calc():
    
    print(avg_knn_accuracy)
    print('\nAverage KNN Accuracy: ', round((np.mean(avg_knn_accuracy) * 100),4), '%\n')
    print('Inter KNN Std: ', round(np.std(avg_knn_accuracy)*100,6), '\n')
    
    print(avg_rf_accuracy)
    print('\nAverage RF Accuracy: ', round((np.mean(avg_rf_accuracy) * 100),4), '%\n')
    print('Inter RF Std: ', round(np.std(avg_rf_accuracy)*100,6), '\n')
    
    print(avg_clf_accuracy)
    print('\nAverage CLF Accuracy: ', round((np.mean(avg_clf_accuracy) * 100),4), '%\n')
    print('Inter CLF Std: ', round(np.std(avg_clf_accuracy)*100,6), '\n')
    
    print(avg_rbf_accuracy)
    print('\nAverage RBF Accuracy: ', round((np.mean(avg_rbf_accuracy) * 100),4), '%\n')
    print('Inter RBF Std: ', round(np.std(avg_rbf_accuracy)*100,6), '\n')

def inter_normal():
    # Determine k
    k_knn = 10
    d_rf = 8
    gam_svm = 0.1
    c_svm = 1000
    status = False
    pca = PCA(0.9)

    p = 20
    algorithm_default(p, k_knn, d_rf, gam_svm, c_svm, status, pca, 1)

def inter_stroke():
    # Determine k
    k_knn = 8
    d_rf = 10
    gam_svm = 0.1
    c_svm = 650
    status = False
    pca = PCA(0.9)
    
    p = 21
    algorithm_default(p, k_knn, d_rf, gam_svm, c_svm, status, pca, 1)

if __name__ =="__main__":
    filename = [''] * 22
    filename2 = [''] * 22

    knn_acc_mean, rf_acc_mean, clf_acc_mean, rbf_acc_mean = ([] for i in range(4))
    knn_acc_std, rf_acc_std, clf_acc_std, rbf_acc_std  = ([] for i in range(4))
    knn_acc_pca_mean, rf_acc_pca_mean, clf_acc_pca_mean, rbf_acc_pca_mean = ([] for i in range(4))
    knn_acc_pca_std, rf_acc_pca_std, clf_acc_pca_std, rbf_acc_pca_std = ([] for i in range(4))
    knn_acc, rf_acc, clf_acc, rbf_acc = ([] for i in range(4))
    avg_knn_accuracy, avg_rf_accuracy, avg_clf_accuracy, avg_rbf_accuracy = ([] for i in range(4))

    load_data()
    
    c = 11
    max = 0.0
    max_k = 0
    f = 0
    for u in range(10, 20):
        max_rf = 0
        bool = True
        print('Subject: ', u+1)
        df2 = pd.read_csv(filename2[u])
    
        df2 = df2[(df2['Label'].values != 0)]

        label2 = df2['Label'].values

        df2.drop(['Label'], axis=1, inplace=True)
        df2.drop(['Prob'], axis=1, inplace=True)

        data2 = df2.values
        #print(data2)
        #inter_normal()
        inter_stroke()
        c+=1
    
    avg_acc_calc()