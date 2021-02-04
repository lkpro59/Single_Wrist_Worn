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

def algorithm_default(u, k_knn, d_rf, gam_svm, c_svm, status, pca, check):
    df = pd.read_csv(filename[u])

    Label = df['Label'].values

    df.drop(['Label'], axis=1, inplace=True)
    df.drop(['Prob'], axis=1, inplace=True)

    data = df.values

    knn_acc, rf_acc, clf_acc, rbf_acc = ([] for i in range(4))
    
    temp_index = 0

    kf2 = KFold(n_splits=5, shuffle=status, random_state=21).split(data2)
    
    for train_index, test_index in kf2:
        if temp_index == 0:
            temp = data2[test_index]
            label_temp = label2[test_index]
        elif temp_index == 1:
            temp2 = data2[test_index]
            label_temp2 = label2[test_index]
        elif temp_index == 2:
            temp3 = data2[test_index]
            label_temp3 = label2[test_index]
        elif temp_index == 3:
            temp4 = data2[test_index]
            label_temp4 = label2[test_index]
        elif temp_index == 4:
            temp5 = data2[test_index]
            label_temp5 = label2[test_index]
        temp_index+=1

    temp_index = 0

    kf = KFold(n_splits=5, shuffle=status, random_state=21).split(data)
    for train_index, test_index in kf:
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = Label[train_index], Label[test_index]
        
        if temp_index == 0:
            x_test = np.concatenate((x_test, temp))
            y_test = np.concatenate((y_test, label_temp))
        elif temp_index == 1:
            x_test = np.concatenate((x_test, temp2))
            y_test = np.concatenate((y_test, label_temp2))
        elif temp_index == 2:
            x_test = np.concatenate((x_test, temp3))
            y_test = np.concatenate((y_test, label_temp3))
        elif temp_index == 3:
            x_test = np.concatenate((x_test, temp4))
            y_test = np.concatenate((y_test, label_temp4))
        elif temp_index == 4:
            x_test = np.concatenate((x_test, temp5))
            y_test = np.concatenate((y_test, label_temp5))

        temp_index += 1

        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        knn_score, rf_score, clf_score, rbf_score = machine_learning(x_train_scaled, x_test_scaled, y_train, y_test, k_knn, d_rf, gam_svm, c_svm)
        knn_acc.append(knn_score)
        rf_acc.append(rf_score)
        clf_acc.append(clf_score)
        rbf_acc.append(rbf_score)

        #print('knn: ',np.mean(knn_acc)*100,'rf: ',np.mean(rf_acc)*100,'clf: ',np.mean(clf_acc)*100,'rbf: ',np.mean(rbf_acc)*100)
    intra_acc_calc(knn_acc, rf_acc, clf_acc, rbf_acc)

def machine_learning(x_train_scaled, x_test_scaled, y_train, y_test, k_knn, d_rf, gam_svm, c_svm):
    knn = KNeighborsClassifier(n_neighbors=k_knn)
    knn.fit(x_train_scaled, y_train)
    y_pred = knn.predict(x_test_scaled)
    knn_acc = accuracy_score(y_test, y_pred)
    count = 0
    
    for i in range(len(y_pred)):
        if y_pred[i] == 2:
            count+=1
    #knn_acc = count/len(y_pred)

    rf = RandomForestClassifier(bootstrap=True, n_estimators = 100, class_weight = "balanced")
    rf.fit(x_train_scaled, y_train)
    y_pred = rf.predict(x_test_scaled)
    rf_acc = accuracy_score(y_test, y_pred)

    #true = pd.Categorical(list(np.where(np.array(y_test) == 1, 'Non-functional','Functional')), categories = ['Non-functional','Functional'])
    #pred = pd.Categorical(list(np.where(np.array(y_pred) == 1, 'Non-functional','Functional')), categories = ['Non-functional','Functional'])

    #print('\n', pd.crosstab(true, pred, rownames=['Predict'], colnames=['Actual'], margins=True, margins_name="Total"))
    
    count = 0
    
    for i in range(len(y_pred)):
        if y_pred[i] == 2:
            count+=1
    rf_acc_count = count/len(y_pred)

    #print("Movement: ", rf_acc_count*100)
    
    clf = LinearSVC(random_state=21)
    clf.fit(x_train_scaled, y_train)
    y_pred = clf.predict(x_test_scaled)
    clf_acc = accuracy_score(y_test, y_pred)
    
    count = 0
    
    for i in range(len(y_pred)):
        if y_pred[i] == 2:
            count+=1
    #clf_acc = count/len(y_pred)
           
    param = {
        'kernel': ['rbf'],
        'gamma': np.linspace(0.1,10,25),
        'C': np.linspace(0.1,25,25)
    }

    grid_search = GridSearchCV(estimator=SVC(), param_grid=param)

    grid_search.fit(x_train_scaled, y_train)
    best_params = grid_search.best_params_
    print(best_params)
    rbf = SVC(kernel='rbf', gamma = best_params['gamma'], C= best_params['C'])

    #rbf = SVC(kernel='rbf', gamma=gam_svm, C=c_svm)
    rbf.fit(x_train_scaled, y_train)
    y_pred = rbf.predict(x_test_scaled)
    rbf_acc = accuracy_score(y_test, y_pred)

    count = 0
    
    for i in range(len(y_pred)):
        if y_pred[i] == 2:
            count+=1
    #rbf_acc = count/len(y_pred)

    return knn_acc, rf_acc, clf_acc, rbf_acc

def intra_acc_calc(knn_acc, rf_acc, clf_acc, rbf_acc):
    print('K-nearest neighbor intra avg acc: ', round((np.mean(knn_acc) * 100),4), '%')
    print('K-nearest neighbor intra std: ', round(np.std(knn_acc)*100,6), '\n')

    print('Random forest intra avg acc: ', round((np.mean(rf_acc) * 100),4), '%')
    print('Random forest intra std: ', round(np.std(rf_acc)*100,6), '\n')
    
    print('Linear SVM classfier intra avg acc: ', round((np.mean(clf_acc) * 100),4), '%')
    print('Linear SVM classfier intra std: ', round(np.std(clf_acc)*100,6), '\n')

    print('RBF SVM classfier intra avg acc: ', round((np.mean(rbf_acc) * 100),4), '%')
    print('RBF SVM classfier intra std: ', round(np.std(rbf_acc)*100,6), '\n')
    
    global avg_knn_acc
    avg_knn_acc.append(np.mean(knn_acc))

    global avg_rf_acc
    avg_rf_acc.append(np.mean(rf_acc))

    global avg_clf_acc
    avg_clf_acc.append(np.mean(clf_acc))

    global avg_rbf_acc
    avg_rbf_acc.append(np.mean(rbf_acc))

def avg_acc_calc():
    print('\nAverage KNN acc: ', round((np.mean(avg_knn_acc) * 100),4), '%\n')
    print('Intra KNN Std: ', round(np.std(avg_knn_acc)*100,6), '\n')
    print(avg_knn_acc)
    
    print('\nAverage RF acc: ', round((np.mean(avg_rf_acc) * 100),4), '%\n')
    print('Intra RF Std: ', round(np.std(avg_rf_acc)*100,6), '\n')
    print(avg_rf_acc)
    
    print('\nAverage CLF acc: ', round((np.mean(avg_clf_acc) * 100),4), '%\n')
    print('Intra CLF Std: ', round(np.std(avg_clf_acc)*100,6), '\n')
    print(avg_clf_acc)
    
    print('\nAverage RBF acc: ', round((np.mean(avg_rbf_acc) * 100),4), '%\n')
    print('Intra RBF Std: ', round(np.std(avg_rbf_acc)*100,6), '\n')
    print(avg_rbf_acc)

def intra_normal():
    # Determine k
    k_knn = 5
    d_rf = 9
    gam_svm = 0.1
    c_svm = 550
    status = True
    pca = PCA(0.9)

    algorithm_default(u, k_knn, d_rf, gam_svm, c_svm, status, pca, 0)

def intra_stroke():
    # Determine k
    k_knn = 5
    d_rf = 4
    gam_svm = 1
    c_svm = 800
    status = True
    pca = PCA(0.9)
    
    algorithm_default(u, k_knn, d_rf, gam_svm, c_svm, status, pca, 0)

if __name__ =="__main__":
    filename = [''] * 22
    filename2 = [''] * 22

    knn_acc_mean, rf_acc_mean, clf_acc_mean, rbf_acc_mean = ([] for i in range(4))
    knn_acc_std, rf_acc_std, clf_acc_std, rbf_acc_std  = ([] for i in range(4))
    knn_acc_pca_mean, rf_acc_pca_mean, clf_acc_pca_mean, rbf_acc_pca_mean = ([] for i in range(4))
    knn_acc_pca_std, rf_acc_pca_std, clf_acc_pca_std, rbf_acc_pca_std = ([] for i in range(4))
    avg_knn_acc, avg_rf_acc, avg_clf_acc, avg_rbf_acc = ([] for i in range(4))

    load_data()
        
    for u in range(10, 20):

        df2 = pd.read_csv(filename2[u])
    
        #df2 = df2[(df2['Prob'].values != 1) & (df2['Label'].values != 0)]

        label2 = df2['Label'].values

        df2.drop(['Label'], axis=1, inplace=True)
        df2.drop(['Prob'], axis=1, inplace=True)
    
        data2 = df2.values
        
        print('Subject: ',u+1)
        #intra_normal()
        intra_stroke()
    avg_acc_calc()