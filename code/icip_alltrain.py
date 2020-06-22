# !/usr/bin/env python
# coding: utf-8
'''
@File    :   boost.py
@Time    :   2020/04/13 13:46:21
@Author  :   Wang Kai 
@Version :   1.0
@Contact :   wk15@mail.ustc.edu.cn
'''
# This is used to build catboost model using extracted features

import argparse
import gc
import time
import os
import math

import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

random_seed = 2020
num_class=30
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


all_popularity_filepath="/home/wangkai/ICIP/feature/label/popularity_TRAIN_20337.csv"
cluster_center_filepath="/home/wangkai/ICIP/feature/label/cluster_center.csv"
cluser_label_filepath="/home/wangkai/ICIP/feature/label/cluster_label_20337.csv"

# random
train_popularity_filepath="/home/wangkai/ICIP/feature/label/train_label_random.csv"
validate_popularity_filepath="/home/wangkai/ICIP/feature/label/validate_label_random.csv"

# # postdate
# train_popularity_filepath="/home/wangkai/ICIP/feature/label/train_label_datetaken.csv"
# validate_popularity_filepath="/home/wangkai/ICIP/feature/label/validate_label_datetaken.csv"

number_columns=["PhotoCount","MeanViews","Contacts","GroupsCount","NumSets","GroupsAvgPictures","GroupsAvgMembers","Ispro","HasStats","AvgGroupsMemb","AvgGroupPhotos","NumGroups"] # 12
text_columns=["Tags","Title","Description"] # 3 
first_columns=["FlickrId","UserId"] # 2

train_feature_filepath={
    "original":"/home/wangkai/ICIP/feature/train/train_feature_20337.csv",
    "fasttext":"/home/wangkai/ICIP/feature/train/FastText_tags+des_20337.csv",
    "tfidf":"/home/wangkai/ICIP/feature/train/Tfidf_tags+des_20337.csv",
    "lsa":"/home/wangkai/ICIP/feature/train/LSA_tags+title+des_20337.csv",
    "lda":"/home/wangkai/ICIP/feature/train/LDA_tags+title+des_20337.csv",
    "wordchar":"/home/wangkai/ICIP/feature/train/wordchar_tags+title+des_20337.csv",
    "userid":"/home/wangkai/ICIP/feature/train/UserId256_20337.csv",
    "image":"/home/wangkai/ICIP/feature/train/ResNext101_image_20337.csv"
}
test_feature_filepath={
    "original":"/home/wangkai/ICIP/feature/test/test_feature_7693.csv",
    "fasttext":"/home/wangkai/ICIP/feature/test/FastText_tags+des_7693.csv",
    "tfidf": "/home/wangkai/ICIP/feature/test/Tfidf_tags+des_7693.csv",
    "lsa": "/home/wangkai/ICIP/feature/test/LSA_tags+title+des_7693.csv",
    "lda": "/home/wangkai/ICIP/feature/test/LDA_tags+title+des_7693.csv",
    "wordchar":"/home/wangkai/ICIP/feature/test/wordchar_tags+title+des_7693.csv",
    "userid":"/home/wangkai/ICIP/feature/test/UserId256_7693.csv",
    "image":"/home/wangkai/ICIP/feature/test/ResNext101_image_7693.csv"
}

def clutser(num_class=num_class):
    df_popularity=pd.read_csv(all_popularity_filepath)
    # 归一化
    normalized_popularity=df_popularity.iloc[:,1:].div(df_popularity["Day30"],axis=0)
    # 聚类的label
    kmeans=KMeans(n_clusters=num_class,init="k-means++",n_init=100,max_iter=10000,random_state=random_seed,n_jobs=-1,algorithm="auto").fit(normalized_popularity)
    df_label=pd.DataFrame({"FlickrId":df_popularity["FlickrId"],"label":kmeans.labels_})
    df_label.to_csv(cluser_label_filepath,index=False)
    # 聚类中心
    df_cluster_center=pd.DataFrame(kmeans.cluster_centers_)
    df_cluster_center.columns=["day"+str(i+1) for i in range(30)]
    df_cluster_center.insert(0,column="label",value=np.arange(num_class))
    df_cluster_center.to_csv(cluster_center_filepath,index=False)
    
def load_feature(feature_list,flag="train"):
    feature_path = train_feature_filepath if flag=="train" else test_feature_filepath
    for i, feature_name in enumerate(feature_list):
        print("Loading {} ...".format(feature_name))
        feature = pd.read_csv(feature_path[feature_name])
        print("feature: {}, len:{}".format(feature_name,len(feature.columns)-1))
        if i == 0:
            all_feature = feature
        else:
            all_feature = pd.merge(all_feature,feature)
    useless = text_columns
    all_feature.drop(useless,axis=1,inplace=True)
    print(all_feature)
    return all_feature


def calssify_catboost(train):
    cat_features=["UserId"]
    # cat_features=[]
    train_data=catboost.Pool(train.iloc[:,1:-31],train["label"],cat_features=cat_features)

    model=catboost.CatBoostClassifier(iterations=4000, learning_rate=0.003, depth=6, objective="MultiClass", classes_count=num_class, eval_metric="Accuracy", l2_leaf_reg=3.0, min_data_in_leaf=1, boosting_type="Plain", thread_count=-1, task_type="GPU",devices="0", random_state=random_seed, verbose=300, early_stopping_rounds=500)
    model=model.fit(train_data,plot=False)
    # predict label
    preds=model.predict(train_data)
    preds=preds.flatten()
    print("\nValidate\nACC: {}\tTotal right: {}".format(
                np.sum(preds==train["label"])/len(preds), np.sum(preds==train["label"])))
    # # feature importance
    # df_important=pd.DataFrame({"feature_name":model.feature_names_,"importance":model.feature_importances_})
    # df_important=df_important.sort_values(by=["importance"],ascending=False)
    # print(df_important)
    
    # # for train dateset
    df_predict_label=pd.DataFrame({"FlickrId":train["FlickrId"],"preds_label":preds})
    return model,df_predict_label

def regression_catboost(train):
    cat_features=["UserId"]
    # cat_features=[]
    p_train=np.log(train["Day30"]/4+1)
    # p_train,p_validate=train["Day30"],validate["Day30"]
    train_data=catboost.Pool(train.iloc[:,1:-31],p_train,cat_features=cat_features)

    model=catboost.CatBoostRegressor(iterations=50000, learning_rate=0.003, depth=6, objective="MAPE", eval_metric="MAPE",custom_metric=["RMSE","MAE","MAPE"], l2_leaf_reg=3.0, min_data_in_leaf=1, boosting_type="Plain", thread_count=-1, task_type="GPU",devices="0", random_state=random_seed, verbose=300, early_stopping_rounds=1000,fold_permutation_block=1,bagging_temperature=0)
    # model=catboost.CatBoostRegressor(iterations=100000, learning_rate=0.1, depth=6, objective="RMSE", eval_metric="RMSE",custom_metric=["RMSE","MAE","MAPE"], l2_leaf_reg=3.0, min_data_in_leaf=1, boosting_type="Plain", use_best_model=True, thread_count=-1, task_type="CPU",devices="0", random_state=random_seed, verbose=300, early_stopping_rounds=500)
    model.fit(train_data,plot=False)

    preds_p_validate=model.predict(train_data)
    preds_day30=(np.exp(preds_p_validate)-1)*4

    # src,_=spearmanr(train["Day30"],preds_day30)

    # df_important=pd.DataFrame({"feature_name":model.feature_names_,"importance":model.feature_importances_})
    # df_important=df_important.sort_values(by=["importance"],ascending=False)
    # print(df_important)

    # # for train
    df_predict_day30=pd.DataFrame({"FlickrId":train["FlickrId"],"preds_day30":preds_day30})
    return model,df_predict_day30




def train(classify_feature_list,regression_feature_list):
    df_label=pd.read_csv(cluser_label_filepath)
    df_popularity=pd.read_csv(all_popularity_filepath)
    train_label=pd.merge(df_label,df_popularity,on="FlickrId",how="inner")
 

    # Classify 
    train_classify_feature=load_feature(classify_feature_list,flag="train")
    train=pd.merge(train_classify_feature,train_label,on="FlickrId",how="inner")
    classify_model,df_predict_label=calssify_catboost(train)
    # df_predict_label.to_csv("/home/wangkai/ICIP/temp/temp_predict_label.csv",index=False)
    # df_predict_label=pd.read_csv("/home/wangkai/ICIP/temp/temp_predict_label.csv")

    # Regression
    train_regression_feature=load_feature(regression_feature_list,flag="train")
    train=pd.merge(train_regression_feature,train_label,on="FlickrId",how="inner")
    regression_model,df_predict_day30=regression_catboost(train)

    df_preds=pd.merge(df_predict_label,df_predict_day30,on="FlickrId",how="inner")
    df_cluster_center=pd.read_csv(cluster_center_filepath)
    df_temp=pd.merge(df_preds,df_cluster_center,how="left",left_on="preds_label",right_on="label")

    # FlickrId, preds, truth
    df_preds_result=pd.concat([df_temp["FlickrId"],df_temp.iloc[:,-30:].mul(df_temp["preds_day30"],axis=0),train.iloc[:,-30:]],axis=1)
    print(df_preds_result.iloc[:,1:31])
    print(df_preds_result.iloc[:,31:])

    columns=["FlickrId"]+["preds_day"+str(i+1) for i in range(30)]+["truth"+str(i+1) for i in range(30)]
    df_preds_result.columns=columns
    # analysis
    y_preds=np.array(df_preds_result.iloc[:,1:31])
    y_true=np.array(df_preds_result.iloc[:,31:])
    # 对于预测结果
    rmse_errors=np.sqrt([mean_squared_error(y_true[i], y_preds[i]) for i in range(y_true.shape[0])])
    trmse=stats.trim_mean(rmse_errors,0.25)
    median_rmse=np.median(rmse_errors)
    src,_=spearmanr(y_true[:,-1],y_preds[:,-1])
    print("\n Predict:")
    print("RMSE(trimmed 0.25): {}".format(trmse))
    print("RMSE(median): {}".format(median_rmse))
    print("SRC: {}".format(src))
   
    # classify_model=1
    return classify_model,regression_model






def parse_arguments():
    parser = argparse.ArgumentParser(description=" ICIP Catboost model")
    parser.add_argument("-classify_f", "--classify_feature", type=str,
                        choices=["original","fasttext","tfidf","lsa","lda","wordchar","userid","image"],
                        nargs="?",
                        const=["original"],
                        default=["original","userid"],
                        help="which feature will be used for classify")
    parser.add_argument("-reg_f", "--regression_feature", type=str,
                        choices=["original","fasttext","tfidf","lsa","lda","wordchar","userid","image"],
                        nargs="?",
                        const=["original"],
                        default=["original","userid","lda","wordchar"],
                        help="which feature will be used for regression")
    parser.add_argument("-output", "--submission_path", type=str,
                        default="/home/wangkai/ICIP/submission",
                        help="ICIP file(.csv) will be submit path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    clutser(num_class=num_class)
    # train classify model regression model
    calssify_model,regression_model = train(args.classify_feature,args.regression_feature)
    
    # predict class label
    test_classify_feature=load_feature(args.classify_feature,flag="test")
    predict_label=calssify_model.predict(test_classify_feature.iloc[:,1:])
    df_predict_label=pd.DataFrame({"FlickrId":test_classify_feature["FlickrId"],"preds_label":predict_label.flatten()})

    # predict Day30
    test_regression_feature=load_feature(args.regression_feature,flag="test")
    preds_p_test=regression_model.predict(test_regression_feature.iloc[:,1:])
    preds_day30=(np.exp(preds_p_test)-1)*4
    df_predict_day30=pd.DataFrame({"FlickrId":test_regression_feature["FlickrId"],"preds_day30":preds_day30})

    df_preds=pd.merge(df_predict_label,df_predict_day30,on="FlickrId",how="inner")
    df_cluster_center=pd.read_csv(cluster_center_filepath)
    df_temp=pd.merge(df_preds,df_cluster_center,how="left",left_on="preds_label",right_on="label")
    df_preds_result=pd.concat([df_temp["FlickrId"],df_temp.iloc[:,-30:].mul(df_temp["preds_day30"],axis=0)],axis=1)
    df_preds_result.columns=pd.read_csv(all_popularity_filepath).columns
    
    print(df_preds_result)
    submission_filepath=os.path.join(args.submission_path,"2_submission.csv")
    df_preds_result.to_csv(submission_filepath,index=False)

    
