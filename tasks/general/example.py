# -*- coding: utf-8 -*-

# @File   : example.py
# @Author : Yuvv
# @Date   : 2018/5/5

import re
from celery import shared_task
from celery.utils.log import get_logger
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tasks.core import MLPMAsyncTask
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import HashingTF, Tokenizer,IndexToString,StringIndexer,VectorIndexer
from pyspark.sql import SparkSession,Row,functions
from pyspark.ml.linalg import Vectors,Vector
from pyspark.ml.classification import DecisionTreeClassifier,LinearSVC,LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os
from .ml_handle.utils import get_train_model,get_result_and_imgs
os.environ['JAVA_HOME']='/usr/jdk8'

LOGGER = get_logger('celery.MLPMAsyncTask')


@shared_task(base=MLPMAsyncTask)
def word_count(content: str) -> dict:
    """
    简单的单词统计示例程序
    :param content: 带统计单词的字串
    :return: 统计结果字典
    """
    r = {}
    for word in re.split(r'\s+', content):
        if r.get(word, None) is None:
            r[word] = 1
        else:
            r[word] += 1
    return r


@shared_task(base=MLPMAsyncTask)
def spark_word_count(file_path: str) -> dict:
    """
    简单的spark单词统计示例程序
    :param file_path: 要统计的文本文件路径
    :return: 统计结果字典
    """

    from operator import add as op_add
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("PythonWordCount").getOrCreate()

    lines = spark.read.text(file_path).rdd.map(lambda r: r[0])
    counts = lines.flatMap(lambda x: re.split(r'\s+', x)) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(op_add)
    output = counts.collect()

    d = {}
    for word, count in output:
        d[word] = count

    spark.stop()

    return d


@shared_task(base=MLPMAsyncTask)
def test_of_wj(content:str) -> dict:
    r={}
    for word in re.split(r'\s+',content):
        if r.get(word, None) is None:
            r[word]=1
        else:
            r[word]+=1
    return content

def splitDataset(df: pd.DataFrame, test_size):
    '''
    划分数据集
    :param df:
    :param test_size:
    :return:训练集，测试集
    '''
    if not 'Y' in df.columns and not 'y' in df.columns:
        return
    X = df.drop('Y', axis=1)
    Y = df['Y']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=618, test_size=test_size)
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    return x_train, x_test, y_train, y_test

def svmTrain(paras):
    '''
    SVM算法训练
    :param paras: 参数列表
    :return: clf
    '''
    clf = svm.SVC()
    return clf

def mlpcTrain(paras):
    '''
    :param paras:
    :return:
    '''
    clf = MLPClassifier(hidden_layer_sizes=(4, 12, 4), max_iter=1000)
    return clf


def gpcTrain(paras):
    '''

    :param paras:
    :return:
    '''
    clf = GaussianProcessClassifier(1.0 * RBF(1.0))
    return clf


def lrTrain(paras):
    '''

    :param paras:
    :return:
    '''
    clf = LogisticRegression()
    return clf


def linearReTrain(paras):
    '''

    :param paras:
    :return:
    '''
    clf = LinearRegression()
    return clf


def bayesReTrain(paras):
    '''

    :param paras:
    :return:
    '''
    clf = BayesianRidge()
    return clf


def ridgeTrain(paras):
    '''

    :param paras:
    :return:
    '''
    clf = Ridge()
    return clf


def regressionModel(alg, filename, paras):
    clf = ''
    rs = {}
    rs['alg']=alg
    if alg == 'Linear':
        clf = linearReTrain(paras)
    elif alg == 'Bayes':
        clf = bayesReTrain(paras)
    elif alg == 'Ridge':
        clf = ridgeTrain(paras)
    else:
        rs['status']='alg failed'
        return rs
    fileType = filename.split('.')[-1]
    print(fileType)
    if fileType == 'xlsx' or fileType == 'xls':
        df = pd.read_excel(filename)
    elif fileType == 'csv' or fileType == 'txt':
        df = pd.read_csv(filename)
    else:
        rs['status']='dataType failed'
        return rs
    if not 'y' in df.columns and not 'Y' in df.columns:
        rs['status']='data failed'
        return rs
    X = df.drop('Y', axis=1)
    Y = df['Y']
    features = list(X.columns)
    clf.fit(X, Y)
    coef, dis = clf.coef_.tolist(), clf.intercept_
    result = 'Y='
    for index, feature in enumerate(features):
        result += str(coef[index]) + '*' + feature + '+'
    result += str(dis)
    rs['status']='success'
    rs['result']=result
    return rs


@shared_task(base=MLPMAsyncTask)
def handle(alg, filename, params, label, features):
    clf = get_train_model(alg, params)
    result = get_result_and_imgs(alg, clf,label,features,filename)
    rs = {} 
    rs["status"] = 'success'
    rs['result'] = result
    return rs

#@shared_task(base=MLPMAsyncTask)
#def handle(alg, filename, paras):
#    '''
#    算法入口函数
#    :param alg:算法名称
#    :param paras: 参数列表
#    :param filename: 数据集路径
#    :return: 准确率，模型
#    '''
#    print('???')
#    rs={}
#    rs['alg']=alg
#    clf = ''
#    if alg in ['Linear','Bayes','Ridge']:
#        rs = regressionModel(alg, filename, paras)
#        return rs
#    fileType = filename.split('.')[-1]
#    if fileType == 'xlsx' or fileType == 'xls':
#        df = pd.read_excel(filename)
#    elif fileType == 'csv' or filename == 'txt':
#        df = pd.read_csv(filename)
#    else:
#        rs['status']='datatype failed'
#        return rs
#    if not 'y' in df.columns and not 'Y' in df.columns:
#        rs['status']='data failed'
#        return rs
#    print('at split')
#    #result = sparkClassifier(alg,df,paras)
#    #if result== False:
#    #    rs['status']='alg failed'
#    #rs['result']='acc='+str(result)
#    #rs['status']='success'
#    #return rs 
#    x_train, x_test, y_train, y_test = splitDataset(df, 0.3)
#    if alg == 'SVM':
#        clf = svmTrain(paras)
#    elif alg == 'MLPC':
#        clf = mlpcTrain(paras)
#    elif alg == 'GPC':
#        clf = gpcTrain(paras)
#    else:
#        rs['status']='alg failed'
#        return rs
#    print('at fit')
#
#    clf.fit(x_train, y_train)
#    print('at predict')
#    prediction = clf.predict(x_test)
#    acc = metrics.accuracy_score(y_test, prediction)
#    result = 'acc=' + str(acc)
#    rs['status']='success'
#    rs['result']=result
#    return rs
#



def transData2RDD(df,features):
    rel = {}
    if 'Y' in features:
        rel['label'] = df['Y']
        index=features.drop('Y')
    elif 'y' in features:
        rel['label'] = df['Y']
        index=features.drop('Y')
    rel['features'] = Vectors.dense([df[feature] for feature in index])
    return rel

def sparkClassifier(alg, df, params):
    spark = SparkSession.builder.master('local').appName('test').getOrCreate()
    features = df.columns
    # rdd和dataframe转换
    data = spark.createDataFrame(df)
    data = data.rdd
    data = data.map(lambda p:Row(**transData2RDD(p,features))).toDF()

    if alg=='SVM':
        dtClassifier = DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
    elif alg=='MLPC':
        dtClassifier = NaiveBayes(smoothing=1.0,modelType='multinomial').setLabelCol('indexedLabel').setFeaturesCol('indexedFeatures')
    elif alg=='GPC':
        dtClassifier = LogisticRegression(regParam=0.01).setLabelCol('indexedLabel').setFeaturesCol('indexedFeatures')
    else:
         return False

    # 获取标签列和特征列并重命名，划分数据集
    labelIndexer = StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
    featureIndexer = VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(5).fit(data)
    labelConverter = IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    trainingData, testData = data.randomSplit([0.7, 0.3])

    # 构造工作流并进行训练和预测
    pipelinedClassifier = Pipeline().setStages([labelIndexer, featureIndexer, dtClassifier, labelConverter])
    modelClassifier = pipelinedClassifier.fit(trainingData)
    predictionsClassifier = modelClassifier.transform(testData)
    # 评估

    evaluatorClassifier = MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    accuracy = evaluatorClassifier.evaluate(predictionsClassifier)
    return accuracy


if __name__=="__main__":
    df = pd.read_excel('/srv/sites/mlpm-jobs/media/_fs/test.xlsx')
    #handle.delay("SVM",'/srv/sites/mlpm-jobs/media/_fs/test.xlsx',{})
    result = sparkClassifier('SVM',df,{})
    print(result)

