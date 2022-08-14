import sys
sys.path.append('..')
from pyanalysis import *
import pandas as pd
from sklearn.svm import LinearSVR

def main():
    csv_name='./data/data.csv'
    data=pd.read_csv(csv_name)

    #计算统计量 max min std mean 等
    # statistic_addition(data).to_excel('./result/统计量.xlsx')

    #计算Pearson相关系数 查找特征相关性
    corr_data=data.corr(method='pearson')
    corr_data=np.round(corr_data,2) #保留2位小数
    # heatmap(corr_data,(10,10)) #热力图

    #lasso回归预测 使用回归系数降维特征
    coef_=np.round(lasso_regression(data,label_col='y').coef_,5)
    mask=list(coef_!=0) #转换成布尔值 0为false
    mask.append(True) #添加最后一列y
    low_data=data.iloc[:,mask] #筛选降维后的数据
    # low_data.to_excel('./result/new_low_data.xlsx')

    low_data.index=range(1994,2014) #将行索引改成1994-2013
    low_data.loc[2014]=None #添加2014
    low_data.loc[2015]=None #添加2015
    #遍历n维特征 对每一列特征进行灰度预测系统预测
    for c in low_data.columns[:-1]:
        f=GM11(low_data.loc[range(1994,2014),c])[0] #得到每一列的预测函数
        low_data.loc[2014,c]=f(len(low_data)-1) #带入自变量 这里自变量从1开始 所以2014为倒数第二个自变量
        low_data.loc[2015,c]=f(len(low_data))
        low_data[c]=low_data[c].round(2) #保留2位小数
    # low_data.to_excel('./result/GM.xlsx')

    #使用支持向量机回归预测y值
    data_train=low_data.copy()
    #计算1994-2013 x1-y的mean std
    mean=data_train.iloc[:-2].mean()
    std=data_train.iloc[:-2].std()
    #归一化 这里对 y也进行归一化
    data_train.iloc[:-2]=normalization(data_train.iloc[:-2])
    #建立线性支持向量机回归模型
    svr_model=linear_svr(data_train.iloc[:-2],label_col='y')
    #对新数据进行归一化，这里mean std是训练数据的mean std
    data_train.iloc[-2:,:-1]=(data_train.iloc[-2:,:-1]-mean.iloc[:-1])/std.iloc[:-1]
    #对预测值还原 同样使用训练数据y的 mean std
    low_data['y_pred']=svr_model.predict(data_train.iloc[:,:-1].values)*std.iloc[-1]+mean.iloc[-1]
    low_data.to_excel('./result/pred.xlsx')









if __name__=='__main__':
    main()