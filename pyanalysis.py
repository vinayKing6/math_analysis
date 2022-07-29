from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import export_graphviz
import os
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras import optimizers
from statsmodels import *
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

def excel_describe(excel_name, index_name: Union[str, int, None] = 0, sheet_name: Union[str, int, None] = 0):
    data = pd.read_excel(excel_name, index_col=index_name, sheet_name=sheet_name)
    print('data length: {}'.format(len(data)))
    print(data.describe())
    return data, data.describe()


# 异常值处理箱型图
def boxplot(data):
    _data = data.copy()
    # 正常显示一些图标
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure()
    p = _data.boxplot(return_type='dict')
    x = p['fliers'][0].get_xdata()
    y = p['fliers'][0].get_ydata()
    y.sort()

    for i in range(len(x)):
        if i > 0:
            plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.05 - 0.8 / (y[i] - y[i - 1]), y[i]))
        else:
            plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.08, y[i]))

    plt.show()


# 频率分布直方图
def distribution_histogram(data, col, cut, figsize=(10, 6), xlabel='分层', title='频率分布直方图'):
    _data = data.copy()
    data_size = len(_data)
    max = _data.max(axis=0)[col]
    min = _data.min(axis=0)[col]
    n = round((max - min) / cut)
    bins = []
    for i in range(n + 1):
        bins.append(i * 500)
    labels = []
    for i in range(n):
        labels.append('[{0},{1})'.format(i * 500, (i + 1) * 500))
    _data['distribution'] = pd.cut(_data[col], bins, labels=labels)
    plt.figure(figsize=figsize)  # 设置图框大小尺寸
    plt.hist(_data['sale'], bins)
    plt.xticks(range(0, bins[-1], cut))
    plt.xlabel(xlabel)
    plt.grid()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title(title, fontsize=20)
    plt.show()


# 饼图
def pie_chart(data, labels, figsize=(8, 6), title='饼图', xlabel='数据'):
    plt.figure(figsize=figsize)
    plt.pie(data, labels=labels)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.title(title)
    plt.xlabel(xlabel)
    plt.axis('equal')
    plt.show()


# 条形图
def bar_chart(data, labels, figsize=(8, 4), title='条形图', xlabel='标签', ylabel='数据'):
    plt.figure(figsize=figsize)
    plt.bar(labels, data)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# 折现对比图
def line_compair_chart(xdata, ydata, labels, figsize=(8, 4), title='对比图', xlabel='标签', ylabel='数据'):
    colors = ['green', 'red', 'blue', 'skyblue', 'black']
    markers = ['o', 's', 'x', 'v', '>', '<', '^']
    plt.figure(figsize=figsize)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    for i in range(len(labels)):
        plt.plot(xdata, ydata[i], color=colors[i], label=labels[i], marker=markers[i])
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# 周期图
def period_chart(xdata, ydata, x_locator=7, figsize=(8, 4), title='周期图', xlabel='标签', ylabel='数据'):
    plt.figure(figsize=figsize)
    plt.plot(xdata, ydata)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 设置x轴间隔
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(x_locator))
    plt.title(title)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()


# 贡献度帕累托图 80%盈利产品
def pareto_chart(data, labels=None, figsize=(8, 4), title='帕累托图', xlabel='标签', ylabel='数据'):
    _data = data.copy()
    if labels is not None:
        _data.index = labels
    _data = _data.sort_values(ascending=False)
    # print(_data)
    # 正常显示图标
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=figsize)
    _data.plot(kind='bar')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    # 累积和比例
    p = 1.0 * _data.cumsum() / _data.sum()
    print(p)
    p.plot(color='r', secondary_y=True, style='-o', linewidth=2)
    plt.annotate(format(p[6], '.4%'), xy=(6, p[6]), xytext=(6 * 0.9, p[6] * 0.9),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.show()


# 散点图，查看线性相关性
def scatter_chart(xdata, ydata, figsize=(8, 4), title='散点图', xlabel='数据1', ylabel='数据2'):
    plt.figure(figsize=figsize)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(xdata, ydata)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# 最大-最小标准化 (x-min)/(max-min)
def max_min_normal(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data


# 零-均值标准化数据 (x-mean)/std
def mean_std_normal(data):
    data = (data - data.mean()) / data.std()
    return data


# 小数定标标准化 x/10^k k取决于绝对值的最大值
def decimal_normal(data):
    data = data / 10 ** np.ceil(np.log10(data.abs().max()))
    return data


# 标准化数据 默认零-均值标准化
def normalization(data, method=mean_std_normal):
    _data = data.copy()
    return method(data)


# range=max-min(极差) var=std/mean(变异系数=标准差/均值) dis=75%-25%(四分位数间距)
def statistic_addition(data):
    _data = data.copy()
    statistics = _data.describe()

    statistics.loc['range'] = statistics.loc['max'] - statistics.loc['min']
    statistics.loc['var'] = statistics.loc['std'] / statistics.loc['mean']
    statistics.loc['dis'] = statistics.loc['75%'] - statistics.loc['25%']

    return statistics


# 拉格朗日插值
def lagrange_interp(x, y, a):
    s = 0.0
    for i in range(len(y)):
        t = y[i]
        for j in range(len(y)):
            if i != j:
                t *= (a - x[j]) / (x[i] - x[j])
        s += t
    return s


# 牛顿插值法 差商
def difference_quotient(x, y):
    n = len(x)
    a = np.zeros([n, n], dtype=float)
    for i in range(n):
        a[i][0] = y[i]
    for j in range(1, n):
        for i in range(j, n):
            a[i][j] = (a[i][j - 1] - a[i - 1][j - 1]) / (x[i] - x[i - j])
    return a


def newton_interp(x, y, _x):
    a = difference_quotient(x, y)
    n = len(x)
    s = a[n - 1][n - 1]
    j = n - 2
    while j >= 0:
        s = np.polyadd(np.polymul(s, np.poly1d(
            [x[j]], True)), np.poly1d([a[j][j]]))
        j -= 1
    _y = np.polyval(s, _x)
    return _y


# 插值
def interp_column(s, n, k=5, method=lagrange_interp):
    # 默认k=5,取n-5,n+5范围插值
    y = s[list(range(n - k, n)) + list(range(n + 1, n + k + 1))]
    y = y[y.notnull()]  # 剔除null值
    return method(y.index, list(y), n)


def interp(data, method=lagrange_interp):
    _data = data.copy()
    for col in _data.columns:
        for j in range(len(_data[col])):
            if (_data[col].isnull())[j]:
                _data[col][j] = interp_column(_data[col], j, method=method)
    return _data


# 等宽离散化，将数据离散成k份相似成分
def equal_width_cluster(data, k):
    return pd.cut(data, k, labels=range(k))


# 等频率离散化
def equal_fraguency_cluster(data, k):
    w = [1.0 * i / k for i in range(k + 1)]
    w = data.describe(percentiles=w)[4:4 + k + 1]
    w[0] = w[0] * (1 - 1e-10)
    return pd.cut(data, w, labels=range(k))


# 一维 k-means聚类离散化
def k_means_cluster_1d(data, k):
    kmodel = KMeans(n_clusters=k)
    kmodel.fit(np.array(data).reshape((len(data), 1)))
    c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)
    w = c.rolling(2).mean()
    w = w.dropna()
    w = [0] + list(w[0]) + [data.max()]
    return pd.cut(data, w, labels=range(k))

#多特征k-means聚类离散化
def k_means_cluster_nd(data,k,to_excel_name,iteration=500):
     kmodel=KMeans(n_clusters=k,max_iter=iteration,random_state=1234)
     kmodel.fit(data)

     #print result
     r1=pd.Series(kmodel.labels_).value_counts()
     r2=pd.DataFrame(kmodel.cluster_centers_)
     r=pd.concat([r2,r1],axis=1)
     r.columns=list(data.columns)+['类别数目']
     print(r)

     r=pd.concat([data,pd.Series(kmodel.labels_,index=data.index)],axis=1)
     r.columns=list(data.columns)+['聚类类别']
     r.to_excel(to_excel_name)

     return r



#离散化数据，分类
def cluster(data, k, method=k_means_cluster_nd,save_density_fig=False, is_draw=False,to_excel_name='k_means_result.xlsx'):
    _data = data.copy()
    try:
        d = method(_data, k, to_excel_name)
    except Exception:
        d=method(_data,k)

    if save_density_fig:
        for i in range(k):
            density_plot(_data[d['聚类类别']==i]).savefig('%s.png'%(i))

    if is_draw:
        cluster_plot(_data, d, k)
    return d

#画图离散化数据
def cluster_plot(data, d, k, figsize=(8, 4), title='离散数据图', xlabel='数据1', ylabel='数据2'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=figsize)
    for j in range(0, k):
        plt.plot(data[d == j], [j for i in d[d == j]], 'o')
    plt.ylim(-0.5, k - 0.5)
    plt.show()

#聚类密度图
def density_plot(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    p=data.plot(kind='kde',linewidth=2,subplots=True,sharex=False)
    [p[i].set_ylabel('密度') for i in range(len(p))]
    plt.legend()
    return plt

#主成分分析 将多维数据降维
def pca(data,ratio=0.97):
    instance=PCA()
    instance.fit(data)
    n=0
    s=0
    for i in instance.explained_variance_ratio_:
        s=i+s
        n=n+1
        if s >ratio:
            break
    # print(instance.explained_variance_ratio_)
    # print(n)
    n_pca=PCA(n)
    n_pca.fit(data)
    low_d=n_pca.transform(data)
    return pd.DataFrame(low_d)

#logistic 回归 分类
def logistic_regression(data,label_col):
    x=data.drop(columns=[label_col],axis=1).values
    y=data[label_col].values
    model=LogisticRegression()
    model.fit(x,y)
    print('accuracy: {}'.format(model.score(x,y)))
    return model

#决策树分类
def dtc(data,label_col,export_name='dtc_export.dot',pdf_name='dtc.pdf',to_pdf=True):
    _data=data.copy()
    x=_data.drop(columns=[label_col],axis=1).values.astype(int)
    y=_data[label_col].values.astype(int)
    _dtc=DTC(criterion='entropy')
    _dtc.fit(x,y)
    x=pd.DataFrame(x)
    with open(export_name,'w') as f:
        f=export_graphviz(_dtc,feature_names=_data.columns[:len(_data.columns)-1],out_file=f)
    if to_pdf:
        command='dot -Tpdf {0} -o {1}'.format(export_name,pdf_name)
        os.system(command)
    return _dtc

# one hot 编码
def one_hot(data, capacity):
    result = np.zeros((len(data), capacity))
    for i, pos in enumerate(data):
        result[i, pos] = 1.
    return result

#bp神经网络 默认二分类
def bp_network(data,label_col,classes=2,epochs=500,batch_size=128,preprocess=True):
    _data=data.copy()
    x_train=_data.drop(columns=[label_col],axis=1).values
    y_train=_data[label_col].values.astype(int)
    y_train=one_hot(y_train,classes)
    print(y_train)
    if preprocess:
        normalization(x_train)#归一化

    model = Sequential()
    model.add(Dense(64, input_shape=(x_train.shape[1],), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer=optimizers.adam_v2.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

#apriori 寻找元素之间关联性(有点像有向图) support支持度=P(AB) confidence置信度=P(B|A)=P(AB)/P(A)
def apriori(data,support=0.2,confidence=0.5,ms='----'):
    _data=data.copy()
    ct=lambda x:pd.Series(1,index=x[pd.notnull(x)])
    b=list(map(ct,_data.values))
    new_data=pd.DataFrame(b).fillna(0)
    del b
    return find_rule(new_data,support,confidence,ms)



def connect_string(x, ms):
    x = list(map(lambda i: sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            if x[i][:l - 1] == x[j][:l - 1] and x[i][l - 1] != x[j][l - 1]:
                r.append(x[i][:l - 1] + sorted([x[j][l - 1], x[i][l - 1]]))
    return r


# 寻找关联规则的函数
def find_rule(d, support, confidence, ms='--'):
    result = pd.DataFrame(index=['support', 'confidence'])  # 定义输出结果

    support_series = 1.0 * d.sum() / len(d)  # 支持度序列
    print(d)
    print(support_series)
    column = list(support_series[support_series > support].index)  # 初步根据支持度筛选
    k = 0

    while len(column) > 1:
        k = k + 1
        print('\n正在进行第%s次搜索...' % k)
        column = connect_string(column, ms)
        print('数目：%s...' % len(column))
        sf = lambda i: d[i].prod(axis=1, numeric_only=True)  # 新一批支持度的计算函数

        # 创建连接数据，这一步耗时、耗内存最严重。当数据集较大时，可以考虑并行运算优化。
        d_2 = pd.DataFrame(list(map(sf, column)), index=[ms.join(i) for i in column]).T

        support_series_2 = 1.0 * d_2[[ms.join(i) for i in column]].sum() / len(d)  # 计算连接后的支持度
        column = list(support_series_2[support_series_2 > support].index)  # 新一轮支持度筛选
        support_series = support_series.append(support_series_2)
        column2 = []

        for i in column:  # 遍历可能的推理，如{A,B,C}究竟是A+B-->C还是B+C-->A还是C+A-->B？
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j] + i[j + 1:] + i[j:j + 1])

        cofidence_series = pd.Series(index=[ms.join(i) for i in column2])  # 定义置信度序列

        for i in column2:  # 计算置信度序列
            cofidence_series[ms.join(i)] = support_series[ms.join(sorted(i))] / support_series[ms.join(i[:len(i) - 1])]

        for i in cofidence_series[cofidence_series > confidence].index:  # 置信度筛选
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(['confidence', 'support'], ascending=False)  # 结果整理，输出
    print('\n结果为：')
    print(result)
    return result

#时间序列预测建模，非平稳序列差分转换成平稳序列（隔k个值的相关性趋于0），使用ARIMA模型预测,diff==成为平稳序列的差分次数
def arima(data,diff=1):
    _data=data.copy()
    D_data=_data.diff().dropna()#差分序列，按行进行
    print('差分序列白噪声检验结果：',acorr_ljungbox(_data,lags=1))
    _data=_data.astype('float64')
    pmax=int(len(D_data)/10)
    qmax=int(len(D_data)/10)
    bic_matrix=[]
    for p in range(pmax+1):
        tmp=[]
        for q in range(qmax+1):
            try:
                tmp.append(sm.tsa.arima.ARIMA(_data,order=(p,diff,q)).fit().bic)
            except Exception as e:
                print(e)
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix=pd.DataFrame(bic_matrix)
    print(bic_matrix)
    p,q=bic_matrix.stack().idxmin()
    print('BIC的最小值p,q为：{0}，{1}'.format(p,q))
    model=sm.tsa.arima.ARIMA(_data,order=(p,diff,q)).fit()
    print('模型报告：\n',model.summary())
    return model

#自相关图，偏自相关图
def diff_draw(data,diff):
    _data=data.copy()
    for i in range(diff):
        _data=_data.diff().dropna()
    plot_acf(_data)
    plot_pacf(_data)
    plt.show()

def test():
    # excel_name = './source/chapter3/demo/data/catering_sale.xls'
    # index_col = '日期'
    # data, data_describe = excel_describe(excel_name, index_col)
    # # 箱型图
    # # boxplot(data)
    #
    # excel_name2 = './source/chapter3/demo/data/catering_sale.xls'
    # data2 = pd.read_excel(excel_name2, names=['date', 'sale'])
    # # print(data2.columns)
    # # 频率分布图
    # # distribution_histogram(data2,'sale',500)
    #
    # excel_name3 = './source/chapter3/demo/data/dish_sale.xls'
    # data3 = pd.read_excel(excel_name3, index_col=[0], names=['A', 'B', 'C'])
    # print(data3)
    # # 饼图
    # # pie_chart(data=data3['A'],labels=data3.index)
    #
    # # 条形图
    # # bar_chart(data=data3['A'],labels=data3.index,title='A部门月销量条形图')
    #
    # excel_name4 = './source/chapter3/demo/data/dish_sale_b.xls'
    # data4 = pd.read_excel(excel_name4, index_col=0)
    # # 折线对比图
    # # line_compair_chart(data4.index,ydata=[data4[col] for col in data4.columns],labels=data4.columns)
    # # 统计量
    # print(statistic_addition(data4))
    #
    # excel_name5 = './source/chapter3/demo/data/user.csv'
    # data5 = pd.read_csv(excel_name5)
    # excel_name6 = './source/chapter3/demo/data/Steal user.csv'
    # data6 = pd.read_csv(excel_name6)
    # # 周期图
    # # period_chart(data5['Date'],data5['Eletricity'],title='正常用户')
    # # period_chart(data6['Date'],data6['Eletricity'],title='窃电用户')
    #
    # excel_name7 = './source/chapter3/demo/data/catering_dish_profit.xls'
    # data7 = pd.read_excel(excel_name7, index_col='菜品名')
    # data7.columns = ['id', 'profits']
    # # 帕累托图
    # # pareto_chart(data7['profits'], figsize=(10, 6))
    #
    # excel_name8 = './source/chapter3/demo/data/catering_sale_all.xls'
    # data8 = pd.read_excel(excel_name8, index_col='日期')
    # # 散点图
    # # scatter_chart(data3['B'],data3['C'])
    # x = [3, 5, 6, 7, 8]
    # y = [4]
    # z = np.array(x) * np.array(y)
    # # scatter_chart(data8['翡翠蒸香茜饺'],data8['香煎韭菜饺'])
    # # scatter_chart(x,z)
    # # print(data8.corr(method='spearman'))  # 相关系数矩阵,pearson spearman等方法
    # # print(data8.cov())  # 协方差矩阵
    #
    # # 插值
    # # data9 = interp(data2)
    # # data9.to_excel('./lagrange_interp.xlsx')
    # # interp(data9,method=newton_interp)
    # # data9.to_excel('./newton_intero.xlsx')
    #
    # # 标准化
    # excel_name10 = './source/chapter3/demo/data/normalization_data.xls'
    # data10 = pd.read_excel(excel_name10)
    # # print(normalization(data10,method=mean_std_normal))
    # # print(normalization(data10,method=max_min_normal))
    # # print(normalization(data10,method=decimal_normal))
    #
    # excel_name11 = './source/chapter3/demo/data/discretization_data.xls'
    # data11 = pd.read_excel(excel_name11, names=['data'])
    # #一维数据离散化
    # # print(cluster(data11['data'], 4,method=k_means_cluster_1d, is_draw=True))
    # # print(cluster(data11['data'], 4,method=equal_width_cluster, is_draw=True))
    # # print(cluster(data11['data'], 4,method=equal_fraguency_cluster, is_draw=True))
    #
    # excel_name12='./source/chapter3/demo/data/principal_component.xls'
    # data12=pd.read_excel(excel_name12)
    # #主成分分析
    # # pca(data12).to_excel('principal_component_result.xls')
    #
    # excel_name13='source/chapter5/demo/data/bankloan.xls'
    # data13=pd.read_excel(excel_name13)
    # # x=pca(data13.iloc[:,:8])
    # # y=data13.iloc[:,8]
    # # data13=pd.concat([x,y],axis=1)
    # #logistic回归 分类预测
    # # logistic_model=logistic_regression(data13,'违约')
    # # print(logistic_model.predict(data13.iloc[0:5,:8]))
    #
    # #决策树分类
    # excel_name14='source/chapter5/demo/data/sales_data.xls'
    # data14=pd.read_excel(excel_name14,index_col='序号')
    # data14.columns=['weather','weekend','off','sales']
    # # print(data14)
    # # 1代表高，好，是，反之-1
    # data14.replace(['是','高','好'],1,inplace=True)
    # data14=data14[data14==1].replace(np.nan,0)
    # # print(data14)
    # # dtc(data14,label_col='sales')
    #
    # #测试聚类与决策树分类
    # # nor_data13=normalization(data13.iloc[:,:len(data13.columns)-1])
    # # nor_data13=abs(nor_data13)
    # # gen_class=[]
    # # print(nor_data13)
    # # for col in nor_data13.columns:
    # #     gen_class.append(cluster(nor_data13[col],k=4,is_draw=False))
    # # gen_class.append(data13.iloc[:,-1])
    # # new_pd=pd.concat(gen_class,axis=1)
    # # dtc(new_pd,label_col='违约')
    # # new_pd.to_excel('test_cluster_dtc.xlsx')
    #
    # #bp神经网络分类
    # # bp_network(data14,label_col='sales',epochs=1000) 0.77
    # # bp_network(data13,label_col='违约',epochs=1000) 0.84
    #
    # #k-means多特征分类
    # excel_name15='source/chapter5/demo/data/consumption_data.xls'
    # data15=pd.read_excel(excel_name15,index_col=0)
    # # cluster(data15,k=4,save_density_fig=True)
    #
    # #apriori关联度
    # excel_name16='source/chapter5/demo/data/menu_orders.xls'
    # data16=pd.read_excel(excel_name16)
    # # apriori(data16)

    #时间序列建模，首先确定是否是平稳序列，或差分后是否是平稳序列
    excel_name17='source/chapter5/demo/data/arima_data.xls'
    data17=pd.read_excel(excel_name17,index_col=0).dropna()
    # diff_draw(data17,0) 画出自相关、偏相关图
    # diff_draw(data17,1)
    # period_chart(data17.index,data17.iloc[:,0])
    # model=arima(data17,diff=1) #输入时间序列数据和差分次数
    # print(model.forecast(5)) #预测后五天数据


if __name__ == '__main__':
    test()

