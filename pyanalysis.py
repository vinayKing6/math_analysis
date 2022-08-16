from typing import Union

from scipy.stats import rankdata
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
import os
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras import optimizers
from statsmodels import *
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns


def excel_describe(excel_name, index_name: Union[str, int, None] = 0, sheet_name: Union[str, int, None] = 0):
    data = pd.read_excel(excel_name, index_col=index_name, sheet_name=sheet_name)
    print('data length: {}'.format(len(data)))
    print(data.describe())
    return data, data.describe()


# 异常值处理箱型图
def boxplot(data, figsize=(5, 10), title='箱型图'):
    _data = data.copy()
    # 正常显示一些图标
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=figsize)
    plt.title(title)
    p = _data.boxplot(return_type='dict', patch_artist=True)
    x = p['fliers'][0].get_xdata()
    y = p['fliers'][0].get_ydata()
    y.sort()

    # for i in range(len(x)):
    #     if i > 0:
    #         plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.05 - 0.8 / (y[i] - y[i - 1]), y[i]))
    #     else:
    #         plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.08, y[i]))

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
        bins.append(i * cut)
    labels = []
    for i in range(n):
        labels.append('[{0},{1})'.format(i * cut, (i + 1) * cut))
    _data['distribution'] = pd.cut(_data[col], bins, labels=labels)
    plt.figure(figsize=figsize)  # 设置图框大小尺寸
    plt.hist(_data[col], bins)
    plt.xticks(range(0, bins[-1], cut))
    plt.xlabel(xlabel)
    plt.grid()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title(title, fontsize=20)
    plt.show()


# 直方图
def histogram(data, col, bins='auto', figsize=(10, 6), xlabel='分层', ylabel='数据', title='直方图'):
    _data = data.copy()
    plt.figure(figsize=figsize)  # 设置图框大小尺寸
    plt.hist(_data[col], bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title(title, fontsize=20)
    plt.show()


# 饼图
def pie_chart(data, labels, figsize=(8, 6), title='饼图', xlabel='数据'):
    plt.figure(figsize=figsize)
    plt.pie(data, labels=labels, autopct='%1.1f%%')
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


# 热力图，查看多特征相关性
def heatmap(corr_data, figsize=(10, 10), title='热力图'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots(figsize=figsize)
    sns.heatmap(corr_data, annot=True, vmax=1, square=True, cmap='Blues')
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


# 比例变换标准化 x/max(Xj)
def scale_transform_normal(data):
    data = data / data.max()
    return data


# 向量归一化 Xij/norm(Xj) norm--->范数 根号（Xij^2求和）
def vector_normal(data):
    for i in range(len(data.columns)):
        data.iloc[:, i] = data.iloc[:, i] / np.linalg.norm(data.iloc[:, i])
    return data


# 标准化数据 默认零-均值标准化
def normalization(data, method=mean_std_normal):
    _data = data.copy()
    return method(data)


# range=max-min(极差) var=std/mean(变异系数=标准差/均值) dis=75%-25%(四分位数间距) null空值
def statistic_addition(data):
    _data = data.copy()
    statistics = _data.describe()

    statistics.loc['range'] = statistics.loc['max'] - statistics.loc['min']
    statistics.loc['var'] = statistics.loc['std'] / statistics.loc['mean']
    statistics.loc['dis'] = statistics.loc['75%'] - statistics.loc['25%']
    statistics.loc['null'] = len(_data) - statistics.loc['count']

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
    return pd.cut(data, w, labels=range(k)), kmodel


# 用于寻找k-means聚类中最优k,即折线折点最大处
def find_k(data, K=None):
    TSSE = []
    if K is None:
        K = len(data)
    for k in range(1, K + 1):
        print(k)
        SSE = []
        model = KMeans(n_clusters=k)
        model.fit(data)
        labels = model.labels_
        centers = model.cluster_centers_
        for label in set(labels):
            SSE.append(np.sum((data.values[labels == label, :] - centers[label, :]) ** 2))
        TSSE.append(np.sum(SSE))

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')
    plt.plot(range(1, K + 1), TSSE, 'b*-')
    plt.xlabel('簇的个数')
    plt.ylabel('簇内离差平方之和')
    plt.show()


# 多特征k-means聚类离散化
def k_means_cluster_nd(data, k, to_excel_name, iteration=500):
    kmodel = KMeans(n_clusters=k, max_iter=iteration, random_state=1234)
    kmodel.fit(data)

    # print result
    r1 = pd.Series(kmodel.labels_).value_counts()
    r2 = pd.DataFrame(kmodel.cluster_centers_)
    r = pd.concat([r2, r1], axis=1)
    r.columns = list(data.columns) + ['类别数目']
    print(r)

    r = pd.concat([data, pd.Series(kmodel.labels_, index=data.index)], axis=1)
    r.columns = list(data.columns) + ['聚类类别']
    r.to_excel(to_excel_name)

    return r, kmodel


# 离散化数据，分类
def cluster(data, k, method=k_means_cluster_nd, save_density_fig=False, save_radar_fig=False, is_draw=False,
            to_excel_name='k_means_result.xlsx'):
    _data = data.copy()
    model = ''
    try:
        d, model = method(_data, k, to_excel_name)
    except Exception:
        d = method(_data, k)

    if save_density_fig:
        for i in range(k):
            density_plot(_data[d['聚类类别'] == i]).savefig('%s.png' % (i))

    if is_draw:
        cluster_plot(_data, d, k)

    if save_radar_fig:
        radar_chart(model.cluster_centers_, data.columns).savefig('radar.png')

    return d, model


# 分类数据雷达图 数据必须标准化 消除量纲
def radar_chart(centers_data, labels):
    labels=list(labels)
    cluster_center = pd.DataFrame(centers_data, columns=labels)
    legen = cluster_center.index
    lstype = ['-', '--', (0, (3, 5, 1, 5, 1, 5)), ':', '-.']
    kinds = list(cluster_center.iloc[:, 0])
    # 由于雷达图要保证数据闭合，因此再添加L列，并转换为 np.ndarray
    cluster_center = pd.concat([cluster_center, cluster_center[[cluster_center.columns[0]]]], axis=1)
    centers = np.array(cluster_center.iloc[:, 0:])

    # 分割圆周长，并让其闭合
    n = len(labels)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angle = np.concatenate((angle, [angle[0]]))

    # 绘图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)  # 以极坐标的形式绘制图形
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 画线
    for i in range(len(kinds)):
        ax.plot(angle, centers[i], linestyle=lstype[i], linewidth=2, label=kinds[i])
    # 添加属性标签
    labels.append(labels[0])
    ax.set_thetagrids(angle * 180 / np.pi, labels)
    plt.title('群体特征分析雷达图')
    plt.legend(legen)
    return plt


# 画图离散化数据
def cluster_plot(data, d, k, figsize=(8, 4), title='离散数据图', xlabel='数据1', ylabel='数据2'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=figsize)
    for j in range(0, k):
        plt.plot(data[d == j], [j for i in d[d == j]], 'o')
    plt.ylim(-0.5, k - 0.5)
    plt.show()


# 聚类密度图
def density_plot(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    p = data.plot(kind='kde', linewidth=2, subplots=True, sharex=False)
    [p[i].set_ylabel('密度') for i in range(len(p))]
    plt.legend()
    return plt


# 基于k-means聚类的离群点检测图 threshold阈值区分离群点
def outliers_chart(data, k, threshold=2, figsize=(8, 4), title='离散数据图', xlabel='标签', ylabel='数据'):
    _data = data.copy()
    _data = normalization(_data)  # 标准化

    model = KMeans(n_clusters=k, max_iter=500)
    model.fit(_data)
    r = pd.concat([_data, pd.Series(model.labels_, index=_data.index)], axis=1)
    r.columns = list(_data.columns) + ['聚类类别']

    norm = []
    for i in range(k):
        norm_tmp = r.iloc[:, :len(r.columns) - 1][r['聚类类别'] == i] - model.cluster_centers_[i]
        norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)
        norm.append(norm_tmp / norm_tmp.median())
    norm = pd.concat(norm)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    norm[norm <= threshold].plot(style='go')
    discrete_points = norm[norm > threshold]
    discrete_points.plot(style='ro')

    for i in range(len(discrete_points)):
        x = discrete_points.index[i]
        n = discrete_points.iloc[i]
        plt.annotate('(%s,%0.2f)' % (x, n), xy=(x, n), xytext=(x, n))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# 主成分分析 将多维数据降维 每条主成分向量由原来的列向量加权得来 主成分大小通过权重反应原来的列向量对主成分向量的影响 详情见司守奎
def pca(data, ratio=0.97):
    instance = PCA()
    instance.fit(data)
    n = 0
    s = 0
    for i in instance.explained_variance_ratio_:
        s = i + s
        n = n + 1
        if s > ratio:
            break
    print('coefficients=', instance.explained_variance_ratio_)
    print('selected numbers=', n)
    n_pca = PCA(n)
    n_pca.fit(data)
    low_d = n_pca.transform(data)
    return pd.DataFrame(low_d)


# logistic 回归 二分类
def logistic_regression(data, label_col):
    x = data.drop(columns=[label_col], axis=1).values
    y = data[label_col].values
    model = LogisticRegression(solver='lbfgs')
    model.fit(x, y)
    print('accuracy: {}'.format(model.score(x, y)))
    print('coefficient: ', model.coef_)  # 回归系数
    print('intercept: ', model.intercept_)  # 截距
    return model


# 线性回归 预测真实值 y=B0+B1X1+B2X2.....+e e--->截距
def linear_regression(data, label_col):
    x = data.drop(columns=[label_col], axis=1).values
    y = data[label_col].values
    model = LinearRegression()
    model.fit(x, y)
    print('accuracy: {}'.format(model.score(x, y)))
    print('coefficient: ', model.coef_)  # 回归系数
    print('intercept: ', model.intercept_)  # 截距
    return model


# Lasso回归 预测真实值 ---> 在较多强相关性特征时，可以使用回归系数降维特征 回归系数为0时表示对预测值无参考意义 C---->惩罚系数
def lasso_regression(data, label_col, C=1000):
    x = data.drop(columns=[label_col], axis=1).values
    y = data[label_col].values
    model = Lasso(C)
    model.fit(x, y)
    print('accuracy: {}'.format(model.score(x, y)))
    print('coefficient: ', model.coef_)  # 回归系数
    print('intercept: ', model.intercept_)  # 截距
    return model


# 决策树分类
def dtc(data, label_col, export_name='dtc_export.dot', pdf_name='dtc.pdf', to_pdf=True):
    _data = data.copy()
    x = _data.drop(columns=[label_col], axis=1).values.astype(int)
    y = _data[label_col].values.astype(int)
    _dtc = DTC(criterion='entropy')
    _dtc.fit(x, y)
    x = pd.DataFrame(x)
    with open(export_name, 'w') as f:
        f = export_graphviz(_dtc, feature_names=_data.columns[:len(_data.columns) - 1], out_file=f)
    if to_pdf:
        command = 'dot -Tpdf {0} -o {1}'.format(export_name, pdf_name)
        os.system(command)
    return _dtc


# one hot 编码
def one_hot(data, capacity):
    result = np.zeros((len(data), capacity))
    for i, pos in enumerate(data):
        result[i, pos] = 1.
    return result


# bp神经网络 默认二分类 建议有大量数据时使用 数据量太小过拟合严重
def bp_network(data, label_col, classes=2, epochs=500, batch_size=128, preprocess=True):
    _data = data.copy()
    x_train = _data.drop(columns=[label_col], axis=1).values
    y_train = _data[label_col].values.astype(int)
    y_train = one_hot(y_train, classes)
    print(y_train)
    if preprocess:
        normalization(x_train)  # 归一化

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


# apriori 寻找元素之间关联性(有点像有向图) support支持度=P(AB) confidence置信度=P(B|A)=P(AB)/P(A)
def apriori(data, support=0.2, confidence=0.5, ms='----'):
    _data = data.copy()
    ct = lambda x: pd.Series(1, index=x[pd.notnull(x)])
    b = list(map(ct, _data.values))
    new_data = pd.DataFrame(b).fillna(0)
    del b
    return find_rule(new_data, support, confidence, ms)


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


# 时间序列预测建模，非平稳序列差分转换成平稳序列（隔k个值的相关性趋于0），使用ARIMA模型预测,diff==成为平稳序列的差分次数
def arima(data, diff=1):
    _data = data.copy()
    D_data = _data.diff().dropna()  # 差分序列，按行进行
    print('差分序列白噪声检验结果：', acorr_ljungbox(_data, lags=1))
    _data = _data.astype('float64')
    pmax = int(len(D_data) / 10)
    qmax = int(len(D_data) / 10)
    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(sm.tsa.arima.ARIMA(_data, order=(p, diff, q)).fit().bic)
            except Exception as e:
                print(e)
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix)
    print(bic_matrix)
    p, q = bic_matrix.stack().idxmin()
    print('BIC的最小值p,q为：{0}，{1}'.format(p, q))
    model = sm.tsa.arima.ARIMA(_data, order=(p, diff, q)).fit()
    print('模型报告：\n', model.summary())
    return model


# 自相关图，偏自相关图
def correlation_draw(data, diff):
    _data = data.copy()
    for i in range(diff):
        _data = _data.diff().dropna()
    plot_acf(_data)
    plot_pacf(_data)
    plt.show()


'''
综合评价方法 TOPSIS求综合评价值 熵值法求特征权重 秩和比求权重
'''


# TOPSIS C+=[MAX(j) for j in columns] C-=[MIN(j) for j in columns]
#  [S+(i)=[((b(ij)-C+(j))^2 for j in columns)^(1/2)] for i in rows] so the same as S- but with C-
#  [f(i)=S-(i)/(S+(i) + S-(i) ) for i in rows] f即评价值
def topsis(data):
    _data = data.copy().values
    cplus = _data.max(axis=0)  # C+正理想解
    cminus = _data.min(axis=0)  # C-负理想解
    print("正理想解=", cplus, "负理想解=", cminus)
    d1 = np.linalg.norm(_data - cplus, axis=1)  # S+
    d2 = np.linalg.norm(_data - cminus, axis=1)  # S-
    print('S+ =', d1, 'S- =', d2)
    f = d2 / (d1 + d2)
    return pd.Series(f)


# 灰色关联度 求评价值f和系数xs rho分辨系数
def grey_relational_degree(data, rho=0.5):
    _data = data.copy().values
    t = _data.max(axis=0) - _data
    mmin = t.min()  # 与每一列最大值的插值的每一列的最小值的最小值
    mmax = t.max()
    xs = (mmin + rho * mmax) / (t + rho * mmax)
    f = xs.mean(axis=1)  # 每一行均值
    return pd.DataFrame(xs), pd.Series(f)


# 熵值法 求权重w 综合评价值f
def entropy(data):
    _data = data.copy().values
    n, m = _data.shape
    cs = _data.sum(axis=0)
    P = _data / cs
    e = -(P * np.log(P)).sum(axis=0) / np.log(n)
    g = 1 - e
    w = g / g.sum()
    f = (P * w).sum(axis=1)
    return pd.Series(w), pd.Series(f)


# 秩和比 求综合评价值
def rank_sum_ratio(data):
    _data = data.copy().values
    n, m = _data.shape
    R = [rankdata(_data[:, i]) for i in np.arange(m)]
    R = np.array(R).T
    RSR = R.mean(axis=1) / n
    return pd.Series(RSR)


'''
判别分析 KNeighborsClassifier Fisher 贝叶斯-----> 多分类
'''


# KNeighborsClassifier
def knn(x, y, classes):
    v = np.cov(x.T)
    params = {}
    params.update(V=v)
    model = KNeighborsClassifier(classes, metric='mahalanobis', metric_params=params)
    model.fit(x, y)
    print('accuracy', model.score(x, y))
    return model


# Fisher 判别分类
def fisher(x, y, classes):
    v = np.cov(x.T)  # 计算协方差矩阵
    model = LDA()
    model.fit(x, y)
    print('accuracy', model.score(x, y))
    return model


# 贝叶斯判别分类
def beyes(x, y, classes):
    v = np.cov(x.T)  # 计算协方差矩阵
    model = GaussianNB()
    model.fit(x, y)
    print('accuracy', model.score(x, y))
    return model


# 判别法分析 分类
def discriminant_classifier(data, label_col, classes=2, method=knn):
    _data = data.copy()
    x_train = _data.drop(columns=[label_col], axis=1).values.astype(float)
    y_train = _data[label_col].values.astype(int)
    return method(x_train, y_train, classes)


# 一维数据(若多维，请循环列数)灰色预测系统 ---->多用于时间序列预测 关键思想---->累加法、微分方程 详情见司守奎
def GM11(data):
    x0 = data.values
    x1 = x0.cumsum()  # 1-AGO序列
    z1 = (x1[:len(x1) - 1] + x1[1:]) / 2.0  # 紧邻均值（MEAN）生成序列
    z1 = z1.reshape((len(z1), 1))
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Yn = x0[1:].reshape((len(x0) - 1, 1))
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn)  # 计算参数
    f = lambda k: (x0[0] - b / a) * np.exp(-a * (k - 1)) - (x0[0] - b / a) * np.exp(-a * (k - 2))  # 还原值
    delta = np.abs(x0 - np.array([f(i) for i in range(1, len(x0) + 1)]))
    C = delta.std() / x0.std()
    P = 1.0 * (np.abs(delta - delta.mean()) < 0.6745 * x0.std()).sum() / len(x0)
    return f, a, b, x0[0], C, P  # 返回灰色预测函数、a、b、首项、方差比、小残差概率


# 支持向量机分类 svc kernal-->核函数 C--->惩罚系数 gamma 核函数参数r
def svc(data, label_col, kernal=('linear', 'rbf'), C=[1], rate=0.1, cv_num=5):
    _data = data.copy()
    x_train = _data.drop(columns=[label_col], axis=1).values
    y_train = _data[label_col].values.astype(int)
    # k-折验算
    id = 0
    max = 0
    maxid = 0
    result = []
    for k in kernal:
        for c in C:
            gama = rate
            while (gama <= 1):
                clf = SVC(kernel=k, C=c, gamma=gama)
                # 开始进行模型的拟合，训练
                clf.fit(x_train, y_train)
                # 拟合分数
                score = clf.score(x_train, y_train)
                result.append([k, c, gama, score])
                # print(id, result[id])
                if max < score:
                    max = score
                    maxid = id

                gama = gama + rate
                id = id + 1

    print("the best model:")
    print(result[maxid])

    # 使用得到的超参数进行模型的训练
    clf = SVC(kernel=result[maxid][0], C=result[maxid][1], gamma=result[maxid][2])

    clf.fit(x_train, y_train)

    return clf


# 支持向量机回归 预测真实值
def svr(data, label_col, kernal='linear', C=1, gama='auto'):
    _data = data.copy()
    x_train = _data.drop(columns=[label_col], axis=1).values
    y_train = _data[label_col].values.astype(int)
    # 使用得到的超参数进行模型的训练
    clf = SVR(kernel=kernal, C=C, gamma=gama)
    clf.fit(x_train, y_train)

    return clf


def linear_svr(data, label_col):
    _data = data.copy()
    x_train = _data.drop(columns=[label_col], axis=1).values
    y_train = _data[label_col].values
    # 使用得到的超参数进行模型的训练
    clf = LinearSVR()
    clf.fit(x_train, y_train)

    return clf
