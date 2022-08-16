import pandas as pd

from pyanalysis import *

def test():
    excel_name = './source/chapter3/demo/data/catering_sale.xls'
    index_col = '日期'
    data, data_describe = excel_describe(excel_name, index_col)
    # 箱型图
    boxplot(data)
    #
    excel_name2 = './source/chapter3/demo/data/catering_sale.xls'
    data2 = pd.read_excel(excel_name2, names=['date', 'sale'])
    # # print(data2.columns)
    # 频率分布图
    # distribution_histogram(data2,'sale',cut=500)
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
    # excel_name10 = 'source/程序及数据/程序及数据/09第9章  综合评价方法/Pdata9_1_1.txt'
    # data10 = pd.read_table(excel_name10,sep='\s+',header=None)
    # # # print(normalization(data10,method=mean_std_normal))
    # print(normalization(data10,method=max_min_normal))
    # print(normalization(data10,method=vector_normal))
    # print(normalization(data10,scale_transform_normal))
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
    # 主成分分析
    # print(pca(data12))
    #
    # excel_name13='source/chapter5/demo/data/linear_reg.xlsx'
    # excel_name13='source/程序及数据/程序及数据/12第12章  回归分析/Pdata12_6.txt'
    # excel_name13='source/程序及数据/程序及数据/12第12章  回归分析/Pdata12_9.txt'
    # data13=pd.read_table(excel_name13,sep='\s+',header=None)
    # print(data13)
    # # x=pca(data13.iloc[:,:8])
    # # y=data13.iloc[:,8]
    # # data13=pd.concat([x,y],axis=1)
    # #logistic回归 分类预测
    # logistic_model=logistic_regression(data13,3)
    # print(logistic_model.predict(data13.iloc[0:19,:3]))

    #linear线性回归 预测

    # linear_regression(data13,4)


    #
    # #决策树分类
    # excel_name14='source/chapter5/demo/data/sales_data.xls'
    # excel_name14='source/程序及数据/程序及数据/11第11章  多元分析/Pdata11_2.xlsx'
    # data14=pd.read_excel(excel_name14,index_col=0)
    # data14.columns=[i for i in range(len(data14.columns))]
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
    # print(data14)
    # data14.iloc[:19,5]-=1
    # print(data14)
    # bp_network(data14.iloc[:19,:],classes=3,label_col=5,epochs=1000)
    # bp_network(data13,label_col='违约',epochs=1000)
    #
    # #k-means多特征分类
    # excel_name15='source/chapter5/demo/data/consumption_data.xls'
    # data15=pd.read_excel(excel_name15,index_col=0)
    # data15=normalization(data15)
    # find_k(data15,10) #寻找最优k
    # cluster(data15,k=3,save_density_fig=True,save_radar_fig=True)
    #
    #apriori关联度
    # excel_name16='source/chapter5/demo/data/menu_orders.xls'
    # data16=pd.read_excel(excel_name16)
    # apriori(data16)

    # 时间序列建模，首先确定是否是平稳序列，或差分后是否是平稳序列
    # excel_name17 = 'source/chapter5/demo/data/arima_data.xls'
    # data17 = pd.read_excel(excel_name17, index_col=0).dropna()
    # correlation_draw(data17, 0)  # 画出自相关、偏相关图
    # correlation_draw(data17, 1)
    # period_chart(data17.index,data17.iloc[:,0])
    # model=arima(data17,diff=1) #输入时间序列数据和差分次数
    # print(model.forecast(5)) #预测后五天数据

    #离群点检测图
    # outliers_chart(data15,k=3)

    #综合评价
    # excel_name18='source/程序及数据/程序及数据/09第9章  综合评价方法/Pdata9_1_1.txt'
    # data18=pd.read_table(excel_name18,sep='\s+',header=None)
    # data18=normalization(data18,method=scale_transform_normal)
    # print(data18)
    #
    # #topsis
    # print(topsis(data18))

    #灰色关联度
    # coef_,res=grey_relational_degree(data18)
    # print(coef_,res)

    #熵值法
    # w,res=entropy(data18)
    # print(w,res)

    #秩和比
    # print(rank_sum_ratio(data18))

    #判别法分析
    # excel_name19='source/程序及数据/程序及数据/11第11章  多元分析/Pdata11_2.xlsx'
    excel_name19='source/chapter5/demo/data/bankloan.xls'
    data19=pd.read_excel(excel_name19)
    data19.iloc[:,:-1]=normalization(data19.iloc[:,:-1],method=scale_transform_normal)
    # data19.columns=[i for i in range(len(data19.columns))]

    # print(discriminant_classifier(data19.iloc[:20,:],3,5).predict(data19.iloc[-2:,:5].values))
    # print(discriminant_classifier(data19.iloc[:-5,:],2,'违约',method=fisher).predict(data19.iloc[-5:,:-1].values))
    # print(discriminant_classifier(data19.iloc[:-5,:],2,'违约',method=beyes).predict(data19.iloc[-5:,:-1].values))

    #支持向量机分类
    print(svc(data19.iloc[:-5,:],label_col='违约',kernal=('linear','rbf'),C=[1,10,15],rate=0.01).predict(data19.iloc[-5:,:-1].values))

    #灰色预测系统 data必须是时间序列
    # for c in data19.columns[:-1]:
    #     f=GM11(data19.loc[range(1994,2014),c])[0] #得到每一列的预测函数
    #     data19.loc[2014,c]=f(len(data19)-1) #带入自变量 这里自变量从1开始 所以2014为倒数第二个自变量
    #     data19.loc[2015,c]=f(len(data19))




if __name__=='__main__':
    test()
