import sys
from datetime import datetime

sys.path.append('..')
from pyanalysis import *
import pandas as pd

def main():
    file_name='./data/air_data.csv'
    data=pd.read_csv(file_name,encoding='utf-8')

    #统计量描述
    statistic=statistic_addition(data)
    # statistic.T.to_excel('./result/statistic.xlsx')

    #会员人数年份直方图
    ffp=data['FFP_DATE'].apply(lambda x:datetime.strptime(x,'%Y/%m/%d'))
    ffp_year=pd.DataFrame(ffp.map(lambda x:x.year))
    # histogram(ffp_year,col='FFP_DATE',xlabel='年份',ylabel='入会人数')

    #会员男女比例图
    male=pd.value_counts(data['GENDER'])['男']
    female=pd.value_counts(data['GENDER'])['女']
    # pie_chart([male,female],labels=['男','女'],title='会员性别比例')

    #会员级别人数条形图
    lv_counts=pd.value_counts(data['FFP_TIER'])
    # bar_chart(lv_counts,lv_counts.index,xlabel='会员级别',ylabel='会员人数')

    #会员年龄箱型图
    age=data['AGE'].dropna().astype('int64')
    age=pd.DataFrame(age)
    # boxplot(age,title='会员年龄箱型图')

    lte=pd.DataFrame(data['LAST_TO_END'])
    fc=pd.DataFrame(data['FLIGHT_COUNT'])
    sks=pd.DataFrame(data['SEG_KM_SUM'])
    # boxplot(lte,title='最后一次乘机至结束的时长箱型图')
    # boxplot(fc,title='飞行次数箱型图')
    # boxplot(sks,title='总飞行公里数箱型图')

    #会员兑换积分次数直方图
    ec=pd.DataFrame(data['EXCHANGE_COUNT'])
    # histogram(ec,col='EXCHANGE_COUNT',bins=5,title='会员兑换积分次数直方图',xlabel='兑换次数',ylabel='会员人数')

    #会员总累计积分箱型图
    ps=pd.DataFrame(data['Points_Sum'])
    # boxplot(ps,title='会员总累计积分箱型图')

    #相关性分析
    data_corr=data[['FFP_TIER','FLIGHT_COUNT','LAST_TO_END','SEG_KM_SUM','EXCHANGE_COUNT','Points_Sum']]
    age=data['AGE'].fillna(0)
    data_corr['AGE']=age.astype('int64')
    data_corr['ffp_year']=ffp_year.values
    corr=data_corr.corr(method='pearson')
    # print(corr)
    # heatmap(corr)

    #数据清理 丢弃异常值
    print(data.shape)
    data_notnull=data.dropna()
    print(data_notnull.shape)
    index1=data_notnull['SUM_YR_1']!=0
    index2=data_notnull['SUM_YR_2']!=0
    index3=(data_notnull['SEG_KM_SUM']>0)&(data_notnull['avg_discount']!=0)
    index4=data_notnull['AGE']<100
    clean_data=data_notnull[(index1 | index2) & index3 & index4]
    print(clean_data.shape)
    # clean_data.to_excel('./result/clean_data.xlsx')

    #RFM模型挖掘价值客户
    data_select=clean_data[['FFP_DATE','LOAD_TIME','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]
    print(data_select.head())
    #数据变换
    L=pd.to_datetime(data_select['LOAD_TIME'])-pd.to_datetime(data_select['FFP_DATE'])
    L=L.astype('str').str.split().str[0]
    L=L.astype('int')/30
    data_select['L']=L
    data_features=data_select.drop(columns=['FFP_DATE','LOAD_TIME'])
    print(data_features.head())

    #标准化数据
    data_scale=normalization(data_features)
    data_scale.columns=['R','F','M','C','L']
    # data_scale.to_excel('./result/data_scale.xlsx')
    print(data_scale.head())

    #k-means聚类 分类客户群体
    # find_k(data_scale,10) 查找折点 这里k选取5
    cluster_data,kmodel=cluster(data_scale,5,to_excel_name='./result/k-means.xlsx',save_density_fig=True,save_radar_fig=True)




if __name__=='__main__':
    main()

