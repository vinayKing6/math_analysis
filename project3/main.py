import sys
sys.path.append('..')
from pyanalysis import *
import numpy as np
import pandas as pd

def main():
    inputfile='./data/GoodsOrder.csv'
    inputfile2='./data/GoodsTypes.csv'
    data=pd.read_csv(inputfile,encoding='gbk')
    data2=pd.read_csv(inputfile2,encoding='gbk')
    # print(statistic_addition(data['id']))

    #查看商品销量数据
    group=data.groupby(['Goods']).count().reset_index()
    sorted_data=group.sort_values('id',ascending=False)
    # bar_chart(sorted_data['id'][:10],sorted_data.index[:10])

    data_nums=data.shape[0]
    #添加种类
    sort_links=pd.merge(sorted_data,data2)
    sort_links=sort_links.groupby(['Types'])['id'].sum().reset_index()
    sort_links=sort_links.sort_values('id',ascending=False).reset_index()
    # pie_chart(sort_links['id'][:10],sort_links['Types'][:10])

    #查看非酒精类饮料比例
    merge_data=pd.merge(sorted_data,data2)
    select=merge_data.loc[merge_data['Types']=='非酒精饮料']
    # pie_chart(select['id'],select['Goods'],title='非酒精饮料比例图')

    #数据处理
    data['Goods']=data['Goods'].apply(lambda x:','+x)
    data=data.groupby(['id']).sum().reset_index()
    data['Goods']=data['Goods'].apply(lambda x:[x[1:]])
    data_list=list(data['Goods'])
    data_iteration=[]
    for p in data_list:
        sp=p[0].split(',')
        data_iteration.append(sp)
    pd.DataFrame(data_iteration).to_excel('./result/data_transform.xlsx')
    inputfile3='./result/data_transform.xlsx'
    clean_data=pd.read_excel(inputfile3,index_col=0)
    print(clean_data.head())
    apriori(clean_data,support=0.01,confidence=0.1).to_excel('./result/final.xlsx')



if __name__=='__main__':
    main()