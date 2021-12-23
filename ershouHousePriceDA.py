import requests, time
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# 设置请求头部信息
headers = {
'Accept':'application/json, text/javascript, */*; q=0.01',
'Accept-Encoding':'gzip, deflate, br',
'Accept-Language':'zh-CN,zh;q=0.8',
'Connection':'keep-alive',
'Referer':'http://www.baidu.com/link?url=_andhfsjjjKRgEWkj7i9cFmYYGsisrnm2A-TN3XZDQXxvGsM9k9ZZSnikW2Yds4s&amp;amp;wd=&amp;amp;eqid=c3435a7d00006bd600000003582bfd1f'
}

# 设置二手房列表页URL固定部分
url = 'http://gz.lianjia.com/ershoufang/pg'

# 循环爬取二手房列表页页面信息
for i in range(1, 100):
    if i == 1:
        i = str(i)
        entireURL = (url + i + '/')
        res = requests.get(url=entireURL, headers=headers)
        html = res.content
    else:
        i = str(i)
        entireURL = (url + i + '/')
        res = requests.get(url=entireURL, headers=headers)
        html2 = res.content
        html = html + html2
    # 设置每页请求间隔时间
    time.sleep(0.5)

# 对爬取的页面信息进行解析
htmlResolve = BeautifulSoup(html, 'html.parser')

# 提取房源总价格信息
price = htmlResolve.find_all("div", attrs={"class": "priceInfo"})
tp = []
for p in price:
    totalPrice = p.span.string
    tp.append(totalPrice)

# 提取房源单价信息
unitPriceInfo = htmlResolve.find_all("div", attrs={"class": "unitPrice"})
upi = []
for up in unitPriceInfo:
    unitPrice = up.get_text()
    upi.append(unitPrice)

# 提取房源位置信息
positionInfo = htmlResolve.find_all("div", attrs={"class": "positionInfo"})
pi = []
for ps in positionInfo:
    position = ps.get_text()
    pi.append(position)

# 提取房源户型、面积、朝向等属性信息
houseInfo = htmlResolve.find_all("div", attrs={"class": "houseInfo"})
hi = []
for h in houseInfo:
    house = h.get_text()
    hi.append(house)

# 提取房源关注度信息
followInfo = htmlResolve.find_all("div", attrs={"class": "followInfo"})
fi = []
for f in followInfo:
    follow = f.get_text()
    fi.append(follow)

'''
# 标题
titleInfo = htmlResolve.find_all("div", attrs={"class": "title"})
ti = []
for t in titleInfo:
    title = t.get_text()
    ti.append(title)
'''


# 创建数据表
house = pd.DataFrame({"totalprice": tp, "unitprice": upi, "positioninfo": pi, "houseinfo": hi, "followinfo": fi})


# 对房源属性信息进行特征构造
houseinfo_split = pd.DataFrame((x.split('|') for x in house.houseinfo), index=house.index, columns=["huxing","mianji","chaoxiang","zhuangxiu","louceng","louling","louxing","spec"])

# 对房源关注度信息进行特征构造
followinfo_split = pd.DataFrame((y.split('/') for y in house.followinfo), index=house.index, columns=["guanzhu", "fabu"])

# 将构造的特征字段拼接到原数据表后
house = pd.merge(house, houseinfo_split, right_index=True, left_index=True)
house = pd.merge(house, followinfo_split, right_index=True, left_index=True)


# 对数值型房源单价信息进行特征构造
unitprice_num_split = pd.DataFrame((z.split('元') for z in house.unitprice), index=house.index, columns=["danjia_num","danjia_danwei"])

# 将构造的特征字段拼接到原数据表后
house = pd.merge(house, unitprice_num_split, right_index=True, left_index=True)

# 定义函数，提取字符串中的数字
def get_num(string):
    return (re.findall("\d+\.?\d*", string)[0])

# 提取房源面积信息中的数字
house["mianji_num"] = house["mianji"].apply(get_num)

# 提取房源关注度信息中的数字
house["guanzhu_num"] = house["guanzhu"].apply(get_num)

# 去除提取的数字字段两端的空格
house["danjia_num"] = house["danjia_num"].map(str.strip)
house["mianji_num"] = house["mianji_num"].map(str.strip)
house["guanzhu_num"] = house["guanzhu_num"].map(str.strip)

# 将数值字段的格式转换为float
house["danjia_num"] = house["danjia_num"].str.replace(',', '').astype(float)
house["totalprice"] = house["totalprice"].astype(float)
house["mianji_num"] = house["mianji_num"].astype(float)
house["guanzhu_num"] = house["guanzhu_num"].astype(float)


# 查看房源面积数据的范围
house["mianji_num"].min(), house["mianji_num"].max()

# 对房源面积数据进行分组
bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
group_mianji = ['小于50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450', '450-500', '500-550']
house['group_mianji'] = pd.cut(house['mianji_num'], bins, labels=group_mianji)

# 按房源面积分组对房源数量进行汇总
group_mianji = house.groupby('group_mianji')['group_mianji'].agg(len)

# 绘制房源面积分布图
plt.rc('font', family='STXihei', size=15)
ygroup = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plt.barh([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], group_mianji, color='#205bc3', alpha=0.8, align='center', edgecolor='white')
plt.ylabel('面积分组（单位：平米）')
plt.xlabel('数量')
plt.title('房源面积分布图')
plt.legend(['数量'], loc='upper right')
plt.grid(color='#92a1a2', linestyle='--', linewidth=1, axis='y', alpha=0.4)
plt.yticks(ygroup, ('小于50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450', '450-500', '500-550'))

# 查看绘制的分布图
plt.show()


# 查看房源总价格数据的范围
house["totalprice"].min(), house["totalprice"].max()

# 对房源总价格数据进行分组
bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
group_totalprice = ['小于500', '500-100', '1000-1500', '1500-2000', '2000-2500', '2500-3000', '3000-3500']
house['group_totalprice'] = pd.cut(house['totalprice'], bins, labels=group_totalprice)

# 按房源总价格分组对房源数量进行汇总
group_totalprice = house.groupby('group_totalprice')['group_totalprice'].agg(len)

# 绘制房源总价分布图
plt.rc('font', family='STXihei', size=15)
ygroup = np.array([1, 2, 3, 4, 5, 6, 7])
plt.barh([1, 2, 3, 4, 5, 6, 7], group_totalprice, color='#205bc3', alpha=0.8, align='center', edgecolor='white')
plt.ylabel('总价分组（单位：万元）')
plt.xlabel('数量')
plt.title('房源总价分布图')
plt.legend(['数量'], loc='upper right')
plt.grid(color='#92a1a2', linestyle='--', linewidth=1, axis='y', alpha=0.4)
plt.yticks(ygroup, ('小于500', '500-100', '1000-1500', '1500-2000', '2000-2500', '2500-3000', '3000-3500'))

# 查看绘制的分布图
plt.show()


# 查看房源关注度数据的范围
house["guanzhu_num"].min(), house["guanzhu_num"].max()

# 对房源关注度数据进行分组
bins = [0, 1000, 2000, 3000, 4000, 5000, 6000]
group_guanzhu = ['小于1000', '1000-2000', '2000-3000', '3000-4000', '4000-5000', '5000-6000']
house['group_guanzhu'] = pd.cut(house['guanzhu_num'], bins, labels=group_guanzhu)

# 按房源关注度分组对房源数量进行汇总
group_guanzhu = house.groupby('group_guanzhu')['group_guanzhu'].agg(len)

# 绘制房源关注度分布图
plt.rc('font', family='STXihei', size=15)
ygroup = np.array([1, 2, 3, 4, 5, 6])
plt.barh([1, 2, 3, 4, 5, 6], group_guanzhu, color='#205bc3', alpha=0.8, align='center', edgecolor='white')
plt.ylabel('关注度分组')
plt.xlabel('数量')
plt.title('房源关注度分布图')
plt.legend(['数量'], loc='upper right')
plt.grid(color='#92a1a2', linestyle='--', linewidth=1, axis='y', alpha=0.4)
plt.yticks(ygroup, ('小于1000', '1000-2000', '2000-3000', '3000-4000', '4000-5000', '5000-6000'))

# 查看绘制的分布图
plt.show()


# 写入Excel
house.to_excel('ershouHousePrice.xls')


# 使用房源总价格、面积和关注度三个字段进行聚类
house_type = np.array(house[['totalprice', 'mianji_num', 'guanzhu_num']])

# 设置质心数量参数值为3
cls_house = KMeans(n_clusters=3)

# 计算聚类结果
cls_house = cls_house.fit(house_type)

# 查看分类结果的中心坐标
cls_house.cluster_centers_

# 在原数据表中标注房子所属类别
house['label'] = cls_house.labels_
