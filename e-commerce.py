#!/usr/bin/env python
# coding: utf-8
# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from datetime import date,timedelta
get_ipython().run_line_magic('matplotlib', 'inline')
os.getcwd()

customers_df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-berezin-33/olist_customers_dataset.csv')
items_df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-berezin-33/olist_order_items_dataset.csv', parse_dates=['shipping_limit_date'])
orders_df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-berezin-33/olist_orders_dataset.csv', parse_dates=['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date'])


# Для решения задачи проведи предварительное исследование данных и #сформулируй, что должно считаться покупкой. 
# Обосновать свой выбор ты можешь с помощью фактов оплат, статусов #заказов и других имеющихся данных.

customers_df.info()
items_df.info()
orders_df.info()

orders_df.order_status.value_counts()

# Исходя из полученных значений, проверим все ли товары имеют оплату.
order_approved = orders_df.groupby(['customer_id', 'order_status'], as_index=False).agg({'order_approved_at' : 'count'}) /
.sort_values('order_approved_at',  ascending=False)
order_approved

# В результате получаем что 160 позиций еще не оплачены.
order_approved.order_approved_at.value_counts()

# Сделали отбор клиентов которые еще не оплатили товар.
order_approved_null = order_approved.query('order_approved_at == 0').groupby(['customer_id', 'order_status'], as_index=False)                                     .agg({'order_approved_at' : 'count'})
order_approved_null

# Клиенты которые произвели оплату.
order_approved_not_null = order_approved.query('order_approved_at == 1').groupby(['customer_id', 'order_status'], as_index=False)                                         .agg({'order_approved_at' : 'count'})
order_approved_not_null

# Клиенты которые не оплатили товар по статусам заказа.
order_approved_null.order_status.value_counts()

# В итоге: Продажей в нашем случае будем считать товары, по которым прошла оплата в указанную дату, 
# не зависимо от того был ли товар возвращен или нет. 
# Т.к. даже товары из категории delivered были не оплачены, но доставлены клиенту.
# Оплаченных позиций получилось 99 281. 

# 1.Сколько у нас пользователей, которые совершили покупку только один раз?

# Для определения этого значения возьмем две таблицы customers_df и orders_df и соеденим их.
# Затем проведем группировку по колонке customer_unique_id и посчитаем для нее значения order_id, 
# Это нужно для того, чтобы собрать все заказы от конкретного покупателя, после этого применяем
# функцию value_counts(), для определения количества пользователей с одной покупкой.
# В нашем случае 93 099 пользоватлей совершили одну покупку. 

df_cust_orders = customers_df.merge(orders_df, how='inner', on='customer_id')

df_cust_orders.info()

total_orders = df_cust_orders.groupby('customer_unique_id', as_index=False) \
                             .agg({'order_id' : 'count'}) \
                             .sort_values('order_id', ascending=False)

total_orders['order_id'].value_counts()

# 2. Сколько заказов в месяц в среднем не доставляется по разным причинам (вывести детализацию по причинам)? 

# Что бы узнать какое количество не доставленного товара мы имеем, нам необходимо отфильтровать колонку
# order_delivered_customer_date, т.к. в ней находятся все заказы которые были доставленны клиентам. 
# При проверке колонок на пусты значения данная колонка показала 2965 пустых значение.
# Из этого можно сделать вывод что такое количество товаров не было доставлено.

undelivered = df_cust_orders[df_cust_orders['order_delivered_customer_date'].isnull()]

undelivered["year_month"] = undelivered["order_estimated_delivery_date"].dt.to_period("M")

value = undelivered.query('order_status != "delivered"').groupby(['year_month', 'order_status'], as_index=False) \
                   .agg({'customer_city' : 'count'}) \
                   .pivot(index='year_month', columns='order_status', values='customer_city').fillna(0).mean().round()
value

# 3. По каждому товару определить, в какой день недели товар чаще всего покупается.

# Для этого используем таблицу items_df, в которой указанны product_id —  ид товара,
# order_item_id —  идентификатор товара внутри одного заказа, так же сделаем дополнительную колонку с днями недели.
# После этого, делаем группировку по колонке day и order_item_id, считаем product_id, и делаем таблицу с по дням
# и идентификатору товара.
# В итоге имеем 21 позицию с товарми и распределением по дням недели.

items_df.head()

items_df['day'] = items_df['shipping_limit_date'].dt.day_name()

items_df.groupby(['day', 'order_item_id'], as_index=False) \
        .agg({'product_id' : 'count'}) \
        .pivot(index='order_item_id', columns='day', values='product_id').fillna(0)

# 4. Сколько у каждого из пользователей в среднем покупок в неделю (по месяцам)? 

# Нужно соеденить две таблицы items_df и orders_df. Это нужно для того что бы мы могли посчитать кол-во заказов для 
# каждого пользователя, это можно сделать используя колонку price и customer_id. Так же необходимо выбрать месяца, от
# от общей даты, что бы было визуально понятно какую сумму потратил клиент за это время.

df_item_orders = items_df.merge(orders_df, how='inner', on='order_id')

df_item_orders['month_number'] = df_item_orders['order_estimated_delivery_date'].dt.month

# Произвели расчет среднего чека для каждого пользователя по месяцам.
df_item_orders.query('order_status == "delivered"').groupby(['customer_id','month_number'])['price'].agg(['mean', 'count'])               .sort_values('count', ascending=False)

# 5. Используя pandas, проведи когортный анализ пользователей. 
# В период с января по декабрь выяви когорту с самым высоким retention на 3й месяц. 

orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
items_df = items_df.set_index('order_id')
orders_df = orders_df.set_index('order_id')
items_orders = orders_df.join(items_df)

items_orders.reset_index(inplace=True)
items_orders['Period'] = items_orders.order_purchase_timestamp.dt.strftime('%Y-%m')

items_orders.set_index('customer_id', inplace=True)
items_orders['CohortGroup'] = items_orders.groupby(level=0)['order_purchase_timestamp'].min().dt.strftime('%Y-%m')
items_orders.reset_index(inplace=True)

grouped = items_orders.groupby(['CohortGroup', 'Period'])
cohorts = grouped.agg({'customer_id': pd.Series.nunique,
                        'price': 'sum',
                        'order_id': 'count'})
cohorts.rename(columns={'customer_id': 'TotalClients',
                         'order_id': 'TotalOrders'}, inplace=True)

def cohort_period(df):
     df['CohortPeriod'] = np.arange(len(df)) + 1
     return df
cohorts = cohorts.groupby(level=0).apply(cohort_period)

cohorts.head(10)
cohorts.tail(12)

cohorts = cohorts.reset_index()
cohorts['CohortGroupYear'] = cohorts['CohortGroup'].apply(lambda x: x.split('-')[0])
tt = cohorts.groupby('CohortGroupYear').agg({'price': 'mean','TotalOrders':'mean','TotalClients':'mean'})
tt['ratio'] = tt['TotalOrders'] / tt['TotalClients']
tt

# Из проведенного выше анализа видно, что покупки освершенные в марте 2018 выше чем за тот же период 2017 года. 
# Это говорит о росте дохода и приросте клиентов. Не смотря на то что в марте 2018 года, число покупок увеличелось
# на 63,9%, общегодовой прирост прибыли не был столь значительным и составляет 30%, но стоит учесть
# тот факт, что инофрмацию 2018 год мы имее только по 10 месяцам, в то время как за 2017 имеем полные 12 месяцев.

# 6.Построй RFM-сегментацию пользователей, чтобы качественно оценить свою аудиторию. 
# В кластеризации можешь выбрать следующие метрики: 
# R - время от последней покупки пользователя до текущей даты 
# F - суммарное количество покупок у пользователя за всё время, 
# M - сумма покупок за всё время.
# Подробно опиши, как ты создавал кластеры. 
# Для каждого RFM-сегмента построй границы метрик recency, frequency и monetary для интерпретации этих кластеров. 
 
last_date = items_orders['order_delivered_carrier_date'].max() + timedelta(days=1)
rfmTable = items_orders.reset_index().groupby('customer_id').agg({'order_delivered_carrier_date': lambda x: (last_date - x.max()).days,
                                                 'order_id': lambda x: len(x), 
                                                 'price': lambda x: x.sum()})
rfmTable.rename(columns={'order_delivered_carrier_date': 'recency', 
                          'order_id': 'frequency', 
                          'price': 'monetary_value'}, inplace=True)

rfmTable.head()

# Разбиваем на диапазоны от 1 до 5. Эта разбивка позволит определить клиентов по частоте совершаемых действий.
# Где 5 наивысшая отметка, а 1 соответсвенно низшая.
# Создадим функцию, которая разобьет наши данные по квантилям, для каждого кластера.
# После чего создадим соответсвующие колонки с по каждому кластеру и одну общую колонку, 
# которая будет объеденять все кластеры.
quantiles = rfmTable.quantile(q=[0.20, 0.40, 0.60, 0.80])
quantiles = quantiles.to_dict()
segmented_rfm = rfmTable

def RScore(x,p,d):
     if x <= d[p][0.20]:
         return 1
     elif x <= d[p][0.40]:
         return 2
     elif x <= d[p][0.60]: 
         return 3
     elif x<=d[p][0.80]:
         return 4
     else:
         return 5

segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))
segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(RScore, args=('frequency',quantiles,))
segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(RScore, args=('monetary_value',quantiles,))
segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)

# Данный график показвает на сколько давно были совершены последние покупки. Рассчитывали среднее число recency.
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
segmented_rfm.groupby('RFMScore').agg('recency').mean().plot(kind='bar', colormap='Blues_r')
plt.show()

segmented_rfm.head()

# По итогу мы получили достаточно информативную таблицу, с колнками recency, frequency, monetary_value
# которые описывают показатели в деталях, на сколько поздно была совершена покупка, сколько позициый было куплено и
# сколько денег было потрачено. Так же были сделаны колнки которые отражают эти значения в RFM-сегментации.
# Так например пользователь с id = 00012a2ce6f8dcda20d059ce98491703, имеет RFMScore 413, это говрит нам о том, что
# данный клиент совершал покупку достаточно недавно, поторатив средюю сумму на приобретение одного товара. 
# Такого клинета можно дополнительно простимулировать к покупкам. 

# Градация RFM:
# R=5, F=5, M=5 — Платят чаcто, много и недавно. 
# R=1, F=1, M=1 — Платят мало, редко и давно. Скорее всего потерянные клиенты. Данную группу клиентов можно попробовать
# реанимировать, но только в том случае если издержки на возварт не будут превышать потенциальную прибыль.
# R=1/2, F=4/5, M=4/5 — Лояльные пользователи на грани ухода. Предлагаем им бонус, скидку и пытаемся их вернуть.
# R=4/5, F=1, M=1/2/3/4/5 — Пользователи недавно совершили покупку. Данных клиентов желательно стимулировать бонусами и акциями, 
# чтобы они совершали еще покупки.