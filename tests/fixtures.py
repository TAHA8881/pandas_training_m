
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Тестовые данные отличаются от примеров в учебнике и имеют разные индексы.
def orders(seed=0):
    if seed == 0:
        return pd.DataFrame({
            'order_id':['o1','o2','o3','o4','o5','o6'],
            'client_id':[1,2,1,3,2,4],
            'city':['Москва','Тверь','Москва','Казань','Тверь','Казань'],
            'category':['чай','кофе','чай','книги','кофе','чай'],
            'price':[120.,240.,90.,310.,50.,140.],
            'qty':[2,1,4,1,3,0],
            'discount':[0,10,5,0,20,0],
            'status':['paid','new','paid','paid','cancelled','new'],
            'date':['2026-01-03','2026-01-04','2026-01-04','2026-02-01','2026-02-02','2026-02-05'],
            'comment':['  Быстрая доставка  ','без звонка','Срочно','подарок','',None],
        }, index=['z','b','a','d','x','q'])
    return pd.DataFrame({
        'order_id':['u10','u20','u30','u40'],
        'client_id':[8,9,8,7],
        'city':['Омск','Пермь','Омск','Рязань'],
        'category':['игры','чай','игры','книги'],
        'price':[45.,0.,1200.,12.], 'qty':[3,2,1,5],
        'discount':[15,0,0,50],
        'status':['new','paid','paid','cancelled'],
        'date':['2026-03-11','2026-03-11','2026-03-15','2026-04-01'],
        'comment':['  забирать самому','НЕТ','скидка?', '  ok  '],
    }, index=['q','x','a','m'])

def dirty(seed=0):
    if seed==0:
        return pd.DataFrame({
            'item':[' A-1 ','b-2',' A-1 ','C-3',None,' d-4 '],
            'price':['120.50','', '120.50','ошибка', '0','-4.00'],
            'quantity':['2','3','2',None,'7','1'],
            'category':[' FOOD ','food',' FOOD ','TOY',None,'toy'],
            'email':['A@EXAMPLE.COM',' b@x.ru ', 'A@EXAMPLE.COM', 'bad','', None],
        }, index=[11,2,3,8,10,15])
    return pd.DataFrame({
        'item':['x-8',' X-8 ','y-9',None],
        'price':['10',' 9.5 ','NA','10'],
        'quantity':['1',None,'0','4'],
        'category':['Home',' home ','TECH','TECH'],
        'email':[' M@TEST.COM ','no-at','b@x.io',None]
    },index=[90,20,30,40])

def clients(seed=0):
    if seed==0:
        return pd.DataFrame({'client_id':[1,2,3,5],'segment':['vip','base','vip','base'],'region':['Центр','Север','Юг','Восток']})
    return pd.DataFrame({'client_id':[7,8,9,10],'segment':['new','vip','new','vip'],'region':['Восток','Юг','Юг','Центр']})

def products(seed=0):
    if seed==0:
        return pd.DataFrame({'product_id':[101,102,103,104],'category':['tea','coffee','tea','book'],'cost':[30,50,20,90],'price':[100,160,80,250]})
    return pd.DataFrame({'product_id':[201,202,203],'category':['book','toy','toy'],'cost':[14,55,0],'price':[20,150,17]})

def events(seed=0):
    if seed==0:
        return pd.DataFrame({'user':['a','a','b','a','b','c'], 'event':['visit','buy','visit','visit','buy','visit'],
                             'ts':['2026-01-01 09:00','2026-01-02 10:30','2026-01-01 12:00','2026-01-04 09:00','2026-01-05 11:00','2026-01-06 19:30'],
                             'value':[0,150,0,0,75,0]})
    return pd.DataFrame({'user':['q','z','q','z'], 'event':['visit','visit','buy','buy'],
                         'ts':['2026-02-10 10:00','2026-02-10 11:00','2026-02-11 12:00','2026-02-12 13:00'],
                         'value':[0,0,20,55]})

def metrics(seed=0):
    if seed==0:
        return pd.DataFrame({'strategy':['seq','seq','parallel','parallel','adaptive','adaptive'],
                            'time_s':[9.,12.,7.,6.,8.,10.], 'success':[True,False,True,True,False,True],
                            'cpu':[35,42,65,61,48,51], 'requests':[1,2,2,2,1,2],
                            'product':['a','b','a','c','b','c']})
    return pd.DataFrame({'strategy':['seq','parallel','adaptive','seq','adaptive'],
                         'time_s':[14.,4.,6.,9.,7.], 'success':[False,True,True,True,False],
                         'cpu':[28,79,52,39,45], 'requests':[1,3,2,1,2],
                         'product':['x','y','x','y','z']})

def ledger(seed=0):
    if seed == 0:
        return pd.DataFrame({'date':['2026-01-01','2026-01-02','2026-01-04','2026-01-07','2026-01-08'],
                             'shop':['A','A','A','B','B'],'amount':[10.,25.,30.,40.,15.]})
    return pd.DataFrame({'date':['2026-05-01','2026-05-03','2026-05-04','2026-05-05'],
                         'shop':['Z','Z','Y','Y'],'amount':[5.,0.,55.,20.]})

def np_array(seed=0):
    return pd.DataFrame({'a':list(range(20 if seed==0 else 9)),
                         'b': [str(x) for x in range(20 if seed==0 else 9)]})

def nested(seed=0):
    if seed==0:
        return [{'id':'a','user':{'country':'RU'},'lines':[{'sku':'S1','qty':2},{'sku':'S2','qty':1}]},
                {'id':'b','user':{'country':'KZ'},'lines':[{'sku':'S3','qty':4}]}]
    return [{'id':'x','user':{'country':'BY'},'lines':[{'sku':'T9','qty':3}]}]


def csv_orders(tmp,seed=0):
    path=Path(tmp)/'orders.csv'
    orders(seed).reset_index(drop=True).to_csv(path,index=False,encoding='utf-8')
    return path

def csv_dirty(tmp,seed=0):
    path=Path(tmp)/'dirty.csv'
    dirty(seed).to_csv(path,index=False,encoding='utf-8')
    return path

def sqlite_db(tmp,seed=0):
    import sqlite3
    path=Path(tmp)/'shop.db'
    with sqlite3.connect(path) as conn:
        orders(seed).reset_index(drop=True).to_sql('orders',conn,index=False,if_exists='replace')
        clients(seed).to_sql('clients',conn,index=False,if_exists='replace')
    return path
