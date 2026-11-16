
from pathlib import Path
import pandas as pd
import numpy as np
from tests.fixtures import orders, dirty, clients, products, events, metrics, ledger, np_array, nested, csv_orders, csv_dirty, sqlite_db
from pandas.testing import assert_frame_equal, assert_series_equal
from copy import deepcopy
import math
import tempfile

# Сгенерированные эталонные функции используются только для сравнения результатов.
ORACLES = {}
CASES = {}

def register(task_id, oracle, case_specs):
    ORACLES[task_id] = oracle
    CASES[task_id] = case_specs

STRICT_DTYPES = {"m3_07", "m5_16", "m5_19", "m6_04", "m6_07", "m6_08"}

def similar(actual, expected, strict=False):
    if isinstance(expected,pd.DataFrame):
        assert isinstance(actual,pd.DataFrame), f'Ожидался DataFrame, получен {type(actual).__name__}'
        assert_frame_equal(actual, expected, check_dtype=strict, check_exact=False, rtol=1e-7, atol=1e-9, check_freq=False)
    elif isinstance(expected,pd.Series):
        assert isinstance(actual,pd.Series), f'Ожидалась Series, получен {type(actual).__name__}'
        assert_series_equal(actual, expected, check_dtype=strict, check_exact=False, rtol=1e-7, atol=1e-9, check_freq=False)
    elif isinstance(expected,dict):
        assert isinstance(actual,dict), f'Ожидался dict, получен {type(actual).__name__}'
        assert set(actual)==set(expected), f'Нужны ключи {set(expected)}, получены {set(actual)}'
        for key in expected: similar(actual[key],expected[key],strict)
    elif isinstance(expected,(tuple,list)):
        assert isinstance(actual,(tuple,list)), f'Ожидался список или кортеж, получен {type(actual).__name__}'
        assert len(actual)==len(expected), f'Длина результата {len(actual)}, ожидается {len(expected)}'
        for a,b in zip(actual,expected): similar(a,b,strict)
    elif isinstance(expected,np.ndarray):
        assert isinstance(actual,np.ndarray), f'Ожидался numpy.ndarray, получен {type(actual).__name__}'
        np.testing.assert_allclose(actual,expected,equal_nan=True)
    elif isinstance(expected,(float,np.floating)) and math.isnan(float(expected)):
        assert pd.isna(actual), 'Ожидалось пропущенное значение'
    elif isinstance(expected,(bool,np.bool_)):
        assert isinstance(actual,(bool,np.bool_)), f'Ожидался bool, получен {type(actual).__name__}'
        assert bool(actual)==bool(expected), f'Получено {actual!r}, ожидалось {expected!r}'
    elif isinstance(expected,(int,float,np.integer,np.floating)):
        assert isinstance(actual,(int,float,np.integer,np.floating)) and not isinstance(actual,bool),f'Ожидалось число, получено {type(actual).__name__}'
        assert math.isclose(float(actual),float(expected),rel_tol=1e-7,abs_tol=1e-9), f'Получено {actual}, ожидалось {expected}'
    else:
        assert actual==expected, f'Получено {actual!r}, ожидалось {expected!r}'

def validate(task_id, student_fn):
    if task_id not in ORACLES: raise KeyError(f'Неизвестный ID: {task_id}')
    for k, spec in enumerate(CASES[task_id],1):
        with tempfile.TemporaryDirectory() as d:
            tmp=Path(d)
            args=eval(spec, globals(), {'tmp':tmp})
            if not isinstance(args,tuple): raise RuntimeError(f'Ошибка тест-кейса {task_id}: ожидается tuple')
            before=deepcopy(args)
            expected=ORACLES[task_id](*deepcopy(args))
            actual=student_fn(*args)
            similar(actual,expected,strict=(task_id in STRICT_DTYPES))
            # Кроме задач на запись файла, функции не должны изменять аргументы.
            for cur,original in zip(args,before):
                if isinstance(cur,(pd.DataFrame,pd.Series)):
                    similar(cur,original,strict=(task_id in STRICT_DTYPES))
    return len(CASES[task_id])

register('m2_01', lambda df,labels,columns: df.loc[labels,columns], ["(orders(0), ['x','a'], ['city','price'])", "(orders(1), ['m','q'], ['qty','status'])"])

register('m2_02', lambda df,n: df.iloc[-n:,:3], ['(orders(0), 2)', '(orders(1), 3)'])

register('m2_03', lambda df,label,column: df.at[label,column], ["(orders(0),'a','price')", "(orders(1),'m','city')"])

register('m2_04', lambda df: df.set_index("order_id"), ['(orders(0),)', '(orders(1),)'])

register('m2_05', lambda df: df.reset_index(drop=True), ['(orders(0),)', '(orders(1),)'])

register('m2_06', lambda df,min_price: df.loc[(df.status=="paid") & (df.price>=min_price)], ['(orders(0), 100)', '(orders(1), 1000)'])

register('m2_07', lambda df,cities: df.loc[df.city.isin(cities) & (df.qty>0)], ["(orders(0), ['Москва','Казань'])", "(orders(1), ['Омск','Рязань'])"])

register('m2_08', lambda df,bad_statuses: df.loc[~df.status.isin(bad_statuses)].reset_index(drop=True), ["(orders(0), ['cancelled','new'])", "(orders(1), ['new'])"])

register('m2_09', lambda df,low,high: df.loc[df.price.between(low,high)].sort_values("price",ascending=False), ['(orders(0), 90,240)', '(orders(1), 10,1200)'])

register('m2_10', lambda df,city,threshold: df.loc[(df.city==city) | ((df.status=="paid") & (df.qty>=threshold))], ["(orders(0), 'Тверь', 4)", "(orders(1),'Пермь',1)"])

register('m2_11', lambda df,cutoff: df.query("qty >= @cutoff and status != 'cancelled'"), ['(orders(0),2)', '(orders(1),3)'])

register('m2_12', lambda df,substring: df.loc[df.city.str.contains(substring,case=False,na=False,regex=False)], ["(orders(0),'МОС')", "(orders(1),'ом')"])

register('m2_13', lambda df: df.assign(clean_comment=df.comment.fillna("").str.strip()).sort_values("clean_comment"), ['(orders(0),)', '(orders(1),)'])

register('m2_14', lambda df: df.sort_values(["city","qty","order_id"],ascending=[True,False,True]), ['(orders(0),)', '(orders(1),)'])

register('m2_15', lambda df,prefix: df.loc[(df.status=="paid") & df.category.str.startswith(prefix,na=False),["order_id","category"]], ["(orders(0),'ч')", "(orders(1),'и')"])

register('m2_16', lambda df: df.assign(line_total=lambda x:x.price*x.qty*(1-x.discount/100),big=lambda x:x.line_total>=200), ['(orders(0),)', '(orders(1),)'])

register('m2_17', lambda df: df.assign(priority=np.select([df.qty>=4,df.qty>=2],["urgent","normal"],default="low")), ['(orders(0),)', '(orders(1),)'])

register('m2_18', lambda df,cap: df.assign(capped_price=df.price.where(df.price<=cap,cap)), ['(orders(0),150)', '(orders(1),100)'])

register('m2_19', lambda df: df.assign(status_ru=df.status.replace({"new":"Новый","paid":"Оплачен","cancelled":"Отменён"})), ['(orders(0),)', '(orders(1),)'])

register('m2_20', lambda df: df.assign(net_price=lambda x:(x.price*(1-x.discount/100)).mask(x.status=="cancelled",0)), ['(orders(0),)', '(orders(1),)'])

register('m3_01', lambda df: df.isna().sum().sort_values(ascending=False,kind="stable"), ['(dirty(0),)', '(dirty(1),)'])

register('m3_02', lambda df,required: df.dropna(subset=required), ["(dirty(0),['item','quantity'])", "(dirty(1),['item','email'])"])

register('m3_03', lambda df: df.assign(category=df.category.fillna("unknown")), ['(dirty(0),)', '(dirty(1),)'])

register('m3_04', lambda df: df.drop_duplicates(subset=["item"],keep="last"), ['(dirty(0),)', '(dirty(1),)'])

register('m3_05', lambda df: df.isna().mean()*100, ['(dirty(0),)', '(dirty(1),)'])

register('m3_06', lambda df: df.assign(price_num=pd.to_numeric(df.price,errors="coerce")), ['(dirty(0),)', '(dirty(1),)'])

register('m3_07', lambda df: df.assign(quantity=pd.to_numeric(df.quantity,errors="coerce").astype("Int64")), ['(dirty(0),)', '(dirty(1),)'])

register('m3_08', lambda values: pd.to_numeric(values.str.strip().str.replace(",",".",regex=False),errors="coerce"), ["(pd.Series([' 12,5','3.25','нет',None]),)", "(pd.Series(['0,0',' -4,20 ','10']),)"])

register('m3_09', lambda df: df.loc[~pd.to_numeric(df.price,errors="coerce").ge(0)], ['(dirty(0),)', '(dirty(1),)'])

register('m3_10', lambda df: df.assign(total=pd.to_numeric(df.price,errors="coerce")*pd.to_numeric(df.quantity,errors="coerce")), ['(dirty(0),)', '(dirty(1),)'])

register('m3_11', lambda df: df.assign(category_key=df.category.str.strip().str.lower().fillna("unknown")), ['(dirty(0),)', '(dirty(1),)'])

register('m3_12', lambda df: df.assign(email_norm=df.email.str.strip().str.lower()), ['(dirty(0),)', '(dirty(1),)'])

register('m3_13', lambda df: df.assign(item_number=pd.to_numeric(df.item.str.strip().str.extract(r"-(\d+)",expand=False),errors="coerce")), ['(dirty(0),)', '(dirty(1),)'])

register('m3_14', lambda df: df.email.str.strip().str.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+",na=False), ['(dirty(0),)', '(dirty(1),)'])

register('m3_15', lambda df: df.assign(item_prefix=df.item.str.strip().str.upper().str.extract(r"^([A-Z]+)-\d+$",expand=False)), ['(dirty(0),)', '(dirty(1),)'])

register('m3_16', lambda df: {"rows":len(df),"missing_cells":int(df.isna().sum().sum()),"duplicate_items":int(df.item.duplicated().sum())}, ['(dirty(0),)', '(dirty(1),)'])

register('m3_17', lambda df: df.loc[pd.to_numeric(df.price,errors="coerce").ge(0) & pd.to_numeric(df.quantity,errors="coerce").ge(0)], ['(dirty(0),)', '(dirty(1),)'])

register('m3_18', lambda df: df.assign(item=df.item.str.strip().str.upper(),category=df.category.str.strip().str.lower()), ['(dirty(0),)', '(dirty(1),)'])

register('m3_19', lambda df: (df.loc[pd.to_numeric(df.price,errors="coerce").ge(0) & df.item.str.strip().fillna("").ne("")],df.loc[~(pd.to_numeric(df.price,errors="coerce").ge(0) & df.item.str.strip().fillna("").ne(""))]), ['(dirty(0),)', '(dirty(1),)'])

register('m3_20', lambda df: {"bad_price":int((~pd.to_numeric(df.price,errors="coerce").ge(0)).sum()),"missing_item":int(df.item.str.strip().fillna("").eq("").sum())}, ['(dirty(0),)', '(dirty(1),)'])

register('m4_01', lambda df:{"orders":len(df),"units":int(df.qty.sum()),"revenue":float((df.price*df.qty).sum()),"unique_clients":int(df.client_id.nunique())}, ['(orders(0),)', '(orders(1),)'])

register('m4_02', lambda df:df.status.value_counts(dropna=False), ['(orders(0),)', '(orders(1),)'])

register('m4_03', lambda df:df.price.quantile([0.25,0.5,0.75]), ['(orders(0),)', '(orders(1),)'])

register('m4_04', lambda df:float(df.status.eq("paid").mean()*100), ['(orders(0),)', '(orders(1),)'])

register('m4_05', lambda df:df.assign(revenue=df.price*df.qty).groupby("city")["revenue"].sum(), ['(orders(0),)', '(orders(1),)'])

register('m4_06', lambda df:df.groupby("city",as_index=False).agg(orders=("qty","size"),units=("qty","sum"),avg_price=("price","mean")), ['(orders(0),)', '(orders(1),)'])

register('m4_07', lambda df:df.assign(city_share=lambda x:(x.price*x.qty)/(x.price*x.qty).groupby(x.city).transform("sum")), ['(orders(0),)', '(orders(1),)'])

register('m4_08', lambda df:df.assign(revenue=df.price*df.qty).groupby(["city","status"],as_index=False).agg(orders=("qty","size"),revenue=("revenue","sum")), ['(orders(0),)', '(orders(1),)'])

register('m4_09', lambda df,min_count:df.groupby("city").filter(lambda g:len(g)>=min_count), ['(orders(0),2)', '(orders(1),2)'])

register('m4_10', lambda df:df.assign(price_rank=df.groupby("city").price.rank(ascending=False,method="min")), ['(orders(0),)', '(orders(1),)'])

register('m4_11', lambda df,reference:df.merge(reference,on="client_id",how="left",validate="many_to_one"), ['(orders(0),clients(0))', '(orders(1),clients(1))'])

register('m4_12', lambda df,reference:df.loc[~df.client_id.isin(reference.client_id),["order_id","client_id"]].reset_index(drop=True), ['(orders(0),clients(0))', '(orders(1),clients(1))'])

register('m4_13', lambda a,b:pd.concat([a,b],ignore_index=True), ['(orders(0).iloc[:3],orders(0).iloc[3:])', '(orders(1).iloc[:2],orders(1).iloc[2:])'])

register('m4_14', lambda left,right:left.merge(right,on="client_id",how="outer",indicator=True), ["(orders(0)[['client_id','order_id']],clients(0))", "(orders(1)[['client_id','order_id']],clients(1))"])

register('m4_15', lambda df,reference:df.join(reference.set_index("client_id"),on="client_id",validate="many_to_one"), ['(orders(0),clients(0))', '(orders(1),clients(1))'])

register('m4_16', lambda df:pd.pivot_table(df,index="city",columns="status",values="order_id",aggfunc="size",fill_value=0), ['(orders(0),)', '(orders(1),)'])

register('m4_17', lambda df:pd.crosstab(df.city,df.category), ['(orders(0),)', '(orders(1),)'])

register('m4_18', lambda df:df.melt(id_vars="order_id",value_vars=["price","qty"],var_name="metric",value_name="value"), ['(orders(0),)', '(orders(1),)'])

register('m4_19', lambda df:df.assign(revenue=df.price*df.qty).groupby(["city","category"]).revenue.sum(), ['(orders(0),)', '(orders(1),)'])

register('m4_20', lambda df:df.groupby(["city","status"]).size().unstack(fill_value=0), ['(orders(0),)', '(orders(1),)'])

register('m5_01', lambda df:df.assign(month=pd.to_datetime(df.date).dt.to_period("M")), ['(orders(0),)', '(orders(1),)'])

register('m5_02', lambda start,end:int((pd.Timestamp(end)-pd.Timestamp(start)).days), ["('2026-01-05','2026-02-01')", "('2026-05-12','2026-05-01')"])

register('m5_03', lambda df:pd.to_datetime(df.date).dt.dayofweek, ['(orders(0),)', '(orders(1),)'])

register('m5_04', lambda df,start,finish:df.loc[pd.to_datetime(df.date).between(pd.Timestamp(start),pd.Timestamp(finish))], ["(orders(0),'2026-01-04','2026-02-01')", "(orders(1),'2026-03-11','2026-03-15')"])

register('m5_05', lambda times:pd.to_datetime(times).dt.tz_localize("UTC").dt.tz_convert("Europe/Amsterdam").dt.hour, ["(pd.Series(['2026-01-01 12:00','2026-07-01 12:00']),)", "(pd.Series(['2026-03-15 22:00','2026-11-10 03:00']),)"])

register('m5_06', lambda df:df.assign(date=pd.to_datetime(df.date)).set_index("date").amount.resample("D").sum(), ['(ledger(0),)', '(ledger(1),)'])

register('m5_07', lambda df:df.assign(month=pd.to_datetime(df.date).dt.to_period("M"),revenue=df.price*df.qty).groupby("month").revenue.sum(), ['(orders(0),)', '(orders(1),)'])

register('m5_08', lambda df:df.sort_values("date").assign(delta=lambda x:x.amount.diff()), ['(ledger(0),)', '(ledger(1),)'])

register('m5_09', lambda df:df.sort_values(["shop","date"]).assign(running_total=lambda x:x.groupby("shop").amount.cumsum()), ['(ledger(0),)', '(ledger(1),)'])

register('m5_10', lambda values:values.pct_change()*100, ['(pd.Series([100.,110.,132.]),)', "(pd.Series([20.,15.,30.,45.],index=list('abcd')),)"])

register('m5_11', lambda values,window:values.rolling(window,min_periods=1).mean(), ['(pd.Series([2,8,4,10,12]),3)', "(pd.Series([1,9,5],index=['a','b','c']),2)"])

register('m5_12', lambda values,window:values.shift(1).rolling(window,min_periods=1).mean(), ['(pd.Series([4.,8.,2.,6.]),2)', '(pd.Series([10.,30.,50.]),3)'])

register('m5_13', lambda values:values.expanding(min_periods=1).max(), ['(pd.Series([3,1,7,2]),)', '(pd.Series([-10,-9,-20,0]),)'])

register('m5_14', lambda values,span:values.ewm(span=span,adjust=False).mean(), ['(pd.Series([9.,7.,3.,11.]),2)', '(pd.Series([1.,4.,5.]),3)'])

register('m5_15', lambda df:df.sort_values(["shop","date"]).assign(moving_avg=lambda x:x.groupby("shop").amount.transform(lambda s:s.rolling(2,min_periods=1).mean())), ['(ledger(0),)', '(ledger(1),)'])

register('m5_16', lambda labels:labels.astype(pd.CategoricalDtype(["low","medium","high"],ordered=True)), ["(pd.Series(['medium','low','high','unknown']),)", "(pd.Series(['high','high','low'],index=['z','x','a']),)"])

register('m5_17', lambda df:df.assign(month=pd.to_datetime(df.date).dt.to_period("M"),revenue=df.price*df.qty).groupby("month").revenue.sum(), ['(orders(0),)', '(orders(1),)'])

register('m5_18', lambda df:df.groupby("status").size().rename("orders"), ['(orders(0),)', '(orders(1),)'])

register('m5_19', lambda df:df.assign(category=df.category.astype("category")), ['(orders(0),)', '(orders(1),)'])

register('m5_20', lambda df:(lambda x: (x.loc.__setitem__((x.status.eq("cancelled"),"qty"),0),x)[1])(df.copy()), ['(orders(0),)', '(orders(1),)'])

register('m6_01', lambda df:(df.price*df.qty*(1-df.discount/100)).rename("total"), ['(orders(0),)', '(orders(1),)'])

register('m6_02', lambda df,threshold:((df.status=="paid") & (df.price*df.qty>=threshold)).rename("flag"), ['(orders(0),180)', '(orders(1),1000)'])

register('m6_03', lambda df:(df.qty/df.groupby("city").qty.transform("max").replace(0,np.nan)).rename("normalized"), ['(orders(0),)', '(orders(1),)'])

register('m6_04', lambda df:df.assign(status=df.status.astype("category")), ['(orders(0),)', '(orders(1),)'])

register('m6_05', lambda df:df.assign(is_returned=df.status.eq("cancelled"),status_code=df.status.map({"paid":2,"new":1,"cancelled":0}).astype(int)), ['(orders(0),)', '(orders(1),)'])

register('m6_06', lambda df:int(df.memory_usage(deep=True).sum()), ['(orders(0),)', '(orders(1),)'])

register('m6_07', lambda df:df.assign(qty=pd.to_numeric(df.qty,downcast="integer")), ['(orders(0),)', '(orders(1),)'])

register('m6_08', lambda df:df.assign(city=df.city.astype("category"),category=df.category.astype("category")), ['(orders(0),)', '(orders(1),)'])

register('m6_09', lambda df,keep:df.loc[:,keep].copy(), ["(orders(0), ['order_id','status','qty'])", "(orders(1), ['order_id','city'])"])

register('m6_10', lambda df:df.memory_usage(index=False,deep=True), ['(orders(0),)', '(orders(1),)'])

register('m6_11', lambda path:pd.read_csv(path,usecols=["order_id","qty"],dtype={"order_id":"string"})[["order_id","qty"]], ['(csv_orders(tmp,0),)', '(csv_orders(tmp,1),)'])

register('m6_12', lambda path,size:sum(float((c.price*c.qty).sum()) for c in pd.read_csv(path,chunksize=size)), ['(csv_orders(tmp,0),2)', '(csv_orders(tmp,1),3)'])

register('m6_13', lambda path,size:(lambda chunks:sum(x[0] for x in chunks)/sum(x[1] for x in chunks))([(float(c.price.sum()),int(c.price.count())) for c in pd.read_csv(path,chunksize=size)]), ['(csv_orders(tmp,0),4)', '(csv_orders(tmp,1),3)'])

register('m6_14', lambda path,size:len(set().union(*(set(c.client_id.dropna()) for c in pd.read_csv(path,chunksize=size)))), ['(csv_orders(tmp,0),2)', '(csv_orders(tmp,1),2)'])

register('m6_15', lambda path:pd.read_csv(path,usecols=["city","price","qty"])[["city","price","qty"]].query("qty > 0").reset_index(drop=True), ['(csv_orders(tmp,0),)', '(csv_orders(tmp,1),)'])

register('m6_16', lambda df:{key:str(value) for key,value in df.dtypes.items()}, ['(orders(0),)', '(orders(1),)'])

register('m6_17', lambda df:float(df.memory_usage(deep=True).sum()/1024), ['(orders(0),)', '(orders(1),)'])

register('m6_18', lambda df,selected:df.loc[:,selected].copy(), ["(orders(0),['city','price','qty'])", "(orders(1),['client_id','status'])"])

register('m6_19', lambda df:{str(m):g.drop(columns=["month"]) for m,g in df.assign(month=pd.to_datetime(df.date).dt.to_period("M")).groupby("month",sort=True)}, ['(orders(0),)', '(orders(1),)'])

register('m6_20', lambda df:df.assign(revenue=df.price*df.qty*(1-df.discount/100)).loc[:,["order_id","client_id","city","revenue"]].reset_index(drop=True), ['(orders(0),)', '(orders(1),)'])

register('m7_01', lambda db_path:(lambda conn:pd.read_sql_query("SELECT order_id,client_id,price,qty FROM orders WHERE status = ? ORDER BY order_id",conn,params=("paid",)))(__import__("sqlite3").connect(db_path)), ['(sqlite_db(tmp,0),)', '(sqlite_db(tmp,1),)'])

register('m7_02', lambda db_path,min_price:(lambda c:pd.read_sql_query("SELECT order_id, price FROM orders WHERE price >= ? ORDER BY price DESC, order_id",c,params=(min_price,)))(__import__("sqlite3").connect(db_path)), ['(sqlite_db(tmp,0),120)', '(sqlite_db(tmp,1),40)'])

register('m7_03', lambda db_path:(lambda c:pd.read_sql_query("SELECT city, SUM(price*qty) AS revenue FROM orders GROUP BY city ORDER BY city",c))(__import__("sqlite3").connect(db_path)), ['(sqlite_db(tmp,0),)', '(sqlite_db(tmp,1),)'])

register('m7_04', lambda db_path:(lambda c:pd.read_sql_query("SELECT o.order_id,o.client_id,c.segment FROM orders AS o LEFT JOIN clients AS c ON o.client_id=c.client_id ORDER BY o.order_id",c))(__import__("sqlite3").connect(db_path)), ['(sqlite_db(tmp,0),)', '(sqlite_db(tmp,1),)'])

register('m7_05', lambda db_path,min_orders:(lambda c:pd.read_sql_query("SELECT city, COUNT(*) AS orders FROM orders GROUP BY city HAVING COUNT(*) >= ? ORDER BY orders DESC, city ASC",c,params=(min_orders,)))(__import__("sqlite3").connect(db_path)), ['(sqlite_db(tmp,0),2)', '(sqlite_db(tmp,1),1)'])

register('m7_06', lambda rows:pd.json_normalize(rows), ['(nested(0),)', '(nested(1),)'])

register('m7_07', lambda rows:pd.concat([pd.DataFrame(rows).rename(columns={"id":"order_id"}).explode("lines",ignore_index=True)[["order_id"]].reset_index(drop=True),pd.json_normalize(pd.DataFrame(rows).explode("lines",ignore_index=True).lines)[["sku","qty"]]],axis=1), ['(nested(0),)', '(nested(1),)'])

register('m7_08', lambda rows:pd.json_normalize(rows).groupby("user.country").size().rename("orders"), ['(nested(0),)', '(nested(1),)'])

register('m7_09', lambda pages:pd.concat(pages,ignore_index=True).drop_duplicates(subset=["order_id"],keep="last").reset_index(drop=True), ['([orders(0).iloc[:3].reset_index(drop=True),orders(0).iloc[2:].reset_index(drop=True)],)', '([orders(1).iloc[:2].reset_index(drop=True),orders(1).iloc[1:].reset_index(drop=True)],)'])

register('m7_10', lambda rows,price_map:(lambda r:r.assign(total=r.qty*r.sku.map(price_map)).groupby("order_id").total.sum())(pd.concat([pd.DataFrame(rows).rename(columns={"id":"order_id"}).explode("lines",ignore_index=True)[["order_id"]].reset_index(drop=True),pd.json_normalize(pd.DataFrame(rows).explode("lines",ignore_index=True).lines)[["sku","qty"]]],axis=1)), ["(nested(0), {'S1':10,'S2':20,'S3':5})", "(nested(1), {'T9':6})"])

register('m7_11', lambda df:df.loc[df.status.ne("cancelled")].assign(revenue=lambda x:x.price*x.qty*(1-x.discount/100))[["order_id","client_id","revenue","status"]].reset_index(drop=True), ['(orders(0),)', '(orders(1),)'])

register('m7_12', lambda df:df.drop_duplicates(subset="order_id",keep="last").sort_values("order_id").reset_index(drop=True), ['(pd.concat([orders(0),orders(0).iloc[:2]],ignore_index=True),)', '(pd.concat([orders(1),orders(1).iloc[1:]],ignore_index=True),)'])

register('m7_13', lambda df:df.assign(timestamp=pd.to_datetime(df.ts),value=df.value.astype(float)), ['(events(0),)', '(events(1),)'])

register('m7_14', lambda df:(df.loc[df.qty.ge(0)&df.price.ge(0)&df.status.isin(["paid","new","cancelled"])],df.loc[~(df.qty.ge(0)&df.price.ge(0)&df.status.isin(["paid","new","cancelled"]))]), ['(orders(0).assign(qty=lambda d:d.qty.where(d.qty>0,-1)),)', "(orders(1).assign(status=lambda d:d.status.mask(d.status.eq('new'),'bad')), )"])

register('m7_15', lambda raw,clean:{"input_rows":len(raw),"output_rows":len(clean),"rejected_rows":max(0,len(raw)-len(clean))}, ['(orders(0),orders(0).iloc[2:])', '(orders(1),orders(1).iloc[:3])'])

register('m7_16', lambda df,required:all(x in df.columns for x in required), ["(orders(0),['order_id','qty'])", "(orders(1),['unknown','price'])"])

register('m7_17', lambda df,column:bool(df[column].is_unique), ["(orders(0),'order_id')", "(pd.concat([orders(1),orders(1).iloc[:1]]),'order_id')"])

register('m7_18', lambda df:{"valid_prices":bool(df.price.ge(0).all()),"valid_quantities":bool(df.qty.ge(0).all()),"unique_ids":bool(df.order_id.is_unique)}, ['(orders(0),)', '(orders(1).assign(price=lambda x:x.price.mask(x.price==0,-5)),)'])

register('m7_19', lambda before,after:{"added":len(set(after.order_id)-set(before.order_id)),"removed":len(set(before.order_id)-set(after.order_id))}, ['(orders(0).iloc[:4],orders(0).iloc[2:])', '(orders(1).iloc[:2],orders(1).iloc[1:])'])

register('m7_20', lambda df:float((df.price.ge(0)&df.qty.ge(0)&df.status.notna()).mean()) if len(df) else 1.0, ['(orders(0),)', '(orders(1).assign(status=lambda x:x.status.mask(x.qty==2,None)),)'])

register('m8_01', lambda df:df.loc[df.status.eq("paid")].assign(revenue=lambda x:x.price*x.qty*(1-x.discount/100))[["order_id","client_id","city","date","revenue"]].reset_index(drop=True), ['(orders(0),)', '(orders(1),)'])

register('m8_02', lambda sales,client_ref:sales.merge(client_ref,on="client_id",how="left",validate="many_to_one"), ["(pd.DataFrame({'order_id':['a','b'],'client_id':[1,404],'city':['A','B'],'date':['2026-01-01','2026-01-02'],'revenue':[20.,40.]}),clients(0))", "(pd.DataFrame({'order_id':['x','y'],'client_id':[8,100],'city':['C','D'],'date':['2026-03-01','2026-04-02'],'revenue':[8.,10.]}),clients(1))"])

register('m8_03', lambda sales:sales.groupby("city",as_index=False).agg(orders=("revenue","size"),revenue=("revenue","sum"),avg_check=("revenue","mean")).sort_values(["revenue","city"],ascending=[False,True]).reset_index(drop=True), ["(orders(0).loc[orders(0).status.eq('paid')].assign(revenue=lambda x:x.price*x.qty*(1-x.discount/100)),)", "(orders(1).loc[orders(1).status.eq('paid')].assign(revenue=lambda x:x.price*x.qty*(1-x.discount/100)),)"])

register('m8_04', lambda sales:sales.assign(month=pd.to_datetime(sales.date).dt.to_period("M")).groupby("month").revenue.sum(), ["(orders(0).loc[orders(0).status.eq('paid')].assign(revenue=lambda x:x.price*x.qty*(1-x.discount/100)),)", "(orders(1).loc[orders(1).status.eq('paid')].assign(revenue=lambda x:x.price*x.qty*(1-x.discount/100)),)"])

register('m8_05', lambda df:(lambda p:{"paid_orders":len(p),"paid_revenue":float((p.price*p.qty*(1-p.discount/100)).sum()),"avg_check":float((p.price*p.qty*(1-p.discount/100)).sum()/len(p)) if len(p) else 0.,"cancelled_orders":int(df.status.eq("cancelled").sum())})(df.loc[df.status.eq("paid")]), ['(orders(0),)', '(orders(1),)'])

register('m8_06', lambda df:df.groupby("strategy",as_index=False).agg(runs=("time_s","size"),median_s=("time_s","median"),success_rate=("success","mean"),mean_cpu=("cpu","mean")), ['(metrics(0),)', '(metrics(1),)'])

register('m8_07', lambda df:df.groupby("strategy").time_s.quantile(0.9).rename("p90_time_s"), ['(metrics(0),)', '(metrics(1),)'])

register('m8_08', lambda df:df.groupby(["strategy","product"],as_index=False).agg(runs=("success","size"),successful=("success","sum")), ['(metrics(0),)', '(metrics(1),)'])

register('m8_09', lambda df:(lambda med:((med.loc["seq"]-med)/med.loc["seq"]*100).rename("speedup_pct"))(df.groupby("strategy").time_s.median()), ['(metrics(0),)', '(metrics(1),)'])

register('m8_10', lambda df:df.groupby("strategy",as_index=False).agg(median_s=("time_s","median"),mean_cpu=("cpu","mean"),mean_requests=("requests","mean"),success_pct=("success",lambda x:x.mean()*100)), ['(metrics(0),)', '(metrics(1),)'])

register('m8_11', lambda df:df.assign(item=df.item.str.strip().str.upper(),category=df.category.str.strip().str.lower().fillna("unknown")), ['(dirty(0),)', '(dirty(1),)'])

register('m8_12', lambda df:df.assign(price_num=pd.to_numeric(df.price,errors="coerce"),qty_num=pd.to_numeric(df.quantity,errors="coerce")), ['(dirty(0),)', '(dirty(1),)'])

register('m8_13', lambda df:df.loc[df.item.str.strip().fillna("").ne("") & pd.to_numeric(df.price,errors="coerce").ge(0) & pd.to_numeric(df.quantity,errors="coerce").ge(0)], ['(dirty(0),)', '(dirty(1),)'])

register('m8_14', lambda df:df.assign(item=df.item.str.strip().str.upper()).drop_duplicates(subset=["item"],keep="last").reset_index(drop=True), ['(dirty(0),)', '(dirty(1),)'])

register('m8_15', lambda df:(lambda ok:{"source_rows":len(df),"valid_rows":int(ok.sum()),"unique_valid_items":int(df.loc[ok,"item"].str.strip().str.upper().nunique())})(df.item.str.strip().fillna("").ne("") & pd.to_numeric(df.price,errors="coerce").ge(0) & pd.to_numeric(df.quantity,errors="coerce").ge(0)), ['(dirty(0),)', '(dirty(1),)'])
