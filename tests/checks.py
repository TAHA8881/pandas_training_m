"""Автопроверка практики модуля 01. Для студентов: этот файл изменять не нужно."""
from __future__ import annotations
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
import pandas.testing as pdt


def eq(got, expected):
    if isinstance(expected, pd.DataFrame):
        assert isinstance(got, pd.DataFrame), "Ожидается DataFrame"
        pdt.assert_frame_equal(got, expected, check_dtype=False)
    elif isinstance(expected, pd.Series):
        assert isinstance(got, pd.Series), "Ожидается Series"
        pdt.assert_series_equal(got, expected, check_dtype=False, check_names=True)
    elif isinstance(expected, float):
        assert math.isclose(float(got), expected, rel_tol=1e-9, abs_tol=1e-9)
    else:
        assert got == expected, f"Ожидалось {expected!r}, получено {got!r}"


def cases(tid, fn):
    if tid == "m1_01":
        for rows in [[{'sku': 7, 'name': 'Кисть', 'stock': 3}, {'sku': 8, 'name': 'Краска', 'stock': 0}], [{'sku': 3, 'name': 'Мяч', 'stock': 17}]]:
            eq(fn(rows), pd.DataFrame(rows, columns=['sku', 'name', 'stock']))
    elif tid == "m1_02":
        for column in ['price','cost']:
            df=pd.DataFrame({'price':[11, 25, 7], 'cost':[5, 8, 1]})
            orig=df.copy(deep=True)
            eq(fn(df,column),df[column]); pdt.assert_frame_equal(df,orig)
    elif tid == "m1_03":
        for d in [{'Пермь':13,'Омск':9},{'Уфа':0,'Тула':100,'Казань':4}]:
            eq(fn(d),pd.Series(d,name='sales'))
    elif tid == "m1_04":
        for shape in [(4,3),(1,2),(0,2)]:
            df=pd.DataFrame([[1]*shape[1]]*shape[0], columns=[f'c{i}' for i in range(shape[1])])
            eq(fn(df),shape)
    elif tid == "m1_05":
        for left,right in [(pd.Series([30,10],index=['b','a']),pd.Series([3,7],index=['a','b'])),(pd.Series([8, 6, 4],index=['z','x','y']),pd.Series([2,3,5],index=['y','z','x']))]:
            eq(fn(left,right),left+right)
    elif tid == "m1_06":
        df=pd.DataFrame({'v':range(10)}, index=list('abcdefghij'))
        for n in [1,3,8]: eq(fn(df,n),df.head(n))
    elif tid == "m1_07":
        for df in [pd.DataFrame({'a':[1,None,3],'b':[None,'ok',''] }),pd.DataFrame({'x':[1,2],'y':[3,4]})]:
            eq(fn(df),df.isna().sum())
    elif tid == "m1_08":
        for df in [pd.DataFrame({'m':[1,2], 'n':[3,4]}),pd.DataFrame(columns=['first','second','third'])]:
            eq(fn(df), {'rows':len(df),'columns':len(df.columns),'names':list(df.columns)})
    elif tid == "m1_09":
        for df,col in [(pd.DataFrame({'tag':['a',None,'b','a']}),'tag'),(pd.DataFrame({'status':['x','y','x','z','z','z']}),'status')]:
            eq(fn(df,col),df[col].value_counts(dropna=False))
    elif tid == "m1_10":
        for values in [[1.,3.,None,9.],[12.,-4.,8.,2.]]:
            df=pd.DataFrame({'score':values})
            eq(fn(df,'score'),{'mean':df['score'].mean(),'median':df['score'].median(),'min':df['score'].min(),'max':df['score'].max()})
    elif tid == "m1_11":
        df=pd.DataFrame({'a':[1,2],'b':[3,4],'c':[5,6]}); old=df.copy(deep=True)
        for cols in [['c','a'],['b'],['b','a','c']]:eq(fn(df,cols),df[cols])
        pdt.assert_frame_equal(df,old)
    elif tid == "m1_12":
        for df in [pd.DataFrame({'units':[2,7],'price':[11.,3.]}),pd.DataFrame({'units':[0,5,3],'price':[8.,1.5,7.]})]:
            old=df.copy(deep=True);exp=df.assign(revenue=df['units']*df['price'])
            eq(fn(df),exp);pdt.assert_frame_equal(df,old)
    elif tid == "m1_13":
        for df in [pd.DataFrame({'price':[100.,90.],'discount_pct':[10,0]}),pd.DataFrame({'price':[199.,20.],'discount_pct':[25,100]})]:
            old=df.copy(deep=True);exp=df.assign(final_price=df['price']*(1-df['discount_pct']/100))
            eq(fn(df),exp);pdt.assert_frame_equal(df,old)
    elif tid == "m1_14":
        for old,new,dead in [('original','renamed','temp'),('code','item_code','deleted')]:
            df=pd.DataFrame({old:[1,2],dead:[3,4],'keep':[5,6]}); snap=df.copy(deep=True)
            eq(fn(df,old,new,dead),df.rename(columns={old:new}).drop(columns=[dead]))
            pdt.assert_frame_equal(df,snap)
    elif tid == "m1_15":
        for df in [pd.DataFrame({'department':['Б','А','А','Б'],'quantity':[2,5,1,8]}, index=[9,7,3,2]),pd.DataFrame({'department':['Z','A','Z'],'quantity':[4,8,2]},index=[4,6,8])]:
            eq(fn(df),df.sort_values(['department','quantity'],ascending=[True,False]))
    elif tid == "m1_16":
        with TemporaryDirectory() as tmp:
            for ids in [['0003','0018'],['00001','20007','00700']]:
                path=Path(tmp)/'in.csv'; pd.DataFrame({'order_id':ids,'quantity':list(range(1,len(ids)+1))}).to_csv(path,sep=';',index=False)
                out=fn(path)
                assert isinstance(out,pd.DataFrame)
                assert out['order_id'].astype(str).tolist()==ids, 'Ведущие нули потерялись'
                eq(out['quantity'].tolist(),list(range(1,len(ids)+1)))
    elif tid == "m1_17":
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'book.xlsx'
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame({'code':[1,2], 'rate':[1.5, 3.0]}).to_excel(writer,sheet_name='A',index=False)
                pd.DataFrame({'product':['a','b']}).to_excel(writer,sheet_name='B',index=False)
            for sh in ['A','B']:eq(fn(path,sh),pd.read_excel(path,sheet_name=sh))
    elif tid == "m1_18":
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.json'
            for rows in [[{'city':'Псков','value':2},{'city':'Тверь','value':3}],[{'name':'А','level':1}]]:
                path.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
                eq(fn(path),pd.DataFrame(rows))
    elif tid == "m1_19":
        for df in [pd.DataFrame({'order_id':['1','2'], 'product_id':[4,3], 'quantity':[2,1], 'unit_price':[10,90]}),pd.DataFrame({'order_id':['x'], 'product_id':[8], 'quantity':[7], 'unit_price':[1.25]})]:
            old=df.copy(deep=True)
            exp=df[['order_id','product_id']].copy();exp['line_total']=df['quantity']*df['unit_price']
            eq(fn(df),exp);pdt.assert_frame_equal(df,old)
    elif tid == "m1_20":
        with TemporaryDirectory() as tmp:
            for df in [pd.DataFrame({'id':['001','002'],'name':['чай','кофе']}),pd.DataFrame({'v':[2,4], 'total':[12.,9.]})]:
                path=Path(tmp)/'report.csv'; ret=fn(df,path)
                assert Path(ret)==path,'Верните переданный путь'
                assert path.is_file(),'Файл не сохранён'
                read=pd.read_csv(path,dtype=str)
                assert read.columns.tolist()==df.columns.tolist(),'В CSV появились лишние столбцы (возможно, индекс)'
                assert read.shape==df.shape
                assert read.iloc[0,0]==str(df.iloc[0,0]),'Значения не совпадают'
    else:
        raise ValueError(f'Неизвестное задание: {tid}')


def check(task_id, fn):
    """Печатает результат и помогает найти ошибку в конкретном задании."""
    try:
        cases(task_id, fn)
    except NotImplementedError:
        print(f'◻ {task_id}: решение ещё не написано. Замените raise NotImplementedError на свой код.')
        return False
    except Exception as exc:
        print(f'✗ {task_id}: проверка не пройдена. {type(exc).__name__}: {exc}')
        return False
    print(f'✓ {task_id}: все проверки пройдены!')
    return True












# --- COURSE EXTENSION ---
from tests.advanced_checks import validate as _validate_advanced
from pathlib import Path as _Path
import json as _json

_original_module_one_check = check
_TASK_IDS = [f'm1_{i:02d}' for i in range(1,21)] + [f'm{m}_{i:02d}' for m in range(2,9) for i in range(1,(15 if m==8 else 20)+1)]
_PROGRESS_FILE = _Path(__file__).resolve().parents[1] / 'output' / 'progress.json'

def _record_success(task_id):
    _PROGRESS_FILE.parent.mkdir(parents=True,exist_ok=True)
    try:
        state=_json.loads(_PROGRESS_FILE.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        state={}
    state[task_id]=True
    _PROGRESS_FILE.write_text(_json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')

def check(task_id, fn):
    """Run self-tests on at least two variants and save local progress."""
    if task_id.startswith('m1_'):
        ok=_original_module_one_check(task_id,fn)
        if ok:_record_success(task_id)
        return ok
    try:
        number=_validate_advanced(task_id,fn)
    except NotImplementedError:
        print(f'◻ {task_id}: добавьте собственную реализацию вместо raise NotImplementedError')
        return False
    except Exception as exc:
        print(f'✗ {task_id}: тест не пройден — {type(exc).__name__}: {str(exc)[:600]}')
        return False
    print(f'✓ {task_id}: все проверки пройдены ({number} независимых набора данных)')
    _record_success(task_id)
    return True

def show_progress():
    try: state=_json.loads(_PROGRESS_FILE.read_text(encoding='utf-8'))
    except (ValueError,OSError): state={}
    completed={k for k,v in state.items() if v}
    print('Прогресс Pandas Academy:')
    for mod in range(1,9):
        keys=[k for k in _TASK_IDS if k.startswith(f'm{mod}_')]
        n=sum(k in completed for k in keys)
        print(f'  Модуль {mod:02d}: {n}/{len(keys)}')
    print(f'ИТОГО: {sum(k in completed for k in _TASK_IDS)}/{len(_TASK_IDS)}')
