"""Простая автопроверка ячеек pandas Academy без функций-обёрток.

Студент работает непосредственно с входными данными в ноутбуке,
сохраняет результат в переменную result и запускает check_result().
"""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import math
import json
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from tests.advanced_checks import ORACLES, similar, STRICT_DTYPES

_PROGRESS_FILE = Path(__file__).resolve().parents[1] / 'output' / 'progress.json'


def _module_one(task_id, p):
    if task_id == 'm1_01':
        return pd.DataFrame(p['records'], columns=['sku','name','stock'])
    if task_id == 'm1_02':
        return p['df'][p['column']]
    if task_id == 'm1_03':
        return pd.Series(p['sales_by_city'], name='sales')
    if task_id == 'm1_04':
        return p['df'].shape
    if task_id == 'm1_05':
        return p['left']+p['right']
    if task_id == 'm1_06':
        return p['df'].head(p['n'])
    if task_id == 'm1_07':
        return p['df'].isna().sum()
    if task_id == 'm1_08':
        return {'rows':len(p['df']), 'columns':len(p['df'].columns), 'names':list(p['df'].columns)}
    if task_id == 'm1_09':
        return p['df'][p['column']].value_counts(dropna=False)
    if task_id == 'm1_10':
        s=p['df'][p['column']]
        return {'mean':s.mean(),'median':s.median(),'min':s.min(),'max':s.max()}
    if task_id == 'm1_11':
        return p['df'][p['fields']]
    if task_id == 'm1_12':
        return p['df'].assign(revenue=p['df']['units']*p['df']['price'])
    if task_id == 'm1_13':
        return p['df'].assign(final_price=p['df']['price']*(1-p['df']['discount_pct']/100))
    if task_id == 'm1_14':
        return p['df'].rename(columns={p['old_name']:p['new_name']}).drop(columns=[p['obsolete']])
    if task_id == 'm1_15':
        return p['df'].sort_values(['department','quantity'],ascending=[True,False])
    if task_id == 'm1_16':
        return pd.read_csv(p['path'],sep=';',dtype={'order_id':'string'})
    if task_id == 'm1_17':
        return pd.read_excel(p['path'], sheet_name=p['sheet'])
    if task_id == 'm1_18':
        return pd.read_json(p['path'])
    if task_id == 'm1_19':
        out=p['df'][['order_id','product_id']].copy()
        out['line_total']=p['df']['quantity']*p['df']['unit_price']
        return out
    if task_id == 'm1_20':
        # Don't create the file prematurely: the student must save it.
        return {'special_export':True, 'path':Path(p['path']), 'df':p['df']}
    raise KeyError(f'Нет задания: {task_id}')


def expected_for(task_id, **inputs):
    """Store expected result BEFORE the student works on the input data.

    This makes accidental in-place modifications visible to the checker.
    """
    p=deepcopy(inputs)
    if task_id.startswith('m1_'):
        return _module_one(task_id,p)
    oracle=ORACLES.get(task_id)
    if oracle is None:
        raise KeyError(f'Нет задания: {task_id}')
    return oracle(**p)


def _progress(task_id):
    _PROGRESS_FILE.parent.mkdir(parents=True,exist_ok=True)
    try: state=json.loads(_PROGRESS_FILE.read_text(encoding='utf-8'))
    except (ValueError,OSError): state={}
    state[task_id]=True
    _PROGRESS_FILE.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')


def check_result(task_id, result, expected):
    """Check only the displayed input example; no functions or arguments needed."""
    if result is None:
        print(f'◻ {task_id}: напишите код и сохраните результат в result.')
        return False
    try:
        if isinstance(expected,dict) and expected.get('special_export') is True:
            path=expected['path']
            assert Path(result)==path, f'Нужно вернуть путь {path}'
            assert path.is_file(), f'Файл {path} не создан'
            saved=pd.read_csv(path,dtype=str)
            orig=expected['df']
            assert saved.columns.tolist()==orig.columns.tolist(), 'Колонки не совпали (лишний индекс?)'
            assert saved.shape==orig.shape, 'Размер сохранённой таблицы не совпадает'
            for col in saved.columns:
                for a,b in zip(saved[col], orig[col]):
                    assert a==str(b), f'Ошибка при сохранении в колонке {col}'
        elif task_id.startswith('m1_'):
            _compare_simple(result,expected)
        else:
            similar(result,expected,strict=(task_id in STRICT_DTYPES))
    except Exception as exc:
        print(f'✗ {task_id}: {type(exc).__name__}: {str(exc)[:600]}')
        return False
    print(f'✓ {task_id}: верно! Результат совпадает с ожидаемым выводом.')
    _progress(task_id)
    return True


def _compare_simple(actual,expected):
    if isinstance(expected,pd.DataFrame):
        assert isinstance(actual,pd.DataFrame),'Ожидается DataFrame'
        assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=1e-7,atol=1e-9)
    elif isinstance(expected,pd.Series):
        assert isinstance(actual,pd.Series),'Ожидается Series'
        assert_series_equal(actual,expected,check_dtype=False,check_exact=False,rtol=1e-7,atol=1e-9)
    elif isinstance(expected,dict):
        assert isinstance(actual,dict),'Ожидается словарь'
        assert set(actual)==set(expected),'Неверные ключи словаря'
        for key in expected: _compare_simple(actual[key],expected[key])
    elif isinstance(expected,(tuple,list)):
        assert isinstance(actual,(tuple,list)),'Ожидается список или кортеж'
        assert len(actual)==len(expected),'Неверная длина'
        for a,b in zip(actual,expected): _compare_simple(a,b)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        assert math.isclose(float(actual),float(expected),rel_tol=1e-7,abs_tol=1e-9), f'Ожидается {expected}, получено {actual}'
    else:
        assert actual==expected,f'Ожидается {expected!r}, получено {actual!r}'


def show_progress():
    try: state=json.loads(_PROGRESS_FILE.read_text(encoding='utf-8'))
    except (ValueError,OSError): state={}
    complete={k for k,v in state.items() if v}
    print('Прогресс Pandas Academy:')
    for mod in range(1,9):
        total=15 if mod==8 else 20
        n=sum(f'm{mod}_{i:02d}' in complete for i in range(1,total+1))
        print(f'  Модуль {mod:02d}: {n}/{total}')
    print(f'ИТОГО: {len(complete.intersection({f"m{m}_{i:02d}" for m in range(1,9) for i in range(1,(15 if m==8 else 20)+1)}))}/155')
