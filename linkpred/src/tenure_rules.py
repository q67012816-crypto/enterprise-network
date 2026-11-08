"""兼任上任时间的解析规则（DEV_RESULTS 第 20.1 节；修订 A8.1）。interlock_tenure.py（开发阶段）和 confirm_tenure.py（2027 年确认运行）共用，规则只在这里定义。"""
import re
import pandas as pd

def months_between(d0, d1):
    """d1 − d0 的整月数（向下取整）。"""
    m = (d1.year - d0.year) * 12 + d1.month - d0.month
    return m - (d1.day < d0.day)

def classify(h, fy, rdate):
    """返回（类别，是否排除）。类别：duration / start / lower / short / unparsed。"""
    s = re.sub(r'\s+', '', h or ''); ye = pd.Timestamp(f'{fy}-12-31'); rd = pd.Timestamp(rdate)
    if s in ('', '-', '--', '―', '－'): return 'unparsed', None
    if '신임' in s: return 'short', True
    m = re.fullmatch(r'(\d+(?:\.\d+)?)년(이상|이내|미만)?', s) or None
    if m and m.group(2) == '이상': return 'lower', False
    if m and m.group(2) in ('미만', '이내'): return ('short', True) if float(m.group(1)) <= 1 else ('lower', False)
    m2 = re.fullmatch(r'(?:(\d+)년)?(?:(\d+)개월)?(?:\(.*\))?', s)
    if m2 and (m2.group(1) or m2.group(2)):
        T = int(m2.group(1) or 0) * 12 + int(m2.group(2) or 0)
    elif m: T = int(float(m.group(1)) * 12)
    elif re.fullmatch(r'\d+(\.\d+)?', s) and float(s) < 100: T = int(float(s) * 12)
    else: T = None
    if T is not None:
        if T == 0: return 'short', True
        return 'duration', bool((rd - pd.DateOffset(months=T)) > ye)   # 上任时间最晚 = 提交日 − T 个月
    d = re.match(r'(\d{4}|\d{2})[.\-/년](\d{1,2})(?:[.\-/월](\d{1,2}))?', s)
    if d:
        y = int(d.group(1)); y = y + 2000 if y < 100 else y; mo = int(d.group(2))
        if 1 <= mo <= 12 and 1950 <= y <= 2030: return 'start', (y, mo) > (fy, 12)
    d = re.match(r'(\d{4})(?:년)?[~\-]', s)
    if d and 1950 <= int(d.group(1)) <= 2030: return 'start', int(d.group(1)) > fy
    return 'unparsed', None
