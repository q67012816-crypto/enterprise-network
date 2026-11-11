"""正例首次取得日相对预测日 C_t（t+1 年 4 月 30 日）的分类规则（DEV_RESULTS 第 21.1 节；修订 A8.2）。
label_audit.py（开发阶段）和 confirm_datesplit.py（2027 年确认运行）共用，规则只在这里定义。
parse_date 由调用方传入 build_v3.py 自己的解析函数。"""
import re
import pandas as pd

def fy_bounds(stlm, fy1):
    """t+1 财年区间 (start, end]：按 build_v3.row_newness 的口径由结算日推出；结算日缺失时按 12 月 31 日。"""
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', stlm or ''); end = tuple(map(int, m.groups())) if m else (fy1, 12, 31)
    return (end[0] - 1, end[1], end[2]), end

def row_date_cat(acq, stlm, t, parse_date):
    """一条出资行的类别：before / after / earlier / later / year_only / missing。"""
    fy1 = t + 1; start, end = fy_bounds(stlm, fy1); cut = (fy1, 4, 30); d = parse_date(acq)
    if d is None: return 'missing'
    if d[1] is None:
        return 'year_only' if d[0] == fy1 else ('earlier' if d[0] < fy1 else 'later')
    if d <= start: return 'earlier'
    if d > end: return 'later'
    return 'before' if d <= cut else 'after'

def pair_rows(ROWS, t, directions):
    """build_v3 生成这条正边时用到的 t+1 年出资行（点对的所有方向）。"""
    R1 = ROWS[t + 1]; out = []
    for dr in str(directions).split('+'):
        a, b = dr.split('>'); out.append(R1[(R1.src_corp_code == a) & (R1.t == b)])
    return pd.concat(out)

def pair_date_class(dc):
    """正例的类别：任一行 before → before；否则任一行 after → after；否则 unknown:<earlier > later > year_only > missing>；没有行 → unknown:no_row。"""
    if 'before' in dc: return 'before'
    if 'after' in dc: return 'after'
    return 'unknown:' + next((c for c in ('earlier', 'later', 'year_only', 'missing') if c in dc), 'no_row')
