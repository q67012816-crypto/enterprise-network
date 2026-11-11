"""探索性（冻结前追加，2026-11-06）：高管履历中的前任职公司。只算 fy 2019–2023（训练年和验证年；不计算测试年）。
数据：data/interim/exec_panel_raw.csv 的 main_career 文本（第 3 项已下载，不需要新数据）；as-of 规则同兼任：FY t 只用提交日 ≤ C_t 的年报。
公司名识别（唯一匹配，不做模糊匹配）：
  - 名称来源：合格上市公司的现名、股票简称、已核实的曾用名（verified_former，只用 C_t 前提交的改名记录）；
  - 名称和履历文本都用 paper_data._key 同样的方式归一化（去公司形态词、空白，小写）；
  - 键长 < 3 的名称不用（如"현대""동원"，作为子串误配太多）；同一个键对应多家公司的不用；
  - 在文本中按位置找最长匹配，匹配后跳过已覆盖的字符；以拉丁字母/数字开头或结尾的键要求词边界；本人当前任职的公司不计。
特征（a = 投资方，o = 候选企业）：
  car_ao  a 的现任人员（FY t）中，履历提到 o 的人数（log1p），去掉 FY t 同时在 o 任职（直接兼任）的人
  car_oa  反向
输出：out_dev/career_feats.csv（索引与 pairs.csv 行号一致）。用法：python src/career_feats.py"""
import collections, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import _key, _suffix, rd

L = Path(__file__).resolve().parents[1]; P0 = L.parent; D = L / 'data_v3x'; O = L / 'out_dev'
YEARS = range(2019, 2024); MINLEN = 3
CAR = ['car_ao', 'car_oa']

def name_index(t):
    """FY t 可用的 名称键 → 公司编号（只保留唯一对应的键）。"""
    U = set(rd(P0 / 'data/processed/universe_firms.csv').corp_code); Ov = rd(P0 / 'data/interim/company_overview.csv')
    sys.path.insert(0, str(P0 / 'src')); from isolated_audit import verified_former
    owners = collections.defaultdict(set)
    for c, a, b in zip(Ov.corp_code, Ov.corp_name, Ov.stock_name):
        if c in U:
            for n in (a, b): owners[_key(n)].add(c)
    for c, names in verified_former(U, f'{t + 1}0430')[0].items():
        for n in names: owners[_key(n)].add(c)
    return {k: next(iter(v)) for k, v in owners.items() if len(k) >= MINLEN and len(v) == 1}

_an = lambda ch: ch.isascii() and ch.isalnum()

def norm_text(s):
    return _suffix.sub('', (s or '').replace('（', '(').replace('）', ')')).lower()

def mentions(text, idx, maxlen):
    """文本中按位置找最长匹配的公司键，返回公司编号集合。"""
    out, i, n = set(), 0, len(text)
    while i < n:
        hit = None
        for L_ in range(min(maxlen, n - i), MINLEN - 1, -1):
            k = text[i:i + L_]; c = idx.get(k)
            # 以拉丁字母或数字开头（结尾）的键，前（后）一个字符不能也是拉丁字母或数字（去空白后英文单词会粘连，如 "University MBA" 中的 "tym"）
            if c is not None and not (_an(k[0]) and i > 0 and _an(text[i - 1])) and not (_an(k[-1]) and i + L_ < n and _an(text[i + L_])):
                hit = (c, L_); break
        if hit: out.add(hit[0]); i += hit[1]
        else: i += 1
    return out

def main():
    X = pd.read_csv(D / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str}); X = X[X.fy.isin(YEARS)]
    E = pd.read_csv(P0 / 'data/interim/exec_panel_raw.csv', dtype=str, keep_default_na=False)
    b = E.birth_ym.str.replace(r'\D', '', regex=True)
    E = E[E.api_status.eq('000') & E.nm.ne('') & b.str.len().eq(6)].assign(pid=lambda d: d.nm.str.replace(r'\s+', '', regex=True) + '|' + d.birth_ym.str.replace(r'\D', '', regex=True),
                                                                               fy=lambda d: pd.to_numeric(d.bsns_year))
    E = E[E.rcept_no.str[:8].le((E.fy + 1).astype(str) + '0430')]
    rows, stats = [], []
    for t, g in X.groupby('fy'):
        idx = name_index(t); maxlen = max(map(len, idx)); Et = E[E.fy == t]
        cache = {}; ment = collections.defaultdict(set)   # (公司, 人) → 履历中提到的公司
        firms_of = collections.defaultdict(set)
        for c, p, txt in zip(Et.src_corp_code, Et.pid, Et.main_career):
            firms_of[p].add(c)
            if txt not in cache: cache[txt] = mentions(norm_text(txt), idx, maxlen)
            ment[(c, p)] |= cache[txt]
        cnt = collections.Counter()                         # (现任公司 a, 提到的公司 o) → 人数，去掉直接兼任和本人现任公司
        for (c, p), ms in ment.items():
            for o in ms - firms_of[p]: cnt[(c, o)] += 1
        stats.append((t, len(idx), len(cache), sum(1 for v in ment.values() if v), len(ment)))
        for i, a, o in zip(g.index, g.anchor, g.other):
            rows.append(pd.Series(dict(car_ao=np.log1p(cnt[(a, o)]), car_oa=np.log1p(cnt[(o, a)])), name=i))
        print(t, 'name keys', len(idx), 'unique texts', len(cache), 'persons with ≥1 listed-firm mention', stats[-1][3], 'of', stats[-1][4], flush=True)
    F = pd.DataFrame(rows)[CAR]; F.to_csv(O / 'career_feats.csv', index_label='row')
    J = X.join(F); B = (J[CAR] > 0).assign(any=lambda d: d.any(axis=1), label=J.label)
    print('share of pairs with feature > 0, by label:'); print(B.groupby('label').mean().T.round(4).to_string())
    IL = pd.read_csv(O / 'interlock_feats.csv', index_col='row').reindex(J.index)
    il_any = (IL[['il_direct', 'il_2hop', 'il_hist']] > 0).any(axis=1); car_any = B['any']
    P = J.label == 1
    print('positives: career', round(float(car_any[P].mean()), 4), '| interlock', round(float(il_any[P].mean()), 4), '| both', round(float((car_any & il_any)[P].mean()), 4),
          '| career but no interlock', round(float((car_any & ~il_any)[P].mean()), 4))
    deg = {t: collections.Counter(list(g_.src) + list(g_.dst)) for t in YEARS for g_ in [pd.read_csv(D / f'graph_{t}.csv', dtype=str)]}
    iso = pd.Series([min(deg[t][a], deg[t][o]) == 0 for t, a, o in zip(J.fy, J.anchor, J.other)], index=J.index)
    print('positives with career signal: isolated', round(float(car_any[P & iso].mean()), 4), '| connected', round(float(car_any[P & ~iso].mean()), 4))

if __name__ == '__main__': main()
