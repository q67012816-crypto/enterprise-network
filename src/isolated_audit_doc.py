"""STEP2 refinement: classify baseline-isolated test firms using the ORIGINAL FY2024 사업보고서 document (as filed, rcept_dt<=C)
in addition to the API table. Rules (deterministic, no fuzzy):
 names: exact / suffix-normalised / stock-name of current names, unique only; then former names from name_history.csv
 (source_verified=Y, filed <=C), unique across universe and not colliding with any current name.
 categories (priority): ii_name_match_failed > i_api_table_incomplete (doc row resolves to an observed universe firm, held>0,
 API produced no edge) > i_doc_not_obtained_or_parse_failed > iv_filing_lists_no_relation > iii_external_unlisted >
 iii_listed_unmatched_未验证 (listed but not matched: foreign/KONEX/delisted or name variant)."""
import collections, json, pandas as pd
from pathlib import Path
from isolated_audit import verified_former, main as api_audit, build_names, resolve, norm, read, PLACEHOLDER, C
from fetch_docs import originals
R = Path(__file__).resolve().parents[1]

def main():
    V, E, observed, iso, nofin = api_audit()
    U = read(R / 'data/processed/universe_firms.csv'); O = read(R / 'data/interim/company_overview.csv'); codes = set(U.corp_code)
    names = build_names(O, codes); cur = set(names['norm'])
    former, _ = verified_former(codes)
    o = originals(); D = read(R / 'data/processed/doc_financials.csv').set_index('rcept_no')
    I = read(R / 'data/processed/doc_invest_rows.csv'); I = I[I.rcept_no.isin(set(o.values))]
    I = I[~I.inv_prm.str.strip().isin(PLACEHOLDER)].copy()
    num = lambda c: pd.to_numeric(c.str.replace(',', '', regex=False), errors='coerce')
    I['held'] = num(I.end_pct).gt(0) | num(I.end_qty).gt(0) | num(I.end_book).gt(0)
    def res2(v):
        t, k = resolve(names, v)
        if t or k == 'ambiguous': return t, k
        c = former.get(norm(v), set())
        return (next(iter(c)), 'former_name') if len(c) == 1 and norm(v) not in cur else ('', 'unmatched')
    rr = {s: res2(s) for s in I.inv_prm.unique()}
    I['target'] = I.inv_prm.map(lambda s: rr[s][0]); I['rule'] = I.inv_prm.map(lambda s: rr[s][1])
    api_pairs = set(zip(E.src_corp_code, E.target))
    ok = I.target.ne('') & I.held & I.corp_code.isin(observed) & I.target.isin(observed) & I.corp_code.ne(I.target)
    new = I[ok & ~pd.Series([p in api_pairs for p in zip(I.corp_code, I.target)], index=I.index)]
    hit = collections.defaultdict(list)
    for r in new.to_dict('records'):
        for k in (r['corp_code'], r['target']): hit[k].append((r['rule'], f"{r['corp_code']}->{r['target']} '{r['inv_prm']}' {r['end_pct']}% ({r['rcept_no']})"))
    prev = read(R / 'data/processed/isolated_audit.csv').set_index('corp_code')
    rows = []
    for k in iso:
        ev = json.loads(prev.evidence[k]); ev['api_category'] = prev.category[k]; rc = o.get(k, '')
        ev['doc_rcept_no'] = rc; d = D.loc[rc] if rc in D.index else None
        if k in hit:
            h = hit[k]; ev['doc_edges'] = [x for _, x in h][:5]
            cat = 'ii_name_match_failed' if any(r == 'former_name' for r, _ in h) else 'i_api_table_incomplete'
        elif d is None or d.doc_status != 'ok': cat = 'i_doc_not_obtained_or_parse_failed'; ev['doc_status'] = 'no_original_by_C' if not rc else (d.doc_status if d is not None else 'not_downloaded')
        else:
            s = I[I.rcept_no.eq(rc)]; ev.update(doc_section=d.invest_section, doc_says_none=d.invest_says_none, doc_real_rows=len(s))
            if len(s) == 0:
                cat = 'iv_filing_lists_no_relation' if d.invest_says_none == 'True' or d.invest_section == 'True' else 'i_doc_not_obtained_or_parse_failed'
                if d.invest_section != 'True': ev['note'] = '타법인출자 section not found in document'
            else:
                ev['doc_listed'] = dict(collections.Counter(s.listed)); ev['doc_rules'] = dict(collections.Counter(s.rule))
                lu = s[s.listed.eq('상장') & s.target.eq('')]
                cat = 'iii_listed_unmatched_未验证' if len(lu) else 'iii_external_unlisted'
                if len(lu): ev['listed_unmatched'] = list(lu.inv_prm.head(5))
        rows.append(dict(corp_code=k, feature_fy='2024', category=cat, evidence=json.dumps(ev, ensure_ascii=False)))
    out = pd.DataFrame(rows); out.to_csv(R / 'data/processed/isolated_audit.csv', index=False)
    new.to_csv(R / 'data/processed/doc_recovered_edges_2024.csv', index=False)
    summ = json.load(open(R / 'reports/isolated_audit.json'))
    summ.update(categories_doc=out.category.value_counts().to_dict(), doc_new_edges_total=len(new),
                doc_new_edges_by_rule=new.rule.value_counts().to_dict(),
                doc_new_edges_touching_isolated=int(new.corp_code.isin(iso).sum() + new.target.isin(iso).sum()))
    json.dump(summ, open(R / 'reports/isolated_audit.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
if __name__ == '__main__': main()
