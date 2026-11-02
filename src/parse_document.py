"""Parse as-filed separate (OFS) key accounts from a DART document.xml (dart4 schema).
Uses the XBRL tag (ACODE) and context (ACONTEXT) that DART embeds in financial-statement cells.
Rule: current-year context 'CFY{fy}' + SeparateMember axis; if a document has no Separate/Consolidated
axis at all (single statement filer) the axis-less CFY context is accepted and flagged. Unit from nearest
preceding '(단위 : ...)'. Fallback by Korean row label only when the tag is absent (flagged)."""
import re, zipfile
TAGS = {'assets': ['ifrs-full_Assets'], 'liabilities': ['ifrs-full_Liabilities'], 'equity': ['ifrs-full_Equity'],
        'capital': ['ifrs-full_IssuedCapital'], 'revenue': ['ifrs-full_Revenue', 'ifrs-full_RevenueFromContractsWithCustomers'],
        'operating_income': ['dart_OperatingIncomeLoss'], 'net_income': ['ifrs-full_ProfitLoss']}
LABELS = {'assets': ['자산총계'], 'liabilities': ['부채총계'], 'equity': ['자본총계'], 'capital': ['자본금'],
          'revenue': ['매출액', '수익(매출액)', '영업수익', '매출'], 'operating_income': ['영업이익', '영업이익(손실)', '영업손실'],
          'net_income': ['당기순이익', '당기순이익(손실)', '당기순손실']}
UNIT = {'원': 1, '천원': 1e3, '백만원': 1e6, '억원': 1e8}
CELL = re.compile(r'<TE\b([^>]*)>(.*?)</TE>', re.S)
def num(s):
    s = re.sub(r'<[^>]+>', '', s).replace('\u3000', '').replace(',', '').strip()
    neg = s.startswith('(') and s.endswith(')') or s.startswith('-') or s.startswith('△')
    s = s.strip('()-△ ')
    try: v = float(s)
    except ValueError: return None
    return -v if neg else v

def main_xml(zpath):
    z = zipfile.ZipFile(zpath); names = [n for n in z.namelist() if re.fullmatch(r'\d{14}\.xml', n)]
    return z.read(names[0]).decode('utf-8', 'replace') if names else None

def parse(text, fy):
    out, flags = {}, []
    units = [(m.start(), UNIT.get(m.group(1).replace(' ', ''), None)) for m in re.finditer(r'\(\s*단위\s*:\s*([^)\s]+)\s*\)', text)]
    def unit_at(pos):
        u = [x for p, x in units if p < pos]; return u[-1] if u and u[-1] else None
    has_axis = 'SeparateMember' in text or 'ConsolidatedMember' in text
    for m in CELL.finditer(text):
        a = m.group(1); code = re.search(r'ACODE="([^"]+)"', a); ctx = re.search(r'ACONTEXT="([^"]+)"', a)
        if not code or not ctx or not ctx.group(1).startswith(f'CFY{fy}'): continue
        c = ctx.group(1)
        sep = 'SeparateMember' in c and c.count('Axis') == 1
        if not (sep or (not has_axis and 'Axis' not in c)): continue
        for k, tags in TAGS.items():
            if code.group(1) in tags and k not in out:
                v = num(m.group(2)); u = unit_at(m.start())
                if v is not None and u: out[k] = v * u
    if not has_axis and out: flags.append('no_axis_single_statement')
    return out, flags

def company_name(text):
    m = re.search(r'<COMPANY-NAME AREGCIK="(\d+)">([^<]*)</COMPANY-NAME>', text)
    return (m.group(1), m.group(2).strip()) if m else (None, None)

def strip(s): return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', s)).replace('\u3000', ' ').strip()

def invest_section(text):
    """Rows of '타법인출자 현황' tables (ACODE INV_*). Returns (rows, section_found, says_none)."""
    starts = [m.end() for m in re.finditer(r'<TITLE[^>]*>[^<]*타법인\s*출자\s*현황[^<]*</TITLE>', text)]
    if not starts: return [], False, False
    rows, none = [], False
    for st in starts:
        nxt = re.search(r'<TITLE[^>]*ATOC="Y"', text[st:]); seg = text[st: st + (nxt.start() if nxt else 200000)]
        if re.search(r'해당\s*사항\s*(이\s*)?없|해당\s*없음', strip(seg)[:3000]): none = True
        for tr in re.findall(r'<TR\b.*?</TR>', seg, re.S):
            if 'ACODE="INV_PRM"' not in tr: continue
            cells = dict((a, strip(v)) for a, v in re.findall(r'<TE[^>]*ACODE="(INV_[A-Z]+)"[^>]*>(.*?)</TE>', tr, re.S))
            plain = strip(re.sub(r'<TE[^>]*ACODE="INV_[A-Z]+"[^>]*>.*?</TE>', ' | ', tr, flags=re.S))
            listed = '상장' if re.search(r'(?<!비)상장', plain) else ''
            if '비상장' in plain: listed = '비상장'
            rows.append(dict(inv_prm=cells.get('INV_PRM', ''), listed=listed, end_qty=cells.get('INV_LTN', ''),
                             end_pct=cells.get('INV_LPR', ''), end_book=cells.get('INV_LTM', '')))
        if rows: break   # first table with rows (detail section usually last; summary has no INV_PRM)
    return rows, True, none

def rename_snippets(text):
    """Text snippets around '상호 변경' (for manual verification of former names)."""
    s = strip(text); out = []
    for m in re.finditer(r'상\s*호\s*(?:를|의)?\s*변\s*경', s):
        out.append(s[max(0, m.start() - 80): m.end() + 60])
    return out[:20]

FORMER = [re.compile(r'[\(（]\s*(?:舊|구)\s*[:：]?\s*([^)）]{2,40})[\)）]'),
          re.compile(r'([가-힣A-Za-z0-9&()㈜\.\s]{2,40}?)\s*(?:에서|→|->|⇒)\s*([가-힣A-Za-z0-9&()㈜\.\s]{2,40}?)\s*(?:으로|로)?\s*상\s*호')]
def former_names(snips):
    """Deterministic patterns only: '(구 X)' and 'X 에서/→ Y (으)로 상호'. Returns list of (old, new_or_empty, snippet)."""
    out = []
    for sn in snips:
        for m in FORMER[0].finditer(sn): out.append((m.group(1).strip(), '', sn))
        for m in FORMER[1].finditer(sn): out.append((m.group(1).strip(), m.group(2).strip(), sn))
    return out

def rename_table(text):
    """'상호의 변경' table: rows (date YYYYMMDD, before, after). Header must contain 변경전/변경후 (spaces ignored)."""
    out = []
    for m in re.finditer(r'상\s*호\s*의\s*변\s*경', text):
        tm = re.search(r'<TABLE\b.*?</TABLE>', text[m.end(): m.end() + 20000], re.S)
        if not tm: continue
        trs = [[strip(c) for c in re.findall(r'<T[DEH]\b[^>]*>(.*?)</T[DEH]>', tr, re.S)] for tr in re.findall(r'<TR\b.*?</TR>', tm.group(0), re.S)]
        if not trs: continue
        h = [x.replace(' ', '') for x in trs[0]]
        if '변경전' not in h or '변경후' not in h: continue
        ib, ia = h.index('변경전'), h.index('변경후'); idt = next((i for i, x in enumerate(h) if '일' in x), 0)
        for r in trs[1:]:
            if len(r) <= max(ib, ia): continue
            d = re.findall(r'\d+', r[idt]) if idt < len(r) else []
            ds = f"{d[0]}{int(d[1]):02d}{int(d[2]):02d}" if len(d) >= 3 and len(d[0]) == 4 else ''
            out.append((ds, r[ib], r[ia]))
        if out: break
    return out

def parse_xbrl(xml, fy):
    """Same context rule as parse() on the XBRL instance (fnlttXbrl.xml member *.xbrl). Values are unscaled."""
    out = {}; has_axis = 'SeparateMember' in xml or 'ConsolidatedMember' in xml
    for k, tags in TAGS.items():
        for tg in tags:
            el = tg.replace('_', ':', 1)
            for m in re.finditer(r'<' + re.escape(el) + r'\b[^>]*contextRef="([^"]+)"[^>]*>([^<]*)<', xml):
                c = m.group(1)
                if not c.startswith(f'CFY{fy}'): continue
                if not (('SeparateMember' in c and c.count('Axis') == 1) or (not has_axis and 'Axis' not in c)): continue
                try: out[k] = float(m.group(2)); break
                except ValueError: pass
            if k in out: break
    return out
