import json,sys
for o in sys.argv[1:]:
    print('=====',o)
    try: v=json.load(open(f'{o}/validation_selection.json')); print('sel',v['selected_C'],'bestH',v['best_heuristic'],{k:round(x,4) for k,x in v['val_AP'].items()})
    except Exception as e: print(e)
    f=[x for x in ['test_exploratory.json','test_frozen.json'] if __import__('os').path.exists(f'{o}/{x}')][0]; d=json.load(open(f'{o}/{f}'))
    for c,e in d.items():
        if c=='test_ranking':
            for k in ['M0_random','CN','AA','Jaccard','PA','M1','M1+scores']: print(' rank',k,{a:round(b,3) for a,b in e[k].items() if a in('Hits10','Hits50','MRR','n_investors')})
            continue
        print(c,e['n_pos'],e['informative'], {k:round(e['AP'][k],4) for k in ['M0_random','M0_same_industry','M0_same_market','CN','AA','Jaccard','PA','M1','M1+scores']} if 'AP' in e else '')
        if 'diff' in e: print('   ',{k:(round(v['point'],4),round(v['lo'],4),round(v['hi'],4)) for k,v in e['diff'].items()})
