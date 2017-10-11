from os import listdir, makedirs, path
import shutil
import re
import time

in_folder = 'data_plot/results'
out_folder = "data_plot/gm1"

regex_keylist = ["seed size", "MATCHED", "CORRECT", "Precision", "F1-score", "Recall", "WRONG"]

def retrieve(f):
    param_dict = {}
    fr = open(f, 'r')
    rows = fr.read()
    fr.close()
    for k in regex_keylist:
        m = re.search('%s =\s+(\d+[.\d]*|$)' % k, rows)
        if m:
            if k == "seed size":
                param_dict[k] = int(m.group(1))
            else:
                param_dict[k] = m.group(1)
    
    return param_dict

def list_files():
    return [path.join(in_folder, f) for f in listdir(in_folder) if path.isfile(path.join(in_folder, f))]

def proceed():
    res = []
    keys_cn = len(regex_keylist)
    for fname in list_files():
        ret = retrieve(fname)
        if len(ret) == keys_cn:
            res.append(ret)
    res = sorted(res, key = lambda d: d['seed size'])
    return res

def store_results(ps):
    try:
        shutil.rmtree(out_folder)
    except FileNotFoundError:
        pass
    makedirs(out_folder)
    seed_key = regex_keylist[0]
    for k in regex_keylist[1:]:
        with open(path.join(out_folder, k.lower()+'.txt'), 'w') as f:
            for p in ps:
                f.write("%d %s\n" % (p[seed_key], p[k]))



ps = proceed()
store_results(ps)