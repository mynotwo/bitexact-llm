"""python cmp8.py REF.hash PLUGIN.hash [PLUGIN.json REF.json]: rows (seed < 4, t) must match in sha16 and token."""
import sys
def load(p):
    d = {}
    for ln in open(p):
        s, t, h, y = ln.split(); d[(int(s), int(t))] = (h, int(y))
    return d
r, p = load(sys.argv[1]), load(sys.argv[2])
p = {k: v for k, v in p.items() if k in r}
print("ref rows", len(r), "plugin rows matched keys", len(p), "diff", sum(r[k] != p.get(k) for k in r))
sys.exit(int(any(r[k] != p.get(k) for k in r)))
