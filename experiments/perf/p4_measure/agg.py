import json,glob,re,statistics as st,os,sys
O=sys.argv[1]
def med(x): return st.median(x)
print("## PROF (decode ms/step, median of 3)")
for b in (128,32):
  for T in (0,1):
    ds=[json.load(open(f)) for f in sorted(glob.glob(f"{O}/prof8_b{b}_T{T}/r*/int.json"))]
    if not ds: continue
    g=lambda f:med([f(d) for d in ds])
    bk=lambda k:g(lambda d:d['decode']['buckets_ms_per_step'][k])
    wall=g(lambda d:d['timing']['decode_step_ms_median']); ker=g(lambda d:d['decode']['kernel_ms_per_step']); busy=g(lambda d:d['decode']['busy_ms_per_step'])
    print(f"B={b} T={T} n={len(ds)} wall={wall:.3f} kernel={ker:.3f} gemm={bk('gemm_int8'):.3f} elem={bk('elementwise'):.3f} attn={bk('attention'):.3f} samp={bk('sampling_logits'):.3f} other={bk('other'):.3f} wall-busy={wall-busy:.3f} idle={g(lambda d:d['decode']['gpu_idle_ms_per_step']):.3f} tok/s={g(lambda d:d['tok_per_s']):.0f} kern/step={g(lambda d:d['decode']['kernels_per_step']):.0f}")
ds=[json.load(open(f)) for f in sorted(glob.glob(f"{O}/prof8_b128_T0/r*/int.json"))]
if ds:
  print("## top10 B=128 T=0 (rep1)")
  for t in ds[0]['decode']['top'][:10]: print(f"{t['ms_per_step']:.3f} ms x{t['calls_per_step']:.0f} [{t['bucket']}] {t['name'][:110]}")
print("## BENCH8 tok/s median [min-max]")
rows={}
for f in glob.glob(f"{O}/bench8/*.log"):
  m=re.match(r"(plugin|stock|bi)_b(\d+)(?:_T(\d))?_r(\d+)",os.path.basename(f))
  for l in open(f,errors="ignore"):
    if l.startswith("RESULT"):
      j=json.loads(l[7:]); v=j.get("tok_per_s") or j.get("tps")
      rows.setdefault((m[1],int(m[2]),m[3]),[]).append(v)
for k in sorted(rows,key=lambda k:(k[0],k[1],str(k[2]))):
  v=rows[k]; print(k,len(v),f"{med(v):.0f} [{min(v):.0f}-{max(v):.0f}]")
print("## SPEC tok/s median [min-max]")
rows={}
for f in glob.glob(f"{O}/spec/*.log"):
  m=re.match(r"(\w+)_B(\d+)_T([\d.]+)_k(\d)_r(\d)",os.path.basename(f))
  for l in open(f,errors="ignore"):
    if l.startswith("RESULT"):
      j=json.loads(l[7:]); rows.setdefault(m.groups()[:4],[]).append(j.get("tok_per_s") or j.get("tps") or j)
for k in sorted(rows):
  v=rows[k]
  if isinstance(v[0],dict): print(k,v[0]); continue
  print(k,len(v),f"{med(v):.0f} [{min(v):.0f}-{max(v):.0f}]")
