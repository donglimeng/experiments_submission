"""Predeclared global blend grid; A5 is public development, never independent test."""
from core import *
from develop_v6 import get_data
import zipfile
ALPHAS=[0.,.05,.15,.3,.5,1.]
def infer(cols,models,meta,logid):
 context,_=c.r.make_features(cols,'context');x=context[:,FEATURES].astype(np.float32);stats=dict(np.load(V9/'preprocess.npz'));out={}
 for j,t in enumerate(c.TARGETS):
  qq=[]
  for f in range(5):
   z=np.load(V9/'test_inputs'/f'{logid}_fold{f}.npz');qq.append(quantiles(z['x'],j,stats))
  q=np.mean(qq,axis=0)
  for mode in ['nominal','global','conditional']:out[f'{t}_cal_{mode}']=decide(q,meta[t],mode,j)
  for kind in ['plain','weighted']:out[f'{t}_{kind}']=physical(models[f'{t}_{kind}'].predict(x,num_threads=2),j)
 return out
def validation_summary():
 v=np.load(ROOT/'validation.npz');o=np.load(c.MODEL/'work/final80/oof_predictions.npz');fold=o['fold'];rows=[]
 for j,t in enumerate(c.TARGETS):
  yy=v[f'{t}_truth'];g=v[f'{t}_groups'];cont=c.valid(yy,j)&~atom(yy,j);q=np.load(ROOT/f'oof_q_{t}.npy');qval=q[fold==0,4]
  for kind in ['plain','weighted']:
   p=v[f'{t}_{kind}'];ref=v[f'{t}_plain'] if kind=='weighted' else qval;wg=[100*(c.score(yy[(g==a)&cont],p[(g==a)&cont],j)-c.score(yy[(g==a)&cont],ref[(g==a)&cont],j)) for a in np.unique(g) if ((g==a)&cont).sum()]
   gain=100*(c.score(yy[cont],p[cont],j)-c.score(yy[cont],ref[cont],j));rows.append(dict(target=t,branch=kind,gain=gain,positive_wells=int(np.sum(np.array(wg)>0)),negative_wells=int(np.sum(np.array(wg)<0)),min_well_gain=min(wg),supported=bool(gain>0 and np.sum(np.array(wg)>0)>=np.sum(np.array(wg)<0)),reference='matched_plain' if kind=='weighted' else 'legacy_q50',scope='New expert outer16 continuous-label development'))
  truth=o['truth'][:,j];contall=c.valid(truth,j)&~atom(truth,j)
  for mode in ['nominal','global','conditional']:
   p=v[f'{t}_cal_{mode}'];ref=q[:,4] if mode=='nominal' else v[f'{t}_cal_nominal'];gain=100*(c.score(truth[contall],p[contall],j)-c.score(truth[contall],ref[contall],j));rows.append(dict(target=t,branch='cal_'+mode,gain=gain,supported=bool(gain>0),reference='legacy_q50' if mode=='nominal' else 'nominal_distribution',scope='Calibrator crossfit, legacy provider dependencies remain; not nested'))
 c.save(ROOT/'validation_summary.json',rows);return rows
def apply(base,data,seq,j):
 out=base.copy()
 for p in seq:out=blend(out,data[p['branch']],p['alpha'],j)
 return out
def main():
 (ROOT/'cache').mkdir(exist_ok=True);layout=c.load(V11/'official_layout.json');ids=layout['logids'];base=np.load(prior.ROOT/'champion.npy');np.save(ROOT/'baseline.npy',base);meta=c.load(ROOT/'calibration.json');models={f'{t}_{k}':lgb.Booster(model_str=(ROOT/'models'/f'{t}_{k}.txt').read_text(encoding='utf-8')) for t in c.TARGETS for k in ['plain','weighted']};records=[];depths={};testsha={}
 for logid in ids:
  path=c.BASE/'data/test'/f'{logid}.txt';testsha[path.name]=c.sha(path);cols,_,_=c.r.read_well(path,training=False);depths[logid]=np.asarray(cols['DEPTH']);cache=ROOT/'cache'/f'{logid}.npz'
  if cache.exists():d=dict(np.load(cache))
  else:d=infer(cols,models,meta,logid);np.savez_compressed(cache,**d)
  records.append(d);print('INFER',logid,flush=True)
 data={k:np.concatenate([r[k] for r in records]) for k in records[0]};ix=np.load(V11/'development_indices.npy');_,y,audit=get_data();offset=np.r_[0,np.cumsum([a['rows'] for a in audit])];vr=validation_summary();trials=[];params={n:{} for n in ['champion','best_local','accuracy_guard','calibration','weighting']};outputs={n:base.copy() for n in ['baseline',*params]}
 for j,t in enumerate(c.TARGETS):
  bscore=c.score(y[:,j],base[ix,j],j)*100;bm=c.metrics(y[:,j],base[ix,j],j);bw=[c.score(y[a:b,j],base[ix[a:b],j],j)*100 for a,b in zip(offset[:-1],offset[1:])];families={};targetdata={k.removeprefix(t+'_'):v for k,v in data.items() if k.startswith(t+'_')}
  def evaluate(seq):
   p=apply(base[:,j],targetdata,seq,j);m=c.metrics(y[:,j],p[ix],j);g=[c.score(y[a:b,j],p[ix[a:b]],j)*100-z for (a,b),z in zip(zip(offset[:-1],offset[1:]),bw)];supported=all(next(r for r in vr if r['target']==t and r['branch']==s['branch'])['supported'] for s in seq);row=dict(target=t,sequence=seq,**m,gain=m['score']-bscore,min_well_gain=min(g),validation_supported=supported,eligible=min(g)>=-.25-1e-9,accuracy_guard=m['mae']<=bm['mae']+1e-10 and m['rmse']<=bm['rmse']+1e-10);trials.append(row);return p,row
  candidates=[([],base[:,j],dict(score=bscore,eligible=True,validation_supported=True,accuracy_guard=True))]
  for family,branches in [('calibration',['cal_nominal','cal_global','cal_conditional']),('weighting',['plain','weighted'])]:
   members=[]
   for branch in branches:
    for alpha in ALPHAS[1:]:
     seq=[dict(branch=branch,alpha=alpha)];p,r=evaluate(seq);members.append((seq,p,r));candidates.append((seq,p,r))
   good=[v for v in members if v[2]['eligible'] and v[2]['validation_supported'] and v[2]['score']>bscore];families[family]=max(good,key=lambda v:v[2]['score']) if good else candidates[0];outputs[family][:,j]=families[family][1];params[family][t]=families[family][0]
  # Joint grid limited to the individually best supported branch in each family.
  cb=families['calibration'][0];wb=families['weighting'][0]
  if cb and wb:
   for aa in [.05,.15,.3,.5]:
    for bb in [.05,.15,.3,.5]:
     seq=[dict(branch=cb[0]['branch'],alpha=aa),dict(branch=wb[0]['branch'],alpha=bb)];p,r=evaluate(seq);candidates.append((seq,p,r))
  for name in ['champion','best_local','accuracy_guard']:
   good=[v for v in candidates if v[2]['eligible'] and (name=='best_local' or v[2]['validation_supported']) and (name!='accuracy_guard' or v[2]['accuracy_guard'])];winner=max(good,key=lambda v:v[2]['score']);params[name][t]=winner[0];outputs[name][:,j]=winner[1]
  print('SELECT',t,params['champion'][t],flush=True)
 for j,t in enumerate(c.TARGETS):p=base.copy();p[:,j]=outputs['champion'][:,j];outputs['only_'+t]=p
 outputs['v14_reference']=np.load(prior.V14/'champion.npy')
 scores={};perwell=[];transitions=[]
 for name,p in outputs.items():
  assert p.shape==base.shape and np.isfinite(p).all()
  if name!='v14_reference':
   for j in range(3):assert np.array_equal(p[atom(base[:,j],j),j],base[atom(base[:,j],j),j])
  np.save(ROOT/f'{name}.npy',p);metrics={t:c.metrics(y[:,j],p[ix,j],j) for j,t in enumerate(c.TARGETS)};scores[name]=dict(total=sum(w*metrics[t]['score'] for w,t in zip([.3,.35,.35],c.TARGETS)),metrics=metrics);payload=c.payload({i:p[slice(*layout['slices'][i])] for i in ids},depths);stem='best_reference' if name=='champion' else name;out=ROOT/f'{stem}_all10.json';c.save(out,payload)
  with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:z.write(out,arcname='best_reference_all10.json')
  for j,t in enumerate(c.TARGETS):
   valid=c.valid(y[:,j],j);b=c.point(y[valid,j],base[ix[valid],j],j);q=c.point(y[valid,j],p[ix[valid],j],j);transitions.append(dict(variant=name,target=t,zero_rescue=float(np.sum(q[b==0])*100/len(q)),new_zero=float(-np.sum(b[q==0])*100/len(q)),positive_precision=float(np.sum((q-b)[(b>0)&(q>0)])*100/len(q)),changed_valid=int(np.count_nonzero(p[ix[valid],j]!=base[ix[valid],j]))))
   for (a,b),info in zip(zip(offset[:-1],offset[1:]),audit):perwell.append(dict(variant=name,target=t,well=info['name'],**c.metrics(y[a:b,j],p[ix[a:b],j],j)))
 c.save(ROOT/'selection.json',params);c.save(ROOT/'scores.json',scores);c.save(ROOT/'trials.json',trials);c.save(ROOT/'source_manifest.json',dict(test_sha256=testsha,baseline_sha256=c.sha(prior.ROOT/'champion.npy'),calibration_sha256=c.sha(ROOT/'calibration.json'),models={p.name:c.sha(p) for p in (ROOT/'models').glob('*.txt')},baseline='v15 score-first champion',A_truth='Five public wells, official test intervals, development only'));c.pd.DataFrame([dict(variant=n,total=s['total'],**{t:s['metrics'][t]['score'] for t in c.TARGETS}) for n,s in scores.items()]).to_csv(ROOT/'comparison.csv',index=False,encoding='utf-8-sig');c.pd.DataFrame(perwell).to_csv(ROOT/'per_well_comparison.csv',index=False,encoding='utf-8-sig');c.pd.DataFrame(transitions).to_csv(ROOT/'score_transitions.csv',index=False,encoding='utf-8-sig');c.save(ROOT/'RUN_STATUS.json',dict(status='selected'));print('SEARCH_COMPLETE',scores['champion']['total'],flush=True)
if __name__=='__main__':main()
