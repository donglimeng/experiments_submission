"""OOF probability calibration and score-aware continuous experts; official80 only."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'experiments_submission84_tuning_v15_score_gate'))
import core_v15 as prior
# Parent versions insert their own directories; local module names must win.
sys.path.insert(0,str(ROOT))
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
np=prior.np;c=prior.c;V9=prior.V9;V11=prior.V11
LEVELS=np.array([.05,.15,.25,.35,.5,.65,.75,.85,.95])
FEATURES=prior.previous.FEATURES
def atom(y,j):return np.isclose(y,c.ATOMS[j],atol=.0005,rtol=0)
def transform(y,j):return np.log(np.maximum(y,1e-8)) if j==1 else y
def physical(y,j):return c.parent.clip(np.exp(np.clip(y,-18,18)) if j==1 else y,j)
def blend(base,other,a,j):
 out=physical((1-a)*transform(base,j)+a*transform(other,j),j)
 return np.where(atom(base,j),base,out)
def model(j):
 return lgb.LGBMRegressor(objective='quantile',alpha=.5,n_estimators=250,max_depth=5,num_leaves=24,min_child_samples=300,learning_rate=.05,reg_lambda=15,colsample_bytree=.9,max_bin=127,n_jobs=4,verbosity=-1,random_state=16021+j)
def fit_save(x,y,w,j,path):
 if not path.exists():
  m=model(j);m.fit(x,y,sample_weight=w);path.write_text(m.booster_.model_to_string(),encoding='utf-8')
 return lgb.Booster(model_str=path.read_text(encoding='utf-8'))
def weight(y,p,j):
 if j==1:r=np.abs(np.log10(np.maximum(p,1e-8))-np.log10(np.maximum(y,1e-8)))
 else:r=np.abs(p-y)/((.08 if j==0 else .05)*(np.abs(y)+1e-5))
 w=.5+np.clip(1-r/2,0,1);return w/w.mean()
def quantiles(z,j,stats):
 q=z[:,100+9*j:109+9*j]*stats['std'][100+9*j:109+9*j]+stats['mean'][100+9*j:109+9*j]
 q=np.exp(np.clip(q,-18,18)) if j<2 else q*100
 return np.sort(np.column_stack([c.parent.clip(q[:,k],j) for k in range(9)]),axis=1)
def regions(q,cuts):
 width=(q[:,-1]-q[:,0])/np.maximum(q[:,4],.001)
 return (q[:,4]>cuts[0]).astype(int)*2+(width>cuts[1]).astype(int)
def calibrate(q,y,mask):
 q=q[mask];y=y[mask];cuts=[float(np.median(q[:,4])),float(np.median((q[:,-1]-q[:,0])/np.maximum(q[:,4],.001)))];g=regions(q,cuts)
 coverage=(y[:,None]<=q).mean(0);glob=.95*coverage+.05*LEVELS;local=[]
 for k in range(4):
  m=g==k;v=.95*(y[m,None]<=q[m]).mean(0)+.05*LEVELS if m.sum()>=2000 else glob
  local.append((.5*v+.5*glob).tolist())
 return dict(cuts=cuts,global_cdf=glob.tolist(),conditional_cdf=local,counts=[int((g==k).sum()) for k in range(4)])
def decide(q,meta,mode,j):
 """Piecewise-linear inverse CDF, flat tails; fixed 41 candidate values."""
 out=np.empty(len(q));group=regions(q,meta['cuts'])
 for start in range(0,len(q),1024):
  sl=slice(start,min(start+1024,len(q)));a=q[sl];n=len(a)
  levels=np.broadcast_to(LEVELS,(n,9)) if mode=='nominal' else np.broadcast_to(np.array(meta['global_cdf']),(n,9)) if mode=='global' else np.asarray(meta['conditional_cdf'])[group[sl]]
  # Integrate the distribution at fixed probability midpoints, not equal quantile weights.
  masspoints=(np.arange(81)+.5)/81;vals=np.empty((n,81))
  for i in range(n):vals[i]=np.interp(masspoints,levels[i],a[i])
  if j==1:cand=np.exp(np.log(np.maximum(a[:,0],1e-8))[:,None]+np.linspace(0,1,41)*(np.log(np.maximum(a[:,-1],1e-8))-np.log(np.maximum(a[:,0],1e-8)))[:,None])
  else:cand=a[:,0,None]+np.linspace(0,1,41)*(a[:,-1]-a[:,0])[:,None]
  utility=prior.old.point_scores(vals[:,None,:],cand[:,:,None],j).mean(2)
  utility-=1e-10*np.abs(cand-a[:,4,None]);out[sl]=cand[np.arange(n),utility.argmax(1)]
 return out
