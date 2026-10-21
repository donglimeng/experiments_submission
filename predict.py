"""Official10 reproducible prediction: no public truth loaded."""
from core import *
from run_experiments import infer,apply
import argparse
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();manifest=c.load(ROOT/'source_manifest.json');paths=sorted(Path(a.data_dir).glob('*.txt'));assert {p.name:c.sha(p) for p in paths}==manifest['test_sha256'];assert c.sha(prior.ROOT/'champion.npy')==manifest['baseline_sha256'];layout=c.load(V11/'official_layout.json');base=np.load(ROOT/'baseline.npy');params=c.load(ROOT/'selection.json')['champion'];meta=c.load(ROOT/'calibration.json');models={f'{t}_{k}':lgb.Booster(model_str=(ROOT/'models'/f'{t}_{k}.txt').read_text(encoding='utf-8')) for t in c.TARGETS for k in ['plain','weighted']};out={};depths={}
 assert np.array_equal(base,np.load(prior.ROOT/'champion.npy'))
 assert c.sha(ROOT/'calibration.json')==manifest['calibration_sha256']
 assert {p.name:c.sha(p) for p in (ROOT/'models').glob('*.txt')}==manifest['models']
 checks=c.load(ROOT/'integrity_checks.json')
 for path,sha in checks['inherited_quantile_input_sha256'].items():assert c.sha(Path(path))==sha
 for path in paths:
  cols,_,_=c.r.read_well(path,training=False);data=infer(cols,models,meta,path.stem);p=base[slice(*layout['slices'][path.stem])].copy()
  for j,t in enumerate(c.TARGETS):p[:,j]=apply(p[:,j],{k.removeprefix(t+'_'):v for k,v in data.items() if k.startswith(t+'_')},params[t],j)
  out[path.stem]=p;depths[path.stem]=np.asarray(cols['DEPTH']);print('PREDICTED',path.stem,flush=True)
 c.save(Path(a.output),c.payload(out,depths))
if __name__=='__main__':main()
