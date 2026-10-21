from core import *
from run_experiments import infer,apply
from develop_v6 import get_data
import zipfile
def main():
 manifest=c.load(ROOT/'source_manifest.json');train=c.load(ROOT/'training_audit.json');assert {p.name:c.sha(p) for p in (c.BASE/'data/train').glob('*.txt')}==train['train_sha256'];assert {p.name:c.sha(p) for p in (c.BASE/'data/test').glob('*.txt')}==manifest['test_sha256'];assert c.sha(prior.ROOT/'champion.npy')==manifest['baseline_sha256'];assert {p.name:c.sha(p) for p in (ROOT/'models').glob('*.txt')}==manifest['models'];assert c.sha(ROOT/'calibration.json')==manifest['calibration_sha256'];layout=c.load(V11/'official_layout.json');base=np.load(ROOT/'baseline.npy');selection=c.load(ROOT/'selection.json');meta=c.load(ROOT/'calibration.json');models={f'{t}_{k}':lgb.Booster(model_str=(ROOT/'models'/f'{t}_{k}.txt').read_text(encoding='utf-8')) for t in c.TARGETS for k in ['plain','weighted']};parts=[];checks=[];cachehash={}
 for logid in layout['logids']:
  cols,_,_=c.r.read_well(c.BASE/'data/test'/f'{logid}.txt',training=False);data=infer(cols,models,meta,logid);cached=dict(np.load(ROOT/'cache'/f'{logid}.npz'));diff=max(float(np.max(np.abs(data[k]-cached[k]))) for k in data);assert diff<1e-10;sl=slice(*layout['slices'][logid]);p=base[sl].copy()
  for j,t in enumerate(c.TARGETS):p[:,j]=apply(p[:,j],{k.removeprefix(t+'_'):v for k,v in data.items() if k.startswith(t+'_')},selection['champion'][t],j)
  parts.append(p);checks.append(dict(well=logid,rows=len(p),candidate_replay_max_diff=diff))
  for f in range(5):path=V9/'test_inputs'/f'{logid}_fold{f}.npz';cachehash[str(path)]=c.sha(path)
 p=np.concatenate(parts);saved=np.load(ROOT/'champion.npy');assert np.array_equal(p,saved)
 # Independently reconstruct scalar official utility on varied points.
 for j in range(3):
  yy=np.array([.1,1,10,30,80,99.9]);pp=yy*np.array([1,.99,1.02,1.1,.9,1])
  expect=np.array([max(0,1-abs(float(np.log10(max(b,1e-8)))-float(np.log10(max(a,1e-8))))) if j==1 else max(0,1-abs(b-a)/((.08 if j==0 else .05)*(abs(a)+1e-5))) for a,b in zip(yy,pp)])
  assert np.allclose(c.point(yy,pp,j),expect,atol=1e-12)
  assert np.array_equal(blend(np.full(6,c.ATOMS[j]),pp,1,j),np.full(6,c.ATOMS[j]))
 # Verify zipped JSON data, row layout and target-only alternatives.
 for name in ['champion','accuracy_guard','best_local','calibration','weighting','only_POR','only_PERM','only_SW']:
  stem='best_reference' if name=='champion' else name;path=ROOT/f'{stem}_all10.json';payload=c.load(path);result={w['logId']:w for w in payload['resultData']};out=np.load(ROOT/f'{name}.npy')
  assert len(result)==10
  for logid in layout['logids']:
   sl=slice(*layout['slices'][logid]);rr=result[logid]['predictions'];arr=np.array([[r[t] for t in c.TARGETS] for r in rr]);assert np.array_equal(arr,out[sl]);cols,_,_=c.r.read_well(c.BASE/'data/test'/f'{logid}.txt',training=False);assert np.array_equal([r['DEPTH'] for r in rr],cols['DEPTH'])
  with zipfile.ZipFile(path.with_suffix('.zip')) as z:assert z.namelist()==['best_reference_all10.json'];assert z.read('best_reference_all10.json')==path.read_bytes()
  if name.startswith('only_'):
   j=c.TARGETS.index(name[5:]);assert np.array_equal(out[:,j],saved[:,j]);assert np.array_equal(out[:,[k for k in range(3) if k!=j]],base[:,[k for k in range(3) if k!=j]])
 scores=c.load(ROOT/'scores.json');ix=np.load(V11/'development_indices.npy');_,y,audit=get_data()
 for name,s in scores.items():
  pp=np.load(ROOT/f'{name}.npy');total=sum(w*c.score(y[:,j],pp[ix,j],j)*100 for j,w in enumerate([.3,.35,.35]));assert abs(total-s['total'])<1e-9
 c.save(ROOT/'integrity_checks.json',dict(status='passed',raw_context_and_model_replay=checks,champion_max_diff=float(np.max(np.abs(p-saved))),official_test_rows=len(p),public_A_rows=len(y),valid_A_counts={t:int(c.valid(y[:,j],j).sum()) for j,t in enumerate(c.TARGETS)},inherited_quantile_input_sha256=cachehash,preprocess_sha256=c.sha(V9/'preprocess.npz'),oof_source_sha256=c.sha(c.MODEL/'work/final80/oof_predictions.npz'),context_sha256=c.sha(c.MODEL/'work/cache_float64/x.npy'),limits='Calibration quantiles inherit v9 fold caches; raw context and new expert models replayed. Not a full re-training of inherited ensemble. Same-agent separate verification script, not external review.'))
 table='\n'.join(f"|{n}|{s['metrics']['POR']['score']:.6f}|{s['metrics']['PERM']['score']:.6f}|{s['metrics']['SW']['score']:.6f}|{s['total']:.6f}|" for n,s in scores.items());vr=c.load(ROOT/'validation_summary.json');vt='\n'.join(f"|{r['target']}|{r['branch']}|{r['gain']:+.6f}|{r['supported']}|" for r in vr);params=json.dumps(selection['champion'],ensure_ascii=False,indent=2);best=scores['champion'];baseline=scores['baseline'];report=f'''# v16：分布概率校准与评分感知权重

仅官方80井标签训练，指定A5官方test井段开发评分。旧v15分数优先冠军保留。最新用户平台反馈83.0445，本轮平台分未知；本地分不可直接换算平台分。

## 五井本地结果

|方案|POR|PERM|SW|总分|
|---|---:|---:|---:|---:|
{table}

冠军相对v15本地变化：{best['total']-baseline['total']:+.6f}。baseline/v15上轮只提高评分但SW误差变差；本轮另外保留accuracy_guard方案，要求当前五井评分提高且MAE、RMSE均不变差。该约束不能保证未知井泛化。

## 方法与对照

1. 分布校准：继承v9的9分位数输入表示，反标准化并还原单位；该表示曾截断标准化值到[-8,8]，不声称恢复极端值的原始精确分位数。对有效连续标签计算每个预测分位数的实际覆盖率，95%经验覆盖与5%名义水平混合。条件组按预测中位数/相对宽度的中位数分为4组；各组向全局收缩50%，支持少于2000点回退。41候选值、81等概率积分点按官方评分选择输出。比较名义分布、全局校准、条件校准，固定边界判断。仅POR/PERM/SW各自独立计算，PERM按log评分。
2. 评分感知权重：相同94维纯曲线上下文、LightGBM中位数回归250轮、深度5、24叶、同种子，比较普通权重与0.5+clip(1-scaled_error/2,0,1)权重。POR/SW相对误差按8%/5%容差缩放；PERM按log10误差。权重来自分井折外教师，不按误差越大权重越大。PERM回归log标签，POR/SW原单位。
3. 组合：先比较单分支全局融合0/0.05/0.15/0.3/0.5/1，再仅对两个获内部支持的最佳分支做小范围组合。无按井、深度、A真值分段输出。所有分支保留冠军原边界值位置，连续条件专家不表示异常删除边界标签；0.1/0.01/99.9仍是有效标签并参与完整评分。

## 内部开发证据

|参数|分支|相对指定内部对照增益（参数分）|正增益支持|
|---|---|---:|---|
{vt}

普通模型对照旧q50；加权模型对照同参数普通模型。权重路线固定16井作开发，其余64井内部2折生成难度权重，再训练普通/加权专家评估16井；新增模型与教师都不读取外层16井标签。全80最终权重由80井内部2折同参数教师生成。连续标签切片不是全任务成绩，也不是原冠军的独立验证。

校准器按5折排除本折标签再评估，但继承提供者存在跨折标签依赖，且旧标准化已利用80井输入，因此校准路线不称严格嵌套独立验证。全局/条件校准的支持条件为校准层折外分超过名义分布；名义分布对照q50。内部数值只用于方向筛选，不可保证迁移到已融合冠军有效。

champion要求内部指定对照正增益、A5总参数分提高、A单井单参数下降不超过0.25；best_local去掉内部支持条件，仅为探索包。各曲线可以保持原模型。accuracy_guard进一步要求当前A5误差不变差。A5被反复用于开发，不能用本结果保证B榜提升；此次未使用其余5井标签，也没有进行测试同源井标签训练。

## 最终分参数方案

```json
{params}
```

comparison.csv汇总参数分与总分，per_well_comparison.csv记录逐井MAE/RMSE/R²，score_transitions.csv分解救回零分、新增零分、已有正分精度变化（参数内分，乘0.30/0.35/0.35才是总分贡献）。trials.json包含全部预设尝试，不只展示获胜项。

## 复现与核验

环境为原.venv-v085；仅新增CPU树模型，本轮不需要安装依赖。运行train.py、run_experiments.py、check_and_report.py。模型通过UTF-8字符串保存重载以兼容中文路径。推理脚本predict.py限定相同官方10井输入，继承冻结冠军与v9折输入缓存，读取新模型和校准器，不读取A真值。它不是脱离旧项目文件的独立部署包。

80训练井/10测试井指纹、模型重载、原曲线上下文重算、95948点候选及冠军逐项重放、评分标量、边界位置、单目标隔离、JSON深度/行数和ZIP内容核验通过，详情integrity_checks.json。分位数部分继承缓存并保存SHA，不声称本轮完整重训原冠军。
''';(ROOT/'REPORT.md').write_text(report,encoding='utf-8');c.save(ROOT/'RUN_STATUS.json',dict(status='complete',baseline_local=baseline['total'],champion_local=best['total'],gain=best['total']-baseline['total'],platform_score=None));print('VERIFIED',best['total'],flush=True)
if __name__=='__main__':main()
