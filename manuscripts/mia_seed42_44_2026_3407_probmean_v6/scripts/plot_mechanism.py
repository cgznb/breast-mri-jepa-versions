#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from figure_style import FULL_WIDTH,MM,COLORS,panel_label,save_figure
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'data/subgroup_world_summary.csv').open()))
fig,axs=plt.subplots(1,3,figsize=(FULL_WIDTH,72*MM))
for ax,l in zip(axs,'ABC'):panel_label(ax,l)
models=['DirectReadout','ForecastAttention']
ax=axs[0];ax.axhline(0,color='#89949B',ls=':',lw=.8)
for mi,m in enumerate(models):
 r=[next(r for r in rows if r['model']==m and r['epoch']=='20' and r['horizon']==f'horizon_{h}') for h in [1,2,3]]
 vals=[float(x['skill_vs_copy_mean']) for x in r]
 sd=[float(x['skill_vs_copy_seed_sd']) for x in r]
 ax.bar(np.arange(3)+(mi-.5)*.32,vals,width=.30,color=COLORS[m],yerr=sd,capsize=2,error_kw={'lw':.7})
ax.set_xticks(range(3),['1 step','2 steps','3 steps']);ax.set_ylabel('Skill relative to copy');ax.set_title('Forecast skill (epoch 20)');ax.grid(axis='y',alpha=.6)
ax=axs[1]
for mi,m in enumerate(models):
 r=next(r for r in rows if r['model']==m and r['epoch']=='20')
 vals=[float(r['task_predictor_gradient_mean']),float(r['rollout_predictor_gradient_mean'])]
 sd=[float(r['task_predictor_gradient_seed_sd']),float(r['rollout_predictor_gradient_seed_sd'])]
 ax.bar(np.arange(2)+(mi-.5)*.32,vals,width=.30,color=COLORS[m],yerr=sd,capsize=2,error_kw={'lw':.7})
ax.set_xticks(range(2),['pCR loss','Rollout loss']);ax.set_ylabel('Predictor gradient norm');ax.set_title('Gradients (epoch 20)');ax.grid(axis='y',alpha=.6)
ax=axs[2];ax.axhline(0,color='#89949B',ls=':',lw=.8)
for m in models:
 r=sorted([r for r in rows if r['model']==m and r['horizon']=='horizon_3'],key=lambda r:int(r['epoch']))
 xx=np.array([int(x['epoch']) for x in r]); yy=np.array([float(x['skill_vs_copy_mean']) for x in r]);sd=np.array([float(x['skill_vs_copy_seed_sd']) for x in r])
 ax.plot(xx,yy,color=COLORS[m],marker='o',markersize=2.5,label=m);ax.fill_between(xx,yy-sd,yy+sd,color=COLORS[m],alpha=.10,lw=0)
ax.set(xlabel='Training epoch',ylabel='Three-step skill relative to copy');ax.set_title('Three-step trajectory');ax.grid(alpha=.6)
fig.legend(*axs[2].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2)
fig.subplots_adjust(left=.08,right=.99,bottom=.21,top=.79,wspace=.48)
save_figure(fig,ROOT/'figures/forecast_mechanism',tight=False)
print('Saved four-seed training-probe means and seed SD; no holdout forecast claims.')
