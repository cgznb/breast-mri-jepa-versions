#!/usr/bin/env python3
"""Publication figures average separately evaluated seed44 fold models."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure_style import FULL_WIDTH,MM,COLORS,panel_label,save_figure
from analysis_common import SEEDS,FOLDS,load_data
SEED_TAG=",".join(map(str,SEEDS))

ROOT=Path(__file__).resolve().parents[1]
LABELS=['NoJEPA','DirectReadout','ForecastAttention']
TIMES=['T0','T1','T2']
B=2000
RNG=20261103

def read(name):return list(csv.DictReader((ROOT/'data'/name).open()))
def write(name,rows):
    with (ROOT/'data'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[r for r in read('extended_summary.csv') if r['seed']==SEED_TAG and r['policy']=='fixed_0.5']
summ={(r['model'],r['timepoint']):r for r in summary}
data=load_data()
y=data['y'];scores=data['scores'][:,:3]
assert scores.shape==(len(FOLDS),3,3,175) and y.sum()==56
weights=np.random.default_rng(RNG).multinomial(175,np.full(175,1/175),size=B)

def legend(fig,y=1.01):
    fig.legend([Line2D([0],[0],color=COLORS[m],lw=1.7) for m in LABELS],LABELS,ncol=3,loc='upper center',bbox_to_anchor=(.52,y),frameon=False,columnspacing=2)

def curves(y,p,w,grid):
    ix=np.argsort(-p,kind='stable');pp=p[ix];yy=y[ix];ww=w[:,ix]
    ends=np.r_[np.flatnonzero(np.diff(pp)),len(p)-1]
    tp=np.cumsum(ww*yy,axis=1)[:,ends];fp=np.cumsum(ww*(1-yy),axis=1)[:,ends]
    recall=tp/(w*y).sum(axis=1)[:,None];fpr=fp/(w*(1-y)).sum(axis=1)[:,None]
    precision=np.divide(tp,tp+fp,out=np.ones_like(tp,dtype=float),where=tp+fp>0)
    roc=np.array([np.interp(grid,np.r_[0,f],np.r_[0,r]) for f,r in zip(fpr,recall)])
    pr=np.array([q[np.minimum(np.searchsorted(r,grid,side='left'),len(r)-1)] for r,q in zip(recall,precision)])
    roc[:,-1]=1
    return roc,pr

# Plot mean fold-specific curves; never average probabilities across folds.
# Shared patient draws first recompute each fold curve, then average coordinates.
grid=np.linspace(0,1,101)
curve_boot=np.zeros((2,B,3,3,len(grid)));curve_point=np.zeros((2,3,3,len(grid)))
curve_export=[]
for mi in range(3):
    for ti in range(3):
        fold_points=[]
        for si in range(len(FOLDS)):
            a,b=curves(y,scores[si,mi,ti],weights,grid)
            aa,bb=curves(y,scores[si,mi,ti],np.ones((1,175)),grid)
            curve_boot[0,:,mi,ti]+=a/len(FOLDS);curve_boot[1,:,mi,ti]+=b/len(FOLDS)
            fold_points.append([aa[0],bb[0]])
        curve_point[:,mi,ti]=np.mean(fold_points,axis=0)
fig,axs=plt.subplots(2,3,figsize=(FULL_WIDTH,115*MM))
for row in range(2):
    for ti,ax in enumerate(axs[row]):
        panel_label(ax,chr(65+row*3+ti));ax.set_title(TIMES[ti])
        for mi,m in enumerate(LABELS):
            lo,hi=np.quantile(curve_boot[row,:,mi,ti],[.025,.975],axis=0)
            ax.fill_between(grid,lo,hi,color=COLORS[m],alpha=.10,lw=0)
            ax.plot(grid,curve_point[row,mi,ti],color=COLORS[m])
            score=float(summ[m,TIMES[ti]][['auroc','auprc'][row]])
            ax.text(.97,.22-mi*.065 if row==0 else .91-mi*.065,f'{score:.3f}',transform=ax.transAxes,ha='right',color=COLORS[m],fontsize=7)
            curve_export.extend(dict(model=m,seed=44,fold="mean",timepoint=TIMES[ti],curve=['ROC','PR'][row],grid=float(g),point=float(v),ci_low=float(l),ci_high=float(h)) for g,v,l,h in zip(grid,curve_point[row,mi,ti],lo,hi))
        if row==0:ax.plot([0,1],[0,1],ls=':',lw=.8,color='#89949B');ax.set_xlabel('1 − specificity');ax.set_ylabel('Sensitivity')
        else:ax.axhline(.32,ls=':',lw=.8,color='#89949B');ax.set_xlabel('Recall');ax.set_ylabel('Precision')
        ax.set(xlim=(0,1),ylim=(0,1));ax.grid(alpha=.65)
legend(fig,1.01);fig.subplots_adjust(left=.075,right=.985,bottom=.105,top=.89,hspace=.56,wspace=.38)
save_figure(fig,ROOT/'figures/discrimination',tight=False);plt.close(fig)
write('discrimination_curve_bands.csv',curve_export)

# Reliability groups contain 35 scores predictions each.
calibration=[]
fig,axs=plt.subplots(2,3,figsize=(FULL_WIDTH,116*MM))
for ti in range(3):
    ax=axs[0,ti];panel_label(ax,chr(65+ti));ax.set_title(TIMES[ti]);ax.plot([0,1],[0,1],ls=':',color='#89949B',lw=.8)
    for mi,m in enumerate(LABELS):
        fold_x=[];fold_y=[]
        for si,fold in enumerate(FOLDS):
            p=scores[si,mi,ti];groups=np.array_split(np.argsort(p,kind='stable'),5)
            sx=np.array([p[g].mean() for g in groups]);sy=np.array([y[g].mean() for g in groups])
            fold_x.append(sx);fold_y.append(sy)
            for bi in range(5):
                calibration.append(dict(model=m,seed=44,fold=fold,timepoint=TIMES[ti],probability_rank_group=bi+1,
                    mean_predicted=float(sx[bi]),mean_observed=float(sy[bi]),patients=35,
                    role='fold_specific_equal_count_bin'))
        xx=np.mean(fold_x,axis=0);yy=np.mean(fold_y,axis=0)
        ax.plot(xx,yy,'o-',color=COLORS[m],markersize=2.8)
        for bi in range(5):
            calibration.append(dict(model=m,seed=44,fold="mean",timepoint=TIMES[ti],probability_rank_group=bi+1,
                mean_predicted=float(xx[bi]),mean_observed=float(yy[bi]),patients=35,
                role='mean_of_five_fold_specific_bins; membership can differ across folds'))
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted pCR probability',ylabel='Observed pCR proportion');ax.grid(alpha=.6)
    ax=axs[1,ti];panel_label(ax,chr(68+ti));ax.set_title(TIMES[ti])
    dca=[r for r in read('extended_dca.csv') if r['seed']==SEED_TAG and r['timepoint']==TIMES[ti]]
    for m in LABELS:
        rr=sorted([r for r in dca if r['model']==m],key=lambda r:float(r['threshold']))
        xx=np.array([float(r['threshold']) for r in rr]);val=np.array([float(r['net_benefit']) for r in rr])
        ax.plot(xx,val,color=COLORS[m])
    ax.plot(xx,[float(r['treat_all']) for r in rr],ls='--',lw=.85,color='#89949B',label='Treat all')
    ax.axhline(0,ls=':',lw=.9,color='#444444',label='Treat none')
    ax.set(xlim=(.05,.80),ylim=(-.12,.32),xlabel='Hypothetical risk threshold',ylabel='Net benefit');ax.grid(alpha=.6)
axs[1,0].legend(loc='lower left',fontsize=6)
legend(fig,1.01);fig.subplots_adjust(left=.085,right=.985,bottom=.11,top=.89,hspace=.63,wspace=.42)
save_figure(fig,ROOT/'figures/calibration_utility',tight=False);plt.close(fig)
write('reliability_rank_bins.csv',calibration)

contrast=[r for r in read('extended_paired_bootstrap.csv') if r['seed']==SEED_TAG and r['policy']=='fixed_0.5']
comparisons=[('DirectReadout minus NoJEPA','Auxiliary supervision',COLORS['DirectReadout']),('ForecastAttention minus DirectReadout','Forecast readout',COLORS['ForecastAttention']),('ForecastAttention minus NoJEPA','Complete model',COLORS['NoJEPA'])]
fig,axs=plt.subplots(1,3,figsize=(FULL_WIDTH,99*MM))
for ki,(key,title) in enumerate([('auroc','Δ AUROC'),('auprc','Δ average precision'),('f1','Δ F1')]):
    ax=axs[ki];panel_label(ax,chr(65+ki));ax.set_title(title);ax.axvline(0,color='#89949B',ls=':',lw=.9)
    ticks=[];ticklabels=[]
    for ci,(comp,label,color) in enumerate(comparisons):
        for ti,t in enumerate(['T0','T1','T2','Macro']):
            yy=ci*5+ti;r=next(r for r in contrast if r['comparison']==comp and r['timepoint']==t and r['metric']==key)
            val,lo,hi=[float(r[k]) for k in ['delta','ci_low','ci_high']]
            ax.errorbar(val,yy,xerr=[[val-lo],[hi-val]],fmt='s' if t=='Macro' else 'o',markersize=3.5,color=color,capsize=2,lw=1)
            ticks.append(yy);ticklabels.append(t)
        ax.axhline(ci*5+4,color='#E3E9ED',lw=.6)
    ax.set_yticks(ticks,ticklabels if ki==0 else []);ax.invert_yaxis();ax.grid(axis='x',alpha=.6)
    ax.set_xlabel('Difference (95% interval)')
handles=[Line2D([0],[0],marker='o',color=c,label=l,lw=1) for _,l,c in comparisons]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1),ncol=3,fontsize=7)
fig.subplots_adjust(left=.08,right=.985,bottom=.14,top=.87,wspace=.25)
save_figure(fig,ROOT/'figures/paired_comparisons',tight=False);plt.close(fig)

metrics=[('auroc','AUROC'),('auprc','Average precision'),('f1','F1 at 0.5'),('sensitivity','Sensitivity at 0.5'),('specificity','Specificity at 0.5'),('brier','Brier score ↓')]
fig,axs=plt.subplots(2,3,figsize=(FULL_WIDTH,108*MM))
for ax,(key,title),letter in zip(axs.flat,metrics,'ABCDEF'):
    panel_label(ax,letter);ax.set_title(title)
    for m in LABELS:
        vals=np.array([float(summ[m,t][key]) for t in TIMES]);lo=np.array([float(summ[m,t][key+'_patient_ci_low']) for t in TIMES]);hi=np.array([float(summ[m,t][key+'_patient_ci_high']) for t in TIMES])
        sd=np.array([float(summ[m,t][key+'_fold_sd']) for t in TIMES])
        ax.errorbar(range(3),vals,yerr=sd,fmt='o-',color=COLORS[m],capsize=2,lw=1);ax.fill_between(range(3),lo,hi,color=COLORS[m],alpha=.08,lw=0)
    ax.set_xticks(range(3),TIMES);ax.grid(alpha=.6)
legend(fig,1.01);fig.subplots_adjust(left=.085,right=.985,bottom=.09,top=.89,hspace=.58,wspace=.38)
save_figure(fig,ROOT/'figures/landmark_performance',tight=False);plt.close(fig)

subgroups=[r for r in read('subgroup_paired_differences.csv') if r['seed']==SEED_TAG and r['timepoint']=='Macro' and r['family'] in ['HR/HER2','Age']]
fig,axs=plt.subplots(1,2,figsize=(FULL_WIDTH,82*MM),gridspec_kw={'width_ratios':[1.25,1]})
ax=axs[0];panel_label(ax,'A');ax.axvline(0,color='#89949B',ls=':',lw=.8)
for i,r in enumerate(subgroups):
    val,lo,hi=[float(r[k]) for k in ['delta_auroc','ci_low','ci_high']]
    ax.errorbar(val,i,xerr=[[val-lo],[hi-val]],fmt='o',color=COLORS['ForecastAttention'],capsize=2,ms=3.5)
ax.set_yticks(range(len(subgroups)),[r['subgroup']+f" (n={r['n']}, pCR+={r['positives']})" for r in subgroups],fontsize=6.5)
ax.invert_yaxis();ax.set_xlabel('Δ macro AUROC: ForecastAttention − NoJEPA');ax.grid(axis='x',alpha=.6)
ax=axs[1];panel_label(ax,'B');ax.set_title('Operating balance')
xx=[];yy=[]
for m in LABELS:
    for ti,t in enumerate(TIMES):
        r=summ[m,t];x=float(r['specificity']);v=float(r['sensitivity']);xx.append(x);yy.append(v)
        ax.plot(x,v,marker=['o','s','^'][ti],color=COLORS[m],ms=4.5,ls='')
ax.set(xlabel='Specificity at 0.5',ylabel='Sensitivity at 0.5',xlim=(min(xx)-.025,max(xx)+.025),ylim=(min(yy)-.035,max(yy)+.035));ax.grid(alpha=.6)
ax.legend(handles=[Line2D([0],[0],ls='',marker=mark,color='#546674',label=t) for mark,t in zip(['o','s','^'],TIMES)],loc='lower left',fontsize=6.5)
legend(fig,1.015);fig.subplots_adjust(left=.28,right=.975,bottom=.19,top=.85,wspace=.70)
save_figure(fig,ROOT/'figures/subgroup_operating',tight=False);plt.close(fig)

# Average fold-specific histograms of T2-T0 scores. Every fold has 175 paired
# patients; repeated fold predictions do not create independent patients.
fig,axs=plt.subplots(2,3,figsize=(FULL_WIDTH,105*MM),sharey=True)
score_limit=max(.04,np.ceil(np.abs(scores[:,:,2]-scores[:,:,0]).max()*100)/100)
bins=np.linspace(-score_limit,score_limit,22);trajectory=[];histogram=[]
for outcome in [0,1]:
    for mi,m in enumerate(LABELS):
        ax=axs[outcome,mi];panel_label(ax,chr(65+outcome*3+mi));ax.set_title(m if outcome==0 else '')
        selected=y==outcome;fold_hist=[];fold_means=[];fold_increasing=[]
        for si,fold in enumerate(FOLDS):
            delta=(scores[si,mi,2]-scores[si,mi,0])[selected]
            counts=np.histogram(delta,bins=bins)[0];assert counts.sum()==selected.sum()
            hist=counts/selected.sum();fold_hist.append(hist)
            fold_means.append(float(delta.mean()));fold_increasing.append(float((delta>0).mean()))
            trajectory.append(dict(model=m,seed=44,fold=fold,outcome=outcome,n=int(selected.sum()),
                mean_change=float(delta.mean()),fraction_increasing=float((delta>0).mean())))
            histogram.extend(dict(model=m,seed=44,fold=fold,outcome=outcome,n=int(selected.sum()),
                bin_left=float(bins[i]),bin_right=float(bins[i+1]),count=float(count),proportion=float(hist[i])) for i,count in enumerate(counts))
        hist=np.mean(fold_hist,axis=0)
        assert np.isclose(hist.sum(),1)
        trajectory.append(dict(model=m,seed=44,fold="mean",outcome=outcome,n=int(selected.sum()),
            mean_change=float(np.mean(fold_means)),fraction_increasing=float(np.mean(fold_increasing))))
        histogram.extend(dict(model=m,seed=44,fold="mean",outcome=outcome,n=int(selected.sum()),
            bin_left=float(bins[i]),bin_right=float(bins[i+1]),count=float(hist[i]*selected.sum()),proportion=float(hist[i])) for i in range(len(hist)))
        ax.bar((bins[1:]+bins[:-1])/2,hist,width=np.diff(bins)*.88,color=COLORS[m],alpha=.85)
        ax.axvline(0,color='#89949B',ls=':',lw=.8);ax.set_xlabel('pCR probability: T2 − T0');ax.set_ylabel('Mean fold-specific proportion');ax.grid(axis='y',alpha=.6)
        ax.text(.03,.93,('Non-pCR (n=119)' if outcome==0 else 'pCR (n=56)'),transform=ax.transAxes,fontsize=6.5,va='top')
fig.subplots_adjust(left=.085,right=.985,bottom=.11,top=.9,hspace=.58,wspace=.4)
save_figure(fig,ROOT/'figures/outcome_score_changes',tight=False);plt.close(fig)
write('outcome_score_changes.csv',trajectory);write('outcome_score_histograms.csv',histogram)
print('Created six plates from separately evaluated seed44 fold models; average metrics, curve coordinates and histogram proportions only.')
