#!/usr/bin/env python3
"""Independent larger Monte Carlo check of the inherited specificity contrast.

The reported 99,999-draw analysis is retained. This fixed 999,999-draw check
assesses simulation error of the adverse Attention-Direct Macro specificity
contrast, using an independently seeded closed-form negative-class statistic.
"""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,math
import numpy as np
from analysis_common import load_data,SEEDS
from significance_analysis import holm

ROOT=Path(__file__).resolve().parents[1]
B=999999
RNG=20261104

def main():
    data=load_data();y=data['y'];p=data['ensemble']
    # Integer sum of all seed/landmark specificity differences per negative patient.
    averaging_terms=len(SEEDS)*p.shape[2]
    coefficients=((p[:,1]<.5).astype(int)-(p[:,2]<.5).astype(int)).sum(axis=(0,1))
    coefficients=-coefficients*(1-y)
    observed=float(coefficients.sum()/(averaging_terms*(1-y).sum()))
    rows=list(csv.DictReader((ROOT/'data/significance_tests.csv').open()))
    ix=next(i for i,r in enumerate(rows) if r['comparison']=='ForecastAttention minus DirectReadout' and r['timepoint']=='Macro' and r['metric']=='specificity')
    source=rows[ix];assert abs(observed-float(source['delta']))<1e-12
    rng=np.random.default_rng(RNG);extreme=0
    for first in range(0,B,4096):
        weights=rng.multinomial(175,np.full(175,1/175),size=min(4096,B-first))
        negative=(weights*(1-y)).sum(axis=1);assert (negative>0).all()
        differences=(weights*coefficients).sum(axis=1)/(averaging_terms*negative)
        extreme+=int((np.abs(differences-observed)>=abs(observed)-1e-12).sum())
    pvalue=(1+extreme)/(B+1);se=math.sqrt(pvalue*(1-pvalue)/B)
    # Wilson interval describes Monte Carlo exceedance probability, not effect uncertainty.
    z=1.96;n=B;k=extreme;center=(k/n+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    adjusted={}
    for name,selected in [('secondary_81',[i for i,r in enumerate(rows) if r['family']=='secondary_81']),('all_90',list(range(90)))]:
        values=[pvalue if i==ix else float(rows[i]['p_raw']) for i in selected]
        adjusted[name]=float(holm(values)[selected.index(ix)])
    result=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),training_seeds=SEEDS,comparison=source['comparison'],timepoint='Macro',metric='specificity',delta=observed,
        reason='Retain the same adverse Attention-Direct Macro specificity contrast examined in v4; use an independently seeded larger simulation as a separate numerical stability check, irrespective of its new significance.',
        primary_bootstrap_replicates=int(source['bootstrap_replicates']),primary_p_raw=float(source['p_raw']),primary_p_holm_secondary=float(source['p_holm_family']),primary_p_holm_all_90=float(source['p_holm_all_90']),
        extra_bootstrap_replicates=B,extra_rng_seed=RNG,extra_exceedances=extreme,extra_p_raw=pvalue,extra_p_monte_carlo_se=se,monte_carlo_probability_wilson95=[center-half,center+half],
        adjusted_p_with_only_this_entry_replaced=adjusted,
        method='Same null-centered paired patient bootstrap estimand; exact per-draw specificity difference from integer negative-patient contributions, averaged over three seeds and three landmarks; other 89 p-values held fixed.',
        reporting='Separate simulation-stability sensitivity, not a replacement for the uniformly generated 90-test primary table. All hypotheses/families remain exploratory.',
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),aggregate_only=True)
    (ROOT/'data/monte_carlo_stability.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
