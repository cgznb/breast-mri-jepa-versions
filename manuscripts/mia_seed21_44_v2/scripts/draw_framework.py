#!/usr/bin/env python3
"""Draw a print-sized method figure with a real, deidentified development example."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from figure_style import apply_style, COLORS, FULL_WIDTH, MM, TEXT, MUTED, GRID, save_figure

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('/personal_fast_storage/VLA-PCRV1/data/registered_roi32_medium976_v1/cv_manifests/fold1/manifest.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_example():
    manifest = json.loads(MANIFEST.read_text())
    for record in manifest['patients']:
        if record.get('split') != 'train':
            continue
        visits = {v['stage']: v for v in record['visits'] if v and v.get('image')}
        if not set(range(4)).issubset(visits):
            continue
        arrays = {}
        for t in range(3):
            with np.load(visits[t]['image']) as data:
                arrays[t] = {k: data[k].copy() for k in ['image', 'foreground', 'roi']}
        area = arrays[0]['roi'][0].sum(axis=(1, 2))
        if area.max() == 0:
            continue
        z = int(np.argmax(area))
        receipt = {
            'selection': 'First manifest-ordered fold1 training patient with real T0-T3 and nonempty T0 localization; no outcome or prediction used.',
            'display_stages': ['T0', 'T1', 'T2'],
            'slice_selection': 'T0 slice maximizing the predicted T0 localization-mask area; same slice at all three visits.',
            'slice_index_zyx': z,
            'manifest_sha256': digest(MANIFEST),
            'image_sha256_by_stage': {f'T{t}': digest(visits[t]['image']) for t in range(3)},
            'patient_identifiers_exported': False,
            'display_windows': {},
        }
        return arrays, z, receipt
    raise RuntimeError('No complete development example found.')


apply_style()
arrays, z, receipt = select_example()
fig = plt.figure(figsize=(FULL_WIDTH, 167 * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')


def text(x, y, s, size=7.4, ha='center', va='center', color=TEXT, weight='normal', **kwargs):
    return ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, weight=weight, **kwargs)


def title(x, y, letter, s):
    text(x, y, letter, size=10, ha='left', weight='bold')
    text(x + .032, y, s, size=8.6, ha='left', weight='bold')


def box(x, y, w, h, label, color=COLORS['NoJEPA'], size=7.3, fill='white'):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor='#A7B6C1', linewidth=.65))
    ax.plot([x, x + w], [y + h, y + h], color=color, lw=1.5, solid_capstyle='butt')
    text(x + w/2, y + h/2, label, size=size, linespacing=1.3)


def arrow(x1, y1, x2, y2, dashed=False, color=MUTED, lw=.85, connectionstyle='arc3'):
    patch = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7,
                            linewidth=lw, color=color, linestyle=(0, (3, 2)) if dashed else '-',
                            connectionstyle=connectionstyle, shrinkA=1, shrinkB=1)
    ax.add_patch(patch)


title(.025, .982, 'A', 'Observed MRI and continuous visit encoding')
ax.plot([.025, .985], [.663, .663], color=GRID, lw=.65)
xs = [.075, .168, .261]
ys = [.865, .782, .699]
for t in range(3):
    text(xs[t] + .040, .959, f'T{t}', size=7.5, weight='bold')
for phase, rowlabel in enumerate(['Pre', 'Post1', 'Late']):
    text(.061, ys[phase] + .039, rowlabel, size=6.8, ha='right')
    reference = arrays[0]['image'][phase][arrays[0]['foreground'][phase]]
    high = float(np.quantile(reference, .995))
    receipt['display_windows'][rowlabel] = {'vmin': 0., 'vmax': high, 'scope': 'T0 phase foreground 99.5th percentile, fixed across visits'}
    for t in range(3):
        iax = fig.add_axes([xs[t], ys[phase], .080, .078])
        iax.imshow(arrays[t]['image'][phase, z], cmap='gray', vmin=0, vmax=high,
                   origin='upper', interpolation='none')
        iax.set_xticks([]); iax.set_yticks([])
        for spine in iax.spines.values():
            spine.set_visible(False)
text(.205, .681, 'Use only examinations available at landmark k', size=6.7)
box(.412, .792, .155, .126, '3D convolution\nshared by phase\nfine-tuned')
box(.617, .792, .148, .126, 'Phase concatenation\n2 × 4 × 4 pooling\n32 spatial tokens')
box(.817, .792, .163, .126, 'Spatial Transformer\n8 layers; d = 768\nonline encoder $E_\\theta$')
arrow(.354, .851, .408, .851)
arrow(.570, .851, .614, .851)
arrow(.768, .851, .813, .851)
text(.489, .747, 'Continuous features\nno quantizer or decoder', size=7)
text(.693, .747, r'$X_t\in\mathbb{R}^{3\times32\times128\times128}$', size=7.6)
text(.899, .747, r'$Z_t\in\mathbb{R}^{32\times768}$', size=7.6)

title(.025, .641, 'B', 'Multimodal fusion, autonomous forecasting, and training supervision')
box(.035, .505, .193, .108, 'Observed visit tokens\nclinical values + masks\nassigned-arm scenario')
box(.285, .505, .203, .108, 'Fusion Transformer\n8 layers\n8 queries per transition\n4 pCR queries', size=7.1)
box(.548, .505, .228, .108, 'Autonomous latent predictor\n$S_k\\rightarrow\\widehat S_{k+1}\\rightarrow\\cdots\\rightarrow\\widehat S_3$\n8-layer time-causal network', color=COLORS['ForecastAttention'], size=7.3)
box(.831, .505, .149, .108, 'Terminal state\n$\\widehat S_3$\nchange $\\widehat S_3-S_k$', color=COLORS['ForecastAttention'])
arrow(.230, .559, .282, .559)
arrow(.490, .559, .545, .559)
arrow(.779, .559, .828, .559)
box(.035, .357, .193, .082, '', color=COLORS['NoJEPA'], size=7.1)
text(.1315, .415, 'Observed MRI', size=7.1)
text(.1315, .377, 'Future MRI: training only', size=6.6, color=COLORS['DirectReadout'])
box(.285, .357, .203, .082, 'EMA visit encoder $E_{\\bar\\theta}$\n$\\beta=0.999$; stop-gradient', color=COLORS['DirectReadout'], size=7.2)
box(.610, .357, .264, .082, 'Patient-equal masked state L1\n$\\mathcal{L}=\\mathcal{L}_{\\mathrm{BCE}}+0.5\\mathcal{L}_{\\mathrm{roll}}$', color=COLORS['DirectReadout'], size=7.3)
arrow(.231, .415, .282, .415, color=COLORS['NoJEPA'])
arrow(.231, .377, .282, .377, dashed=True, color=COLORS['DirectReadout'])
arrow(.490, .398, .607, .398, dashed=True, color=COLORS['DirectReadout'])
text(.548, .418, 'Future targets', size=6.6, color=COLORS['DirectReadout'])
arrow(.690, .502, .690, .442, dashed=True, color=COLORS['DirectReadout'])
arrow(.488, .438, .567, .501, color=COLORS['ForecastAttention'])
text(.504, .476, 'Current state', size=6.5, ha='right', color=COLORS['ForecastAttention'])
text(.388, .345, r'Online weights update EMA after each step', size=6.6, color=MUTED)
ax.plot([.025, .985], [.325, .325], color=GRID, lw=.65)

title(.025, .302, 'C', 'Forecast-attention readout')
text(.287, .262, r'$\alpha=\operatorname{softmax}\!\left(w^{\mathsf{T}}\operatorname{LN}(\widehat S_3)\right)$', size=8.6)
box(.041, .164, .239, .063, r'$v_k=\sum_n\alpha_n\widehat S_{3,n}$', color=COLORS['ForecastAttention'], size=8.3)
box(.318, .164, .253, .063, r'$d_k=\sum_n\alpha_n(\widehat S_{3,n}-S_{k,n})$', color=COLORS['ForecastAttention'], size=8)
box(.041, .056, .110, .065, 'Direct\n$q_k$', size=7.4)
box(.210, .056, .178, .065, '$[q_k;v_k;d_k]$\nconcatenation', color=COLORS['ForecastAttention'], size=7.2)
box(.435, .056, .109, .065, 'LN + linear\nsigmoid', color=COLORS['ForecastAttention'], size=6.9)
arrow(.153, .089, .207, .089)
arrow(.160, .161, .261, .124)
arrow(.445, .161, .340, .124)
arrow(.391, .089, .432, .089)
arrow(.547, .089, .583, .089)
text(.584, .106, '$p_k$', size=8, ha='left')
text(.298, .031, 'One set of weights pools terminal state and change.', size=6.7, color=MUTED)

title(.624, .302, 'D', 'Matched ablation design')
x_left, x_right = .624, .985
ax.plot([x_left, x_right], [.278, .278], color=MUTED, lw=.65)
text(.633, .257, 'Model', size=7, ha='left', weight='bold')
text(.811, .257, r'$\lambda_{\rm roll}$', size=7.8)
text(.884, .257, 'Readout', size=6.9, weight='bold')
text(.957, .257, 'Pool', size=6.9, weight='bold')
ax.plot([x_left, x_right], [.239, .239], color=GRID, lw=.6)
rows = [('NoJEPA','0','Direct','—'),('DirectReadout','0.5','Direct','—'),
        ('ForecastAttention','0.5','Forecast','Attention'),('ForecastMean','0.5','Forecast','Mean')]
for i,(model,weight,readout,pooling) in enumerate(rows):
    y = .214 - i * .047
    ax.plot([.625,.628], [y-.013,y+.013], color=COLORS[model], lw=2)
    text(.635, y, model, size=6.75, ha='left', color=COLORS[model], weight='bold')
    text(.811, y, weight, size=7)
    text(.884, y, readout, size=6.8)
    text(.957, y, pooling, size=6.7)
ax.plot([x_left, x_right], [.096, .096], color=GRID, lw=.6, linestyle=(0,(2,2)))
ax.plot([x_left, x_right], [.048, .048], color=MUTED, lw=.65)
text(.807, .028, 'ForecastMean: supplementary pooling control', size=6.5, color=MUTED)

save_figure(fig, ROOT/'figures'/'framework', png=True, dpi=600, tight=False)
plt.close(fig)

caption = r'''\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/framework.pdf}
\caption{Forecast-aware latent modeling and matched controls. (A) Real three-phase MRI crops from one development patient illustrate the same baseline-defined plane at T0, T1, and T2; the example and slice were selected without outcomes or model predictions. Only visits available at the prediction landmark are inputs. A shared continuous encoder and spatial Transformer produce visit tokens. (B) Clinical values, availability masks, and an assigned-arm scenario condition fusion and autonomous prediction to T3. The EMA encoder supplies the detached current state; available future MRI supplies training targets only. Dashed orange arrows denote training-target or loss paths. (C) Attention weights pool the predicted terminal state and its change; these features are combined with direct pCR-query features. (D) The primary variants separate removal of future-state supervision from the complete forecast readout. ForecastMean is a supplementary pooling control. MRI tiles are illustrative crops, not generated images, segmentation ground truth, or attention maps.}
\label{fig:framework}
\end{figure}
'''
(ROOT/'figures'/'framework.tex').write_text(caption)
note = '''# Publication figure design and provenance

The new method figure is 182 x 167 mm, vector PDF with embedded TrueType fonts plus a 600-dpi PNG. It uses four aligned panels, straight-edged modules, white background, modest semantic accents, uniform 0.65-0.85-point rules/arrows, and explicit training-target paths. Main labels use 7-8.6-point sans-serif text at the declared physical size; the smallest ancillary annotations are 6.5 points. This is a publication-oriented layout, not a claim that MIA prescribes one visual style.

The model colors are NoJEPA #486581, DirectReadout #D58A36, ForecastAttention #247B73, and ForecastMean #8B6CA8. The shared helpers are in `scripts/figure_style.py`.

The real MRI tiles are read directly from already validated, version-matched ROI32 NPZs. They were not regenerated from metadata or combined with new mask geometry. The mask is a predicted T0 localization mask, used only to choose a representative slice. No mask outline, outcome, patient identifier, or individual prediction is displayed or exported. Display windows are phase-specific T0 foreground 99.5th percentiles, held fixed across visits. Source hashes below make the selected example auditable locally without exporting patient names or identifiers. The same slice index is used for every visit and phase. No disease-response claim is inferred from this illustration.

```json
'''+json.dumps(receipt,indent=2)+'''\n```

Rebuild using `python scripts/draw_framework.py`. Python requires NumPy and Matplotlib and access to the original local manifest and NPZs; the generated PDF is standalone for LaTeX/Overleaf.
'''
(ROOT/'notes'/'figure_design.md').write_text(note)
print(json.dumps({'figure':'framework','width_mm':182,'height_mm':167,'panels':4,'patient_identifiers_exported':False,'font_min_pt':6.5}))
