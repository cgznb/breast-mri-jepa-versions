# Publication figure design and provenance

The new method figure is 182 x 167 mm, vector PDF with embedded TrueType fonts plus a 600-dpi PNG. It uses four aligned panels, straight-edged modules, white background, modest semantic accents, uniform 0.65-0.85-point rules/arrows, and explicit training-target paths. Main labels use 7-8.6-point sans-serif text at the declared physical size; the smallest ancillary annotations are 6.5 points. This is a publication-oriented layout, not a claim that MIA prescribes one visual style.

The model colors are NoJEPA #486581, DirectReadout #D58A36, ForecastAttention #247B73, and ForecastMean #8B6CA8. The shared helpers are in `scripts/figure_style.py`.

The real MRI tiles are read directly from already validated, version-matched ROI32 NPZs. They were not regenerated from metadata or combined with new mask geometry. The mask is a predicted T0 localization mask, used only to choose a representative slice. No mask outline, outcome, patient identifier, or individual prediction is displayed or exported. Display windows are phase-specific T0 foreground 99.5th percentiles, held fixed across visits. Source hashes below make the selected example auditable locally without exporting patient names or identifiers. The same slice index is used for every visit and phase. No disease-response claim is inferred from this illustration.

```json
{
  "selection": "First manifest-ordered fold1 training patient with real T0-T3 and nonempty T0 localization; no outcome or prediction used.",
  "display_stages": [
    "T0",
    "T1",
    "T2"
  ],
  "slice_selection": "T0 slice maximizing the predicted T0 localization-mask area; same slice at all three visits.",
  "slice_index_zyx": 14,
  "manifest_sha256": "4330676fff6015a9abec7b4097e8d9a380479042183feda9790162d5447353e8",
  "image_sha256_by_stage": {
    "T0": "2b003ff381cbd60055778d206d74f5d3757b449662a8fada32aa9014ff7590f8",
    "T1": "9f4997eee780a08bf781507faf722ba450d52e7612463f9ceb5a3c10a26e64bc",
    "T2": "27f959b827b26af654b83274daf351a0e4d6bba2cfd30ecddc980b871c8425a0"
  },
  "patient_identifiers_exported": false,
  "display_windows": {
    "Pre": {
      "vmin": 0.0,
      "vmax": 959.4524536132812,
      "scope": "T0 phase foreground 99.5th percentile, fixed across visits"
    },
    "Post1": {
      "vmin": 0.0,
      "vmax": 1087.19580078125,
      "scope": "T0 phase foreground 99.5th percentile, fixed across visits"
    },
    "Late": {
      "vmin": 0.0,
      "vmax": 1600.6829833984375,
      "scope": "T0 phase foreground 99.5th percentile, fixed across visits"
    }
  }
}
```

Rebuild using `python scripts/draw_framework.py`. Python requires NumPy and Matplotlib and access to the original local manifest and NPZs; the generated PDF is standalone for LaTeX/Overleaf.
