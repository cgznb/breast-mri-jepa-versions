# VQ continuous encoder exception: CC BY-NC 4.0

`src/mri_vla_jepa/vq_encoder.py` adapts the convolution, replication-padding,
residual and GroupNorm encoder architecture and its checkpoint parameter names
from the following sources:

- MeWM, https://github.com/scott-yjyang/MeWM
- cgznb/symm-fm, https://github.com/cgznb/symm-fm
- `workflows/first_post_three_phase/mewm_ispy2/vqgan.py`
- The local adaptations in `MAM/symm-fm-trjepa-v2/src/trjepa/codec.py` and
  `MAM/symm-fm-world-v2/src/symm_world/codec.py`.

The upstream workflow identifies the codec material as derived from MeWM under
Creative Commons Attribution-NonCommercial 4.0 International. Attribution to the
original MeWM and cgznb/symm-fm contributors is retained. This derived front end
is not relicensed by the MIT license covering the original project additions.

License: https://creativecommons.org/licenses/by-nc/4.0/legalcode

Changes in this adaptation: retain only `encoder` and `pre_quant`; preserve
parameter names and continuous numerical inference; omit the discrete
quantizer, codebook, decoder, Lightning trainer and GAN discriminator; expose
gradient-enabled continuous encoding and weights-only checkpoint initialization.
The new visit adapter shares this single-channel front end across three phases,
then pools and projects their concatenated continuous features into VLA tokens.

Source code attribution does not grant rights to medical datasets or external
pretrained weights. Those remain external and their terms must be considered
separately. Commercial use of the derived front end requires rights beyond
those granted by CC BY-NC 4.0. No original author endorses these additions.
