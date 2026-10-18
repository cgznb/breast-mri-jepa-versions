"""Check the existing encoder, decoder, and flow backbone on meta tensors."""

import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, "/home/cxx/ispy2-symmflow3d/src")
from ispy2_symmflow.models.mewm_vqgan import MRIEncoder, MRIDecoder, MeWMVQGANConfig
from ispy2_symmflow.models.velocity import JointVelocityUNet, JointVelocityUNetConfig


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    config = yaml.safe_load(Path("/home/cxx/ispy2-symmflow3d/configs/mewm_all_pairs_5090.yaml").read_text())
    with torch.device("meta"):
        encoder = MRIEncoder(MeWMVQGANConfig())
        decoder = MRIDecoder(MeWMVQGANConfig())
        velocity = JointVelocityUNet(JointVelocityUNetConfig.from_mapping(config["velocity"]))
    results = []
    for shape in [(32, 96, 96), (64, 128, 128), (64, 160, 128), (64, 160, 160), (64, 192, 192), (96, 192, 192), (128, 448, 480)]:
        latent_shape = tuple(v // 4 for v in shape)
        with torch.no_grad():
            encoded = encoder(torch.empty((1, 1, *shape), device="meta"))
            decoded = decoder(torch.empty((1, 64, *latent_shape), device="meta"))
            output = velocity.backbone(x=torch.empty((1, 16, *latent_shape), device="meta"), timesteps=torch.empty(1, device="meta"), context=torch.empty((1, 11, 256), device="meta"))
        assert tuple(encoded.shape) == (1, 64, *latent_shape)
        assert tuple(decoded.shape) == (1, 1, *shape)
        assert tuple(output.shape) == (1, 16, *latent_shape)
        results.append({"image_shape_zyx": shape, "prequant_latent_shape_czyx": (8, *latent_shape), "shape_check": "passed"})
    report = {"method": "meta tensors through existing architecture; no checkpoint loading, image inference, gradient updates, or GPU work", "current_runtime": "codec and data wrappers currently enforce 96x256x256; shape checks do not enable a new training path", "results": results}
    (root / "architecture_shape_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
