import os
import json

import torch
import matplotlib.pyplot as plt
from torch import nn
from generative.networks.nets import DiffusionModelUNet, ControlNet
from flow_matching.solver import ODESolver

from .general_utils import class_name_from_map, normalize_zero_to_one, save_image, save_image_3d
from .motfm_logging import get_logger
from tqdm import tqdm

logger = get_logger(__name__)


###############################################################################
# Model Building
###############################################################################
class MergedModel(nn.Module):
    """
    Merged model that wraps a UNet and an optional ControlNet.
    Takes in x, time in [0,1], and (optionally) a ControlNet condition.
    """

    def __init__(
        self,
        unet: DiffusionModelUNet,
        controlnet: ControlNet = None,
        max_timestep=1000,
        *,
        spatial_dims=None,
        in_channels=None,
        out_channels=None,
        spatial_condition_channels=None,
    ):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.max_timestep = max_timestep
        self.has_controlnet = controlnet is not None
        self.spatial_dims = spatial_dims or getattr(unet, "spatial_dims", None)
        self.in_channels = in_channels or getattr(unet, "in_channels", None)
        self.out_channels = out_channels or getattr(unet, "out_channels", None)
        self.cross_attention_dim = getattr(unet, "cross_attention_dim", None)
        self.spatial_condition_channels = spatial_condition_channels
        if self.spatial_condition_channels is None and controlnet is not None:
            self.spatial_condition_channels = getattr(
                controlnet, "conditioning_embedding_in_channels", None
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        spatial_condition: torch.Tensor = None,
        masks: torch.Tensor = None,
    ):
        """
        Args:
            x: input image tensor [B, C, H, W].
            t: timesteps in [0,1], will be scaled to [0, max_timestep - 1].
            cond: [B,1 , conditions_dim].
            spatial_condition: [B, C, *spatial] ControlNet input.
            masks: Compatibility alias for ``spatial_condition``.

        Returns:
            The network output (e.g. velocity, noise, or predicted epsilon).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a tensor")
        if x.ndim < 3 or x.shape[0] <= 0:
            raise ValueError("x must have shape [B, C, *spatial]")
        if self.spatial_dims is not None and x.ndim != int(self.spatial_dims) + 2:
            raise ValueError("x has the wrong number of spatial dimensions")
        if self.in_channels is not None and x.shape[1] != int(self.in_channels):
            raise ValueError("x has the wrong number of channels")
        if not isinstance(t, torch.Tensor):
            raise TypeError("t must be a tensor")
        if not torch.isfinite(t).all() or torch.any((t < 0) | (t > 1)):
            raise ValueError("t must be finite and in [0, 1]")
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must be scalar or have shape [B]")

        # MONAI uses a discrete timestep embedding internally.
        t = t * (self.max_timestep - 1)
        t = t.floor().long()

        if cond is not None:
            if not isinstance(cond, torch.Tensor):
                raise TypeError("cond must be a tensor")
            if cond.ndim == 2:
                cond = cond.unsqueeze(1)
            if cond.ndim != 3:
                raise ValueError("cond must have shape [B, sequence, features]")
            if cond.shape[0] != x.shape[0]:
                raise ValueError("cond batch size must match x")
            if (
                self.cross_attention_dim is not None
                and cond.shape[-1] != int(self.cross_attention_dim)
            ):
                raise ValueError("cond has the wrong feature dimension")

        if spatial_condition is not None and masks is not None:
            raise ValueError("spatial_condition and masks cannot both be provided")
        control_condition = spatial_condition if spatial_condition is not None else masks

        if self.has_controlnet:
            if control_condition is None:
                raise ValueError(
                    "spatial conditioning is enabled but no spatial condition was provided"
                )
            if not isinstance(control_condition, torch.Tensor):
                raise TypeError("spatial condition must be a tensor")
            if control_condition.ndim != x.ndim:
                raise ValueError("spatial condition has the wrong number of spatial dimensions")
            if control_condition.shape[0] != x.shape[0]:
                raise ValueError("spatial condition batch size must match x")
            if control_condition.shape[2:] != x.shape[2:]:
                raise ValueError("spatial condition spatial shape must match x")
            if (
                self.spatial_condition_channels is not None
                and control_condition.shape[1] != int(self.spatial_condition_channels)
            ):
                raise ValueError("spatial condition has the wrong number of channels")
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                x=x, timesteps=t, controlnet_cond=control_condition, context=cond
            )
            output = self.unet(
                x=x,
                timesteps=t,
                context=cond,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
        else:
            output = self.unet(x=x, timesteps=t, context=cond)

        if not isinstance(output, torch.Tensor):
            raise ValueError("the UNet output must be a tensor")
        if output.shape[0] != x.shape[0] or output.shape[2:] != x.shape[2:]:
            raise ValueError("the UNet output batch and spatial shape must match x")
        if self.out_channels is not None and output.shape[1] != int(self.out_channels):
            raise ValueError("the UNet output has the wrong number of channels")
        return output


def _conditioning_enabled(model_config: dict) -> bool:
    has_canonical = "spatial_conditioning" in model_config
    has_legacy = "mask_conditioning" in model_config
    canonical = model_config.pop("spatial_conditioning", None)
    legacy = model_config.pop("mask_conditioning", None)
    if has_canonical and type(canonical) is not bool:
        raise TypeError("spatial_conditioning must be a boolean")
    if has_legacy and type(legacy) is not bool:
        raise TypeError("mask_conditioning must be a boolean")
    if has_canonical and has_legacy and canonical != legacy:
        raise ValueError("spatial_conditioning and mask_conditioning disagree")
    if has_canonical:
        return canonical
    if has_legacy:
        return legacy
    return False


def _runtime_conditioning_enabled(
    spatial_conditioning, mask_conditioning, *, default: bool
) -> bool:
    if spatial_conditioning is not None and type(spatial_conditioning) is not bool:
        raise TypeError("spatial_conditioning must be a boolean")
    if mask_conditioning is not None and type(mask_conditioning) is not bool:
        raise TypeError("mask_conditioning must be a boolean")
    if (
        spatial_conditioning is not None
        and mask_conditioning is not None
        and spatial_conditioning != mask_conditioning
    ):
        raise ValueError("spatial_conditioning and mask_conditioning disagree")
    if spatial_conditioning is not None:
        return spatial_conditioning
    if mask_conditioning is not None:
        return mask_conditioning
    return default


def _batch_spatial_condition(batch: dict, *, enabled: bool):
    if not enabled:
        return None, None
    has_canonical = "spatial_condition" in batch
    has_legacy = "masks" in batch
    if has_canonical and has_legacy:
        raise ValueError("batch cannot provide both spatial_condition and masks")
    if has_canonical:
        return batch["spatial_condition"], "spatial_condition"
    if has_legacy:
        return batch["masks"], "masks"
    raise KeyError(
        "spatial conditioning is enabled but the batch has no 'spatial_condition' "
        "or legacy 'masks' key"
    )


def build_model(model_config: dict, device: torch.device = None) -> MergedModel:
    """
    Builds a model (UNet only, or UNet+ControlNet) based on the provided model_config.

    Args:
        model_config: Dictionary containing model configuration.
        device: Device to move the model to.

    Returns:
        A MergedModel instance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make a copy so the original config remains unaltered.
    mc = model_config.copy()

    # Resolve public compatibility flags before calling either MONAI constructor.
    spatial_conditioning = _conditioning_enabled(mc)
    max_timestep = mc.pop("max_timestep", 1000)
    cond_embed_in_channels = mc.pop("conditioning_embedding_in_channels", None)
    cond_embed_channels = mc.pop("conditioning_embedding_num_channels", None)

    # Build the base UNet by passing all remaining items as kwargs.
    unet = DiffusionModelUNet(**mc)

    controlnet = None
    if spatial_conditioning:
        controlnet_config = mc.copy()
        controlnet_config.pop("out_channels", None)
        if cond_embed_in_channels is None:
            cond_embed_in_channels = 1
        if cond_embed_channels is None:
            cond_embed_channels = (16,)
        controlnet = ControlNet(
            **controlnet_config,
            conditioning_embedding_in_channels=cond_embed_in_channels,
            conditioning_embedding_num_channels=cond_embed_channels,
        )
        controlnet.load_state_dict(unet.state_dict(), strict=False)

    model = MergedModel(
        unet=unet,
        controlnet=controlnet,
        max_timestep=max_timestep,
        spatial_dims=mc.get("spatial_dims"),
        in_channels=mc.get("in_channels"),
        out_channels=mc.get("out_channels"),
        spatial_condition_channels=cond_embed_in_channels,
    )

    # Print number of trainable parameters.
    logger.info(
        f"Building model with spatial_conditioning={spatial_conditioning} and "
        f"max_timestep={max_timestep}."
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params} trainable parameters.")
    model_size_mb = num_params * 4 / (1024**2)
    logger.info(f"Model size: {model_size_mb:.2f} MB")

    return model.to(device)


def sample_with_solver(
    model,
    x_init,
    solver_config,
    cond=None,
    masks=None,
    spatial_condition=None,
):
    """
    Uses ODESolver (flow-matching) to sample from x_init -> final output.
    solver_config might contain keys:
        {
          "method": "midpoint"/"rk4"/etc.,
          "step_size": float,
          "time_points": int,
        }

    Returns either the full trajectory [time_points, B, C, H, W] if return_intermediates=True
    or just the final state [B, C, H, W].
    """
    solver = ODESolver(velocity_model=model)

    time_points = solver_config.get("time_points", 10)
    # Flow matching integrates the velocity field from t=0 to t=1.
    T = torch.linspace(0, 1, time_points, device=x_init.device)

    method = solver_config.get("method", "midpoint")
    step_size = solver_config.get("step_size", 0.02)

    if spatial_condition is not None and masks is not None:
        raise ValueError("spatial_condition and masks cannot both be provided")
    model_kwargs = {"cond": cond}
    if spatial_condition is not None:
        model_kwargs["spatial_condition"] = spatial_condition
    elif masks is not None:
        model_kwargs["masks"] = masks

    sol = solver.sample(
        time_grid=T,
        x_init=x_init,
        method=method,
        step_size=step_size,
        return_intermediates=True,
        **model_kwargs,
    )
    return sol


def plot_solver_steps(sol, im_batch, mask_batch, class_batch, class_map, outdir, max_plot=4):
    if sol.dim() != 5:  # No intermediates to plot
        return
    n_samples = min(sol.shape[1], max_plot)
    n_steps = sol.shape[0]
    if mask_batch is not None:
        fig, axes = plt.subplots(n_samples, n_steps + 2, figsize=(20, 8))
    else:
        fig, axes = plt.subplots(n_samples, n_steps + 1, figsize=(20, 8))
    if n_samples == 1:
        axes = [axes]
    for i in range(n_samples):
        for t in range(n_steps):
            axes[i][t].imshow(sol[t, i].cpu().numpy().squeeze(), cmap="gray")
            axes[i][t].axis("off")
            if i == 0:
                axes[i][t].set_title(f"Step {t}")
        col = n_steps
        if mask_batch is not None:
            axes[i][col].imshow(mask_batch[i].cpu().numpy().squeeze(), cmap="gray")
            axes[i][col].axis("off")
            if i == 0:
                axes[i][col].set_title("Mask")
            col += 1
        axes[i][col].imshow(im_batch[i].cpu().numpy().squeeze(), cmap="gray")
        axes[i][col].axis("off")
        if i == 0:
            axes[i][col].set_title("Real")
        if class_map and class_batch is not None:
            idx = class_batch[i].argmax().item()
            cls = class_name_from_map(class_map, idx)
            axes[i][col].text(
                0.5,
                -0.15,
                f"Class: {cls}",
                ha="center",
                va="top",
                transform=axes[i][col].transAxes,
                color="red",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "solver_steps.png"), bbox_inches="tight", pad_inches=0)
    plt.close()


def validate_and_save_samples(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    checkpoint_dir: str,
    epoch: int,
    solver_config: dict,
    max_samples=16,
    class_map=None,
    mask_conditioning=None,
    class_conditioning=False,
    spatial_conditioning=None,
):
    spatial_enabled = _runtime_conditioning_enabled(
        spatial_conditioning, mask_conditioning, default=True
    )
    model.eval()
    outdir = os.path.join(checkpoint_dir, f"val_samples_epoch_{epoch}")
    os.makedirs(outdir, exist_ok=True)
    logger.info(
        f"Saving validation samples: epoch={epoch}, max_samples={max_samples}, output_dir={outdir}."
    )
    count, step_plot_done = 0, False
    for batch in tqdm(val_loader, desc="Validating"):
        imgs = batch["images"].to(device)
        is_3d = imgs.dim() == 5
        if class_conditioning:
            if "classes" not in batch:
                raise KeyError(
                    "class_conditioning is enabled but the dataloader batch has no 'classes' key."
                )
            cond = batch["classes"].to(device).unsqueeze(1)
        else:
            cond = None

        masks, spatial_key = _batch_spatial_condition(batch, enabled=spatial_enabled)
        if masks is not None:
            masks = masks.to(device)

        x_init = torch.randn_like(imgs)
        sample_kwargs = {"cond": cond}
        if spatial_key == "spatial_condition":
            sample_kwargs["spatial_condition"] = masks
        elif spatial_key == "masks":
            sample_kwargs["masks"] = masks
        sol = sample_with_solver(model, x_init, solver_config, **sample_kwargs)
        final_imgs = sol[-1] if sol.dim() == imgs.dim() + 1 else sol

        for i in range(final_imgs.size(0)):
            if count >= max_samples:
                break
            gen_img = normalize_zero_to_one(final_imgs[i])
            real_img = normalize_zero_to_one(imgs[i])
            sdir = os.path.join(outdir, f"sample_{count+1:03d}")
            os.makedirs(sdir, exist_ok=True)
            if is_3d:
                save_image_3d(gen_img, os.path.join(sdir, "gen"))
                save_image_3d(real_img, os.path.join(sdir, "real"))
            else:
                save_image(gen_img, os.path.join(sdir, "gen.png"))
                save_image(real_img, os.path.join(sdir, "real.png"))
            if masks is not None:
                cnd_img = normalize_zero_to_one(masks[i])
                if is_3d:
                    save_image_3d(cnd_img, os.path.join(sdir, "mask"))
                else:
                    save_image(cnd_img, os.path.join(sdir, "mask.png"))
            if class_map and "classes" in batch:
                idx = batch["classes"][i].argmax().item()
                with open(os.path.join(sdir, "class.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "class_index": idx,
                            "class_name": class_name_from_map(class_map, idx),
                            "class_map": class_map,
                            "class_conditioning": class_conditioning,
                            "mask_conditioning": spatial_enabled,
                            "spatial_conditioning": spatial_enabled,
                        },
                        f,
                        indent=4,
                    )
            count += 1
        if not step_plot_done:
            # Save only one trajectory grid per validation epoch to limit plotting overhead.
            clz = batch["classes"] if class_map and "classes" in batch else None
            plot_solver_steps(sol, imgs, masks, clz, class_map, outdir)
            step_plot_done = True
        if count >= max_samples:
            break
    logger.info(f"Validation samples saved in: {outdir}")


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    solver_config: dict,
    batch: torch.Tensor,
    device: torch.device,
    class_conditioning: bool = False,
    mask_conditioning: bool = None,
    spatial_conditioning: bool = None,
):
    model.eval()
    imgs = batch["images"].to(device)
    spatial_enabled = _runtime_conditioning_enabled(
        spatial_conditioning, mask_conditioning, default=False
    )
    if class_conditioning:
        if "classes" not in batch:
            raise KeyError("class_conditioning is enabled but the batch has no 'classes' key.")
        cond = batch["classes"].to(device).unsqueeze(1)
    else:
        cond = None

    masks, spatial_key = _batch_spatial_condition(batch, enabled=spatial_enabled)
    if masks is not None:
        masks = masks.to(device)

    x_init = torch.randn_like(imgs)
    sample_kwargs = {
        "model": model,
        "solver_config": solver_config,
        "x_init": x_init,
        "cond": cond,
    }
    if spatial_key == "spatial_condition":
        sample_kwargs["spatial_condition"] = masks
    elif spatial_key == "masks":
        sample_kwargs["masks"] = masks
    sol = sample_with_solver(**sample_kwargs)
    final_imgs = sol[-1] if sol.dim() == imgs.dim() + 1 else sol
    return final_imgs
