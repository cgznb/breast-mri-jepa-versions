"""VLA JEPA with explicit training prediction policy and prefix-only inference.

Teacher forcing can share parallel predictions between JEPA and pCR during
training. Public prediction APIs always use observed history and free rollout.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import RawMRIInput, RawMRISupervision
from .model_config import RawJEPAConfig, RawVLAJEPAConfig
from .encoder import build_visit_encoder
from .flow import ConditionalMRIFlow
from .attention import build_transformer
from .phases import selected_phases, select_phases
from .sigreg import observed_state_sigreg





class TimeCausalWorldPredictor(nn.Module):
    """[z(0->1),S0,z(1->2),S1,z(2->3),S2] -> [predicted S1,S2,S3].

    Within one time block, all action/state positions can attend each other.
    Across blocks attention is causal. A learned root key is always available
    even if an early block is missing. Inactive inputs are zeroed *before*
    embedding, so their NaNs or contents cannot affect active outputs.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.tokens = math.prod(cfg.token_grid)
        self.queries = cfg.state_queries
        self.state_embed = nn.Linear(cfg.dim, cfg.dim)
        self.action_embed = nn.Linear(cfg.dim, cfg.dim)
        self.residual = cfg.world_residual
        self.history_delta = cfg.history_delta
        if self.history_delta:
            self.history_embed = nn.Linear(cfg.dim, cfg.dim)
            self.history_valid = nn.Embedding(2, cfg.dim)
        self.spatial_position = nn.Parameter(torch.randn(1, self.tokens, cfg.dim) * .02)
        self.query_position = nn.Parameter(torch.randn(1, self.queries, cfg.dim) * .02)
        self.time = nn.Embedding(3, cfg.dim)
        self.root = nn.Parameter(torch.randn(1, 1, cfg.dim) * .02)
        self.blocks = build_transformer(cfg, cfg.predictor_depth)
        self.output = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.dim))
        if self.residual:
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)

    def forward(self, source_states, transitions, source_mask, *,
                history_states=None, history_mask=None):
        batch = len(source_states)
        if source_states.shape != (batch, 3, self.tokens, self.dim):
            raise ValueError("World source states must have shape [B,3,N,dim]")
        if transitions.shape != (batch, 3, self.queries, self.dim):
            raise ValueError("World transitions must have shape [B,3,K,dim]")
        if source_mask.shape != (batch, 3) or source_mask.dtype != torch.bool:
            raise ValueError("World source mask must be boolean [B,3]")
        active = source_mask[:, :, None, None]
        source_states = source_states.masked_fill(~active, 0)
        transitions = transitions.masked_fill(~active, 0)
        temporal = self.time.weight[None, :, None]
        state = self.state_embed(source_states) + self.spatial_position[:, None] + temporal
        if self.history_delta:
            if history_states is None or history_mask is None:
                raise ValueError("History-aware prediction requires predecessor states and validity")
            if history_states.shape != source_states.shape or history_mask.shape != source_mask.shape:
                raise ValueError("World history shapes must match source states and mask")
            if history_mask.dtype != torch.bool:
                raise ValueError("World history mask must be boolean")
            valid = history_mask & source_mask
            previous = history_states.masked_fill(~valid[:, :, None, None], 0)
            delta = (source_states - previous).masked_fill(~valid[:, :, None, None], 0)
            state = state + self.history_embed(delta) + self.history_valid(valid.long())[:, :, None]
        action = self.action_embed(transitions) + self.query_position[:, None] + temporal
        per_block = self.queries + self.tokens
        packed = torch.cat((action, state), 2).flatten(1, 2)
        packed = torch.cat((self.root.expand(batch, -1, -1), packed), 1)
        key_padding = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=packed.device),
                                 ~source_mask[:, :, None].expand(-1, -1, per_block).flatten(1)), 1)
        block_time = torch.cat((torch.tensor([-1], device=packed.device),
                                torch.arange(3, device=packed.device).repeat_interleave(per_block)))
        blocked = block_time[None, :] > block_time[:, None]
        hidden = self.blocks(packed, mask=blocked, src_key_padding_mask=key_padding)
        spatial = hidden[:, 1:].reshape(batch, 3, per_block, self.dim)[:, :, self.queries:]
        prediction = self.output(spatial)
        if self.residual:
            prediction = source_states + prediction
        return prediction.masked_fill(~active, 0)


class RawVLAJEPA(nn.Module):
    """Source-only shared backbone with configurable state-based pCR readout."""

    def __init__(self, cfg: RawVLAJEPAConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.encoder = build_visit_encoder(cfg)
        self.teacher_encoder = (copy.deepcopy(self.encoder).requires_grad_(False)
                                if cfg.enable_world_model else None)
        if self.teacher_encoder is not None:
            self.teacher_encoder.eval()
        self.stage = nn.Embedding(4, cfg.dim)
        self.clinical = nn.Linear(2 * cfg.clinical_dim, cfg.dim)
        self.arm = nn.Embedding(14, cfg.dim)
        self.arm_known_at = nn.Embedding(4, cfg.dim)
        # Slot 0 is unused; slot t conditions the transition (t-1) -> t.
        if cfg.enable_world_model:
            self.dynamics_queries = nn.Parameter(torch.randn(4, cfg.state_queries, cfg.dim) * .02)
        else:
            # Preserve shared pCR-query/fusion initialization in paired ablations
            # without retaining any world-model parameter or module.
            torch.randn(4, cfg.state_queries, cfg.dim)
            self.dynamics_queries = None
        self.pcr_queries = nn.Parameter(torch.randn(cfg.pcr_queries, cfg.dim) * .02)
        self.fusion = build_transformer(cfg, cfg.fusion_depth)
        self.fusion_norm = nn.LayerNorm(cfg.dim)
        readout_dim = cfg.dim if cfg.pcr_readout == "direct" else 3 * cfg.dim
        self.pcr_head = nn.Sequential(nn.LayerNorm(readout_dim), nn.Linear(readout_dim, 1))
        self.world_predictor = TimeCausalWorldPredictor(cfg) if cfg.enable_world_model else None
        self.reconstruction = (nn.Conv3d(cfg.dim, len(selected_phases(cfg)), 1)
                               if cfg.enable_world_model else None)
        self.flow = ConditionalMRIFlow(cfg) if cfg.enable_flow else None
        self.pcr_spatial_scorer = None
        if cfg.enable_world_model and cfg.spatial_readout == "attention":
            # Keep both shared weights and subsequent training RNG identical
            # across paired mean/attention ablations at initialization.
            with torch.random.fork_rng(devices=[]):
                self.pcr_spatial_scorer = nn.Sequential(
                    nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 1, bias=False))
                nn.init.zeros_(self.pcr_spatial_scorer[-1].weight)

    def train(self, mode=True):
        super().train(mode)
        if self.teacher_encoder is not None:
            self.teacher_encoder.eval()
        return self

    def initialize_pretrained_encoder(self, path):
        """Initialize the VQ front end and synchronize the entire EMA teacher.

        Called only for a fresh run; inference/resume reconstructs the
        architecture from cfg and loads all weights from its own checkpoint.
        """
        if self.cfg.encoder_type not in {"vq_prequant", "vq_encoder"}:
            raise ValueError("Pretrained VQ initialization requires a VQ encoder_type")
        metadata = self.encoder.initialize_pretrained(path)
        if self.teacher_encoder is not None:
            self.teacher_encoder.load_state_dict(self.encoder.state_dict(), strict=True)
            self.teacher_encoder.requires_grad_(False).eval()
        return metadata

    def _backbone(self, inp, *, validated=False, classify=True):
        if not validated:
            inp.validate()
        if inp.clinical.shape[1] != self.cfg.clinical_dim:
            raise ValueError("Input clinical dimension differs from model configuration")
        batch, tokens_per_visit = len(inp.images), math.prod(self.cfg.token_grid)
        observed = self._encode_selected(self.encoder, inp, inp.images, inp.observed_mask)
        spatial = observed.new_zeros(batch, 4, tokens_per_visit, self.cfg.dim)
        spatial[inp.observed_mask] = observed
        spatial = spatial + self.stage.weight[None, :, None]
        clinical = self.clinical(torch.cat((inp.clinical, inp.clinical_mask.to(observed.dtype)), -1))
        arm = (self.arm(inp.arm_id) + self.arm_known_at(inp.arm_known_at.clamp(0, 3))
               * inp.arm_mask[:, None].to(observed.dtype))
        context = torch.cat((spatial.flatten(1, 2), clinical[:, None], arm[:, None]), 1)
        context_padding = torch.cat((~inp.observed_mask[:, :, None]
                                     .expand(-1, -1, tokens_per_visit).flatten(1),
                                     torch.zeros(batch, 2, dtype=torch.bool, device=observed.device)), 1)
        pcr = self.pcr_queries[None].expand(batch, -1, -1) + self.stage(inp.landmark)[:, None]
        pcr_padding = torch.zeros(batch, self.cfg.pcr_queries, dtype=torch.bool, device=observed.device)
        if self.cfg.enable_world_model:
            dynamic = self.dynamics_queries[None] + self.stage.weight[None, :, None]
            dynamic = dynamic.expand(batch, -1, -1, -1)
            packed = torch.cat((context, dynamic.flatten(1, 2), pcr), 1)
            dynamic_mask = self._internal_query_mask(inp)
            padding = torch.cat((context_padding,
                                 ~dynamic_mask[:, :, None].expand(-1, -1, self.cfg.state_queries).flatten(1),
                                 pcr_padding), 1)
        else:
            packed = torch.cat((context, pcr), 1)
            padding = torch.cat((context_padding, pcr_padding), 1)
        length, context_length = packed.shape[1], context.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=observed.device).triu(1)
        causal[:context_length, :context_length] = False
        hidden = self.fusion_norm(self.fusion(packed, mask=causal, src_key_padding_mask=padding))
        pcr_features = hidden[:, -self.cfg.pcr_queries:]
        output = {"pcr_features": pcr_features}
        if self.cfg.enable_world_model:
            dynamics = hidden[:, context_length:context_length + 4 * self.cfg.state_queries]
            dynamics = dynamics.reshape(batch, 4, self.cfg.state_queries, self.cfg.dim)
            dynamics = dynamics.masked_fill(~dynamic_mask[:, :, None, None], 0)
            output["dynamics_features"] = dynamics
        if not classify:
            return output, observed
        if self.cfg.pcr_readout == "direct":
            features = pcr_features.mean(1)
        elif self.cfg.pcr_readout == "zero_state":
            features = self._zero_state_readout(pcr_features)
            output["pcr_state_features"] = features
        else:
            states, known = self._observed_teacher_states(inp)
            latest = self._latest_stage(inp)
            rows = torch.arange(batch, device=inp.images.device)
            current = states[rows, latest]
            baseline = states[:, 0]
            if self.cfg.pcr_readout == "copy":
                terminal = current
            else:
                states = self._autonomous_states(inp, dynamics, states, known, latest)
                # Keep the original differentiable rollout for the auxiliary loss.
                output["autonomous_states"] = states
                terminal = states[:, 3]
                if self.cfg.pcr_readout == "forecast_detach":
                    terminal = terminal.detach()
            features, diagnostics = self._pcr_state_readout(
                inp, pcr_features, terminal, current, baseline)
            output["pcr_state_features"] = features
            output.update(diagnostics)
        output["pcr_logit"] = self.pcr_head(features).squeeze(-1)
        return output, observed

    @staticmethod
    def _zero_state_readout(pcr_features):
        """Keep the forecast classifier width while ablating both state inputs."""
        query = pcr_features.mean(1)
        return torch.cat((query, torch.zeros_like(query), torch.zeros_like(query)), -1)

    def _pcr_state_readout(self, inp, pcr_features, terminal, current, baseline):
        """Shared TF/inference readout in the EMA teacher's spatial coordinates.

        The terminal weights also pool the change, so both state branches
        describe the same spatial regions. Attention summarizes token readout;
        it does not establish anatomical localization after Transformer mixing.
        """
        if self.cfg.pcr_delta_reference == "t0":
            if not bool(inp.observed_mask[:, 0].all()):
                raise ValueError("pcr_delta_reference=t0 requires an observed T0 MRI for every patient")
            reference = baseline
        else:
            reference = current
        change = terminal - reference
        diagnostics = {}
        if self.pcr_spatial_scorer is None:
            terminal_features = terminal.mean(1)
            change_features = change.mean(1)
        else:
            weights = self.pcr_spatial_scorer(terminal).squeeze(-1).float().softmax(1)
            spatial_weights = weights.to(terminal.dtype).unsqueeze(-1)
            terminal_features = (terminal * spatial_weights).sum(1)
            change_features = (change * spatial_weights).sum(1)
            detached = weights.detach()
            entropy = -(detached * detached.clamp_min(torch.finfo(detached.dtype).tiny).log()).sum(1)
            diagnostics = {"pcr_spatial_attention": detached,
                           "pcr_spatial_attention_entropy": entropy,
                           "pcr_spatial_effective_tokens": entropy.exp()}
        return torch.cat((pcr_features.mean(1), terminal_features, change_features), -1), diagnostics

    def _internal_query_mask(self, inp):
        if not self.cfg.fixed_transition_queries:
            return inp.query_mask
        # These are transition slots, not output requests: all three are present
        # at every prefix. Their conditioning still sees only legal observations.
        return torch.arange(4, device=inp.images.device)[None].gt(0).expand(len(inp.images), -1)

    def _encode_selected(self, encoder, inp, images, selected):
        if not self.cfg.lesion_pooling:
            return encoder(images[selected])
        weights = getattr(inp, "lesion_weights", None)
        if weights is None:
            raise ValueError("Lesion pooling requires observed T0 lesion_weights")
        patient, _ = selected.nonzero(as_tuple=True)
        return encoder(images[selected], lesion_weights=weights[patient])

    @staticmethod
    def _latest_stage(inp):
        stages = torch.arange(4, device=inp.images.device)[None]
        return stages.expand_as(inp.observed_mask).masked_fill(~inp.observed_mask, -1).max(1).values

    @torch.no_grad()
    def _observed_teacher_states(self, inp):
        self._require_world_model()
        states = inp.images.new_zeros(len(inp.images), 4, math.prod(self.cfg.token_grid), self.cfg.dim)
        states[inp.observed_mask] = self._encode_selected(
            self.teacher_encoder, inp, inp.images, inp.observed_mask).to(states.dtype)
        return states, inp.observed_mask.clone()

    def backbone(self, inp: RawMRIInput):
        """Only legal observed data; state readouts may run autonomous predictions."""
        output = self._backbone(inp)[0]
        # The complete rollout is an internal training cache. Public requests
        # expose future states only through the masked forecast API.
        output.pop("autonomous_states", None)
        return output

    def forward(self, inp: RawMRIInput):
        return self.backbone(inp)

    @torch.no_grad()
    def _teacher_states(self, inp, sup):
        self._require_world_model()
        known = inp.observed_mask | sup.future_mask
        scans = torch.zeros_like(inp.images)
        scans[inp.observed_mask] = inp.images[inp.observed_mask]
        # Index before encoding: masked future NaNs are never read by the teacher.
        scans[sup.future_mask] = sup.future[sup.future_mask]
        states = inp.images.new_zeros(len(inp.images), 4, math.prod(self.cfg.token_grid), self.cfg.dim)
        states[known] = self._encode_selected(self.teacher_encoder, inp, scans, known).to(states.dtype)
        return states, known

    def teacher_forcing_loss(self, inp, sup, dynamics, return_details=False):
        """True predecessor -> next state with patient-mean valid-pair reduction.

        For landmark k only transitions j>=k and requested j+1 are active.
        Missing target or source removes that pair; no invented adjacent visit.
        This explicit diagnostic API computes the JEPA objective only. Training
        may additionally share its predictions with the pCR head.
        """
        inp.validate()
        sup.validate(inp)
        return self._teacher_forcing_loss(inp, sup, dynamics, return_details)

    def _teacher_forcing_loss(self, inp, sup, dynamics, return_details=False):
        details = self._teacher_forcing_predictions(inp, sup, dynamics)
        loss = self._teacher_forcing_objective(inp, dynamics, details)
        return (loss, details) if return_details else loss

    def _teacher_forcing_predictions(self, inp, sup, dynamics, *, teacher=None):
        """Encode true states once and predict all valid source blocks together."""
        expected = (len(inp.images), 4, self.cfg.state_queries, self.cfg.dim)
        if dynamics.shape != expected:
            raise ValueError("Dynamics must have shape [B,4,K,dim]")
        states, known = self._teacher_states(inp, sup) if teacher is None else teacher
        stage = torch.arange(3, device=inp.images.device)[None]
        source_mask = known[:, :-1] & (stage >= inp.landmark[:, None]) & inp.query_mask[:, 1:]
        pair_mask = source_mask & known[:, 1:]
        # Terminal prefixes and empty requests need no world-model forward.
        prediction = (self._predict_world(states, dynamics, source_mask, known)
                      if bool(source_mask.any()) else states[:, 1:] * 0 + dynamics.sum() * 0)
        return {"prediction": prediction, "pair_mask": pair_mask,
                "source_mask": source_mask, "teacher_states": states,
                "source_states": states[:, :-1], "targets": states[:, 1:],
                "known_mask": known}

    def _teacher_forcing_objective(self, inp, dynamics, details):
        pair_mask = details["pair_mask"]
        pair_losses = self._state_l1(details["prediction"], details["targets"], inp)
        details["pair_losses"] = pair_losses
        pair_counts = pair_mask.sum(1)
        patient_loss = pair_losses.masked_fill(~pair_mask, 0).sum(1) / pair_counts.clamp_min(1)
        return patient_loss[pair_counts > 0].mean() if (pair_counts > 0).any() else dynamics.sum() * 0

    def _teacher_forced_readout(self, inp, output, teacher, details=None):
        """Classify shared TF predictions, falling back only to observed states.

        A valid source can produce a pCR prediction even without a true target;
        pair validity is exclusively a JEPA supervision constraint.
        """
        pcr_features = output["pcr_features"]
        if self.cfg.pcr_readout in {"direct", "zero_state"}:
            features = (self._zero_state_readout(pcr_features) if self.cfg.pcr_readout == "zero_state"
                        else pcr_features.mean(1))
            if self.cfg.pcr_readout == "zero_state":
                output["pcr_state_features"] = features
            output["pcr_logit"] = self.pcr_head(features).squeeze(-1)
            return output
        states, _ = teacher
        latest = self._latest_stage(inp)
        rows = torch.arange(len(inp.images), device=inp.images.device)
        current = states[rows, latest]
        baseline = states[:, 0]
        terminal, read_stage = current, latest
        fallback = torch.ones_like(latest, dtype=torch.bool)
        if self.cfg.pcr_readout != "copy":
            if details is None:
                raise ValueError("Forecast training readout requires teacher-forced predictions")
            candidates = torch.arange(1, 4, device=inp.images.device)[None]
            predicted_stage = candidates.expand_as(details["source_mask"]).masked_fill(
                ~details["source_mask"], -1).max(1).values
            fallback = predicted_stage < 0
            # A masked gather would still read a prediction for fallback rows;
            # keep the observed terminal state explicit instead.
            terminal = current.clone()
            active = ~fallback
            terminal[active] = details["prediction"][
                rows[active], predicted_stage[active] - 1].to(terminal.dtype)
            read_stage = torch.where(fallback, latest, predicted_stage)
            if self.cfg.pcr_readout == "forecast_detach":
                terminal = terminal.detach()
        features, diagnostics = self._pcr_state_readout(
            inp, pcr_features, terminal, current, baseline)
        output.update(pcr_logit=self.pcr_head(features).squeeze(-1), pcr_state_features=features,
                      pcr_read_stage=read_stage, pcr_observation_fallback=fallback)
        output.update(diagnostics)
        return output

    @staticmethod
    def _source_history(states, known):
        """Latest available predecessor per source, with explicit missingness.

        Callers choose the availability: teacher forcing uses supervised states;
        autonomous rollout uses only observed states and its own earlier outputs.
        """
        history = torch.zeros_like(states[:, :-1])
        valid = torch.zeros_like(known[:, :-1])
        rows = torch.arange(len(states), device=states.device)
        for source in range(1, 3):
            stages = torch.arange(source, device=states.device)[None]
            previous = stages.expand(len(states), -1).masked_fill(~known[:, :source], -1).max(1).values
            active = previous >= 0
            history[active, source] = states[rows[active], previous[active]]
            valid[:, source] = active
        return history, valid

    def _predict_world(self, states, dynamics, source_mask, known):
        if not self.cfg.history_delta:
            return self.world_predictor(states[:, :-1], dynamics[:, 1:], source_mask)
        history, valid = self._source_history(states, known)
        return self.world_predictor(states[:, :-1], dynamics[:, 1:], source_mask,
                                    history_states=history, history_mask=valid)

    def _state_l1(self, prediction, target, inp):
        token_losses = (prediction - target).abs().mean(-1)
        if not self.cfg.lesion_world_weighting:
            return token_losses.mean(-1)
        weights = getattr(inp, "lesion_weights", None)
        if weights is None:
            raise ValueError("Lesion world weighting requires observed T0 lesion_weights")
        # A nonzero background weight retains whole-ROI supervision. Normalize
        # each patient's weights so this ablation does not rescale the objective.
        spatial = F.adaptive_avg_pool3d(weights.float(), self.cfg.token_grid).flatten(1)
        spatial = 1 + self.cfg.lesion_weight_strength * spatial
        return (token_losses.float() * spatial[:, None]).sum(-1) / spatial.sum(-1)[:, None]

    def _autonomous_states(self, inp, dynamics, states=None, known=None, start=None, needs=None):
        """Predict to T3 from actual latest available observed state.

        No supervision object is accepted. Prefix history remains available to
        the explicit history-delta input; predictions replace future states only.
        """
        batch = len(inp.images)
        if states is None:
            states, known = self._observed_teacher_states(inp)
        if start is None:
            start = self._latest_stage(inp)
        if needs is None:
            needs = start < 3
        source_stages = torch.arange(3, device=inp.images.device)[None]
        rows = torch.arange(batch, device=inp.images.device)
        for source_stage in range(3):
            updating = needs & (start <= source_stage)
            if not updating.any():
                continue
            active = ((source_stages >= start[:, None]) &
                      (source_stages <= source_stage) & needs[:, None])
            prediction = self._predict_world(states, dynamics, active, known)
            # A fresh tensor preserves predecessors used by autograd.
            states = states.clone()
            states[rows[updating], source_stage + 1] = prediction[updating, source_stage].to(states.dtype)
            known = known.clone()
            known[rows[updating], source_stage + 1] = True
        return states

    def autonomous_rollout_loss(self, inp, sup, dynamics, *, states=None, return_details=False):
        """Patient-equal L1 over actual future targets after autonomous rollout."""
        inp.validate()
        sup.validate(inp)
        targets, _ = self._teacher_states(inp, sup)
        if states is None:
            states = self._autonomous_states(inp, dynamics)
        losses = self._state_l1(states, targets, inp)
        target_mask = sup.future_mask
        counts = target_mask.sum(1)
        per_patient = losses.masked_fill(~target_mask, 0).sum(1) / counts.clamp_min(1)
        loss = per_patient[counts > 0].mean() if (counts > 0).any() else dynamics.sum() * 0
        if return_details:
            return loss, {"prediction": states, "targets": targets, "target_mask": target_mask,
                          "target_losses": losses, "patient_count": (counts > 0).sum()}
        return loss

    def _rollout(self, inp, dynamics=None):
        """Free rollout from real S_k; no access to supervision or future MRI."""
        self._require_world_model()
        inp.validate()
        batch, count = len(inp.images), math.prod(self.cfg.token_grid)
        needs_future = inp.query_mask.any(1)
        at_landmark = inp.observed_mask.gather(1, inp.landmark[:, None]).squeeze(1)
        if not self.cfg.fixed_transition_queries and (needs_future & ~at_landmark).any():
            raise ValueError("Forecast requires a real MRI at the requested landmark; pCR can still use legal history")
        result = inp.images.new_zeros(batch, 4, count, self.cfg.dim)
        if not needs_future.any():
            return result
        # Sparse output requests may still need intermediate transitions.
        full_query = torch.arange(4, device=inp.images.device)[None] > inp.landmark[:, None]
        if self.cfg.fixed_transition_queries:
            if dynamics is None:
                dynamics = self._backbone(inp, classify=False)[0]["dynamics_features"]
            states = self._autonomous_states(inp, dynamics, needs=needs_future)
            return states.masked_fill(~inp.query_mask[:, :, None, None], 0)
        if dynamics is None or not torch.equal(inp.query_mask, full_query):
            dynamics = self._backbone(replace(inp, query_mask=full_query))[0]["dynamics_features"]
        if self.cfg.history_delta:
            states = self._autonomous_states(inp, dynamics, start=inp.landmark, needs=needs_future)
            return states.masked_fill(~inp.query_mask[:, :, None, None], 0)
        states = inp.images.new_zeros(batch, 4, count, self.cfg.dim)
        current_rows = needs_future.nonzero().flatten()
        with torch.no_grad():
            initial = self.teacher_encoder(inp.images[current_rows, inp.landmark[current_rows]])
        states[current_rows, inp.landmark[current_rows]] = initial.to(states.dtype)
        source_stages = torch.arange(3, device=inp.images.device)[None]
        rows = torch.arange(batch, device=inp.images.device)
        for source_stage in range(3):
            updating = needs_future & (inp.landmark <= source_stage)
            if not updating.any():
                continue
            active = (source_stages >= inp.landmark[:, None]) & (source_stages <= source_stage) & needs_future[:, None]
            prediction = self.world_predictor(states[:, :-1], dynamics[:, 1:], active)
            # Clone avoids changing a predecessor tensor needed by backward.
            states = states.clone()
            states[rows[updating], source_stage + 1] = prediction[updating, source_stage].to(states.dtype)
        return states.masked_fill(~inp.query_mask[:, :, None, None], 0)

    @torch.no_grad()
    def forecast_states(self, inp: RawMRIInput):
        return self._rollout(inp)

    @torch.no_grad()
    def forecast(self, inp: RawMRIInput):
        self._require_world_model()
        output = self._backbone(inp)[0]
        prediction = output.pop("autonomous_states", None)
        if prediction is None:
            prediction = self._rollout(inp, output["dynamics_features"])
        else:
            prediction = prediction.masked_fill(~inp.query_mask[:, :, None, None], 0)
        return {**output, "state_prediction": prediction,
                "query_mask": inp.query_mask.clone()}

    @staticmethod
    def _latest_mri(inp):
        stages = torch.arange(4, device=inp.images.device)[None]
        latest = stages.expand_as(inp.observed_mask).masked_fill(~inp.observed_mask, -1).max(1).values
        return inp.images[torch.arange(len(inp.images), device=inp.images.device), latest]

    def compute_loss(self, inp: RawMRIInput, sup: RawMRISupervision, *,
                     task_weight=1., jepa_weight=1., flow_weight=.1,
                     reconstruction_weight=.1, variance_weight=.01,
                     representation_only=False, variance_definition="legacy", rollout_weight=0.,
                     prediction_training="hybrid", sigreg_weight=0.,
                     sigreg_num_projections=256, sigreg_num_points=17):
        inp.validate()
        if (isinstance(sigreg_weight, bool) or not isinstance(sigreg_weight, (int, float))
                or not math.isfinite(sigreg_weight) or sigreg_weight < 0):
            raise ValueError("sigreg_weight must be finite and nonnegative")
        for name, value, minimum in (("sigreg_num_projections", sigreg_num_projections, 1),
                                     ("sigreg_num_points", sigreg_num_points, 2)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        weights = (task_weight, jepa_weight, flow_weight, reconstruction_weight, variance_weight,
                   rollout_weight, sigreg_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Loss weights must be finite and nonnegative")
        if not self.cfg.enable_world_model and (representation_only or any(weights[1:])):
            raise ValueError("Disabled world model supports BCE-only training with all auxiliary weights zero")
        if variance_definition not in {"legacy", "off", "patient_axis_stage_token_fp32"}:
            raise ValueError("Unsupported variance_definition")
        if variance_definition == "off" and variance_weight:
            raise ValueError("variance_definition=off requires variance_weight=0")
        if prediction_training not in {"hybrid", "teacher_forcing"}:
            raise ValueError("prediction_training must be teacher_forcing or hybrid")
        teacher_forcing = prediction_training == "teacher_forcing"
        if teacher_forcing and rollout_weight:
            raise ValueError("teacher_forcing requires rollout_weight=0; autonomous training is hybrid")
        classify = not representation_only and bool(task_weight) and bool(sup.label_mask.any())
        tf_forecast = (teacher_forcing and classify and
                       self.cfg.pcr_readout in {"forecast", "forecast_detach"})
        sup.validate(inp, future_values=bool(jepa_weight or rollout_weight or
                     tf_forecast or
                     (self.flow is not None and flow_weight and not representation_only)))
        output, observed = self._backbone(inp, validated=True, classify=not teacher_forcing)
        zero = observed.sum() * 0
        task = world = rollout = flow = reconstruction = variance = zero
        pair_count = zero.detach()
        rollout_count = zero.detach()
        details = None
        if teacher_forcing:
            flow_enabled = bool(self.flow is not None and flow_weight and not representation_only)
            needs_prediction = bool(jepa_weight or tf_forecast or flow_enabled)
            needs_teacher = needs_prediction or (classify and self.cfg.pcr_readout not in {"direct", "zero_state"})
            teacher = (self._teacher_states(inp, sup) if needs_prediction else
                       self._observed_teacher_states(inp) if needs_teacher else None)
            if needs_prediction:
                details = self._teacher_forcing_predictions(
                    inp, sup, output["dynamics_features"], teacher=teacher)
            if classify:
                self._teacher_forced_readout(inp, output, teacher, details)
        if not representation_only and task_weight and sup.label_mask.any():
            task = F.binary_cross_entropy_with_logits(output["pcr_logit"][sup.label_mask],
                                                      sup.label[sup.label_mask].to(observed.dtype))
        if jepa_weight:
            if teacher_forcing:
                world = self._teacher_forcing_objective(inp, output["dynamics_features"], details)
            else:
                world, details = self._teacher_forcing_loss(inp, sup, output["dynamics_features"],
                                                           return_details=True)
            pair_count = details["pair_mask"].sum().to(observed.dtype)
        if rollout_weight:
            rollout, details = self.autonomous_rollout_loss(
                inp, sup, output["dynamics_features"], states=output.get("autonomous_states"),
                return_details=True)
            rollout_count = details["target_mask"].sum().to(observed.dtype)
        if reconstruction_weight:
            spatial = observed.transpose(1, 2).reshape(len(observed), self.cfg.dim, *self.cfg.token_grid)
            reconstructed = F.interpolate(self.reconstruction(spatial), size=inp.images.shape[-3:],
                                            mode="trilinear", align_corners=False)
            reconstruction = F.mse_loss(
                reconstructed, select_phases(inp.images[inp.observed_mask], self.cfg).detach())
        if variance_weight:
            variance = self.variance_penalty(observed, inp.observed_mask, variance_definition)
        if not representation_only and self.flow is not None and flow_weight and sup.future_mask.any():
            selected = sup.future_mask.clone()
            if teacher_forcing:
                selected[:, 0] = False
                selected[:, 1:] &= details["pair_mask"]
                forecast = details["prediction"]
            else:
                forecast = self._rollout(inp, output["dynamics_features"])
            if bool(selected.any()):
                x1 = select_phases(sup.future[selected], self.cfg).detach()
                x0 = torch.randn_like(x1)
                tau = torch.rand(len(x1), device=x1.device, dtype=x1.dtype)
                t = tau[:, None, None, None, None]
                x_tau = (1 - t) * x0 + t * x1
                source = select_phases(self._latest_mri(inp), self.cfg)
                source = source[:, None].expand(-1, 4, -1, -1, -1, -1)[selected]
                stage = torch.arange(4, device=x1.device)[None].expand(len(inp.images), -1)[selected]
                latent = forecast[selected[:, 1:]] if teacher_forcing else forecast[selected]
                velocity = self.flow(x_tau, tau, source, latent, stage)
                flow = F.mse_loss(velocity, x1 - x0)
        loss = (task_weight * task + jepa_weight * world + rollout_weight * rollout + flow_weight * flow
                + reconstruction_weight * reconstruction + variance_weight * variance)
        if sigreg_weight:
            sigreg = observed_state_sigreg(observed, inp.observed_mask,
                num_projections=sigreg_num_projections, num_points=sigreg_num_points)
            loss = loss + sigreg_weight * sigreg
        terms = {"loss": loss, "task": task, "jepa": world, "world_model": world,
                      "rollout": rollout,
                      "flow": flow, "reconstruction": reconstruction, "variance": variance,
                      "label_count": sup.label_mask.sum().to(observed.dtype),
                      "future_count": sup.future_mask.sum().to(observed.dtype),
                      "pair_count": pair_count, "rollout_count": rollout_count}
        if sigreg_weight:
            terms["sigreg"] = sigreg
            patient_counts = inp.observed_mask.sum(0).detach().float()
            terms["sigreg_stage_count"] = (patient_counts >= 2).sum().float()
            for stage in range(inp.observed_mask.shape[1]):
                terms[f"sigreg_T{stage}_patient_count"] = patient_counts[stage]
        if teacher_forcing:
            stage = output.get("pcr_read_stage")
            for index in range(4):
                terms[f"pcr_stage_T{index}_count"] = (
                    (stage == index).sum().to(observed.dtype) if stage is not None else zero.detach())
            fallback = output.get("pcr_observation_fallback")
            terms["pcr_observation_fallback_count"] = (
                fallback.sum().to(observed.dtype) if fallback is not None else zero.detach())
        return loss, terms

    @staticmethod
    def variance_penalty(observed, observed_mask, definition="legacy"):
        if definition == "legacy":
            std = (observed.flatten(0, 1).var(0, unbiased=False) + 1e-4).sqrt()
            return F.relu(1 - std).mean()
        if definition == "off":
            return observed.sum() * 0
        if definition != "patient_axis_stage_token_fp32":
            raise ValueError("Unsupported variance_definition")
        # Compare patients at matching stage/token/channel; spatial diversity cannot satisfy this term.
        states = observed.new_zeros(*observed_mask.shape, *observed.shape[1:], dtype=torch.float32)
        states[observed_mask] = observed.float()
        losses = []
        for stage in range(4):
            valid = observed_mask[:, stage]
            if valid.sum() >= 2:
                std = (states[valid, stage].var(0, unbiased=False) + 1e-4).sqrt()
                losses.append(F.relu(1 - std).mean())
        return torch.stack(losses).mean() if losses else observed.float().sum() * 0

    @torch.no_grad()
    def update_teacher(self):
        if self.teacher_encoder is None:
            return
        frozen_frontend = (self.cfg.encoder_type in {"vq_prequant", "vq_encoder"} and
                           not self.encoder.pretrained_trainable)
        for (_, teacher), (name, student) in zip(self.teacher_encoder.named_parameters(),
                                                 self.encoder.named_parameters(), strict=True):
            if frozen_frontend and name.startswith("frontend."):
                # Preserve exact frozen coordinates; EMA still tracks adapters.
                teacher.copy_(student)
            else:
                teacher.lerp_(student, 1 - self.cfg.ema_decay)
        for teacher, student in zip(self.teacher_encoder.buffers(), self.encoder.buffers(), strict=True):
            teacher.copy_(student)

    def _require_world_model(self):
        if not self.cfg.enable_world_model:
            raise ValueError("World prediction is disabled in this direct pCR model")

    @torch.no_grad()
    def generate(self, inp: RawMRIInput, steps=4, seed=0, method="heun"):
        """Optional MRI flow, conditioned on free forecasts, never teacher forcing."""
        if self.flow is None:
            raise ValueError("Generation is disabled in this model configuration")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if method not in {"euler", "heun"}:
            raise ValueError("method must be euler or heun")
        training = self.training
        self.eval()
        try:
            output = self.forecast(inp)
            selected = inp.query_mask
            images = torch.zeros_like(select_phases(inp.images, self.cfg))
            if selected.any():
                noise_device = "cpu" if inp.images.device.type == "mps" else inp.images.device
                generator = torch.Generator(device=noise_device).manual_seed(seed)
                value = torch.randn(images[selected].shape, dtype=inp.images.dtype,
                                    device=noise_device, generator=generator).to(inp.images.device)
                source = select_phases(self._latest_mri(inp), self.cfg)
                source = source[:, None].expand(-1, 4, -1, -1, -1, -1)[selected]
                future = output["state_prediction"][selected]
                stage = torch.arange(4, device=inp.images.device)[None].expand(len(inp.images), -1)[selected]
                dt = 1 / steps
                for step in range(steps):
                    tau = value.new_full((len(value),), step * dt)
                    velocity = self.flow(value, tau, source, future, stage)
                    if method == "heun":
                        proposal = value + dt * velocity
                        velocity = .5 * (velocity + self.flow(proposal, tau + dt, source, future, stage))
                    value = value + dt * velocity
                images[selected] = value
            return {"future_images": images, "pcr_probability": output["pcr_logit"].sigmoid(),
                    "query_mask": inp.query_mask.clone(), "phase_order": list(selected_phases(self.cfg))}
        finally:
            self.train(training)

# SIGReg integration v1: observed patient-axis stage-token regularization.
