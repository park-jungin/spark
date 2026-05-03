"""Implementation of additional projectors for additional inputs to the VLA models."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class NoisyActionProjector(nn.Module):
    """
    [Diffusion] Projects noisy action inputs into the LLM's embedding space.

    Note that since each action is tokenized into 7 tokens in OpenVLA (rather
    than having 1 token per action), each noisy action token will have dimension 1
    instead of 7.
    """
    def __init__(self, llm_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.action_token_dim = 1

        self.fc1 = nn.Linear(self.action_token_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, noisy_actions: torch.Tensor = None) -> torch.Tensor:
        # noisy_actions: (bsz, num_action_tokens=chunk_len*action_dim, 1)
        projected_features = self.fc1(noisy_actions)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class AlignProjector(nn.Module):
    """
    calculate the alignment between LLM and VGGT embeddings.
    """
    def __init__(
            self, 
            llm_dim: int, 
            vggt_dim: int,
            align_loss_type: str = "cosine",
            use_vlm_norm: bool = False,
        ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.vggt_dim = vggt_dim
        self.align_loss_type = align_loss_type

        self.fc1 = nn.Linear(self.llm_dim, 2 * self.vggt_dim, bias=True)
        self.fc2 = nn.Linear(2 * self.vggt_dim, 2 * self.vggt_dim, bias=True)
        self.act_fn1 = nn.GELU()
        
        self.vlm_norm = nn.LayerNorm(llm_dim) if use_vlm_norm else None

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def align_dimension(self, LLM_embedding: torch.Tensor = None) -> torch.Tensor:
        if self.vlm_norm is not None:
            LLM_embedding = self.vlm_norm(LLM_embedding)
        projected_features = self.fc1(LLM_embedding)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features
    
    def compute_align_loss_cosine(self, vision_hidden, vggt_hidden):
        # vision_hidden has a shape of (bs, N, D)
        def mean_flat(x):
            return torch.mean(x, dim=list(range(1, len(x.size()))))
        align_loss = 0
        bsz = vision_hidden.shape[0]
        for _vision, _vggt in zip(vision_hidden, vggt_hidden):
            _vision = torch.nn.functional.normalize(_vision, dim=-1)
            _vggt = torch.nn.functional.normalize(_vggt, dim=-1)
            # align_loss += 1 - torch.mean(vision_hidden * vggt_hidden).sum(dim=-1).mean()
            align_loss += 1 - mean_flat((_vision * _vggt).sum(dim=-1))  # Cosine similarity loss
        align_loss /= bsz  # Average over batch size
        return align_loss
    
    def forward(self, LLM_emb, target_emb):
        if self.align_loss_type == "cosine":
            # project vla dimension and calculate align loss
            with torch.autocast("cuda", dtype=torch.bfloat16):
                LLM_emb = self.align_dimension(LLM_emb)
            align_loss = self.compute_align_loss_cosine(LLM_emb, target_emb).mean()  # mean for sequence length
            return align_loss
        else:
            raise NotImplementedError(f"Align loss type {self.align_loss_type} is not implemented.")


class DualPathFusionProjector(nn.Module):
    """Fuses frozen-base and LoRA-expert visual embeddings for action prediction."""

    def __init__(self, llm_dim: int, input_dim: int = None) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.input_dim = llm_dim if input_dim is None else input_dim
        fusion_dim = 2 * self.input_dim
        self.fc1 = nn.Linear(fusion_dim, fusion_dim, bias=True)
        self.fc2 = nn.Linear(fusion_dim, llm_dim, bias=True)
        self.act_fn = nn.GELU()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, base_features: torch.Tensor, expert_features: torch.Tensor) -> torch.Tensor:
        fused = torch.cat((base_features, expert_features), dim=-1)
        fused = self.fc1(fused)
        fused = self.act_fn(fused)
        fused = self.fc2(fused)
        return fused


class SinglePathProjector(nn.Module):
    """Projects a single visual path to the LLM embedding space (separate-path ablation baseline)."""

    def __init__(self, llm_dim: int, input_dim: int = None) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.input_dim = llm_dim if input_dim is None else input_dim
        self.fc1 = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.fc2 = nn.Linear(self.input_dim, llm_dim, bias=True)
        self.act_fn = nn.GELU()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.fc1(features)
        projected = self.act_fn(projected)
        projected = self.fc2(projected)
        return projected


class AttentionResponseAligner(nn.Module):
    """Aligns attention responses Attn(Qvla, Kvla, Vvla) and Attn(Qvla, Kvggt, Vvggt)."""

    def __init__(
        self,
        vla_dim: int,
        vggt_dim: int,
        hidden_dim: int = 512,
        loss_type: str = "mse",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.vla_dim = vla_dim
        self.vggt_dim = vggt_dim
        # Kept for config/checkpoint compatibility; projections are intentionally removed.
        self.hidden_dim = hidden_dim
        self.loss_type = loss_type
        self.temperature = temperature

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        attn_scores = torch.softmax(attn_logits, dim=-1)
        return torch.matmul(attn_scores, value)

    def _loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(source, target, reduction="mean")
        if self.loss_type == "l1":
            return F.l1_loss(source, target, reduction="mean")
        if self.loss_type == "kl":
            source_log_prob = F.log_softmax(source / self.temperature, dim=-1)
            target_prob = F.softmax(target / self.temperature, dim=-1)
            return F.kl_div(source_log_prob, target_prob, reduction="batchmean")
        raise NotImplementedError(f"Attention response loss type `{self.loss_type}` is not implemented.")

    def forward(
        self,
        q_vla: torch.Tensor,
        k_vla: torch.Tensor,
        v_vla: torch.Tensor,
        k_vggt: torch.Tensor,
        v_vggt: torch.Tensor,
    ) -> torch.Tensor:
        if q_vla.shape[-1] != k_vggt.shape[-1] or q_vla.shape[-1] != v_vggt.shape[-1]:
            raise ValueError(
                "AttentionResponseAligner requires matching VLA/VGGT QKV dims when projections are disabled. "
                f"Got VLA dim={q_vla.shape[-1]}, VGGT K dim={k_vggt.shape[-1]}, VGGT V dim={v_vggt.shape[-1]}. "
                "Use DINO alignment branch (or re-enable projections for mismatched dims)."
            )
        q = q_vla
        k_self = k_vla
        v_self = v_vla
        k_cross = k_vggt.to(dtype=q.dtype)
        v_cross = v_vggt.to(dtype=q.dtype)

        self_attn_response = self._attention(q, k_self, v_self)
        cross_attn_response = self._attention(q, k_cross, v_cross)
        return self._loss(self_attn_response, cross_attn_response)


class KnowledgeRouter(nn.Module):
    """
    Task-conditioned token router for visual/3D tokens before feeding into the LLM.

    Architecture:
      1) Project candidate visual tokens into text embedding space.
      2) Cross-attention with visual tokens as Query and contextualized text tokens as Key/Value to produce
         text-conditioned visual token embeddings.
      3) Token-wise MLP gate (Linear-GELU-Linear) producing keep probabilities.

    During training, online pseudo-labels are built from detached visual->text attention
    scores (top-k by keep ratio), and focal BCE is used for supervision.
    """

    def __init__(
        self,
        text_dim: int,
        token_dim: int,
        num_heads: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        temperature: float = 1.0,
        focal_gamma: float = 2.0,
        effective_num_beta: float = 0.999,
    ) -> None:
        super().__init__()
        if text_dim % num_heads != 0:
            raise ValueError(
                f"KnowledgeRouter requires text_dim % num_heads == 0, got text_dim={text_dim}, num_heads={num_heads}."
            )

        self.text_dim = text_dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.temperature = temperature
        self.focal_gamma = focal_gamma
        self.effective_num_beta = effective_num_beta

        # Project candidate tokens into language-token space before cross-attention.
        self.token_proj = nn.Linear(token_dim, text_dim, bias=True)
        # Kept for checkpoint compatibility with earlier router versions.
        self.query_proj = nn.Linear(text_dim, text_dim, bias=True)
        for param in self.query_proj.parameters():
            param.requires_grad = False
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fc1 = nn.Linear(text_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self.act_fn = nn.GELU()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T)
        mask = mask.to(dtype=x.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _build_online_pseudo_targets(
        self,
        token_scores: torch.Tensor,
        target_keep_ratio: float,
        min_keep_tokens: int,
    ) -> torch.Tensor:
        # token_scores: (B, N)
        bsz, num_tokens = token_scores.shape
        keep_tokens = int(round(float(target_keep_ratio) * float(num_tokens)))
        keep_tokens = max(int(min_keep_tokens), keep_tokens)
        keep_tokens = min(num_tokens, keep_tokens)
        keep_tokens = max(1, keep_tokens)

        _, topk_idx = torch.topk(token_scores, k=keep_tokens, dim=1, largest=True, sorted=False)
        targets = torch.zeros_like(token_scores)
        targets.scatter_(1, topk_idx, 1.0)
        return targets

    @staticmethod
    def _build_default_positions(batch_size: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        return torch.arange(num_tokens, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    @staticmethod
    def _apply_rope(tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, D), positions: (B, T)
        dim = tokens.shape[-1]
        if dim % 2 != 0:
            return tokens

        half_dim = dim // 2
        inv_freq = torch.arange(half_dim, device=tokens.device, dtype=torch.float32)
        inv_freq = torch.pow(10000.0, -inv_freq / float(half_dim))
        angles = positions.to(dtype=torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
        cos = torch.cos(angles).to(dtype=tokens.dtype)
        sin = torch.sin(angles).to(dtype=tokens.dtype)

        even = tokens[..., 0::2]
        odd = tokens[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        rotated = torch.empty_like(tokens)
        rotated[..., 0::2] = rotated_even
        rotated[..., 1::2] = rotated_odd
        return rotated

    def _focal_binary_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        probs = probs.float().clamp(min=eps, max=1.0 - eps)
        targets = targets.float()

        bce = -(targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs))
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)

        # Effective-number class weighting (Dr.LLM-style initialization).
        n_pos = targets.sum(dtype=torch.float32).clamp_min(1.0)
        n_all = torch.tensor(float(targets.numel()), device=targets.device, dtype=torch.float32)
        n_neg = (n_all - n_pos).clamp_min(1.0)
        beta = torch.tensor(float(self.effective_num_beta), device=targets.device, dtype=torch.float32).clamp(
            min=0.0, max=0.999999
        )

        alpha_pos_raw = (1.0 - beta) / (1.0 - torch.pow(beta, n_pos)).clamp_min(eps)
        alpha_neg_raw = (1.0 - beta) / (1.0 - torch.pow(beta, n_neg)).clamp_min(eps)
        alpha_norm = (alpha_pos_raw + alpha_neg_raw).clamp_min(1e-8)
        alpha_pos = alpha_pos_raw / alpha_norm
        alpha_neg = alpha_neg_raw / alpha_norm
        alpha = torch.where(targets > 0.5, alpha_pos, alpha_neg)

        focal = alpha * ((1.0 - pt) ** float(self.focal_gamma)) * bce
        return torch.nan_to_num(focal.mean(), nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        text_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        text_mask: torch.Tensor = None,
        candidate_positions: torch.Tensor = None,
        target_keep_ratio: float = 0.5,
        min_keep_tokens: int = 8,
        hard_routing: bool = False,
        compute_loss: bool = True,
    ) -> dict:
        # text_tokens: (B, Tq, D), candidate_tokens: (B, N, D)
        if text_mask is None:
            text_mask = torch.ones(
                text_tokens.shape[0],
                text_tokens.shape[1],
                device=text_tokens.device,
                dtype=torch.bool,
            )
        if candidate_positions is None:
            candidate_positions = self._build_default_positions(
                batch_size=candidate_tokens.shape[0],
                num_tokens=candidate_tokens.shape[1],
                device=candidate_tokens.device,
            )
        if candidate_positions.shape != candidate_tokens.shape[:2]:
            raise ValueError(
                "candidate_positions must have shape (B, N) matching candidate_tokens[:, :, :]. "
                f"Got candidate_positions={tuple(candidate_positions.shape)}, "
                f"candidate_tokens={tuple(candidate_tokens.shape)}."
            )

        # Use contextualized language embeddings directly as cross-attention K/V.
        # (No text projection in forward.)
        projected_text_tokens = text_tokens.to(dtype=candidate_tokens.dtype)
        projected_candidate_tokens = self.token_proj(candidate_tokens).to(dtype=candidate_tokens.dtype)

        # Text tokens are already contextualized by the LLM stack; keep their representation unchanged here.
        projected_candidate_tokens = self._apply_rope(projected_candidate_tokens, candidate_positions)

        attn_output, attn_weights = self.cross_attn(
            query=projected_candidate_tokens,
            key=projected_text_tokens,
            value=projected_text_tokens,
            key_padding_mask=~text_mask.bool(),
            need_weights=True,
            average_attn_weights=True,
        )

        # Per-visual-token text-conditioned features from cross-attention.
        gate_input = projected_candidate_tokens + attn_output
        gate_logits = self.fc2(self.act_fn(self.fc1(gate_input))).squeeze(-1).float()
        gate_logits = torch.nan_to_num(gate_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        gate_probs = torch.sigmoid(gate_logits / max(float(self.temperature), 1e-6))

        if hard_routing:
            hard_mask = (gate_probs >= 0.5).to(dtype=gate_probs.dtype)
            # Straight-through estimator to keep gradients through probabilities.
            gates = hard_mask + (gate_probs - gate_probs.detach())
        else:
            gates = gate_probs

        gated_tokens = candidate_tokens * gates.to(dtype=candidate_tokens.dtype).unsqueeze(-1)
        keep_tokens = int(round(float(target_keep_ratio) * float(candidate_tokens.shape[1])))
        keep_tokens = max(int(min_keep_tokens), keep_tokens)
        keep_tokens = min(candidate_tokens.shape[1], keep_tokens)
        keep_tokens = max(1, keep_tokens)

        _, selected_indices = torch.topk(gate_probs, k=keep_tokens, dim=1, largest=True, sorted=False)
        selected_indices, _ = torch.sort(selected_indices, dim=1)
        selected_gate_probs = torch.gather(gate_probs, dim=1, index=selected_indices)
        selected_idx_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, candidate_tokens.shape[-1])
        selected_candidate_tokens = torch.gather(candidate_tokens, dim=1, index=selected_idx_expanded)
        pruned_gated_tokens = selected_candidate_tokens * selected_gate_probs.to(selected_candidate_tokens.dtype).unsqueeze(-1)

        # Build online pseudo labels from detached visual->text attention scores.
        # attn_weights: (B, N_visual, T_text) when average_attn_weights=True.
        text_mask_float = text_mask.to(dtype=attn_weights.dtype).unsqueeze(1)
        text_mask_denom = text_mask_float.sum(dim=2).clamp_min(1.0)
        token_scores = ((attn_weights.float() * text_mask_float).sum(dim=2) / text_mask_denom).detach()
        token_scores = torch.nan_to_num(token_scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        pseudo_targets = self._build_online_pseudo_targets(
            token_scores=token_scores,
            target_keep_ratio=target_keep_ratio,
            min_keep_tokens=min_keep_tokens,
        )

        zero = gate_probs.new_zeros(())
        router_cls_loss = zero
        router_budget_loss = zero
        router_entropy_loss = zero
        if compute_loss:
            router_cls_loss = self._focal_binary_loss(gate_probs, pseudo_targets)
            router_budget_loss = torch.abs(gate_probs.mean(dim=1) - float(target_keep_ratio)).mean()
            safe_probs = gate_probs.clamp(min=1e-6, max=1.0 - 1e-6)
            entropy = -(safe_probs * torch.log(safe_probs) + (1.0 - safe_probs) * torch.log(1.0 - safe_probs))
            # Minimize negative entropy to encourage non-collapsed routing.
            router_entropy_loss = -entropy.mean()
            router_cls_loss = torch.nan_to_num(router_cls_loss, nan=0.0, posinf=0.0, neginf=0.0)
            router_budget_loss = torch.nan_to_num(router_budget_loss, nan=0.0, posinf=0.0, neginf=0.0)
            router_entropy_loss = torch.nan_to_num(router_entropy_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "gated_tokens": gated_tokens,
            "gate_probs": gate_probs,
            "pseudo_targets": pseudo_targets,
            "router_cls_loss": router_cls_loss,
            "router_budget_loss": router_budget_loss,
            "router_entropy_loss": router_entropy_loss,
            "keep_ratio": gate_probs.mean(),
            "selected_indices": selected_indices,
            "selected_gate_probs": selected_gate_probs,
            "selected_candidate_tokens": selected_candidate_tokens,
            "pruned_gated_tokens": pruned_gated_tokens,
        }
