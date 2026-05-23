"""
GCM (Generative Causal Mediation) gradient-based attribution for translation.

Implements arXiv:2602.16080 Eq. 1 with the repository sign convention:

    IE_hat = grad_z [logp(r_orig | p_orig) - logp(r_cf | p_orig)] . (z_orig - z_cf)

at every (a) attention head's pre-o_proj activation and (b) SAE feature in
layer 16, patched at the LAST source-token position.

Sign convention:
    M(z) = logp(r_orig | p_orig, z) - logp(r_cf | p_orig, z)
    grad = dM/dz |_{z = z_orig}
    delta_z = z_orig - z_cf
    IE = grad . delta_z  (elementwise per component)
    => Positive IE means: this component, when in its z_orig state vs z_cf
       state, increases the model's preference for r_orig over r_cf. So |IE|
       measures component importance for distinguishing the two translations,
       and sign tells which translation it favors (positive = orig/correct,
       negative = counterfactual/incorrect).

Departures from the original GCM paper, all intentional:
    * Patching at the LAST source-token position only (not all source positions).
      The paper sums over all source positions; here we follow the per-position
      single-anchor design discussed in the experiment plan.
    * Two separate `model.trace()` blocks (one per response) instead of a single
      multi-invoke trace, because nnsight 0.5 batches multi-invoke into one
      forward pass with left-padding, which would shift the patched index when
      the two responses have different lengths. See REDTEAM.md C2.
    * Heads patched at o_proj.OUTPUT (via the equivalent delta = (z_leaf - z_orig)
      @ W_o.T), since writing to o_proj.input is unsupported in nnsight (per the
      official docs and REDTEAM.md C3).
    * Score the first response token (no chat-template artifact to skip).

Implementation notes (see REDTEAM.md for the bugs this corrects):

  * Two SEPARATE `model.trace(...)` blocks per pair (one per response), each
    with a single invoke. Avoids the multi-invoke left-padding bug.
  * `metric_proxy.backward()` is called OUTSIDE the trace context. After the
    trace exits, the saved scalar is a regular Tensor whose autograd graph
    still reaches `z_leaf` (the leaf injected via `f_patched[idx,:] = z_leaf`
    + `submodule.output[0][:] = ...`). Doing backward INSIDE the trace via
    `with metric.backward():` keeps the entire nnsight proxy graph alive
    during backprop and OOMs after one pair on 44 GB GPUs.
  * Loops inside `with model.trace():` MUST use explicit `for` (not list
    comprehension). nnsight raises `ExitTracingException` via sys.settrace
    to exit the body; mid-comprehension this leaves the target unbound.
  * Patches via the `delta = patched_decode - clean_decode` perturbation
    pattern from `lang_probing_src/interventions/ablate.py` — never writes
    to `module.input` (unsupported per official nnsight docs).
  * For heads: read o_proj.input + o_proj.weight, compute the equivalent
    o_proj output delta in Python, write to o_proj.output (supported).
  * Tokenization via joint-tokenize + longest-common-prefix, so the
    response can begin with any character (RTL/CJK-safe).
  * Truncation in token space (no decode->retokenize round-trip).
  * Float32 leaf tensors for numerically-clean gradients.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from lang_probing_src.config import LAYER_NUM, SAE_DIM, TRACER_KWARGS


# ---------------------------------------------------------------------------
# Tokenization + metric helpers
# ---------------------------------------------------------------------------


@dataclass
class TokenizedPrompt:
    input_ids: torch.Tensor       # [1, S]
    prompt_len: int                # number of prompt tokens (including BOS)
    last_src_idx: int              # = prompt_len - 1
    response_start: int            # = prompt_len
    response_end: int              # = S
    decoded_last_src: str
    decoded_first_response: str


def tokenize_pair(
    tokenizer,
    prompt: str,
    response: str,
    device,
    max_response_tokens: Optional[int] = None,
) -> TokenizedPrompt:
    """
    Joint-tokenize (prompt + response), then locate the prompt boundary via
    longest-common-prefix with the prompt tokenized alone. This is robust to:
      - BPE merges across the prompt/response seam (rare).
      - Non-Latin scripts (Arabic, Hebrew, Hindi, CJK) where the natural
        first response token is NOT a leading-space-prefixed word.

    If max_response_tokens is set, the JOINT id sequence is truncated to
    `prompt_len + max_response_tokens` (token-space, no decode roundtrip).
    """
    if not prompt:
        raise ValueError("Empty prompt")
    if not response:
        raise ValueError("Empty response")

    prompt_alone_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids[0]
    joint_ids = tokenizer(prompt + response, add_special_tokens=True, return_tensors="pt").input_ids[0]

    # Longest common prefix between prompt_alone and joint
    prompt_len = 0
    for k in range(min(len(prompt_alone_ids), len(joint_ids))):
        if prompt_alone_ids[k].item() == joint_ids[k].item():
            prompt_len = k + 1
        else:
            break

    if prompt_len == 0:
        raise ValueError("Tokenization seam: prompt and joint share no common prefix")
    if prompt_len >= len(joint_ids):
        raise ValueError(f"Empty response after tokenization (prompt_len={prompt_len}, joint_len={len(joint_ids)})")

    # Truncate response in token space
    if max_response_tokens is not None and (len(joint_ids) - prompt_len) > max_response_tokens:
        joint_ids = joint_ids[: prompt_len + max_response_tokens]

    full_ids = joint_ids.unsqueeze(0).to(device)  # [1, S]

    decoded_last_src = tokenizer.decode([full_ids[0, prompt_len - 1].item()])
    assert decoded_last_src in (":", " "), (
        f"tokenize_pair: decoded_last_src is {decoded_last_src!r} — expected ':' or ' '. "
        f"The prompt template must end with 'LangName: ' (trailing space after colon). "
        f"This assert catches tokenization drift from prompt template changes."
    )
    return TokenizedPrompt(
        input_ids=full_ids,
        prompt_len=prompt_len,
        last_src_idx=prompt_len - 1,
        response_start=prompt_len,
        response_end=full_ids.shape[1],
        decoded_last_src=decoded_last_src,
        decoded_first_response=tokenizer.decode([full_ids[0, prompt_len].item()]),
    )


# Logit scale applied before softmax. Cohere/Aya define config.logit_scale
# (0.0625) which the HF forward multiplies into the logits AFTER lm_head; since
# we read lm_head.output directly (pre-scale) we must reapply it, or Aya's
# metric/IE are inflated ~16x. Llama has no logit_scale -> stays 1.0. Set via
# set_logit_scale() once the model is loaded.
_LOGIT_SCALE = 1.0


def set_logit_scale(scale: float) -> None:
    """Set the global logit scale (call once after model load)."""
    global _LOGIT_SCALE
    _LOGIT_SCALE = float(scale)


def sum_response_logprobs(logits: torch.Tensor, input_ids: torch.Tensor, response_start: int) -> torch.Tensor:
    """
    Sum log p(input_ids[t] | input_ids[<t]) over t in [response_start, S).
    logits: [1, S, V]   input_ids: [1, S]
    Returns scalar tensor (with grad if logits has grad).
    """
    seq_len = input_ids.shape[1]
    n_response = seq_len - response_start
    if n_response <= 0:
        raise ValueError(f"Empty response: response_start={response_start} seq_len={seq_len}")
    pred_logits = logits[:, response_start - 1 : seq_len - 1, :]      # [1, n_response, V]
    targets = input_ids[:, response_start:seq_len]                     # [1, n_response]
    if _LOGIT_SCALE != 1.0:
        pred_logits = pred_logits * _LOGIT_SCALE
    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, n_response]
    return gathered.sum()


# ---------------------------------------------------------------------------
# SAE feature attribution at L16
# ---------------------------------------------------------------------------


def gcm_attribute_sae(
    model,
    submodule,           # model.model.layers[LAYER_NUM]
    autoencoder,
    tokenizer,
    prompt_orig: str,
    response_orig: str,
    prompt_cf: str,
    response_cf: str,
    device,
    max_response_tokens: Optional[int] = None,
    *,
    cache_prompt_orig: Optional[str] = None,
    cache_response_orig: Optional[str] = None,
    cache_prompt_cf: Optional[str] = None,
    cache_response_cf: Optional[str] = None,
    m_orig_clean: Optional[float] = None,
    m_cf_clean: Optional[float] = None,
):
    """
    GCM attribution at L16 SAE features, patched at the last source-token of
    the orig prompt. See module docstring for the math.

    `cache_prompt_*` overrides decouple "where z is *cached from*" from "the
    prompt where z is patched and the metric is scored". Phase 1 leaves these
    None (cache prompts default to prompt_orig / prompt_cf). For the null
    control: prompt_orig == prompt_cf == p_A (the actual source); responses
    are tgt_B and tgt_C; cache prompts are p_B and p_C respectively, so
    z_orig := z_B and z_cf := z_C.
    """
    # --- Tokenize: scoring prompts (same prompt prefix for both, by design)
    #     and (possibly distinct) cache prompts for z_orig and z_cf.
    pr_or = tokenize_pair(tokenizer, prompt_orig, response_orig, device, max_response_tokens)
    pr_or_rc = tokenize_pair(tokenizer, prompt_orig, response_cf, device, max_response_tokens)
    pr_cache_orig = (
        pr_or if cache_prompt_orig is None
        else tokenize_pair(
            tokenizer,
            cache_prompt_orig,
            cache_response_orig if cache_response_orig is not None else response_orig,
            device, max_response_tokens,
        )
    )
    pr_cache_cf = tokenize_pair(
        tokenizer,
        cache_prompt_cf if cache_prompt_cf is not None else prompt_cf,
        cache_response_cf if cache_response_cf is not None else response_cf,
        device, max_response_tokens,
    )

    # --- Cache z_orig (SAE features at last_src_idx of cache_prompt_orig) ---
    # autoencoder.encode() called on an nnsight proxy returns [S, SAE_DIM] (batch dim
    # collapsed by nnsight), so index with 2-D indexing: f[pos, :] not f[:, pos, :].
    with model.trace(pr_cache_orig.input_ids, **TRACER_KWARGS), torch.no_grad():
        x = submodule.output[0]
        f = autoencoder.encode(x)
        z_orig_proxy = f[pr_cache_orig.last_src_idx, :].clone().save()
    z_orig = z_orig_proxy.detach().clone()  # [SAE_DIM]

    # --- Cache z_cf at last_src_idx of cache_prompt_cf ---
    with model.trace(pr_cache_cf.input_ids, **TRACER_KWARGS), torch.no_grad():
        x = submodule.output[0]
        f = autoencoder.encode(x)
        z_cf_proxy = f[pr_cache_cf.last_src_idx, :].clone().save()
    z_cf = z_cf_proxy.detach().clone()      # [SAE_DIM]

    # --- Clean (un-patched) metrics for sanity ---
    # If the caller pre-computed these (e.g. run.py with --components both),
    # skip the two forward passes to avoid ~20% wasted compute.
    if m_orig_clean is None or m_cf_clean is None:
        with model.trace(pr_or.input_ids, **TRACER_KWARGS), torch.no_grad():
            logits = model.lm_head.output
            m_orig_clean_proxy = sum_response_logprobs(logits, pr_or.input_ids, pr_or.response_start).save()
        with model.trace(pr_or_rc.input_ids, **TRACER_KWARGS), torch.no_grad():
            logits = model.lm_head.output
            m_cf_clean_proxy = sum_response_logprobs(logits, pr_or_rc.input_ids, pr_or_rc.response_start).save()
        m_orig_clean = float(m_orig_clean_proxy.item())
        m_cf_clean = float(m_cf_clean_proxy.item())

    # --- Build z_leaf (float32 for numerically-clean gradients) ---
    z_leaf = z_orig.detach().clone().to(torch.float32).requires_grad_(True)

    # --- Trace 1: prompt_orig + r_orig ; patch z at last_src_idx ; backward ---
    # Do backward OUTSIDE the trace (matches the working pattern in
    # counterfactual_attribution/run.py). Backward inside `with model.trace():`
    # via `with metric.backward():` keeps the entire nnsight proxy graph alive
    # during backprop and OOMs after one pair on a 44 GB GPU.
    with model.trace(pr_or.input_ids, **TRACER_KWARGS):
        x_clean = submodule.output[0]                       # [1, S, d_model]
        f_clean = autoencoder.encode(x_clean)               # [S, SAE_DIM]  (batch dim dropped by nnsight)
        f_patched = f_clean.clone()
        f_patched[pr_or.last_src_idx, :] = z_leaf.to(f_patched.dtype)
        decoded_clean = autoencoder.decode(f_clean)         # [S, d_model]
        decoded_patched = autoencoder.decode(f_patched)     # [S, d_model]
        # delta is [S, d_model]; broadcasts with [1, S, d_model] output
        submodule.output[0][:] = submodule.output[0] + (decoded_patched - decoded_clean)
        m_orig_p_value = sum_response_logprobs(
            model.lm_head.output, pr_or.input_ids, pr_or.response_start
        ).save()

    m_orig_p_value.backward()
    assert z_leaf.grad is not None and z_leaf.grad.abs().sum() > 0, (
        "SAE leaf received zero gradient after m_orig backward — "
        "the patch did not connect to the loss (C1/C4 regression?)."
    )
    grad_orig = z_leaf.grad.detach().clone()  # [SAE_DIM]
    m_orig_patched = float(m_orig_p_value.item())
    z_leaf.grad = None  # zero between traces so the next backward starts clean

    # --- Trace 2: prompt_orig + r_cf ; same patch ; backward ---
    with model.trace(pr_or_rc.input_ids, **TRACER_KWARGS):
        x_clean = submodule.output[0]
        f_clean = autoencoder.encode(x_clean)               # [S, SAE_DIM]
        f_patched = f_clean.clone()
        f_patched[pr_or_rc.last_src_idx, :] = z_leaf.to(f_patched.dtype)
        decoded_clean = autoencoder.decode(f_clean)
        decoded_patched = autoencoder.decode(f_patched)
        submodule.output[0][:] = submodule.output[0] + (decoded_patched - decoded_clean)
        m_cf_p_value = sum_response_logprobs(
            model.lm_head.output, pr_or_rc.input_ids, pr_or_rc.response_start
        ).save()

    m_cf_p_value.backward()
    assert z_leaf.grad is not None and z_leaf.grad.abs().sum() > 0, (
        "SAE leaf received zero gradient after m_cf backward — "
        "the patch did not connect to the loss (C1/C4 regression?)."
    )
    grad_cf = z_leaf.grad.detach().clone()
    m_cf_patched = float(m_cf_p_value.item())
    z_leaf.grad = None

    # --- Compose: gradient of M = m_orig - m_cf is grad_orig - grad_cf ---
    grad = grad_orig - grad_cf                     # [SAE_DIM]
    delta_z = (z_orig - z_cf).to(torch.float32)    # [SAE_DIM]
    ie = (grad * delta_z).cpu()
    M_patched = m_orig_patched - m_cf_patched

    # Sanity: at z_leaf = z_orig, the patched orig metric should equal the clean orig metric.
    # The SAE clones decoded_clean and reconstructs identically, so this should hold to ~bf16
    # roundoff. Drift > 0.5 nat means the patch is not behaving like an identity.
    sanity_orig_drift = abs(m_orig_patched - m_orig_clean)

    return {
        "ie": ie,
        "grad": grad.cpu(),
        "delta_z": delta_z.cpu(),
        "metric_orig_clean": m_orig_clean,
        "metric_cf_clean": m_cf_clean,
        "metric_orig_patched": m_orig_patched,
        "metric_cf_patched": m_cf_patched,
        "metric_diff_patched": M_patched,
        "sanity_orig_drift": sanity_orig_drift,
        "z_orig_norm": float(z_orig.float().norm().item()),
        "z_cf_norm": float(z_cf.float().norm().item()),
        "decoded_last_src_orig": pr_cache_orig.decoded_last_src,
        "decoded_last_src_cf": pr_cache_cf.decoded_last_src,
        "n_response_orig": pr_or.response_end - pr_or.response_start,
        "n_response_cf": pr_or_rc.response_end - pr_or_rc.response_start,
    }


# ---------------------------------------------------------------------------
# Per-attention-head attribution
# ---------------------------------------------------------------------------


def gcm_attribute_heads(
    model,
    tokenizer,
    prompt_orig: str,
    response_orig: str,
    prompt_cf: str,
    response_cf: str,
    device,
    max_response_tokens: Optional[int] = None,
    *,
    cache_prompt_orig: Optional[str] = None,
    cache_response_orig: Optional[str] = None,
    cache_prompt_cf: Optional[str] = None,
    cache_response_cf: Optional[str] = None,
    m_orig_clean: Optional[float] = None,
    m_cf_clean: Optional[float] = None,
):
    """
    GCM attribution at every attention-head output, patched at last source token.

    Mechanics:
      * Read each layer's o_proj.input (= post-attention concat of per-head
        outputs, [B, S, n_q_heads * head_dim] = [B, S, d_model] for Llama).
      * The "patch from z_orig to z_leaf at last_src_idx" maps to a delta in
        o_proj.output of magnitude (z_leaf - z_orig) @ W_o.T at that position.
        We compute that delta in Python and add it to o_proj.output (writing
        to o_proj.output is supported; writing to o_proj.input is not).

    For Llama-3.1-8B (GQA: 32 q-heads, 8 kv-heads) the o_proj input is still
    n_q_heads * head_dim wide, so the per-q-head reshape [n_q_heads, head_dim]
    is correct. Note that q-heads come in groups of 4 sharing a kv-head.

    `cache_prompt_*` overrides decouple where z is *cached from* from the
    scoring prompt. Phase 1 leaves them None. See gcm_attribute_sae docstring.
    """
    layers = model.model.layers
    cfg = model.model.config  # use the underlying HF config to avoid nnsight forwarding edge cases
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    d_model = cfg.hidden_size
    head_dim = d_model // n_heads

    pr_or = tokenize_pair(tokenizer, prompt_orig, response_orig, device, max_response_tokens)
    pr_or_rc = tokenize_pair(tokenizer, prompt_orig, response_cf, device, max_response_tokens)
    pr_cache_orig = (
        pr_or if cache_prompt_orig is None
        else tokenize_pair(
            tokenizer,
            cache_prompt_orig,
            cache_response_orig if cache_response_orig is not None else response_orig,
            device, max_response_tokens,
        )
    )
    pr_cache_cf = tokenize_pair(
        tokenizer,
        cache_prompt_cf if cache_prompt_cf is not None else prompt_cf,
        cache_response_cf if cache_response_cf is not None else response_cf,
        device, max_response_tokens,
    )

    # --- Cache z_orig per layer at last_src_idx of cache_prompt_orig ---
    # List comprehensions inside with model.trace() are broken in nnsight 0.5:
    # ExitTracingException fires via sys.settrace mid-comprehension, leaving the
    # target variable unbound. Use an explicit pre-initialized list + for loop.
    z_orig_proxies = []
    with model.trace(pr_cache_orig.input_ids, **TRACER_KWARGS), torch.no_grad():
        for i in range(n_layers):
            z_orig_proxies.append(
                layers[i].self_attn.o_proj.input[:, pr_cache_orig.last_src_idx, :].clone().save()
            )
    z_orig = torch.stack([p.detach().squeeze(0) for p in z_orig_proxies])  # [n_layers, d_model]

    # --- Cache z_cf per layer at last_src_idx of cache_prompt_cf ---
    z_cf_proxies = []
    with model.trace(pr_cache_cf.input_ids, **TRACER_KWARGS), torch.no_grad():
        for i in range(n_layers):
            z_cf_proxies.append(
                layers[i].self_attn.o_proj.input[:, pr_cache_cf.last_src_idx, :].clone().save()
            )
    z_cf = torch.stack([p.detach().squeeze(0) for p in z_cf_proxies])

    # --- Clean metric values for sanity ---
    # If the caller pre-computed these (e.g. run.py with --components both),
    # skip the two forward passes to avoid ~20% wasted compute.
    if m_orig_clean is None or m_cf_clean is None:
        with model.trace(pr_or.input_ids, **TRACER_KWARGS), torch.no_grad():
            m_orig_clean_proxy = sum_response_logprobs(model.lm_head.output, pr_or.input_ids, pr_or.response_start).save()
        with model.trace(pr_or_rc.input_ids, **TRACER_KWARGS), torch.no_grad():
            m_cf_clean_proxy = sum_response_logprobs(model.lm_head.output, pr_or_rc.input_ids, pr_or_rc.response_start).save()
        m_orig_clean = float(m_orig_clean_proxy.item())
        m_cf_clean = float(m_cf_clean_proxy.item())

    # --- Build leaf in float32 ---
    z_leaf = z_orig.detach().clone().to(torch.float32).requires_grad_(True)

    # --- Trace 1: prompt_orig + r_orig --- backward OUTSIDE trace (memory hygiene) ---
    with model.trace(pr_or.input_ids, **TRACER_KWARGS):
        for i in range(n_layers):
            o_in_clean = layers[i].self_attn.o_proj.input        # [B, S, d_model]
            o_in_patched = o_in_clean.clone()
            o_in_patched[:, pr_or.last_src_idx, :] = z_leaf[i].unsqueeze(0).to(o_in_clean.dtype)
            W_o = layers[i].self_attn.o_proj.weight              # Parameter, [d_model, d_model]
            o_out_clean = o_in_clean @ W_o.T
            o_out_patched = o_in_patched @ W_o.T
            layers[i].self_attn.o_proj.output[:] = (
                layers[i].self_attn.o_proj.output + (o_out_patched - o_out_clean)
            )
        m_orig_p_value = sum_response_logprobs(
            model.lm_head.output, pr_or.input_ids, pr_or.response_start
        ).save()

    m_orig_p_value.backward()
    assert z_leaf.grad is not None, (
        "heads leaf received None gradient after m_orig backward (C1/C3 regression?)."
    )
    per_layer = z_leaf.grad.view(n_layers, -1).abs().sum(-1)
    assert (per_layer > 0).all(), (
        f"Partial gradient after m_orig backward: layers with zero grad = "
        f"{(per_layer == 0).nonzero().flatten().tolist()}"
    )
    grad_orig = z_leaf.grad.detach().clone()  # [n_layers, d_model]
    m_orig_patched = float(m_orig_p_value.item())
    z_leaf.grad = None

    # --- Trace 2: prompt_orig + r_cf ---
    with model.trace(pr_or_rc.input_ids, **TRACER_KWARGS):
        for i in range(n_layers):
            o_in_clean = layers[i].self_attn.o_proj.input
            o_in_patched = o_in_clean.clone()
            o_in_patched[:, pr_or_rc.last_src_idx, :] = z_leaf[i].unsqueeze(0).to(o_in_clean.dtype)
            W_o = layers[i].self_attn.o_proj.weight
            o_out_clean = o_in_clean @ W_o.T
            o_out_patched = o_in_patched @ W_o.T
            layers[i].self_attn.o_proj.output[:] = (
                layers[i].self_attn.o_proj.output + (o_out_patched - o_out_clean)
            )
        m_cf_p_value = sum_response_logprobs(
            model.lm_head.output, pr_or_rc.input_ids, pr_or_rc.response_start
        ).save()

    m_cf_p_value.backward()
    assert z_leaf.grad is not None, (
        "heads leaf received None gradient after m_cf backward (C1/C3 regression?)."
    )
    per_layer = z_leaf.grad.view(n_layers, -1).abs().sum(-1)
    assert (per_layer > 0).all(), (
        f"Partial gradient after m_cf backward: layers with zero grad = "
        f"{(per_layer == 0).nonzero().flatten().tolist()}"
    )
    grad_cf = z_leaf.grad.detach().clone()
    m_cf_patched = float(m_cf_p_value.item())
    z_leaf.grad = None

    # --- Compose ---
    grad = grad_orig - grad_cf                                # [n_layers, d_model]
    delta_z = (z_orig - z_cf).to(torch.float32)               # [n_layers, d_model]
    grad_h = grad.view(n_layers, n_heads, head_dim)
    delta_h = delta_z.view(n_layers, n_heads, head_dim)
    ie = (grad_h * delta_h).sum(dim=-1)                       # [n_layers, n_heads]
    M_patched = m_orig_patched - m_cf_patched

    sanity_orig_drift = abs(m_orig_patched - m_orig_clean)

    return {
        "ie": ie.cpu(),                                       # [n_layers, n_heads]
        "grad": grad.cpu(),                                   # [n_layers, d_model]
        "delta_z": delta_z.cpu(),                             # [n_layers, d_model]
        "metric_orig_clean": m_orig_clean,
        "metric_cf_clean": m_cf_clean,
        "metric_orig_patched": m_orig_patched,
        "metric_cf_patched": m_cf_patched,
        "metric_diff_patched": M_patched,
        "sanity_orig_drift": sanity_orig_drift,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "head_dim": head_dim,
        "decoded_last_src_orig": pr_cache_orig.decoded_last_src,
        "decoded_last_src_cf": pr_cache_cf.decoded_last_src,
    }
