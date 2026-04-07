---
name: ESMM MMoE PLE plan
overview: After a short Phase 0 (metrics, splits, variance, learning curve), compare baseline K vs shared-bottom ESMM vs MMoE vs PLE under matched training and documented capacity; primary model selection uses CTCVR ROC-AUC with PR-AUC/logloss/ECE as required companions; Phase B tunes schedule and loss balance before routing/capacity, then auxiliary losses last. Incorporates OpenAI + Gemini MCP review (2026).
todos:
  - id: phase0-metrics
    content: "Phase 0: Add eval helpers — CTCVR/CTR PR-AUC, BCE logloss, ECE; log per-task train losses & optional grad norms"
    status: pending
  - id: phase0-splits-seeds
    content: "Phase 0: Document split policy (no leakage); 3 seeds for Phase A final decision; mean±std or min–max in leaderboard"
    status: pending
  - id: phase0-learning-curve
    content: "Phase 0: K_ref learning curve (epochs or steps) to justify 5 epochs vs early-stop; confirm row-group shuffle is sufficient for Adam"
    status: pending
  - id: spec-hparams
    content: Fix Phase A hparams — d_model, expert_hidden, expert counts; total params (embed vs trunk) vs K and shared-bottom
    status: pending
  - id: impl-shared-bottom
    content: Add ESMM_SharedBottom — one shared trunk (matched width/depth to fair param budget) then thin CTR/CVR heads
    status: pending
  - id: impl-models
    content: Add ESMM_MMoE and ESMM_PLE (same forward API); log gate entropy / expert usage; optional load-balancing aux loss if collapse
    status: pending
  - id: trainer-factory
    content: Parameterize train_esmm_parquet_rowgroups — model_ctor, optional λ_ctr/λ_ctcvr; keep K default λ=1
    status: pending
  - id: phase-a-run
    content: Run Phase A legs (K_ref, SharedBottom, MMoE, PLE) × seeds; log full metric table + wall time
    status: pending
  - id: phase-b-rounds
    content: Phase B in order B4→B2→B1→B3 on winner(s); optional GradNorm/uncertainty weighting instead of large λ grid
    status: pending
isProject: false
---

# ESMM: MMoE vs PLE vs baselines (CTCVR-focused) + four improvement rounds

## Consultant review (incorporated)

Suggestions from **OpenAI** and **Gemini** (MCP `chat-with-openai` / `chat-with-gemini`, same prompt) are merged below: **richer evaluation**, **shared-bottom control**, **multi-seed Phase A**, **split/shuffle discipline**, **task loss balance earlier**, **gate / routing diagnostics**, **reordered Phase B** (schedule before capacity), **defer auxiliary clicked-only loss**, optional **load-balancing** and **uncertainty weighting**.

---

## Context (from trail log)

- **Notebook:** [experiments/20260404_ali_cpp_esmm/20260404_esmm_experiment.ipynb](../20260404_esmm_experiment.ipynb)
- **Baseline K (canonical):** CVR_AUC **0.6158**, **CTCVR_AUC 0.5917**, ~7094 s wall, 5 epochs, batch **4096**, row-group Parquet training, fused embedding, default **manual batches + AMP + prefetch** in `train_esmm_parquet_rowgroups`.
- **ESMM loss (default):** `BCE(click, p_ctr) + BCE(click·purchase, p_ctcvr)` with `p_ctcvr = p_ctr · p_cvr`. Trainer should support **`λ_ctr`, `λ_ctcvr`** (default 1,1) for Phase B2 without forking code paths.

### Metrics (model selection and reporting)

| Role | Metrics |
|------|---------|
| **Primary (ranking, user goal)** | **ROC-AUC for CTCVR** on full impression space (existing `evaluate_esmm_ctcvr` pattern). |
| **Required companions** | **PR-AUC (average precision)** for **CTCVR** (heavy class imbalance); **BCE / log loss** for CTR and for CTCVR; **ECE** (or reliability binning) for `p_ctr` and `p_ctcvr` — calibration errors compound multiplicatively. |
| **Diagnostic only** | **CVR ROC-AUC on clicked-only** — useful for negative transfer; **do not** use as sole selector (selection bias vs ESMM’s entire-space objective). |

**Decision rule (Phase A):** Prefer winner on **mean CTCVR ROC-AUC across seeds**; require agreement with **CTCVR PR-AUC** where they conflict, and document logloss/ECE. Define a **minimum worthwhile delta** (e.g. >0.002 ROC-AUC mean) to avoid chasing noise.

---

## Paper-aligned structures (what changes vs K)

| Idea | Role in ESMM | Contrast to K |
|------|----------------|---------------|
| **[MMoE](https://www.kdd.org/kdd2018/acceptedpapers/view/mixing-task-relationships-in-multi-task-learning-with-multi-gate-mixture-)** (Ma et al., KDD 2018) | Shared experts on `x` + **per-task gates** → fused states → thin heads. | K = two full independent towers. |
| **[PLE](https://dl.acm.org/doi/10.1145/3383313.3412236)** (Tencent, RecSys 2020) | **2-level** extraction: shared + task experts; level-2 selectors from level-1 fusions. | Adds task-dedicated capacity + hierarchy vs MMoE. |
| **Shared-bottom ESMM (new control)** | **Single shared trunk** (MLP), then **small separate heads** for CTR/CVR — **no gating**. | Isolates whether gains come from **routing** vs **more shared depth**. Match **total non-embedding params** ~to MMoE/PLE where feasible; always report **embedding vs trunk** param counts. |

*Implementation: two-task gated pattern; PDFs are references, not layout constraints.*

---

## Implementation plan (code shape)

1. **Frozen front-end:** same `unified_emb`, `field_offsets`, `x = concat(flatten(embeddings), dense)`. All models: `forward(sparse_x, dense_x) -> (p_ctr, p_cvr, p_ctcvr)` with **sigmoid + clamp** like K.
2. **New modules** (notebook cell or [notebooks/ad_hoc/esmm_mtl_blocks.py](notebooks/ad_hoc/esmm_mtl_blocks.py)):
   - **`ESMM_SharedBottom`:** one stack (e.g. 360→200→80) shared, then linear (or tiny) heads to two logits. Param-budget match to expert models.
   - **`ESMM_MMoE`:** single-level MMoE first; optional stacked level-2 if Phase B warrants.
   - **`ESMM_PLE`:** 2-level PLE patterned on [ple_experiment/model.py](ple_experiment/model.py); small expert counts initially.
3. **Trainer:** `model_ctor`, optional **`λ_ctr`, `λ_ctcvr`**, log **CTR loss vs CTCVR loss** each epoch (and optional **grad norm** per task). **`torch.compile` off** for Phase A for MMoE/PLE.
4. **Routing diagnostics:** periodically log **gate softmax entropy** and **expert utilization** (mean weight per expert). If **routing collapse** (one expert dominates), add a small **load-balancing** auxiliary (e.g. encourage uniform expert usage) or increase dropout on gates — only if collapse is observed.
5. **Capacity documentation:** table per leg: total params, **embedding params**, **trunk params**, expert counts, `d_model`, `expert_hidden`.

---

## Phase 0 — Validation infrastructure (before Phase A shootout)

Do this **once** before comparing architectures.

1. **Splits:** Document whether train/test are **random row** or **time/user-aware**. Flag **leakage risk** if row groups correlate with time/campaign; if a time column exists, note whether a **time holdout** sanity check is feasible later.
2. **Shuffle / streaming:** Confirm training uses **shuffled row-group order per epoch** and **within-row-group** `randperm` (current manual-batch path). If batches are overly correlated, Adam suffers — fix buffer/shuffle before architecture work.
3. **Learning curve:** Run **K_ref** for **epochs 1…E** (or fixed step budget) once; if metrics still climb sharply at epoch 5, Phase A comparisons may be **under-training**; enable **early stopping** on a **held-out validation** (held-out row groups or split file) from the start for long runs.
4. **Variance:** Fix **≥3 random seeds** for the **final** Phase A leaderboard; optional single-seed smoke first. Report **mean ± std** (or min/max) for key metrics.
5. **Task balance logging:** Even with λ=1, log **raw BCE values**; if CTR term dominates magnitude, plan **B2 earlier** (see below).

---

## Phase A — Structural shootout (4 legs + K_ref)

| Leg | ID | Hypothesis |
|-----|-----|------------|
| A0 | **K_ref** | Same as current K; refresh under new metrics code. |
| A0b | **SharedBottom** | Gains from MMoE/PLE require **gating/experts**, not merely a **wider shared trunk**. |
| A1 | **MMoE** | Task-specific routing improves **CTCVR** vs shared trunk / K. |
| A2 | **PLE** | Task experts + progressive extraction beat MMoE on **CTCVR** or **PR-AUC**. |

**Protocol:** Same Parquet/vocab, **batch 4096**, **Adam lr=1e-3** (unless Phase 0 curve suggests otherwise), **EMBED_DIM=18**, same `expert_hidden` / `d_model` policy for MMoE and PLE. **5 epochs** default, with **early stopping** if Phase 0 showed benefit.

**Tie-break:** If top models within **±0.002** mean CTCVR ROC-AUC, use **CTCVR PR-AUC** and **logloss**; carry **both** into Phase B if still ambiguous.

**Optional Phase A λ smoke:** If Phase 0 logs show **strong CTR dominance**, run **one** extra short leg (e.g. `λ_ctcvr=2`) only on **K_ref** or the current best backbone — avoid a full grid in Phase A.

---

## Phase B — Four rounds (**reordered**: schedule → balance → capacity → auxiliary)

Rationale (consultants): tuning **capacity/routing (old B1)** under wrong **LR/epochs** or **λ** wastes runs. **Auxiliary clicked-only CVR** changes the objective — **last**.

| New step | Theme | Content |
|----------|--------|--------|
| **B4 (first)** | Schedule / budget / early-stop | Cosine or step LR, warmup if needed; **more epochs** only if val proxy improves; **early stopping** on held-out row groups; revisit whether **5 epochs** overfits embeddings on full Ali-scale data. |
| **B2 (second)** | Loss balance & regularization | **`λ_ctr`, `λ_ctcvr`** grid (e.g. 0.5–2.0); **weight decay**; **grad clip**. Alternative: **GradNorm** or **Kendall uncertainty weighting** to reduce manual λ search. |
| **B1 (third)** | Routing / capacity | `num_experts`, shared/task expert counts, `d_model`, dropout; stacked MMoE second level; **load-balancing** loss if gates collapse. |
| **B3 (last)** | Auxiliary objective | **Clicked-only CVR** loss *in addition* to ESMM — use **careful weighting** or **stop-grad** experiments so auxiliary does not break entire-space semantics; **ablate** if CTCVR PR-AUC or calibration worsens. |

**Fallback:** If MMoE and PLE **lose clearly** to **K_ref** and **SharedBottom**, document negative result; optionally try **lightweight soft-sharing** (e.g. cross-stitch–style) before dropping MTL research thread.

**Future (out of scope unless Phase B plateaus):** **AITM**-style sequential CTR→CVR transfer on AliCCP; deeper PLE beyond 2 levels.

---

## Logging / ops

- Extend [logs/20260404_esmm_experiment_trial.md](20260404_esmm_experiment_trial.md) or add **`logs/2026XXXX_esmm_mtl_mmoe_ple.md`** with columns: **CTCVR ROC-AUC**, **CTCVR PR-AUC**, CTR/CTCVR **logloss**, **ECE**, CVR_AUC (clicked), wall time, **seed**, param breakdown.
- Cache keys: **`esmm_mtl_phase_a.json`** (or per-leg keys) so Round 4 K caches stay untouched.
- Colab: **`scp` + `papermill`** only; no git sync to Colab for notebooks.

---

## Risks (updated)

- **OOM / latency:** More experts → reduce `d_model` or expert count before dropping batch size.
- **Metric noise:** Mitigate with **3 seeds** and **PR-AUC + logloss** alongside ROC-AUC.
- **Routing collapse:** Monitor gates; add **load-balancing** only if needed.
- **Split / row-group leakage:** Underweighted in original plan — validate in Phase 0.
- **`torch.compile`:** Off for Phase A on gated models.
