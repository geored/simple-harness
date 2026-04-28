"""
Prompt Sandbox V1 - Core Implementation
Integrates: adaptive baseline, stratified logging, warm-up gating
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PromotionStatus(str, Enum):
    TEST = "TEST"        # Warm-up or insufficient data
    SHADOW = "SHADOW"    # Running in parallel, not serving
    CANARY = "CANARY"    # Serving small traffic slice
    STABLE = "STABLE"    # Full production


class RejectionReason(str, Enum):
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    SEMANTIC_DRIFT = "SEMANTIC_DRIFT"
    JUDGE_REJECTION = "JUDGE_REJECTION"
    WARM_UP_INCOMPLETE = "WARM_UP_INCOMPLETE"


class LogTier(str, Enum):
    FULL = "FULL"           # All details stored
    BORDERLINE = "BORDERLINE"  # Uncertain success, used for baseline retraining
    SAMPLED = "SAMPLED"     # Routine success, kept at reduced rate
    SKIPPED = "SKIPPED"     # Dropped after sampling decision


# ---------------------------------------------------------------------------
# Immutable Artifact + Mutable Delta
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptArtifact:
    """
    Immutable, version-controlled prompt definition.
    Changing any field requires a new artifact with a new artifact_id.
    """
    artifact_id: str
    template: str
    schema_fingerprint: str        # SHA-256 of the expected output JSON schema
    embedding_anchor: np.ndarray   # Reference embedding for this prompt's intent

    # frozen=True won't work with ndarray directly; use __post_init__ workaround
    def __post_init__(self):
        object.__setattr__(
            self,
            "embedding_anchor",
            np.array(self.embedding_anchor, dtype=np.float32)
        )

    def fingerprint(self) -> str:
        """Deterministic identity hash over template + schema."""
        payload = f"{self.template}::{self.schema_fingerprint}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class PromptDelta:
    """
    Mutable tuning layer applied on top of a PromptArtifact.
    Does NOT require a new artifact version for minor adjustments.
    Carries its own version counter for audit purposes.
    """
    delta_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_id: str = ""
    delta_version: int = 1

    # Tuning knobs
    temperature_override: float | None = None
    system_prefix: str = ""
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)
    stop_sequences: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def bump_version(self) -> "PromptDelta":
        """Return a new delta with incremented version, preserving all other fields."""
        import copy
        clone = copy.deepcopy(self)
        clone.delta_id = str(uuid.uuid4())
        clone.delta_version += 1
        return clone


@dataclass
class EvaluationContext:
    """
    Full context for a single sandbox evaluation run.
    Bundles artifact + delta + runtime state for one request cycle.
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact: PromptArtifact | None = None
    delta: PromptDelta | None = None
    raw_input: dict[str, Any] = field(default_factory=dict)
    raw_output: str = ""
    output_embedding: np.ndarray | None = None
    output_schema_fingerprint: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Populated during evaluation
    tier1_passed: bool = False
    tier2_score: float | None = None      # Cosine similarity vs anchor
    tier2_anomaly_score: float | None = None  # IsolationForest decision score
    tier3_judge_score: float | None = None
    rejection_reason: RejectionReason | None = None
    log_tier: LogTier = LogTier.SKIPPED
    added_to_baseline: bool = False       # Refinement 1: tracks baseline update


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SandboxConfig:
    # Warm-up gating (Refinement 3)
    min_success_threshold: int = 50

    # Tier 2 thresholds
    cosine_similarity_floor: float = 0.82   # Below this → semantic drift
    anomaly_score_floor: float = -0.15      # IsolationForest scores below → anomaly
    borderline_band: float = 0.05           # Within this of floor → BORDERLINE log tier

    # Tier 3 throttle
    judge_invocation_rate: float = 0.10     # Only call LLM judge this fraction of time

    # Refinement 2: success logging
    routine_success_sample_rate: float = 0.05   # 5% of clean successes logged

    # Refinement 1: baseline update
    baseline_retrain_every_n: int = 25      # Retrain IsolationForest every N additions


# ---------------------------------------------------------------------------
# Drift Detector with Adaptive Baseline (Refinement 1)
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Isolation Forest over embedding space.
    Supports incremental baseline updates from borderline-accepted outputs.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._baseline_embeddings: list[np.ndarray] = []
        self._model: IsolationForest | None = None
        self._additions_since_retrain: int = 0

    def bootstrap(self, seed_embeddings: list[np.ndarray]) -> None:
        """Load initial golden-set embeddings and fit the first model."""
        if len(seed_embeddings) < 10:
            raise ValueError(
                f"Need at least 10 seed embeddings for a stable baseline, "
                f"got {len(seed_embeddings)}."
            )
        self._baseline_embeddings = [np.array(e, dtype=np.float32) for e in seed_embeddings]
        self._fit()
        logger.info("DriftDetector bootstrapped with %d embeddings.", len(seed_embeddings))

    def score(self, embedding: np.ndarray) -> float:
        """
        Return the IsolationForest decision_function score.
        Positive = inlier, negative = outlier.
        Returns 0.0 if model not yet fitted (safe default during warm-up).
        """
        if self._model is None:
            return 0.0
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        return float(self._model.decision_function(vec)[0])

    def maybe_update_baseline(self, embedding: np.ndarray, log_tier: LogTier) -> bool:
        """
        Refinement 1: Add borderline-accepted outputs to the baseline training set.
        Triggers a model retrain every N additions to avoid stale detectors.
        Returns True if the embedding was added.
        """
        if log_tier != LogTier.BORDERLINE:
            return False

        self._baseline_embeddings.append(np.array(embedding, dtype=np.float32))
        self._additions_since_retrain += 1

        if self._additions_since_retrain >= self.config.baseline_retrain_every_n:
            self._fit()
            self._additions_since_retrain = 0
            logger.info(
                "DriftDetector retrained. Baseline size: %d",
                len(self._baseline_embeddings)
            )
        return True

    def _fit(self) -> None:
        matrix = np.stack(self._baseline_embeddings)
        self._model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(matrix)


# ---------------------------------------------------------------------------
# Lifecycle Manager (Refinement 3: Warm-up Gate)
# ---------------------------------------------------------------------------

@dataclass
class LifecycleState:
    artifact_id: str
    status: PromotionStatus = PromotionStatus.TEST
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    promoted_at: datetime | None = None

    def is_warm_up_complete(self, threshold: int) -> bool:
        return self.success_count >= threshold

    def record_success(self) -> None:
        self.success_count += 1

    def record_failure(self) -> None:
        self.failure_count += 1

    def try_promote_to_shadow(self, threshold: int) -> bool:
        """
        Attempt promotion from TEST → SHADOW.
        Blocked until warm-up threshold is met (Refinement 3).
        """
        if self.status != PromotionStatus.TEST:
            return False
        if not self.is_warm_up_complete(threshold):
            logger.debug(
                "Promotion blocked: %d/%d successes for artifact %s.",
                self.success_count, threshold, self.artifact_id
            )
            return False
        self.status = PromotionStatus.SHADOW
        self.promoted_at = datetime.now(timezone.utc)
        logger.info(
            "Artifact %s promoted to SHADOW after %d successes.",
            self.artifact_id, self.success_count
        )
        return True


# ---------------------------------------------------------------------------
# Tiered Evaluation Funnel
# ---------------------------------------------------------------------------

class TieredEvaluator:
    """
    Three-tier evaluation funnel:
      Tier 1 — Schema validation  (cheap, synchronous)
      Tier 2 — Embedding drift    (medium, local model)
      Tier 3 — LLM Judge          (expensive, throttled)
    """

    def __init__(
        self,
        config: SandboxConfig,
        drift_detector: DriftDetector,
        lifecycle: LifecycleState,
    ):
        self.config = config
        self.detector = drift_detector
        self.lifecycle = lifecycle

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(self, ctx: EvaluationContext) -> EvaluationContext:
        """Run the full funnel and mutate ctx with results. Returns ctx."""

        # Tier 1: Schema
        ctx = self._tier1_schema(ctx)
        if not ctx.tier1_passed:
            return self._finalize_rejection(ctx, RejectionReason.SCHEMA_MISMATCH)

        # Tier 2: Embedding drift
        ctx = self._tier2_drift(ctx)
        if ctx.rejection_reason == RejectionReason.SEMANTIC_DRIFT:
            return self._finalize_rejection(ctx, RejectionReason.SEMANTIC_DRIFT)

        # Tier 3: LLM judge (throttled)
        ctx = self._tier3_judge(ctx)
        if ctx.rejection_reason == RejectionReason.JUDGE_REJECTION:
            return self._finalize_rejection(ctx, RejectionReason.JUDGE_REJECTION)

        # All tiers passed → success path
        return self._finalize_success(ctx)

    # ------------------------------------------------------------------
    # Tier 1 — Schema Validation
    # ------------------------------------------------------------------

    def _tier1_schema(self, ctx: EvaluationContext) -> EvaluationContext:
        """
        Compare output schema fingerprint against the artifact's locked fingerprint.
        In production this would parse and validate against the actual JSON schema.
        """
        expected = ctx.artifact.schema_fingerprint if ctx.artifact else ""
        ctx.tier1_passed = (ctx.output_schema_fingerprint == expected)
        if not ctx.tier1_passed:
            logger.debug("run=%s Tier1 FAIL: schema mismatch.", ctx.run_id)
        return ctx

    # ------------------------------------------------------------------
    # Tier 2 — Embedding Drift Detection
    # ------------------------------------------------------------------

    def _tier2_drift(self, ctx: EvaluationContext) -> EvaluationContext:
        if ctx.output_embedding is None or ctx.artifact is None:
            # No embedding available; skip drift check conservatively
            ctx.tier2_score = 0.0
            ctx.tier2_anomaly_score = 0.0
            return ctx

        # Cosine similarity vs. the artifact's anchor embedding
        similarity = 1.0 - cosine(ctx.output_embedding, ctx.artifact.embedding_anchor)
        ctx.tier2_score = similarity

        # IsolationForest anomaly score
        anomaly = self.detector.score(ctx.output_embedding)
        ctx.tier2_anomaly_score = anomaly

        failed_similarity = similarity < self.config.cosine_similarity_floor
        failed_anomaly = anomaly < self.config.anomaly_score_floor

        if failed_similarity or failed_anomaly:
            ctx.rejection_reason = RejectionReason.SEMANTIC_DRIFT
            logger.debug(
                "run=%s Tier2 FAIL: cosine=%.3f anomaly=%.3f",
                ctx.run_id, similarity, anomaly
            )

        return ctx

    # ------------------------------------------------------------------
    # Tier 3 — LLM Judge (throttled)
    # ------------------------------------------------------------------

    def _tier3_judge(self, ctx: EvaluationContext) -> EvaluationContext:
        """
        Invoke the LLM judge only at the configured sample rate.
        Skipping the judge is treated as a tentative pass, not a failure.
        """
        if np.random.random() > self.config.judge_invocation_rate:
            logger.debug("run=%s Tier3 SKIPPED (throttle).", ctx.run_id)
            return ctx

        score = self._call_llm_judge(ctx)
        ctx.tier3_judge_score = score

        if score < 0.70:   # Configurable threshold
            ctx.rejection_reason = RejectionReason.JUDGE_REJECTION
            logger.debug("run=%s Tier3 FAIL: judge_score=%.3f", ctx.run_id, score)

        return ctx

    def _call_llm_judge(self, ctx: EvaluationContext) -> float:
        """
        Stub — replace with real LLM judge call.
        Returns a quality score in [0.0, 1.0].
        """
        raise NotImplementedError(
            "Implement _call_llm_judge with your LLM client. "
            "Expected return: float in [0.0, 1.0]."
        )

    # ------------------------------------------------------------------
    # Finalisation helpers
    # ------------------------------------------------------------------

    def _finalize_rejection(
        self, ctx: EvaluationContext, reason: RejectionReason
    ) -> EvaluationContext:
        ctx.rejection_reason = reason
        ctx.log_tier = LogTier.FULL   # Always log failures in full
        self.lifecycle.record_failure()
        logger.info("run=%s REJECTED reason=%s", ctx.run_id, reason.value)
        return ctx

    def _finalize_success(self, ctx: EvaluationContext) -> EvaluationContext:
        self.lifecycle.record_success()

        # Refinement 2: Stratified success logging
        ctx.log_tier = self._classify_success_log_tier(ctx)

        # Refinement 1: Feed borderline successes back to drift detector
        if ctx.output_embedding is not None:
            ctx.added_to_baseline = self.detector.maybe_update_baseline(
                ctx.output_embedding, ctx.log_tier
            )

        # Refinement 3: Attempt warm-up promotion
        self.lifecycle.try_promote_to_shadow(self.config.min_success_threshold)

        logger.info(
            "run=%s ACCEPTED log_tier=%s added_to_baseline=%s",
            ctx.run_id, ctx.log_tier.value, ctx.added_to_baseline
        )
        return ctx

    def _classify_success_log_tier(self, ctx: EvaluationContext) -> LogTier:
        """
        Refinement 2: Distinguish routine vs. borderline successes.
        Borderline = passed but close to the rejection threshold.
        Routine successes are sub-sampled; borderline ones are always kept.
        """
        if ctx.tier2_score is not None:
            near_cosine_floor = (
                ctx.tier2_score
                < self.config.cosine_similarity_floor + self.config.borderline_band
            )
        else:
            near_cosine_floor = False

        if ctx.tier2_anomaly_score is not None:
            near_anomaly_floor = (
                ctx.tier2_anomaly_score
                < self.config.anomaly_score_floor + self.config.borderline_band
            )
        else:
            near_anomaly_floor = False

        if near_cosine_floor or near_anomaly_floor:
            return LogTier.BORDERLINE   # Always kept + fed to baseline

        # Routine success: sub-sample
        if np.random.random() < self.config.routine_success_sample_rate:
            return LogTier.SAMPLED

        return LogTier.SKIPPED


# ---------------------------------------------------------------------------
# Factory — wire all components together
# ---------------------------------------------------------------------------

def build_sandbox(
    artifact: PromptArtifact,
    delta: PromptDelta,
    seed_embeddings: list[np.ndarray],
    config: SandboxConfig | None = None,
) -> tuple[TieredEvaluator, LifecycleState]:
    """
    Convenience factory.
    Returns a ready-to-use (evaluator, lifecycle) pair.
    """
    cfg = config or SandboxConfig()

    detector = DriftDetector(cfg)
    detector.bootstrap(seed_embeddings)

    lifecycle = LifecycleState(artifact_id=artifact.artifact_id)

    evaluator = TieredEvaluator(
        config=cfg,
        drift_detector=detector,
        lifecycle=lifecycle,
    )
    return evaluator, lifecycle
