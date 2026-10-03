from dataclasses import dataclass


@dataclass
class EvidenceDecision:
    sufficient: bool
    score: float
    threshold: float
    reason: str


class EvidenceGate:
    """Determine whether retrieved evidence is sufficient."""

    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold

    def evaluate(self, results) -> EvidenceDecision:
        """Evaluate the strongest retrieved evidence."""

        if not results:
            return EvidenceDecision(
                sufficient=False,
                score=0.0,
                threshold=self.threshold,
                reason="No evidence was retrieved.",
            )

        best_score = max(result.score for result in results)

        if best_score < self.threshold:
            return EvidenceDecision(
                sufficient=False,
                score=best_score,
                threshold=self.threshold,
                reason=(
                    "The retrieved evidence does not "
                    "meet the minimum similarity threshold."
                ),
            )

        return EvidenceDecision(
            sufficient=True,
            score=best_score,
            threshold=self.threshold,
            reason=("At least one retrieved chunk meets the evidence threshold."),
        )
