"""Probability calibration, fitted on dev and carried inside the artefact.

A discriminative model answers "which class is more likely". It does not answer
"how likely", and the gap between those two questions is where every downstream
decision goes wrong. A logistic regression fitted on a small, class-imbalanced
split routinely emits 0.99 for something that turns out to be bona fide about
half the time, and a risk band keyed on that number is then wrong in a way no
amount of good EER will reveal.

So the scores are calibrated, and the ordering of this pipeline is what makes the
result trustworthy:

1. Fit the model on train.
2. Fit the calibrator on **dev** scores against **dev** labels.
3. Select the operating point on the *calibrated* dev scores.
4. Apply calibrator and threshold to test, unchanged.

Test is never seen by step 2 or 3. That is the whole point, and it is why
calibration cannot be bolted on afterwards as a convenience.

Two properties are worth stating because they are easy to get wrong:

* **Calibration is monotone, so it cannot change EER or AUC.** Platt scaling and
  isotonic regression both preserve score ordering. What calibration changes is
  the *meaning* of a probability: Brier score, expected calibration error, the
  reliability bins, and the false-positive rate implied by any fixed threshold.
  Anyone reporting a calibration-induced change in EER has made an error.
* **A calibrator fitted on too few samples is worse than none.** Isotonic
  regression will happily fit an exact step function through four points and
  then be confidently wrong everywhere between them. Both implementations
  therefore refuse to fit a dev split that cannot support them, rather than
  emitting a curve that looks rigorous and is not.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from voxshield.training.config import CALIBRATION_METHODS, TrainingError

#: Methods a calibrator must provide to be usable in the runner and the artefact.
CALIBRATION_FORMAT_VERSION = 1

#: Below this many dev samples, isotonic regression is refused. A step function
#: fitted to fewer points than it has knots is memorisation, not calibration.
MIN_ISOTONIC_SAMPLES = 100

#: Below this many dev samples per class, either calibrator is refused.
MIN_CALIBRATION_SAMPLES_PER_CLASS = 5

# Re-exported so callers have one obvious import site for the vocabulary.
__all__ = [
    "CALIBRATION_FORMAT_VERSION",
    "CALIBRATION_METHODS",
    "MIN_CALIBRATION_SAMPLES_PER_CLASS",
    "MIN_ISOTONIC_SAMPLES",
    "Calibrator",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "build_calibrator",
    "calibrator_from_dict",
    "fit_calibrator",
]


@runtime_checkable
class Calibrator(Protocol):
    """Maps raw model scores to calibrated spoof probabilities.

    Implementations are fitted on one split and applied to another, so ``fit``
    and ``transform`` must not share state beyond the fitted parameters.
    """

    method: str

    def fit(self, labels: np.ndarray, scores: np.ndarray) -> Calibrator:
        """Fit on one split.

        Args:
            labels: Binary labels, ``1`` meaning spoof.
            scores: Raw model scores from the same rows.

        Returns:
            ``self``, for chaining.
        """

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities.

        Args:
            scores: Raw model scores.

        Returns:
            Probabilities in ``[0, 1]``, same length as the input.
        """

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready description of the fitted parameters.

        Returns:
            A mapping that :func:`calibrator_from_dict` can rebuild.
        """


def _validate_fit_inputs(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Check that a dev split can support calibration at all.

    Args:
        labels: Binary labels.
        scores: Raw scores.

    Returns:
        The arrays as float64/int64.

    Raises:
        TrainingError: The split is too small, single-classed, or mismatched in
            length. Every one of these would produce a calibrator that cannot be
            checked against anything.
    """
    y = np.asarray(labels, dtype=np.int64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.size != s.size:
        msg = f"labels and scores differ in length: {y.size} vs {s.size}"
        raise TrainingError(msg)
    if y.size == 0:
        msg = "cannot calibrate on an empty split"
        raise TrainingError(msg)
    counts = np.bincount(y, minlength=2)
    if counts[0] == 0 or counts[1] == 0:
        present = int(np.argmax(counts))
        msg = f"calibration needs both classes in the dev split, found only label {present}"
        raise TrainingError(msg)
    smallest = int(counts.min())
    if smallest < MIN_CALIBRATION_SAMPLES_PER_CLASS:
        msg = (
            f"calibration needs at least {MIN_CALIBRATION_SAMPLES_PER_CLASS} dev samples "
            f"per class, found {smallest}. A calibrator fitted on fewer is not "
            "calibration, so none is fitted."
        )
        raise TrainingError(msg)
    return y, s


class PlattCalibrator:
    """Logistic regression mapping the model's log-odds to a probability.

    This is the default because it is hard to get wrong: two parameters, fitted
    by convex optimisation, and no way to memorise a training split. It cannot
    represent a non-monotone mapping, which is the correct assumption here --
    a score that ranked a spoof above a bona fide should keep doing so.

    The model's raw score is passed through the logit before the fit, so the
    linear model operates on the scale where a relationship is actually linear.
    Operating directly on a probability that is already squashed near 0 or 1
    produces a badly conditioned fit and a visibly worse curve.
    """

    method = "platt"

    def __init__(self) -> None:
        """Create an unfitted calibrator."""
        self._coef: float = 1.0
        self._intercept: float = 0.0
        self._fitted = False

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has run.

        Returns:
            ``True`` once parameters are present.
        """
        return self._fitted

    @staticmethod
    def _logit(scores: np.ndarray) -> np.ndarray:
        """Map scores to log-odds, clipping so the logit stays finite.

        Args:
            scores: Raw scores in ``[0, 1]``.

        Returns:
            Log-odds, finite for every input.
        """
        clipped = np.clip(np.asarray(scores, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        return np.log(clipped / (1.0 - clipped))

    def fit(self, labels: np.ndarray, scores: np.ndarray) -> PlattCalibrator:
        """Fit the two-parameter logistic map on one split.

        Args:
            labels: Binary labels, ``1`` meaning spoof.
            scores: Raw model scores from the same rows.

        Returns:
            ``self``.

        Raises:
            TrainingError: The split cannot support calibration.
        """
        y, s = _validate_fit_inputs(labels, scores)
        z = self._logit(s)

        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        estimator.fit(z.reshape(-1, 1), y)
        self._coef = float(estimator.coef_[0][0])
        self._intercept = float(estimator.intercept_[0])
        self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities.

        Args:
            scores: Raw model scores.

        Returns:
            Probabilities in ``[0, 1]``.

        Raises:
            TrainingError: The calibrator is unfitted.
        """
        if not self._fitted:
            msg = "PlattCalibrator.transform called before fit"
            raise TrainingError(msg)
        z = self._logit(scores) * self._coef + self._intercept
        # A numerically stable logistic, so extreme logits saturate rather than
        # overflowing to inf and then to a NaN probability.
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-np.clip(z, 0, 700))),
            np.exp(np.clip(z, -700, 0)) / (1.0 + np.exp(np.clip(z, -700, 0))),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready description of the fitted parameters.

        Returns:
            The method name and both coefficients.
        """
        return {
            "method": self.method,
            "coefficient": self._coef,
            "intercept": self._intercept,
            "fitted": self._fitted,
        }


class IsotonicCalibrator:
    """Monotone step interpolation, for models whose scores are badly scaled.

    More flexible than Platt and correspondingly more dangerous: it will fit an
    exact step function through a small split and be confidently wrong in every
    gap. It is therefore refused below
    :data:`MIN_ISOTONIC_SAMPLES` dev samples, which is the whole reason this
    class is not the default.
    """

    method = "isotonic"

    def __init__(self) -> None:
        """Create an unfitted calibrator."""
        self._x: tuple[float, ...] = ()
        self._y: tuple[float, ...] = ()
        self._fitted = False

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has run.

        Returns:
            ``True`` once knots are present.
        """
        return self._fitted

    def fit(self, labels: np.ndarray, scores: np.ndarray) -> IsotonicCalibrator:
        """Fit the monotone map on one split.

        Args:
            labels: Binary labels, ``1`` meaning spoof.
            scores: Raw model scores from the same rows.

        Returns:
            ``self``.

        Raises:
            TrainingError: The split is too small or single-classed.
        """
        y, s = _validate_fit_inputs(labels, scores)
        if y.size < MIN_ISOTONIC_SAMPLES:
            msg = (
                f"isotonic calibration needs at least {MIN_ISOTONIC_SAMPLES} dev "
                f"samples to be more than memorisation, found {y.size}. Use the "
                "platt method, which cannot overfit in this way."
            )
            raise TrainingError(msg)

        from sklearn.isotonic import IsotonicRegression

        estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        estimator.fit(s, y)
        self._x = tuple(float(v) for v in estimator.X_thresholds_)
        self._y = tuple(float(v) for v in estimator.y_thresholds_)
        self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities.

        Args:
            scores: Raw model scores.

        Returns:
            Probabilities in ``[0, 1]``.

        Raises:
            TrainingError: The calibrator is unfitted.
        """
        if not self._fitted:
            msg = "IsotonicCalibrator.transform called before fit"
            raise TrainingError(msg)
        if len(self._x) < 2:
            # A degenerate fit: everything gets the base rate. Returning a
            # constant is honest, where extrapolating would not be.
            return np.full(np.shape(scores), self._y[0] if self._y else 0.5, dtype=np.float64)
        return np.interp(np.asarray(scores, dtype=np.float64), self._x, self._y)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready description of the fitted knots.

        Returns:
            The method name and the interpolation knots.
        """
        return {
            "method": self.method,
            "x": list(self._x),
            "y": list(self._y),
            "fitted": self._fitted,
        }


class IdentityCalibrator:
    """The no-op calibrator, used when calibration is disabled.

    Exists so the rest of the pipeline has one code path. An ``if calibrator is
    None`` check scattered through the runner is how a later edit ends up
    forgetting to calibrate test.
    """

    method = "none"

    @property
    def fitted(self) -> bool:
        """Always ``True``: this calibrator has nothing to fit.

        Returns:
            ``True``.
        """
        return True

    def fit(self, labels: np.ndarray, scores: np.ndarray) -> IdentityCalibrator:
        """Accept the split and change nothing.

        Args:
            labels: Binary labels. Unused.
            scores: Raw scores. Unused.

        Returns:
            ``self``.
        """
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Return the scores unchanged.

        Args:
            scores: Raw model scores.

        Returns:
            The same values, as float64.
        """
        return np.asarray(scores, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready description.

        Returns:
            The method name only.
        """
        return {"method": self.method, "fitted": True}


def build_calibrator(method: str) -> Calibrator:
    """Construct an unfitted calibrator by name.

    Args:
        method: One of :data:`CALIBRATION_METHODS`.

    Returns:
        An unfitted calibrator.

    Raises:
        TrainingError: The name is not recognised.
    """
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    if method == "none":
        return IdentityCalibrator()
    msg = f"calibration method must be one of {CALIBRATION_METHODS}, got {method!r}"
    raise TrainingError(msg)


def fit_calibrator(method: str, labels: np.ndarray, scores: np.ndarray) -> tuple[Calibrator, str]:
    """Fit a calibrator on dev, degrading to identity rather than failing the run.

    A calibrator that cannot be fitted is not a reason to discard a trained model.
    The metrics that do not depend on calibration, notably EER and AUC, are still
    worth reporting, so the run continues uncalibrated and says so in the reason.
    What it must never do is quietly present uncalibrated scores as calibrated.

    Args:
        method: Requested method.
        labels: Dev labels.
        scores: Dev scores.

    Returns:
        The fitted calibrator and a note explaining any degradation.
    """
    if method == "none":
        return IdentityCalibrator(), "calibration disabled by configuration"
    try:
        calibrator = build_calibrator(method).fit(labels, scores)
    except TrainingError as exc:
        return IdentityCalibrator(), f"calibration fell back to none: {exc}"
    return calibrator, f"calibrated with {method} fitted on the dev split"


def calibrator_from_dict(payload: dict[str, Any] | None) -> Calibrator:
    """Rebuild a fitted calibrator from its serialised form.

    Args:
        payload: The mapping produced by ``to_dict``, or ``None``.

    Returns:
        A fitted calibrator. An absent or unknown payload yields an
        :class:`IdentityCalibrator`, so an artefact written before calibration
        existed still loads rather than failing at read time.

    Raises:
        TrainingError: The payload names a calibrator whose parameters are
            unusable.
    """
    if not payload:
        return IdentityCalibrator()
    method = str(payload.get("method", "none"))
    if method == "platt":
        return _rehydrate_platt(payload)
    if method == "isotonic":
        return _rehydrate_isotonic(payload)
    return build_calibrator(method)


def _rehydrate_platt(payload: dict[str, Any]) -> PlattCalibrator:
    """Restore Platt coefficients written by :meth:`PlattCalibrator.to_dict`.

    Args:
        payload: The serialised parameters.

    Returns:
        A fitted calibrator.

    Raises:
        TrainingError: A parameter is missing or not a number.
    """
    try:
        coefficient = float(payload["coefficient"])
        intercept = float(payload["intercept"])
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"stored platt calibrator is unusable: {exc}"
        raise TrainingError(msg) from exc
    calibrator = PlattCalibrator()
    calibrator._coef = coefficient
    calibrator._intercept = intercept
    calibrator._fitted = bool(payload.get("fitted", True))
    return calibrator


def _rehydrate_isotonic(payload: dict[str, Any]) -> IsotonicCalibrator:
    """Restore isotonic knots written by :meth:`IsotonicCalibrator.to_dict`.

    Args:
        payload: The serialised parameters.

    Returns:
        A fitted calibrator.

    Raises:
        TrainingError: The knots are missing, ragged, or not numbers.
    """
    try:
        x = [float(v) for v in payload["x"]]
        y = [float(v) for v in payload["y"]]
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"stored isotonic calibrator is unusable: {exc}"
        raise TrainingError(msg) from exc
    if len(x) != len(y) or not x:
        msg = f"stored isotonic calibrator has {len(x)} knots for {len(y)} values"
        raise TrainingError(msg)
    calibrator = IsotonicCalibrator()
    calibrator._x = tuple(x)
    calibrator._y = tuple(y)
    calibrator._fitted = bool(payload.get("fitted", True))
    return calibrator
