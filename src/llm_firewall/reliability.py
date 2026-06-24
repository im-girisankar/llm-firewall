"""reliability.py — llm-reliability-kit bridge for OutputGuard.

Provides :func:`reliability_kit_score_fn`, a factory that returns a
:data:`~llm_firewall.guards.ScoreFn` backed by the real HRI faithfulness
scorer from ``llm-reliability-kit``.

Usage::

    from llm_firewall.guards import OutputGuard
    from llm_firewall.reliability import reliability_kit_score_fn

    guard = OutputGuard(score_fn=reliability_kit_score_fn())

``llm-reliability-kit`` is a **soft dependency**: importing this module does
NOT require it to be installed.  The import happens lazily inside the returned
``score_fn`` so that the rest of llm-firewall stays importable in environments
where the kit is absent.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_firewall.guards import ScoreFn


def reliability_kit_score_fn(*, use_nli: bool = False) -> ScoreFn:
    """Return a :data:`~llm_firewall.guards.ScoreFn` backed by ``llm-reliability-kit``.

    The returned callable computes hallucination **risk** as::

        risk = 1 - faithfulness_score(response, context)

    where ``faithfulness_score`` comes from
    ``llm_reliability_kit.faithfulness``.  A score of ``1.0`` means the
    response is entirely unsupported by the context; ``0.0`` means it is fully
    grounded.

    Parameters
    ----------
    use_nli:
        When ``True``, the NLI backend from ``llm_reliability_kit.nli`` is
        used to judge sentence-level support instead of the default lexical
        containment check.  Requires ``pip install 'llm-reliability-kit[nli]'``
        (i.e. ``sentence-transformers`` must be present).

    Returns
    -------
    ScoreFn
        ``score_fn(response, context) -> (risk, reasons)`` where *risk* is in
        ``[0.0, 1.0]`` and *reasons* is a list of sentences that were judged
        unsupported (when the kit exposes them) or a single summary string.

    Raises
    ------
    ImportError
        If ``llm-reliability-kit`` is not installed when the *returned*
        ``score_fn`` is first called.  Install with::

            pip install llm-reliability-kit

        For the NLI path also add the ``[nli]`` extra::

            pip install 'llm-reliability-kit[nli]'
    """

    def score_fn(response: str, context: str) -> tuple[float, list[str]]:
        # Lazy import — llm-reliability-kit is optional.
        try:
            from llm_reliability_kit.faithfulness import (  # type: ignore[import-not-found]
                faithfulness_score,
            )
        except ImportError as exc:
            raise ImportError(
                "reliability_kit_score_fn requires llm-reliability-kit. "
                "Install it with:  pip install llm-reliability-kit\n"
                "For NLI-based scoring also add the [nli] extra:  "
                "pip install 'llm-reliability-kit[nli]'"
            ) from exc

        support_fn: Callable[[str, str], bool] | None = None
        if use_nli:
            try:
                from llm_reliability_kit.nli import (  # type: ignore[import-not-found]
                    make_nli_support_fn,
                )
            except ImportError as exc:
                raise ImportError(
                    "use_nli=True requires sentence-transformers. "
                    "Install it with:  pip install 'llm-reliability-kit[nli]'"
                ) from exc
            support_fn = make_nli_support_fn()

        score = faithfulness_score(response, context, support_fn=support_fn)
        risk = 1.0 - score

        # Build reasons: name unsupported sentences by re-running sentence
        # splitting (the kit's split_sentences is stable and zero-dep).
        reasons: list[str] = []
        if risk > 0.0:
            try:
                from llm_reliability_kit.metrics import (  # type: ignore[import-not-found]
                    normalized_containment,
                    split_sentences,
                )

                sentences = split_sentences(response)
                for sentence in sentences:
                    # A sentence is unsupported when containment < default threshold.
                    if normalized_containment(sentence, context) < 0.25:
                        reasons.append(f"unsupported: {sentence}")
            except ImportError:
                # Fallback: single summary reason so callers always get something.
                reasons = [f"hallucination risk {risk:.2f} (llm-reliability-kit)"]

        return risk, reasons

    return score_fn
