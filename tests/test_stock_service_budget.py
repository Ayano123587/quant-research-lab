"""Service-boundary checks run before reading any data or fitting a model."""

import math

import pytest

from quantlab.stocks import development


@pytest.mark.parametrize("deadline", [0.0, math.nan, math.inf, -math.inf])
def test_expired_or_invalid_budget_precedes_source_reads(monkeypatch, deadline):
    # Emulate only the runtime directory check, never data or model outcomes.
    monkeypatch.setattr(development.Path, "is_dir", lambda self: True)

    def forbidden(*args, **kwargs):
        pytest.fail("Budget rejection must precede source or outcome reads")

    monkeypatch.setattr(development, "load_spec", forbidden)
    with pytest.raises((TimeoutError, ValueError)):
        development.run_stock_development(
            None, None, None, None, None, service_deadline_monotonic=deadline
        )
