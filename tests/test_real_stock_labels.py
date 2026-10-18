"""Real labels test clocks only, including disclosed 2015/2017 source exposure.

No strategies, predictions, fitting, selection of recipes or performance claims.
"""

import os
from datetime import date, datetime
from pathlib import Path

import pytest

from quantlab.data.archive import SUPPLIED_ARCHIVE_SHA256, MarketArchive
from quantlab.data.calendar import XNYSSchedule
from quantlab.stocks.labels import (
    development_selection_rows,
    label_from_history,
    mature_training_rows,
    read_labels,
)

pytestmark = pytest.mark.real_data
ENDPOINT = date(2017, 11, 10)


@pytest.fixture(scope="module")
def inputs():
    path = Path(os.environ.get("QUANTLAB_ARCHIVE", "data/raw/archive.zip"))
    if not path.exists():
        pytest.skip("Supplied real archive absent; set QUANTLAB_ARCHIVE")
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as source:
        histories = {}
        for symbol in ("IBM", "GE", "TSLA", "DGLT"):
            bars = source.bars("stock", symbol)
            histories[bars[0].asset_id] = bars
    schedule = XNYSSchedule(date(2009, 1, 1), date(2018, 2, 28))
    return path, histories, schedule


def label(inputs, asset, signal):
    _, histories, schedule = inputs
    return label_from_history(asset, histories[asset], schedule, signal_session=signal,
                              archive_endpoint=ENDPOINT,
                              archive_sha256=SUPPLIED_ARCHIVE_SHA256)


def test_real_2010_opening_label_and_strict_availability_boundary(inputs):
    _, histories, schedule = inputs
    rows = tuple(label(inputs, asset, date(2010, 1, 29))
                 for asset in ("stock:ibm.us", "stock:ge.us"))
    for row in rows:
        indexed = {b.session: b for b in histories[row.asset_id]}
        assert row.status == "valid"
        assert row.entry_session == date(2010, 2, 1)
        assert row.exit_session == date(2010, 3, 1)
        assert row.return_value == pytest.approx(
            indexed[row.exit_session].open / indexed[row.entry_session].open - 1)
        assert row.entry_row == indexed[row.entry_session]
        assert row.exit_row_id == (row.exit_row.source_member, row.exit_row.row_number,
                                   row.exit_row.row_sha256)
        assert row.label_available_at == schedule.session(row.exit_session).decision_at
        assert row.label_available_at > row.exit_at
    before = mature_training_rows(rows, schedule.session(date(2010, 3, 1)).open_at)
    assert before.accepted == ()
    assert all(r.reason == "IMMATURE_LABEL" for r in before.exclusions)
    exact = mature_training_rows(rows, rows[0].label_available_at)
    assert len(exact.accepted) == 2 and not exact.exclusions
    with pytest.raises(ValueError, match="Timezone-aware"):
        mature_training_rows(rows, datetime(2010, 3, 1))


def test_real_2014_spilled_labels_excluded_by_development_information_cutoff(inputs):
    _, _, schedule = inputs
    signals = (date(2014, 10, 31), date(2014, 11, 28), date(2014, 12, 31))
    rows = tuple(label(inputs, asset, signal) for signal in signals
                 for asset in ("stock:ibm.us", "stock:ge.us"))
    cutoff = schedule.session(date(2014, 12, 31)).decision_at
    selected = development_selection_rows(rows, information_cutoff=cutoff,
                                          signal_start=date(2014, 1, 1),
                                          signal_end=date(2014, 12, 31))
    assert len(selected.accepted) == 2
    assert {r.signal_session for r in selected.accepted} == {signals[0]}
    assert len(selected.exclusions) == 4
    assert all(r.reason == "IMMATURE_LABEL" for r in selected.exclusions)
    nov = next(r for r in rows if r.signal_session == signals[1])
    dec = next(r for r in rows if r.signal_session == signals[2])
    assert nov.entry_session == date(2014, 12, 1) and nov.exit_session == date(2015, 1, 2)
    assert dec.entry_session == date(2015, 1, 2) and dec.exit_session == date(2015, 2, 2)
    restricted = development_selection_rows(rows, information_cutoff=cutoff,
                                            signal_start=date(2014, 11, 1),
                                            signal_end=date(2014, 12, 31))
    assert not restricted.accepted
    assert sum(r.reason == "OUTSIDE_SIGNAL_SCOPE" for r in restricted.exclusions) == 2


def test_actual_inception_and_terminal_absences_are_not_invented_outcomes(inputs):
    missing_entry = label(inputs, "stock:tsla.us", date(2010, 5, 28))
    assert missing_entry.status == "missing_entry"
    assert missing_entry.entry_row is None and missing_entry.exit_row is not None
    # DGLT's actual last source row is 2017-08-16. This tests missing-mark
    # handling; it asserts neither a delisting date nor any terminal return.
    missing_exit = label(inputs, "stock:dglt.us", date(2017, 7, 31))
    assert missing_exit.status == "missing_exit"
    assert missing_exit.entry_row is not None and missing_exit.exit_row is None
    censored = label(inputs, "stock:ibm.us", date(2017, 10, 31))
    assert censored.status == "right_censored"
    assert censored.entry_session == date(2017, 11, 1)
    assert censored.exit_session == date(2017, 12, 1)
    assert censored.entry_row is not None and censored.exit_row is None
    for row in (missing_entry, missing_exit, censored):
        assert row.return_value is None and row.label_available_at is None
    selected = mature_training_rows((missing_entry, missing_exit, censored),
                                    inputs[2].session(date(2018, 1, 2)).decision_at)
    assert not selected.accepted
    assert all(r.reason == "UNRESOLVED_LABEL" for r in selected.exclusions)


def test_authenticated_reader_preserves_keys_and_rejects_duplicate_training_labels(inputs):
    path, histories, schedule = inputs
    signal = date(2010, 1, 29)
    with MarketArchive(path, SUPPLIED_ARCHIVE_SHA256) as source:
        records = read_labels(source, ("IBM", "GE"), schedule, signal_sessions=(signal,),
                              archive_endpoint=ENDPOINT, archive_sha256=SUPPLIED_ARCHIVE_SHA256)
    assert records == tuple(sorted((label(inputs, asset, signal)
                                    for asset in ("stock:ibm.us", "stock:ge.us")),
                                   key=lambda r: (r.signal_session, r.asset_id)))
    def adapter(asset_type, symbol):
        return histories[asset_type + ":" + symbol + ".us"]

    assert records == read_labels(adapter, ("IBM", "GE"), schedule, signal_sessions=(signal,),
                                 archive_endpoint=ENDPOINT,
                                 archive_sha256=SUPPLIED_ARCHIVE_SHA256)
    with pytest.raises(ValueError, match="Duplicate training label key"):
        mature_training_rows(records + records, records[0].label_available_at)
