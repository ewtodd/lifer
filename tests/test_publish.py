"""publish.sort_date: the site orders by this, never by the displayed date
(as text "8/UNK/2021" outranks every ISO date)."""

from lifer.publish import sort_date


def test_iso_dates_pass_through():
    assert sort_date("2026-10-18") == "2026-10-18"


def test_lifelist_dates_become_iso():
    assert sort_date("11/12/21") == "2021-11-12"
    assert sort_date("7/4/2021") == "2021-07-04"


def test_partly_unknown_dates_keep_what_is_known():
    assert sort_date("8/UNK/2021") == "2021-08"
    assert sort_date("UNK/UNK/2021") == "2021"
    # a partial date still sorts before the days of its month, after earlier months
    assert "2021-07-31" < sort_date("8/UNK/2021") < "2021-08-01"


def test_unknown_is_empty():
    assert sort_date("UNK") == ""
    assert sort_date("") == ""
    assert sort_date("TM") == ""
