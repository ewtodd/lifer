"""Unit tests for lifer.lifelist: UK -> UNK date normalisation on import."""
from pathlib import Path

from lifer.lifelist import normalize_date, parse_lifelist


def test_standalone_uk_becomes_unk():
    assert normalize_date("UK") == "UNK"
    assert normalize_date("uk") == "UNK"
    assert normalize_date(" Uk ") == "UNK"


def test_uk_as_date_component_becomes_unk():
    assert normalize_date("7/UK/2021") == "7/UNK/2021"
    assert normalize_date("UK/8/2021") == "UNK/8/2021"
    assert normalize_date("7/uk/21") == "7/UNK/21"


def test_unk_and_full_dates_pass_through():
    assert normalize_date("UNK") == "UNK"
    assert normalize_date("7/UNK/2021") == "7/UNK/2021"
    assert normalize_date("7/14/2021") == "7/14/2021"
    assert normalize_date("2021-07-14") == "2021-07-14"


def test_words_containing_uk_are_untouched():
    # whole-component match only: "uk" as a substring is not rewritten
    assert normalize_date("7/UKS/2021") == "7/UKS/2021"
    assert normalize_date("ukulele") == "ukulele"
    assert normalize_date("7/14/2021UK") == "7/14/2021UK"
    assert normalize_date("U K") == "U K"


def test_empty_stays_empty():
    assert normalize_date("") == ""


def _make_ods(path: Path) -> None:
    """A minimal LIFELIST.ods with the four canonical columns."""
    from odf.opendocument import OpenDocumentSpreadsheet
    from odf.table import Table, TableCell, TableRow
    from odf.text import P

    doc = OpenDocumentSpreadsheet()
    table = Table(name="Sheet1")
    for values in (
        ("SPECIES", "DATE", "LOCATION", "DETAILS"),
        ("American Crow", "UK", "TM", ""),
        ("Brown Pelican", "7/uk/2021", "FL", ""),
        ("Canada Goose", "7/14/2021", "FL", ""),
    ):
        row = TableRow()
        for v in values:
            cell = TableCell()
            cell.addElement(P(text=v))
            row.addElement(cell)
        table.addElement(row)
    doc.spreadsheet.addElement(table)
    doc.save(str(path))


def test_parse_lifelist_normalises_dates(tmp_path):
    p = tmp_path / "LIFELIST.ods"
    _make_ods(p)
    rows = parse_lifelist(p)
    assert [r[1] for r in rows] == ["UNK", "7/UNK/2021", "7/14/2021"]
    # other columns untouched
    assert [r[0] for r in rows] == ["American Crow", "Brown Pelican",
                                    "Canada Goose"]
