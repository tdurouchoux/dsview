import pytest

from dsview.mcp.server import _build_type_filter


def test_build_type_filter_none_types_returns_no_filter():
    assert _build_type_filter("content_type", None, ["Course", "Repository"]) is None


def test_build_type_filter_empty_list_matches_nothing():
    assert _build_type_filter("content_type", [], ["Course", "Repository"]) == ["1=0"]


def test_build_type_filter_builds_in_clause_for_known_types():
    filters = _build_type_filter(
        "content_type",
        ["Course", "Repository"],
        ["Course", "Repository", "Documentation"],
    )

    assert filters == ["content_type IN ('Course','Repository')"]


def test_build_type_filter_rejects_unknown_type():
    with pytest.raises(ValueError):
        _build_type_filter(
            "content_type", ["Not a real type"], ["Course", "Repository"]
        )


def test_build_type_filter_rejects_sql_injection_attempt():
    with pytest.raises(ValueError):
        _build_type_filter(
            "content_type",
            ["Course') OR 1=1--"],
            ["Course", "Repository"],
        )
