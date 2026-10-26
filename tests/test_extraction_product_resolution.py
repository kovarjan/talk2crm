from __future__ import annotations

import asyncio
import json

from app.engine.extraction import _pick_product, resolve_product_rows

CATALOG = {
    "SE Xtreme Quick Elcon 16x6-8/4.33 RES 660": [{"id": "p1", "name": "SE Xtreme Quick Elcon 16x6-8/4.33 RES 660", "mft_part_num": "9.1124.14439"}],
    "kompresor": [{"id": "a", "name": "Kompresor A", "mft_part_num": ""}, {"id": "b", "name": "Kompresor B", "mft_part_num": ""}],
    "9.1124.14439": [{"id": "p1", "name": "SE Xtreme Quick Elcon 16x6-8/4.33 RES 660", "mft_part_num": "9.1124.14439"}],
}


async def lookup(query: str) -> str:
    return json.dumps({"status": "ok", "results": CATALOG.get(query, [])})


def run(rows):
    return asyncio.run(resolve_product_rows(rows, lookup))


def test_name_only_row_gets_template_id_so_the_form_can_price_it() -> None:
    rows = [{"name": "SE Xtreme Quick Elcon 16x6-8/4.33 RES 660", "quantity": 4}]
    assert run(rows) == []
    assert rows[0]["product_template_id"] == {"id": "p1", "name": "SE Xtreme Quick Elcon 16x6-8/4.33 RES 660"}
    assert rows[0]["quantity"] == 4


def test_row_found_by_part_number_in_the_relate_name() -> None:
    rows = [{"product_template_id": {"name": "9.1124.14439"}}]
    assert run(rows) == []
    assert rows[0]["product_template_id"]["id"] == "p1"
    assert rows[0]["name"].startswith("SE Xtreme")


def test_existing_id_is_never_overwritten_or_looked_up() -> None:
    rows = [{"name": "X", "product_template_id": {"id": "keep", "name": "X"}}]
    assert run(rows) == []
    assert rows[0]["product_template_id"]["id"] == "keep"


def test_ambiguous_or_unknown_product_stays_unlinked_and_is_reported() -> None:
    rows = [{"name": "kompresor"}, {"name": "neexistuje"}]
    assert run(rows) == ["kompresor", "neexistuje"]
    assert all("product_template_id" not in r for r in rows)


def test_lookup_failure_does_not_raise() -> None:
    async def broken(query: str) -> str:
        raise RuntimeError("down")

    rows = [{"name": "A"}]
    assert asyncio.run(resolve_product_rows(rows, broken)) == ["A"]


def test_pick_product_prefers_the_single_exact_match() -> None:
    results = [{"id": "1", "name": "Vrtačka"}, {"id": "2", "name": "Vrtačka XL"}]
    assert _pick_product("vrtacka", results)["id"] == "1"
    assert _pick_product("vrt", results) is None
