from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from app.engine.rag import TenantRAGService

TENANT_ID = os.getenv("RAG_BENCH_TENANT_ID", "ai-local")
LIMIT = 5

_CEMAT_IDS = (
    "151cd173-c17e-3622-9fe1-5f2ac1d7ed6b",
    "73c979c2-0efd-eb5c-551a-62847462aebc",
    "c92c789a-ad03-4858-8a1e-fa16c6001787",
)
_TESCAN_IDS = (
    "ed4e5c66-3c98-8b7f-7961-524d6749513a",
    "4ab27509-cb19-487d-3b8c-64e3303521c9",
)
_SEKYROVA_IDS = ("753e8ad7-e827-cae2-39fd-55bb17146044",)
_PREDESLY_IDS = ("8ae00617-a02d-19e1-2edd-6026b14e2046",)
_PETVALDSKY_IDS = ("f134dbd7-91ea-c49d-66c9-5f3e69da202c",)


@dataclass(frozen=True)
class BenchmarkCase:
    layer: Literal["ranking", "command"]
    category: str
    query: str
    expected_module: str | None
    expected_name: str | None
    expected_ids: tuple[str, ...] = ()
    required: bool = True
    intent: str = "entity_lookup"
    notes: str = ""


@dataclass
class Candidate:
    record_id: str
    module: str
    name: str
    score: float


@dataclass
class CaseResult:
    case: BenchmarkCase
    candidates: list[Candidate]
    top1: Candidate | None
    top1_match: bool
    top3_match: bool
    wrong_module_top1: bool
    nonsense_top1: bool
    passed: bool


def _norm(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(TenantRAGService._strip_diacritics(text).split())


def _matches_expected(candidate: Candidate, case: BenchmarkCase) -> bool:
    if case.expected_name is None or case.expected_module is None:
        return False
    if candidate.module != case.expected_module:
        return False
    if case.expected_ids and candidate.record_id in case.expected_ids:
        return True
    return _norm(candidate.name) == _norm(case.expected_name)


def _to_candidates(results: list[dict[str, Any]]) -> list[Candidate]:
    out: list[Candidate] = []
    for result in results:
        payload = result.get("payload") or {}
        record = payload.get("record") or {}
        out.append(
            Candidate(
                record_id=str(payload.get("record_id") or ""),
                module=str(payload.get("module") or ""),
                name=str(payload.get("name") or record.get("name") or ""),
                score=float(result.get("_entity_score") or result.get("score") or 0.0),
            )
        )
    return out


def _evaluate_case(case: BenchmarkCase, candidates: list[Candidate]) -> CaseResult:
    top1 = candidates[0] if candidates else None

    if case.expected_name is None or case.expected_module is None:
        nonsense_top1 = top1 is not None
        return CaseResult(
            case=case,
            candidates=candidates,
            top1=top1,
            top1_match=False,
            top3_match=False,
            wrong_module_top1=False,
            nonsense_top1=nonsense_top1,
            passed=not nonsense_top1,
        )

    top1_match = bool(top1 and _matches_expected(top1, case))
    top3_match = any(_matches_expected(candidate, case) for candidate in candidates[:3])
    wrong_module_top1 = bool(top1 and top1.module and top1.module != case.expected_module)
    return CaseResult(
        case=case,
        candidates=candidates,
        top1=top1,
        top1_match=top1_match,
        top3_match=top3_match,
        wrong_module_top1=wrong_module_top1,
        nonsense_top1=False,
        passed=top1_match,
    )


def _run_cases(rag_service: TenantRAGService, cases: list[BenchmarkCase]) -> list[CaseResult]:
    evaluated: list[CaseResult] = []
    for case in cases:
        entity_type = {
            "Contacts": "contact",
            "Accounts": "account",
        }.get(case.expected_module or "", "any")
        raw_results = rag_service.search_entities(
            tenant_id=TENANT_ID,
            query=case.query,
            entity_type=entity_type,
            limit=LIMIT,
        )
        candidates = _to_candidates(raw_results)
        evaluated.append(_evaluate_case(case, candidates))
    return evaluated


def _rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _summarize(results: list[CaseResult]) -> dict[str, float]:
    required_results = [result for result in results if result.case.required]
    optional_results = [result for result in results if not result.case.required]
    positive_results = [result for result in results if result.case.expected_name is not None]
    negative_results = [result for result in results if result.case.expected_name is None]
    wrong_top1_scores = [
        float(result.top1.score)
        for result in results
        if result.top1 is not None
        and (
            (result.case.expected_name is not None and not result.top1_match)
            or (result.case.expected_name is None and result.nonsense_top1)
        )
    ]

    return {
        "top1": _rate([result.top1_match for result in positive_results]),
        "top3": _rate([result.top3_match for result in positive_results]),
        "wrong_module_rate": _rate([result.wrong_module_top1 for result in positive_results]),
        "nonsense_top1_rate": _rate([result.nonsense_top1 for result in negative_results]),
        "no_match_false_positive_rate": _rate([result.nonsense_top1 for result in negative_results]),
        "avg_wrong_answer_confidence": (
            sum(wrong_top1_scores) / len(wrong_top1_scores)
            if wrong_top1_scores
            else 0.0
        ),
        "required_pass_rate": _rate([result.passed for result in required_results]),
        "optional_pass_rate": _rate([result.passed for result in optional_results]),
    }


def _summary_by_category(results: list[CaseResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.case.category].append(result)

    table: dict[str, dict[str, float]] = {}
    for category, items in grouped.items():
        positive = [item for item in items if item.case.expected_name is not None]
        table[category] = {
            "top1": _rate([item.top1_match for item in positive]),
            "top3": _rate([item.top3_match for item in positive]),
            "wrong_module_rate": _rate([item.wrong_module_top1 for item in positive]),
        }
    return table


def _format_report(results: list[CaseResult]) -> str:
    summary = _summarize(results)
    per_category = _summary_by_category(results)
    lines = [
        "RAG benchmark summary",
        (
            f"Top1={summary['top1']:.0%}, Top3={summary['top3']:.0%}, "
            f"Wrong-module={summary['wrong_module_rate']:.0%}, "
            f"No-match-FP={summary['no_match_false_positive_rate']:.0%}, "
            f"Avg-wrong-score={summary['avg_wrong_answer_confidence']:.1f}, "
            f"Required={summary['required_pass_rate']:.0%}, "
            f"Optional={summary['optional_pass_rate']:.0%}"
        ),
    ]
    for category in sorted(per_category):
        row = per_category[category]
        lines.append(
            f"- {category}: top1={row['top1']:.0%}, top3={row['top3']:.0%}, wrong-module={row['wrong_module_rate']:.0%}"
        )
    return "\n".join(lines)


BENCHMARK_CASES: list[BenchmarkCase] = [
    # --- Přesný název ---
    BenchmarkCase("ranking", "Přesný název", "ČEMAT trading, spol. s r.o.", "Accounts", "ČEMAT trading, spol. s r.o.", _CEMAT_IDS),
    BenchmarkCase("ranking", "Přesný název", "TESCAN GROUP, a.s.", "Accounts", "TESCAN GROUP, a.s.", _TESCAN_IDS),
    BenchmarkCase("ranking", "Přesný název", "Ilona Sekyrová", "Contacts", "Ilona Sekyrová", _SEKYROVA_IDS),
    BenchmarkCase("ranking", "Přesný název", "Libor Předešlý", "Contacts", "Libor Předešlý", _PREDESLY_IDS),
    BenchmarkCase("ranking", "Přesný název", "Karel Pětvaldský", "Contacts", "Karel Pětvaldský", _PETVALDSKY_IDS),
    # --- Bez diakritiky ---
    BenchmarkCase("ranking", "Bez diakritiky", "cemat trading", "Accounts", "ČEMAT trading, spol. s r.o.", _CEMAT_IDS),
    BenchmarkCase("ranking", "Bez diakritiky", "tescan group a.s.", "Accounts", "TESCAN GROUP, a.s.", _TESCAN_IDS),
    BenchmarkCase("ranking", "Bez diakritiky", "ilona sekyrova", "Contacts", "Ilona Sekyrová", _SEKYROVA_IDS),
    BenchmarkCase("ranking", "Bez diakritiky", "libor predesly", "Contacts", "Libor Předešlý", _PREDESLY_IDS),
    BenchmarkCase("ranking", "Bez diakritiky", "karel petvaldsky", "Contacts", "Karel Pětvaldský", _PETVALDSKY_IDS),
    # --- Skloňovaný tvar ---
    BenchmarkCase("ranking", "Skloňovaný tvar", "v čematu", "Accounts", "ČEMAT trading, spol. s r.o.", _CEMAT_IDS),
    BenchmarkCase("ranking", "Skloňovaný tvar", "do tescanu", "Accounts", "TESCAN GROUP, a.s.", _TESCAN_IDS),
    BenchmarkCase("ranking", "Skloňovaný tvar", "ilony sekyrové", "Contacts", "Ilona Sekyrová", _SEKYROVA_IDS),
    BenchmarkCase("ranking", "Skloňovaný tvar", "libora předešlého", "Contacts", "Libor Předešlý", _PREDESLY_IDS),
    BenchmarkCase("ranking", "Skloňovaný tvar", "karla pětvaldského", "Contacts", "Karel Pětvaldský", _PETVALDSKY_IDS),
    # --- Pouze příjmení ---
    BenchmarkCase("ranking", "Pouze příjmení", "sekyrová", "Contacts", "Ilona Sekyrová", _SEKYROVA_IDS),
    BenchmarkCase("ranking", "Pouze příjmení", "předešlý", "Contacts", "Libor Předešlý", _PREDESLY_IDS),
    BenchmarkCase("ranking", "Pouze příjmení", "pětvaldský", "Contacts", "Karel Pětvaldský", _PETVALDSKY_IDS),
    BenchmarkCase("ranking", "Pouze příjmení bez diakritiky", "petvaldsky", "Contacts", "Karel Pětvaldský", _PETVALDSKY_IDS),
    # --- Příjmení ve skloňovaném tvaru ---
    BenchmarkCase("ranking", "Příjmení ve skloňovaném tvaru", "petvaldským", "Contacts", "Karel Pětvaldský", _PETVALDSKY_IDS),
    BenchmarkCase("ranking", "Příjmení ve skloňovaném tvaru", "předešlým", "Contacts", "Libor Předešlý", _PREDESLY_IDS),
    BenchmarkCase("ranking", "Příjmení ve skloňovaném tvaru", "sekyrovou", "Contacts", "Ilona Sekyrová", _SEKYROVA_IDS),
    # --- Tituly / oslovení ---
    BenchmarkCase("ranking", "Tituly / oslovení", "pan pětvaldský", "Contacts", "Karel Pětvaldský", _PETVALDSKY_IDS),
    BenchmarkCase("ranking", "Tituly / oslovení", "panem petvaldským", "Contacts", "Karel Pětvaldský", _PETVALDSKY_IDS),
    BenchmarkCase("ranking", "Tituly / oslovení", "s panem předešlým", "Contacts", "Libor Předešlý", _PREDESLY_IDS),
    BenchmarkCase("ranking", "Tituly / oslovení", "s paní sekyrovou", "Contacts", "Ilona Sekyrová", _SEKYROVA_IDS),
    # --- CRM příkaz - schůzka s kontaktem ---
    BenchmarkCase("command", "CRM příkaz - schůzka s kontaktem", "naplánuj schůzku s panem petvaldským v úterý", "Contacts", "Karel Pětvaldský", _PETVALDSKY_IDS, intent="create_meeting"),
    BenchmarkCase("command", "CRM příkaz - schůzka s kontaktem", "domluv meeting s karlem petvaldskym na utery", "Contacts", "Karel Pětvaldský", _PETVALDSKY_IDS, intent="create_meeting"),
    BenchmarkCase("command", "CRM příkaz - schůzka s kontaktem", "vytvoř schůzku s liborem předešlým zítra", "Contacts", "Libor Předešlý", _PREDESLY_IDS, intent="create_meeting"),
    BenchmarkCase("command", "CRM příkaz - schůzka s kontaktem", "naplánuj call s ilonou sekyrovou", "Contacts", "Ilona Sekyrová", _SEKYROVA_IDS, intent="create_meeting"),
    # --- CRM příkaz - schůzka s firmou ---
    BenchmarkCase("command", "CRM příkaz - schůzka s firmou", "naplánuj schůzku v čematu příští týden", "Accounts", "ČEMAT trading, spol. s r.o.", _CEMAT_IDS, intent="create_meeting"),
    BenchmarkCase("command", "CRM příkaz - schůzka s firmou", "domluv meeting do tescanu", "Accounts", "TESCAN GROUP, a.s.", _TESCAN_IDS, intent="create_meeting"),
    # --- Ambiguous query ---
    BenchmarkCase("ranking", "Ambiguous query", "tescan", "Accounts", "TESCAN GROUP, a.s.", _TESCAN_IDS, required=False),
    BenchmarkCase("ranking", "Ambiguous query", "čemat", "Accounts", "ČEMAT trading, spol. s r.o.", _CEMAT_IDS, required=False),
    # --- Překlep / STT chyba ---
    BenchmarkCase("ranking", "Překlep / STT chyba", "teskan group", "Accounts", "TESCAN GROUP, a.s.", _TESCAN_IDS),
    BenchmarkCase("ranking", "Překlep / STT chyba", "ilona sekirova", "Contacts", "Ilona Sekyrová", _SEKYROVA_IDS),
    BenchmarkCase("ranking", "Překlep / STT chyba", "karel petvaldski", "Contacts", "Karel Pětvaldský", _PETVALDSKY_IDS, required=False),
    BenchmarkCase("ranking", "Překlep / STT chyba", "šemat trejding", "Accounts", "ČEMAT trading, spol. s r.o.", _CEMAT_IDS, required=False),
    BenchmarkCase("ranking", "Překlep / STT chyba", "libor přeďešlý", "Contacts", "Libor Předešlý", _PREDESLY_IDS, required=False),
    # --- Negativní případ ---
    BenchmarkCase("command", "Negativní případ", "naplánuj schůzku s člověkem který neexistuje", None, None, intent="create_meeting"),
    BenchmarkCase("command", "Negativní případ", "schůzka s panem xyzabc", None, None, intent="create_meeting"),
]

RANKING_CASES = [case for case in BENCHMARK_CASES if case.layer == "ranking"]
COMMAND_CASES = [case for case in BENCHMARK_CASES if case.layer == "command"]


def _require_live_benchmark() -> None:
    if os.getenv("RUN_RAG_BENCHMARK") != "1":
        pytest.skip("Set RUN_RAG_BENCHMARK=1 to run live RAG benchmark tests.")


def _benchmark_service() -> TenantRAGService:
    service = TenantRAGService()
    total = service.count(tenant_id=TENANT_ID)
    if total <= 0:
        pytest.skip(f"Tenant {TENANT_ID!r} has no indexed records.")
    return service


def test_strip_diacritics_czech_chars():
    assert TenantRAGService._strip_diacritics("Předešlý") == "predesly"
    assert TenantRAGService._strip_diacritics("ČEMAT trading") == "cemat trading"
    assert TenantRAGService._strip_diacritics("Ilona Sekyrová") == "ilona sekyrova"
    assert TenantRAGService._strip_diacritics("Karel Pětvaldský") == "karel petvaldsky"


def test_strip_diacritics_ascii_passthrough():
    assert TenantRAGService._strip_diacritics("tescan group") == "tescan group"
    assert TenantRAGService._strip_diacritics("ACMARK s.r.o.") == "acmark s.r.o."


def test_strip_diacritics_is_used_to_build_text_ascii():
    record_text = '{"name": "Libor Předešlý", "account_name": "ČEMAT trading, spol. s r.o."}'
    result = TenantRAGService._strip_diacritics(record_text)
    assert "predesly" in result
    assert "cemat" in result
    assert "ě" not in result
    assert "š" not in result


def _make_text_result(module: str, name: str) -> dict:
    return {"score": 1.0, "payload": {"module": module, "record": {"name": name}}}


def test_sort_text_results_accounts_before_meetings():
    results = [
        _make_text_result("Meetings", "TESCAN schůzka"),
        _make_text_result("Accounts", "TESCAN GROUP, a.s."),
    ]
    out = TenantRAGService._sort_text_results(results, "TESCAN")
    assert out[0]["payload"]["module"] == "Accounts"
    assert out[1]["payload"]["module"] == "Meetings"


def test_sort_text_results_contacts_before_meetings():
    results = [
        _make_text_result("Meetings", "Předešlý - schůzka"),
        _make_text_result("Contacts", "Libor Předešlý"),
    ]
    out = TenantRAGService._sort_text_results(results, "Libor Předešlý")
    assert out[0]["payload"]["module"] == "Contacts"


def test_sort_text_results_name_match_first_within_module():
    results = [
        _make_text_result("Accounts", "Jiná firma"),
        _make_text_result("Accounts", "TESCAN GROUP, a.s."),
    ]
    out = TenantRAGService._sort_text_results(results, "tescan")
    assert out[0]["payload"]["record"]["name"] == "TESCAN GROUP, a.s."


def test_sort_text_results_stable_for_equal_priority():
    results = [
        _make_text_result("Accounts", "Alpha s.r.o."),
        _make_text_result("Accounts", "Beta s.r.o."),
    ]
    out = TenantRAGService._sort_text_results(results, "gamma")
    assert out[0]["payload"]["record"]["name"] == "Alpha s.r.o."
    assert out[1]["payload"]["record"]["name"] == "Beta s.r.o."


def test_strip_diacritics_detects_ascii_query():
    ascii_query = "libor predesly"
    assert TenantRAGService._strip_diacritics(ascii_query) == ascii_query

    diacritics_query = "Libor Předešlý"
    assert TenantRAGService._strip_diacritics(diacritics_query) != diacritics_query.lower()


def test_clean_entity_query_removes_meeting_words():
    assert (
        TenantRAGService._clean_entity_query("naplánuj schůzku s panem petvaldským v úterý")
        == "petvaldskym"
    )


def test_entity_stopwords_can_be_extended():
    stopwords = TenantRAGService._build_entity_stopwords("prosím, laskavě")
    assert TenantRAGService._clean_entity_query("prosím najdi laskavě čemat", stopwords=stopwords) == "najdi cemat"


def test_entity_query_variants_include_czech_stem():
    variants = TenantRAGService._entity_query_variants("panem petvaldským")
    assert "petvaldsky" in variants


def test_record_search_name_prefers_full_contact_name():
    assert (
        TenantRAGService._record_search_name({"first_name": "Karel", "last_name": "Pětvaldský"})
        == "Karel Pětvaldský"
    )


def test_entity_score_prefers_contact_name_over_meeting_text():
    results = [
        _make_text_result("Meetings", "Pětvaldský - schůzka"),
        _make_text_result("Contacts", "Karel Pětvaldský"),
    ]
    out = TenantRAGService._sort_entity_results(
        results,
        "panem petvaldským",
        preferred_modules=["Contacts"],
    )
    assert out[0]["payload"]["module"] == "Contacts"


def test_entity_score_returns_match_debug():
    result = _make_text_result("Contacts", "Karel Pětvaldský")
    score, debug = TenantRAGService._score_entity_result_with_debug(
        result,
        "panem petvaldským",
        preferred_modules=["Contacts"],
    )
    assert score > 0
    assert debug["clean_query"] == "panem petvaldskym"
    assert debug["name"] == "Karel Pětvaldský"
    assert debug["module_bonus"] > 0
    assert debug["name_fuzzy"] > 0
    assert debug["token_overlap"] > 0


def test_benchmark_contains_requested_categories():
    categories = {case.category for case in BENCHMARK_CASES}
    assert "CRM příkaz - schůzka s kontaktem" in categories
    assert "CRM příkaz - schůzka s firmou" in categories
    assert "Pouze příjmení" in categories
    assert "Příjmení ve skloňovaném tvaru" in categories
    assert "Tituly / oslovení" in categories
    assert "Ambiguous query" in categories
    assert "Negativní případ" in categories


def test_benchmark_has_required_and_optional_cases():
    assert any(case.required for case in BENCHMARK_CASES)
    assert any(not case.required for case in BENCHMARK_CASES)


def test_benchmark_checks_expected_module_for_positive_cases():
    for case in BENCHMARK_CASES:
        if case.expected_name is None:
            assert case.expected_module is None
        else:
            assert case.expected_module in {"Contacts", "Accounts"}


def test_ranking_benchmark_live():
    _require_live_benchmark()
    service = _benchmark_service()
    results = _run_cases(service, RANKING_CASES)
    summary = _summarize(results)

    # Baseline sanity gates for required ranking cases.
    assert summary["required_pass_rate"] >= 0.50, _format_report(results)
    assert summary["wrong_module_rate"] <= 0.35, _format_report(results)

    if os.getenv("ENFORCE_RAG_RELEASE_GATES") == "1":
        assert summary["required_pass_rate"] >= 0.85, _format_report(results)
        assert summary["top3"] >= 0.95, _format_report(results)
        assert summary["wrong_module_rate"] <= 0.05, _format_report(results)


def test_real_command_benchmark_live():
    _require_live_benchmark()
    service = _benchmark_service()
    results = _run_cases(service, COMMAND_CASES)
    summary = _summarize(results)

    # Baseline sanity gates for CRM command queries.
    assert summary["required_pass_rate"] >= 0.40, _format_report(results)
    assert summary["wrong_module_rate"] <= 0.45, _format_report(results)

    if os.getenv("ENFORCE_RAG_RELEASE_GATES") == "1":
        assert summary["required_pass_rate"] >= 0.80, _format_report(results)
        assert summary["nonsense_top1_rate"] <= 0.10, _format_report(results)
