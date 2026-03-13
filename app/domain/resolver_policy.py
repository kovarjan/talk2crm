from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings


@dataclass(frozen=True)
class ResolverPolicy:
    read_confidence_threshold: float
    mutation_confidence_threshold: float
    ambiguity_gap_threshold: float
    strict_mutation_confirmation: bool



def load_resolver_policy() -> ResolverPolicy:
    settings = get_settings()
    if not settings.resolver_v2_active:
        return ResolverPolicy(
            read_confidence_threshold=0.0,
            mutation_confidence_threshold=0.0,
            ambiguity_gap_threshold=0.0,
            strict_mutation_confirmation=False,
        )
    return ResolverPolicy(
        read_confidence_threshold=float(settings.resolver_read_confidence_threshold),
        mutation_confidence_threshold=float(settings.resolver_mutation_confidence_threshold),
        ambiguity_gap_threshold=float(settings.resolver_ambiguity_gap_threshold),
        strict_mutation_confirmation=bool(settings.resolver_strict_mutation_confirmation),
    )
