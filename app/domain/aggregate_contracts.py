from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, model_validator


AggregateModule = Literal[
    "invoices",
    "invoice_items",
    "quotes",
    "quote_items",
    "opportunities",
]

AggregateOperation = Literal["sum", "avg", "min", "max", "count"]

AggregateMetric = Literal[
    "amount",
    "amount_total",
    "amount_without_vat",
    "vat_amount",
    "quantity",
    "unit_price",
    "discount",
    "margin",
]

AggregateGroupBy = Literal["month", "year", "status", "account", "assigned_user"]

DateField = Literal["date_issued", "date_paid", "due_date", "date_entered", "date_modified"]


class AggregateFilters(BaseModel):
    account_id: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    date_field: DateField = "date_issued"
    statuses: list[str] | None = None
    exclude_cancelled: bool = True
    assigned_user_id: str | None = None


class AggregateRequest(BaseModel):
    module: AggregateModule
    operation: AggregateOperation
    metric: AggregateMetric = "amount"
    filters: AggregateFilters = Field(default_factory=AggregateFilters)
    group_by: AggregateGroupBy | None = None

    @model_validator(mode="after")
    def count_does_not_need_metric(self) -> "AggregateRequest":
        return self


class AggregateResult(BaseModel):
    success: bool
    operation: AggregateOperation
    metric: AggregateMetric
    module: AggregateModule
    value: Decimal | None = None
    currency: str = "CZK"
    record_count: int
    assumptions: list[str] = Field(default_factory=list)
    error: str | None = None

    def to_agent_string(self) -> str:
        if not self.success:
            return f"Chyba při výpočtu: {self.error}"
        if self.record_count == 0:
            return "Žádné záznamy neodpovídají zadaným filtrům."

        op_labels = {
            "sum": "Součet",
            "avg": "Průměr",
            "min": "Minimum",
            "max": "Maximum",
            "count": "Počet",
        }
        label = op_labels.get(self.operation, self.operation)

        if self.operation == "count":
            result_str = f"{label}: {self.record_count} záznamů"
        else:
            formatted = f"{self.value:,.2f}".replace(",", " ").replace(".", ",")
            result_str = f"{label}: {formatted} {self.currency}"

        assumptions_str = ""
        if self.assumptions:
            assumptions_str = " (" + "; ".join(self.assumptions) + ")"

        return f"{result_str} — z {self.record_count} záznamů modulu {self.module}{assumptions_str}"


class AggregateGroup(BaseModel):
    group: str
    value: Decimal | None
    record_count: int


class AggregateBreakdownResult(BaseModel):
    success: bool
    operation: AggregateOperation
    metric: AggregateMetric
    module: AggregateModule
    groups: list[AggregateGroup] = Field(default_factory=list)
    currency: str = "CZK"
    assumptions: list[str] = Field(default_factory=list)
    error: str | None = None
