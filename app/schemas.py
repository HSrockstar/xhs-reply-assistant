from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ModelRequest(StrictModel):
    provider: Literal["codex", "deepseek"] | None = None
    api_key: str = Field(default="", max_length=512, repr=False, exclude=True)
    request_id: str = Field(min_length=8, max_length=128)


class GenerateRequest(ModelRequest):
    question: str = Field(min_length=1, max_length=8000)
    background: str = Field(default="", max_length=8000)
    instructions: str = Field(default="", max_length=2000)
    length: str = Field(default="自然", max_length=100)
    scene: str = Field(default="auto", max_length=50)
    mode: Literal["full", "style", "base"] = "full"

    @model_validator(mode="after")
    def check_size(self):
        if not self.question.strip() or len(self.question) + len(self.background) > 8000:
            raise ValueError("问题不能为空，问题与背景合计最多8000字符")
        return self


class ReferenceUsage(StrictModel):
    example_id: str
    purpose: Literal["content", "style"]
    note: str


class Reply(StrictModel):
    reply_text: str = Field(min_length=1, max_length=12000)
    reference_usage: list[ReferenceUsage]
    supplement_summary: str
    clarification_needed: bool
    review_notes: list[str]


class Evidence(StrictModel):
    example_id: str
    quote: str = Field(min_length=1)


class Observation(StrictModel):
    description: str
    applicable_scenes: list[str]
    evidence_ids: list[str]
    evidence: list[Evidence]


class Strategy(StrictModel):
    observations: list[Observation]
    content_candidates: list[Observation]
    contradictions: list[Observation]
    boundaries: list[str]
    gaps: list[str]


class Rule(StrictModel):
    rule_id: str = Field(min_length=1, max_length=100)
    dimension: str = Field(min_length=1, max_length=50)
    description: str = Field(min_length=1, max_length=1500)
    applicable_scenes: list[str]
    evidence_ids: list[str]
    counterexample_ids: list[str]
    strength: Literal["核心倾向", "场景倾向", "待核实"]
    origin: Literal["observed", "user_defined"]
    enabled: bool


class Expression(StrictModel):
    rules: list[Rule]
    dimensions: list[Observation]
    cross_checks: list[str]
    gaps: list[str]


class ObservationSelection(StrictModel):
    description: str
    applicable_scenes: list[str]
    evidence_ids: list[str] = Field(min_length=1)


class StrategySelection(Strategy):
    observations: list[ObservationSelection]
    content_candidates: list[ObservationSelection]
    contradictions: list[ObservationSelection]


class ExpressionSelection(Expression):
    dimensions: list[ObservationSelection]


def selection_schema(schema, example_ids):
    """将本批依据 ID 写入结构化输出枚举，避免模型抄写长 ID 时出错。"""
    reference = Literal[tuple(example_ids)]
    observation = create_model(
        "ObservationSelection",
        __base__=ObservationSelection,
        evidence_ids=(list[reference], Field(min_length=1)),
    )
    if schema is StrategySelection:
        return create_model(
            "StrategySelection",
            __base__=StrategySelection,
            **{field: (list[observation], ...) for field in ("observations", "content_candidates", "contradictions")},
        )
    rule = create_model(
        "RuleSelection",
        __base__=Rule,
        evidence_ids=(list[reference], Field(min_length=1)),
        counterexample_ids=(list[reference], ...),
    )
    return create_model(
        "ExpressionSelection",
        __base__=ExpressionSelection,
        dimensions=(list[observation], ...),
        rules=(list[rule], ...),
    )


class OrganizeUnit(StrictModel):
    question_ids: list[str]
    context_ids: list[str]
    answer_ids: list[str]


class Organization(StrictModel):
    units: list[OrganizeUnit] = Field(min_length=1, max_length=100)


class ResolveRequest(Organization):
    expected_version: int = Field(ge=1)


class ExamplePatch(StrictModel):
    expected_version: int = Field(ge=1)
    question_text: str | None = Field(default=None, max_length=12000)
    context_text: str | None = Field(default=None, max_length=20000)
    answer_text: str | None = Field(default=None, max_length=20000)
    content_enabled: bool | None = None
    style_enabled: bool | None = None
    status: Literal["ready", "needs_review", "excluded"] | None = None
    topic_tags: list[str] | None = None
    scene_tags: list[str] | None = None


class StyleEdit(StrictModel):
    base_id: str | None = None
    expected_version: int | None = None
    rules: list[Rule] = Field(max_length=40)
    activate: bool = False


class FeedbackRequest(StrictModel):
    job_id: str
    final_text: str = Field(min_length=1, max_length=20000)
    rating: int | None = Field(default=None, ge=1, le=5)
    reasons: str = Field(default="", max_length=4000)
    adopt: bool = False
    request_id: str = Field(min_length=8, max_length=128)


def output_schema(model: type[BaseModel]) -> dict:
    """Codex structured output要求所有属性为required。"""
    schema = model.model_json_schema()

    def walk(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return schema
