"""规范化数值、标注路径并保留实体身份和属性来源。"""

import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation

from kgqa.agent.workflow.models import RuntimeQuestion

NUMBER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w.])")
DATE_RE = re.compile(r"^(-?\d{4,})-(\d{2})(?:-(\d{2}))?(?:T(\d{2}):(\d{2}):(\d{2}))?Z?$")


class EvidenceFormatter:
    @staticmethod
    def numbers(question: str) -> list[str]:
        result = []
        for raw in NUMBER_RE.findall(question):
            value = raw.replace(",", "")
            if len(value) == 4 and value.isdigit() and int(value) < 1700:
                continue
            if value not in result:
                result.append(value)
        return result

    @staticmethod
    def comparable(value: str) -> tuple[str, Decimal] | None:
        date = DATE_RE.fullmatch(value)
        if date:
            year, month, day, hour, minute, second = [int(part or 0) for part in date.groups()]
            magnitude = abs(year) * 10**10 + month * 10**8 + day * 10**6 + hour * 10**4 + minute * 10**2 + second
            return "date", Decimal(-magnitude if year < 0 else magnitude)
        try:
            number = Decimal(value)
            return ("number", number) if number.is_finite() else None
        except InvalidOperation:
            return None

    @staticmethod
    def candidate_names(question: RuntimeQuestion) -> dict[str, str]:
        return {mid: item["name"] for item in question.candidates for mid in item.get("mids", [])}

    @staticmethod
    def annotate(paths: tuple[dict, ...], names: set[str], chain: tuple, value: str) -> tuple[dict, ...]:
        enriched = deepcopy(paths)
        label = chain[-1].split(".")[-1].replace("_", " ")
        for item in enriched:
            edges = item.get("path", [])
            if edges and edges[-1][2] in names:
                original = edges[-1][2]
                display = f"{original} ({label}: {value})"
                item["canonical_tail"] = original
                item["display_to_canonical"] = {display: original}
                edges[-1][2] = display
        return enriched
