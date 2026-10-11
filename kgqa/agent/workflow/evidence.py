"""规范化数值、标注路径并保留实体身份和属性来源。"""

import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation

from kgqa.agent.workflow.models import RuntimeQuestion

NUMBER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w.])")
DATE_RE = re.compile(r"^(-?\d{4,})-(\d{2})(?:-(\d{2}))?(?:T(\d{2}):(\d{2}):(\d{2}))?Z?$")
YEAR_OR_DATE_RE = re.compile(r"^(-?\d{4,})(?:-(\d{2})(?:-(\d{2}))?(?:T(\d{2}):(\d{2}):(\d{2}))?Z?)?$")


def is_date_chain(chain: list[str] | tuple[str, ...] | None) -> bool:
    """判断属性链是否属于日期与时间语义。"""
    if not chain:
        return False
    text = ".".join(chain).casefold()
    keywords = (
        "date", "founded", "opened", "established", "start", "end",
        "death", "birth", "release", "from", "to", "occurance",
    )
    return any(k in text for k in keywords)


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
    def comparable(
        value: str, chain: list[str] | tuple[str, ...] | None = None
    ) -> tuple[str, Decimal] | None:
        """将字符串属性值转换为可排序的数值元组 (类型标签, Decimal 数值)。"""
        if chain and is_date_chain(chain):
            matched = YEAR_OR_DATE_RE.fullmatch(value)
            if matched:
                parts = matched.groups()
                year = int(parts[0])
                month = int(parts[1] or 0)
                day = int(parts[2] or 0)
                hour = int(parts[3] or 0)
                minute = int(parts[4] or 0)
                second = int(parts[5] or 0)
                magnitude = abs(year) * 10**10 + month * 10**8 + day * 10**6 + hour * 10**4 + minute * 10**2 + second
                return "date", Decimal(-magnitude if year < 0 else magnitude)
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
