"""执行带缓存、超时和显式截断状态的无标签 QLever 查询。"""

import csv
import hashlib
import io
import re
from pathlib import Path
from time import perf_counter, sleep

import requests

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.interfaces import KnowledgeSource
from kgqa.agent.workflow.models import QueryResult
from kgqa.agent.workflow.persistence import ArtifactStore

NS = "http://rdf.freebase.com/ns/"
MID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
REL_RE = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+$")


def parse_term(value: str) -> str:
    raw = value.strip()
    if raw.startswith(f"<{NS}") and raw.endswith(">"):
        return raw[len(NS) + 1:-1]
    if raw.startswith('"'):
        end = raw.rfind('"')
        return raw[1:end] if end > 0 else raw
    if re.search(r"@[a-z]{2,3}(?:-[A-Za-z0-9]+)?$", raw):
        return raw.rsplit("@", 1)[0]
    return raw


class QLeverClient(KnowledgeSource):
    def __init__(self, config: WorkflowConfig, cache_dir: Path,
                 endpoint: str = "https://qlever.dev/api/freebase") -> None:
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint
        self.session = requests.Session()
        self.events = []

    def query_key(self, query: str) -> str:
        return hashlib.sha256((self.endpoint + "\n" + query).encode()).hexdigest()

    def query(self, query: str, limit: int) -> QueryResult:
        started = perf_counter()
        key = self.query_key(query)
        cache_path = self.cache_dir / f"{key}.tsv"
        cached = cache_path.exists()
        error = ""
        try:
            text = cache_path.read_text() if cached else self._request(query)
            records = list(csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE))
            rows = tuple({k.lstrip("?"): parse_term(v or "") for k, v in row.items()} for row in records[:limit])
            if not cached:
                ArtifactStore.atomic_text(cache_path, text)
                ArtifactStore.atomic_text(cache_path.with_suffix(".rq"), query)
            status = "truncated" if len(records) > limit else ("ok" if rows else "empty")
        except (requests.RequestException, OSError, ValueError, TypeError, AttributeError) as exc:
            rows, status, error = (), "failed", f"{type(exc).__name__}: {exc}"[:400]
        result = QueryResult(status, rows, key, cached, (perf_counter() - started) * 1000, error)
        self.events.append(result)
        return result

    def _request(self, query: str) -> str:
        for attempt in range(self.config.max_query_retries + 1):
            try:
                response = self.session.post(
                    self.endpoint, data=query.encode(), timeout=self.config.query_timeout,
                    headers={"Accept": "text/tab-separated-values", "Content-Type": "application/sparql-query"},
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                if attempt >= self.config.max_query_retries:
                    raise
                if getattr(exc.response, "status_code", 0) in (400, 401, 403, 404):
                    raise
                sleep(min(0.25 * 2**attempt, 1.0))
        raise RuntimeError("查询重试未返回结果")

    def attributes(self, mids: list[str], chain: tuple[str, ...] | list[str]) -> QueryResult:
        subjects = sorted({mid for mid in mids if MID_RE.fullmatch(mid)})
        if not subjects or not chain or any(not REL_RE.fullmatch(p) for p in chain):
            return QueryResult("failed", error="invalid_mid_or_chain")
        values = " ".join(f"ns:{mid}" for mid in subjects)
        variables = ["?mid", *[f"?v{i}" for i in range(len(chain) - 1)], "?value"]
        triples = " ".join(f"{variables[i]} ns:{p} {variables[i + 1]} ." for i, p in enumerate(chain))
        limit = max(100, len(subjects) * 50)
        query = (f"PREFIX ns: <{NS}> SELECT DISTINCT ?mid ?value WHERE {{ VALUES ?mid {{ {values} }} "
                 f"{triples} }} ORDER BY ?mid ?value LIMIT {limit + 1}")
        return self.query(query, limit)

    def intersect_topic_attribute(self, topics: list[str] | tuple[str, ...],
                                  chain: tuple[str, ...] | list[str],
                                  target_value: str) -> QueryResult:
        subjects = sorted({t for t in topics if MID_RE.fullmatch(t)})
        if not subjects or not chain or any(not REL_RE.fullmatch(p) for p in chain):
            return QueryResult("failed", error="invalid_topic_or_chain")
        topic_clauses = " UNION ".join(f"{{ ns:{t} ?p ?cand . }} UNION {{ ?cand ?p ns:{t} . }}" for t in subjects)
        variables = ["?cand", *[f"?v{i}" for i in range(len(chain) - 1)], f'"{target_value}"']
        triples = " ".join(f"{variables[i]} ns:{p} {variables[i + 1]} ." for i, p in enumerate(chain))
        limit = 10
        query = (f"PREFIX ns: <{NS}> SELECT DISTINCT ?cand WHERE {{ "
                 f"{{ {topic_clauses} }} {triples} "
                 f"FILTER(STRSTARTS(STR(?cand), \"{NS}m.\")) }} ORDER BY ?cand LIMIT {limit + 1}")
        return self.query(query, limit)


    def adjacency(self, frontier: list[str], step: dict) -> QueryResult:
        mids = sorted({mid for mid in frontier if MID_RE.fullmatch(mid)})
        relation = step["predicate"]
        if not mids or not REL_RE.fullmatch(relation):
            return QueryResult("failed", error="invalid_adjacency_request")
        values = " ".join("ns:" + mid for mid in mids)
        head, tail = ("?target", "?source") if step["reverse"] else ("?source", "?target")
        query = (f"PREFIX ns: <{NS}> SELECT DISTINCT ?source ?target WHERE {{ VALUES ?source {{ {values} }} "
                 f"{head} ns:{relation} {tail} . FILTER(STRSTARTS(STR(?target), \"{NS}m.\") || "
                 f"STRSTARTS(STR(?target), \"{NS}g.\")) }} ORDER BY ?source ?target "
                 f"LIMIT {self.config.step_limit + 1}")
        return self.query(query, self.config.step_limit)

    def labels(self, mids: list[str]) -> QueryResult:
        subjects = sorted({mid for mid in mids if MID_RE.fullmatch(mid)})
        if not subjects:
            return QueryResult("empty")
        values = " ".join("ns:" + mid for mid in subjects)
        query = (f"PREFIX ns: <{NS}> SELECT DISTINCT ?mid ?name WHERE {{ VALUES ?mid {{ {values} }} "
                 "?mid ns:type.object.name ?name . FILTER(lang(?name) = 'en') } ORDER BY ?mid ?name")
        return self.query(query, max(len(subjects) * 20, 100))
