"""批量查询并缓存 QLever Freebase 属性值。"""
from __future__ import annotations

import csv
import hashlib
import json
import socket
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import anyio
import httpx2

from kgqa.retrieve.cwq_attribute_audit import SampleInput
from kgqa.retrieve.cwq_attribute_evidence import (
    build_candidate_constrained_query,
    build_sparql_query,
)

_FREEBASE_NS = "http://rdf.freebase.com/ns/"


@dataclass(frozen=True, slots=True)
class QueryJob:
    chain: tuple[str, ...]
    mids: tuple[str, ...]
    cache_path: Path


@dataclass(frozen=True, slots=True)
class StrictQueryJob:
    sample_index: int
    query: str
    cache_path: Path


@dataclass(frozen=True, slots=True)
class RelationQueryJob:
    query: str
    cache_path: Path


class _OfficialInput(TypedDict):
    ID: str
    sparql: str


@dataclass(frozen=True, slots=True)
class OfficialAlignmentError(Exception):
    sample_index: int

    def __str__(self) -> str:
        return f"官方 CWQ 与比较清单在 sample_index={self.sample_index} 不一致"


def build_query_jobs(samples: list[SampleInput], cache_dir: Path) -> tuple[QueryJob, ...]:
    mids_by_chain: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
    for sample in samples:
        chain = tuple(sample["attribute_chain"])
        for candidate in sample["candidates"]:
            mids_by_chain[chain].update(candidate["mids"])
    jobs: list[QueryJob] = []
    for chain, mids in sorted(mids_by_chain.items()):
        digest = hashlib.sha256("\t".join(chain).encode()).hexdigest()[:16]
        jobs.append(QueryJob(chain, tuple(sorted(mids)), cache_dir / f"{digest}.tsv"))
    return tuple(jobs)


def build_strict_query_jobs(
    samples: list[SampleInput],
    official_test: Path,
    cache_dir: Path,
) -> tuple[StrictQueryJob, ...]:
    official: list[_OfficialInput] = json.loads(official_test.read_text(encoding="utf-8"))
    jobs: list[StrictQueryJob] = []
    for sample in samples:
        sample_index = sample["sample_index"]
        record = official[sample_index]
        if record["ID"] != sample["id"]:
            raise OfficialAlignmentError(sample_index)
        mids = tuple(sorted({mid for candidate in sample["candidates"] for mid in candidate["mids"]}))
        query = build_candidate_constrained_query(record["sparql"], mids)
        jobs.append(StrictQueryJob(sample_index, query, cache_dir / f"{sample_index:04d}.tsv"))
    return tuple(jobs)


async def _request_started(request: httpx2.Request) -> None:
    request.extensions["started_at"] = time.perf_counter()


async def _response_received(response: httpx2.Response) -> None:
    started_at = response.request.extensions["started_at"]
    elapsed = time.perf_counter() - started_at
    print(f"QLever {response.status_code} {elapsed:.2f}s", file=sys.stderr)


async def _fetch_query(
    client: httpx2.AsyncClient,
    limiter: anyio.CapacityLimiter,
    cache_path: Path,
    query: str,
) -> None:
    if cache_path.exists():
        return
    async with limiter:
        response = await client.post(
            "",
            content=query,
            headers={"Accept": "text/tab-separated-values", "Content-Type": "application/sparql-query"},
        )
        response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")


def _create_client(endpoint: str) -> httpx2.AsyncClient:
    limits = httpx2.Limits(max_connections=8, max_keepalive_connections=4, keepalive_expiry=30.0)
    timeout = httpx2.Timeout(connect=5.0, read=120.0, write=30.0, pool=10.0)
    transport = httpx2.AsyncHTTPTransport(
        http2=True,
        retries=3,
        limits=limits,
        socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
    )
    hooks = {"request": [_request_started], "response": [_response_received]}
    return httpx2.AsyncClient(
        transport=transport,
        timeout=timeout,
        base_url=endpoint,
        event_hooks=hooks,
        follow_redirects=True,
    )


async def fetch_jobs(endpoint: str, jobs: tuple[QueryJob, ...]) -> None:
    async with _create_client(endpoint) as client:
        limiter = anyio.CapacityLimiter(2)
        async with anyio.create_task_group() as task_group:
            for job in jobs:
                query = build_sparql_query(job.chain, job.mids)
                task_group.start_soon(_fetch_query, client, limiter, job.cache_path, query)


async def fetch_strict_jobs(endpoint: str, jobs: tuple[StrictQueryJob, ...]) -> None:
    async with _create_client(endpoint) as client:
        limiter = anyio.CapacityLimiter(2)
        async with anyio.create_task_group() as task_group:
            for job in jobs:
                task_group.start_soon(_fetch_query, client, limiter, job.cache_path, job.query)


async def fetch_relation_jobs(endpoint: str, jobs: tuple[RelationQueryJob, ...]) -> None:
    async with _create_client(endpoint) as client:
        limiter = anyio.CapacityLimiter(2)
        async with anyio.create_task_group() as task_group:
            for job in jobs:
                task_group.start_soon(_fetch_query, client, limiter, job.cache_path, job.query)


def _parse_cached_values(job: QueryJob) -> dict[str, tuple[str, ...]]:
    values: defaultdict[str, list[str]] = defaultdict(list)
    with job.cache_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            subject = row["?subject"].removeprefix(f"<{_FREEBASE_NS}").removesuffix(">")
            values[subject].append(row["?value"])
    return {mid: tuple(dict.fromkeys(raw_values)) for mid, raw_values in values.items()}


def load_values_by_chain(jobs: tuple[QueryJob, ...]) -> dict[tuple[str, ...], dict[str, tuple[str, ...]]]:
    return {job.chain: _parse_cached_values(job) for job in jobs}


def load_strict_selections(jobs: tuple[StrictQueryJob, ...]) -> dict[int, tuple[str, ...]]:
    selections: dict[int, tuple[str, ...]] = {}
    for job in jobs:
        mids: list[str] = []
        with job.cache_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                mid = row["?x"].removeprefix(f"<{_FREEBASE_NS}").removesuffix(">")
                mids.append(mid)
        selections[job.sample_index] = tuple(dict.fromkeys(mids))
    return selections


def load_relation_support(jobs: tuple[RelationQueryJob, ...]) -> dict[int, frozenset[str]]:
    support: defaultdict[int, set[str]] = defaultdict(set)
    for job in jobs:
        with job.cache_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sample_index = int(row["?sample"].strip('"'))
                candidate_mid = row["?candidate"].removeprefix(f"<{_FREEBASE_NS}").removesuffix(">")
                support[sample_index].add(candidate_mid)
    return {sample_index: frozenset(mids) for sample_index, mids in support.items()}
