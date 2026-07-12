"""统一检索适配器注册表。"""
from kgqa.retrieve.datasets.base import DatasetAdapter
from kgqa.retrieve.datasets.cwq import CWQAdapter
from kgqa.retrieve.datasets.metaqa import MetaQAAdapter
from kgqa.retrieve.datasets.rearev_webqsp import ReaRevWebQSPAdapter
from kgqa.retrieve.datasets.webqsp import WebQSPAdapter

_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "webqsp": WebQSPAdapter,
    "metaqa": MetaQAAdapter,
    "cwq": CWQAdapter,
    "webqsp-rearev": ReaRevWebQSPAdapter,
}


def register_adapter(name: str, cls: type[DatasetAdapter]) -> None:
    _REGISTRY[name] = cls


def get_adapter(name: str, **kwargs) -> DatasetAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"未注册的数据集适配器: {name}（已注册: {sorted(_REGISTRY)}）")
    return _REGISTRY[name](**kwargs)
