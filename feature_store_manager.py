"""
Feature Store Manager — manages feature registration, versioning,
and serving for online and offline ML training pipelines.
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import datetime


@dataclass
class FeatureDefinition:
    name: str
    dtype: str  # "float", "int", "string", "bool"
    source_table: str
    transformation: Optional[str] = None  # SQL or Python expression
    ttl_days: int = 30
    tags: list[str] = field(default_factory=list)
    version: int = 1


@dataclass
class FeatureSet:
    name: str
    entity_key: str  # e.g. "user_id", "product_id"
    features: list[FeatureDefinition]
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    description: str = ""


@dataclass
class FeatureVector:
    entity_id: str
    feature_set: str
    values: dict[str, object]
    served_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


class FeatureStoreManager:
    """
    Lightweight feature store: register feature sets, compute offline
    training datasets, and serve online feature vectors.
    """

    def __init__(self):
        self._registry: dict[str, FeatureSet] = {}
        self._online_store: dict[str, dict[str, FeatureVector]] = {}  # {feature_set: {entity_id: vector}}

    def register(self, feature_set: FeatureSet) -> None:
        key = f"{feature_set.name}:v{max(f.version for f in feature_set.features)}"
        self._registry[feature_set.name] = feature_set
        self._online_store.setdefault(feature_set.name, {})

    def list_feature_sets(self) -> list[str]:
        return list(self._registry.keys())

    def get_feature_set(self, name: str) -> Optional[FeatureSet]:
        return self._registry.get(name)

    def write_online(self, feature_set_name: str, entity_id: str, values: dict[str, object]) -> None:
        if feature_set_name not in self._registry:
            raise KeyError(f"Feature set '{feature_set_name}' not registered")
        self._online_store[feature_set_name][entity_id] = FeatureVector(
            entity_id=entity_id,
            feature_set=feature_set_name,
            values=values,
        )

    def read_online(self, feature_set_name: str, entity_id: str) -> Optional[FeatureVector]:
        return self._online_store.get(feature_set_name, {}).get(entity_id)

    def build_training_dataset(
        self,
        feature_set_name: str,
        records: list[dict],
        label_col: str,
    ) -> pd.DataFrame:
        """
        Assembles training dataset from a list of {entity_id, label, timestamp} records
        by joining with online feature vectors.
        """
        rows = []
        for rec in records:
            vec = self.read_online(feature_set_name, rec["entity_id"])
            if vec is None:
                continue
            row = {"entity_id": rec["entity_id"], label_col: rec[label_col]}
            row.update(vec.values)
            rows.append(row)
        return pd.DataFrame(rows)

    def feature_summary(self, feature_set_name: str) -> dict:
        fs = self.get_feature_set(feature_set_name)
        if not fs:
            return {}
        return {
            "name": fs.name,
            "entity_key": fs.entity_key,
            "num_features": len(fs.features),
            "features": [
                {"name": f.name, "dtype": f.dtype, "ttl_days": f.ttl_days, "version": f.version}
                for f in fs.features
            ],
            "online_entries": len(self._online_store.get(feature_set_name, {})),
        }
