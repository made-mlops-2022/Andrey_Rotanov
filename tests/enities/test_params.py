from typing import List
from dataclasses import dataclass

@dataclass()
class DescriptionFeature:
    name: str
    type: str
    min_value: int
    max_value: int

@dataclass()
class SyntheticDataParams:
    output_path: str
    size: int
    feature: List[DescriptionFeature]
