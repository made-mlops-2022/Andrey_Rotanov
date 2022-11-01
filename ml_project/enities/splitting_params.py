from dataclasses import dataclass


@dataclass()
class SplittingParams:
    val_size: int
    random_state: int
