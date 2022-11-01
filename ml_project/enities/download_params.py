from dataclasses import dataclass, field


@dataclass()
class DownloadParams:
    dataset_name: str
    output_folder: str
    username: str
    api_key: str
