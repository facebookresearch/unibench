from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import PIL.Image

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

class SUN397(VisionDataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.

    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _DATASET_URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _DATASET_MD5 = "8ca2778205c41d23104230ba66911c7a"
    _PARTITION_INFO_URL = "https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip"
    

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        partition_idx: int = 1,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert partition_idx > 0 and partition_idx <= 10, "partition_idx should be between 1 and 10"
        self._data_dir = Path(self.root) / "SUN397"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(self._data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        
        self.partition = self._data_dir / f"Testing_{partition_idx:02}.txt"
        with open(self.partition) as f:
            self.partition_list = [line.strip() for line in f]
            
        self._image_files = []
        for entry in self.partition_list:
            image_path = self._data_dir / entry.lstrip('/')
            if image_path.exists():
                self._image_files.append(image_path)
            else:
                print(f"Image not found: {image_path}")
        self._labels = [
            self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])] for path in self._image_files
        ]
        self.classes = [cl.replace("_", " ").replace("/", " ") for cl in self.classes]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return self._data_dir.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=self.root, md5=self._DATASET_MD5)
        download_and_extract_archive(self._PARTITION_INFO_URL, download_root=self._data_dir)