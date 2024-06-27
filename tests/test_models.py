from molbind.models.components.custom_encoders import ImageEncoder
from molbind.data.components.datasets import ImageDataset
from torch.utils.data import DataLoader
from typing import List


class ImageDatasetWithoutCentralModality(ImageDataset):
    def __init__(self, image_files: List[str], **kwargs) -> None: # noqa: UP006
        super().__init__(image_files=image_files, **kwargs)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        image_as_tensor = self.read_image_to_tensor(self.image_files[idx], repeats=1)
        return image_as_tensor.mean(dim=0)


def test_image_encoder():
    model = ImageEncoder(ckpt_path=None)
    dataset = ImageDatasetWithoutCentralModality(
        image_files=["tests/data/1.png", "tests/data/2.png"],
        central_modality=None, 
        central_modality_data=None
        )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    for batch in dataloader:
        out = model(batch)
        assert out.shape == (2, 512)
