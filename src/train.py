import os

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.deep_fusion_gan.model import DeepFusionGAN
from src.objects.dataset import DFGANDataset
from src.utils import fix_seed


def create_loader(imsize: int, batch_size: int, data_dir: str, split: str) -> DataLoader:
    assert split in ["train", "test"], "Wrong split type, expected train or test"
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()
    ])

    dataset = DFGANDataset(data_dir, split, image_transform)

    return DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)


def train():
    fix_seed()

    data_path = "../data"
    encoder_weights_path = "../text_encoder_weights/text_encoder200.pth"
    image_save_path = "../gen_images"
    gen_path_save = "../gen_weights"

    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(gen_path_save, exist_ok=True)

    train_loader = create_loader(256, 24, data_path, "train")
    model = DeepFusionGAN(n_words=train_loader.dataset.n_words,
                          encoder_weights_path=encoder_weights_path,
                          image_save_path=image_save_path,
                          gen_path_save=gen_path_save)

    model.fit(train_loader)


if __name__ == '__main__':
    train()
