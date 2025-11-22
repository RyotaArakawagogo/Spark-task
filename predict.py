import argparse
import torch
from torchvision import models, transforms
from PIL import Image

# CIFAR-10 クラス名
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def build_model(weights_path, num_classes=10, use_pretrained=False, device="cpu"):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="例: artifacts/baseline/20251115-120000/best.pt"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.weights, num_classes=len(CLASSES), device=device)
    x = preprocess(args.image).to(device)

    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()

    print(CLASSES[pred])


if __name__ == "__main__":
    main()
