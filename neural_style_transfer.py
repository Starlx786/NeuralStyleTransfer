import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert("RGB")
    size = max(image.size)
    scale = max_size / size
    new_size = tuple([int(dim * scale) for dim in image.size])

    transform = transforms.Compose([
        transforms.Resize(new_size),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)

# Display image
def imshow(tensor, title=None):
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")

# Load images
content = load_image("content.webp")
style = load_image("style.webp")

# Load pre-trained VGG19
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Gram Matrix
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram

# Layers
content_layer = "21"
style_layers = ["0", "5", "10", "19", "28"]

content_weight = 1e4
style_weight = 1e2

generated = content.clone().requires_grad_(True)

optimizer = optim.Adam([generated], lr=0.01)

# Training loop
for step in range(300):
    generated_features = {}
    content_features = {}
    style_features = {}

    x = generated
    y = content
    z = style

    for name, layer in vgg._modules.items():
        x = layer(x)
        y = layer(y)
        z = layer(z)

        if name == content_layer:
            content_features[name] = y
            generated_features[name] = x

        if name in style_layers:
            style_features[name] = z
            generated_features[name] = x

    content_loss = torch.mean((generated_features[content_layer] - content_features[content_layer]) ** 2)

    style_loss = 0
    for layer in style_layers:
        gram_g = gram_matrix(generated_features[layer])
        gram_s = gram_matrix(style_features[layer])
        style_loss += torch.mean((gram_g - gram_s) ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss: {total_loss.item():.2f}")

# Show final image
imshow(generated, "Styled Image")
plt.show()
