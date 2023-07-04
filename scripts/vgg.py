import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import gc
torch.backends.cuda.max_split_size_mb = 1  # Set the max_split_size_mb value to adjust memory splitting

class ModifiedVGG(nn.Module):
    def __init__(self):
        super(ModifiedVGG, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier = self.vgg.classifier[:-1]
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x = self.normalize(x)
        x = self.conv(x)
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg.classifier(x)
        return x

def load_and_resize_image(image_path):
    # Open the image using PIL
    image = Image.open(image_path)

    # Resize the image to 256x256 pixels
    resize_transform = transforms.Resize((256, 256))
    image = resize_transform(image)

    # Convert the image to a tensor
    transform = transforms.ToTensor()
    image = transform(image)

    # Add a batch dimension
    image = image.unsqueeze(0)

    return image


def create_model():
    try:
        model = torch.load("models/vgg")
        return model
    except FileNotFoundError:
        pass
    
    model = ModifiedVGG()
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()
    torch.save(model, "models/vgg")
    return model


def compare_images(model, image1, image2):
    #if torch.cuda.is_available():
     #   image1 = image1.cuda()
      #  image2 = image2.cuda()

    image1 = model(image1)
    image2 = model(image2)

    similarity_score = torch.nn.functional.cosine_similarity(image1, image2)

    return similarity_score


def image_similarity_energy(image1, image2):



    image1_gray = torch.mean(image1, dim=1, keepdim=True)
    image2_gray = torch.mean(image2, dim=1, keepdim=True)

    # Calculate energy (mean absolute difference)
    mse = torch.mean((image1_gray - image2_gray) ** 2)
    #mse = torch.mean((image1 - image2) ** 2)

    # Return the negative MSE
    energy = -mse
    return energy

# """ energy_scale = 20
# batch_size = 10  # Number of iterations per batch
# path_1 = "data/Standard_Golden_Retriever.jpeg"
# path_2 = "data/Canadian_Golden_Retriever.jpeg"
# path_3 = "data/Cat03.jpg"
# if torch.cuda.is_available():
#     print("cuda")
# image_1 = load_and_resize_image(path_1)
# image_2 = load_and_resize_image(path_2)
# image_3 = load_and_resize_image(path_3).requires_grad_()
# #model = torch.load("models/vgg").to(torch.device("cuda"))
# model = torch.load("models/vgg")
# score = []

# total_iterations = 50
# num_batches = total_iterations // batch_size

# for batch in range(num_batches):
#     for i in range(batch_size):
#         iteration = batch * batch_size + i
#         similarity, similarity_score = compare_images(model, image_1, image_3)
#         score.append(similarity)
#         print(score[-1])
#         loss = 1 - similarity_score
#         grad = torch.autograd.grad(outputs=loss, inputs=image_3, create_graph=True)[0]
#         grad_norm = torch.norm(grad, p=2)  # Calculate the norm of gradients
#         grad /= (grad_norm + 1e-8)  # Normalize gradients
#         grad = grad * energy_scale
#         image_3 = image_3 - grad

#         vutils.save_image(image_3, f"output/image_{iteration}.png")

#         # Add garbage collection to free up GPU memory
#         del loss, grad, grad_norm
#         gc.collect()
#         #torch.cuda.empty_cache()

#     # Perform garbage collection and empty GPU cache after each batch
#     gc.collect()
#    # torch.cuda.empty_cache()
#  """