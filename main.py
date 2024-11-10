import torch
from torchvision import transforms
import urllib
from PIL import Image
from torchvision import transforms
import random
# Samoyed -> clase verdadera -> clase objetivo -> keeshond
problacion = 1
model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_5m_m1", pretrained=True)
# or any of these variants
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_5m_m2", pretrained=True)
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_9m_m1", pretrained=True)
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_9m_m2", pretrained=True)
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_small_m1_05", pretrained=True)
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_small_m2_05", pretrained=True)
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_small_m1_075", pretrained=True)
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_small_m2_075", pretrained=True)
model.eval()

bestRR = []
bestGR = []
bestBR = []

# Download an example image from the pytorch website
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

for p in range(problacion):
    # sample execution (requires torchvision)
    input_image = Image.open(filename)
    input_image = input_image.resize((224,224))
    input_image.save("clean.jpg")
    input_image = input_image.convert('RGB')
    pixeles = input_image.load()
    for x in range(input_image.width):
        for y in range(input_image.height):
            prob = random.randint(0,10)
            if prob == 1:
                rr = random.randint(0,100)
                gr = random.randint(0,100)
                br = random.randint(0,100)
                r, g, b = pixeles[x, y]
                r, g, b = pixeles[x, y] = ( r - rr, g - gr ,  b - br )
    #xCoord = random.randint(0,input_image.width)
    #yCoord = random.randint(0,input_image.width)
    #rr = random.randint(0,255)
    #gr = random.randint(0,255)
    #br = random.randint(0,255)
    #r, g, b = pixeles[xCoord, yCoord]
    #r, g, b = pixeles[xCoord, yCoord] = ( r - rr, g - gr ,  b - br )
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    #print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    #print(probabilities)

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    masProbable = ''
    for i in range(top5_prob.size(0)):
        if i == 0:
            masProbable = categories[top5_catid[i]]
            print(categories[top5_catid[i]], top5_prob[i].item())
    print(masProbable)

input_image.save('noise.jpg')
