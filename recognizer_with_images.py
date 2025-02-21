import clip
import torch
from PIL import Image
import os

# 1. Load CLIP model and the preprocessing pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. Define your dataset of images and labels
#    Here, we have 3 images with their respective labels (text descriptions).
image_paths = [
    "servilletero.jpg",
    "centrales.jpg",
    "biblioteca.jpg"
]


text_labels = [
    "El Servilletero, a distinctive building with angled geometric shapes",
    "Centrales Comida, a cafeteria/food court area",
    "Biblioteca, a modern library with glass facades"
]

# 3. Precompute text embeddings
#    We'll encode the text labels with the CLIP text encoder.
text_tokens = clip.tokenize(text_labels).to(device)
with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)  # normalize (optional)

# 4. Precompute image embeddings
#    For each image, load and preprocess it, then encode with CLIP image encoder.
image_embeddings = []
directory_path = os.path.dirname(__file__)
for path in image_paths:
    #Images in Images folder direct child of parent folder
    image_path = os.path.join(directory_path, "Images/"+path)
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize (optional)
    image_embeddings.append(embedding)

# Convert list of embeddings to a single tensor for convenience
image_embeddings = torch.cat(image_embeddings, dim=0)

# 5. Create a small dataset structure in Python
dataset = []
for i in range(len(image_paths)):
    dataset.append({
        "image_path": image_paths[i],
        "text_label": text_labels[i],
        "image_embedding": image_embeddings[i],
        "text_embedding": text_embeddings[i]
    })

# 6. Print out the embeddings or store them for later use
for entry in dataset:
    print(f"Image: {entry['image_path']}")
    print(f"Text: {entry['text_label']}")
    print(f"Image Embedding Shape: {entry['image_embedding'].shape}")
    print(f"Text Embedding Shape: {entry['text_embedding'].shape}")
    print("-----------")

