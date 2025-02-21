import clip
import torch
from PIL import Image
import os

# Set device: GPU if available, else CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and the preprocessing pipeline.
model, preprocess = clip.load("ViT-B/32", device=device)

# Step 1: Precompute embeddings for your historical texts.
historical_texts = [
    "Biblioteca, a distinctive building with angled geometric shapes and impressive stairs where students study",
    "Centrales Comida, a cafeteria/food court area with excellent cuisine",
     "Servilletero, simple building with triangular shape and surrounded with beautiful trees"
    # Add more historical descriptions as needed.
]

# Tokenize and encode the historical texts.
text_tokens = clip.tokenize(historical_texts).to(device)
with torch.no_grad():
    historical_embeddings = model.encode_text(text_tokens)

# Step 2: Encode a building image.\
directory_path = os.path.dirname(__file__)
#Images in Images folder direct child of parent folder
image_path = os.path.join(directory_path, 'Images/biblioteca.jpg')
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
with torch.no_grad():
    image_embedding = model.encode_image(image)

# Step 3: Compute similarities between the image and each historical text.
# The dot product (optionally normalized to cosine similarity) can be used.
similarity = (image_embedding @ historical_embeddings.T).softmax(dim=-1)
max_index = similarity.argmax().item()
relevant_text = historical_texts[max_index]

print("Relevant historical information:", relevant_text)
