import torch
from transformers import AutoModel, AutoProcessor

# load the model and processor
model = AutoModel.from_pretrained("jadechoghari/vfusion3d", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("jadechoghari/vfusion3d")

# download and preprocess the image
import requests
from PIL import Image
from io import BytesIO

image_url = 'https://sm.ign.com/ign_nordic/cover/a/avatar-gen/avatar-generations_prsz.jpg'
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# preprocess the image and get the source camera 
image, source_camera = processor(image)


# generate planes (default output)
output_planes = model(image, source_camera)
print("Planes shape:", output_planes.shape)

# generate a 3D mesh
output_planes, mesh_path = model(image, source_camera, export_mesh=True)
print("Planes shape:", output_planes.shape)
print("Mesh saved at:", mesh_path)

# Generate a video
output_planes, video_path = model(image, source_camera, export_video=True)
print("Planes shape:", output_planes.shape)
print("Video saved at:", video_path)
