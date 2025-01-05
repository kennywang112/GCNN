import gdown
import os

# 創建保存模型的目錄
model_dir = "./model/"
os.makedirs(model_dir, exist_ok=True)

# Google Drive 文件ID與文件名的對應
files = {
    "1ytHjZpCYpw0y9scXSMkmXVAPdmJ7GRRC": "face_landmarker.task",
    "1JufgIIjoMIJ1PCx3WaYLa3MrTn12iAi9": "model_Net_Resnet.pth",
    "1P3vMy4c3bQP0_UHLE_kuVLu87PGTN2t2": "model_Net_Alex.pth",
    "1wOT6_aHLounU6o9aLA0YO7VynHQbBDjt": "model_Net_VGG.pth"
}
# 循環下載每個模型
for file_id, file_name in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output = os.path.join(model_dir, file_name)
    print(f"Downloading {file_name}...")
    gdown.download(url, output, quiet=False)
    print(f"{file_name} downloaded to {output}")