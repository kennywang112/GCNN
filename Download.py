import gdown
import os

# 創建保存模型的目錄
model_dir = "./model/"
os.makedirs(model_dir, exist_ok=True)

# Google Drive 文件ID與文件名的對應
files = {
    "1tIgKRb0JKKsWIq5T9fCpi07gKyo6dZwk": "face_landmarker.task",
    "1ax7lpgQjR9-csF8r2UBfNJfQ_RNzYcdS": "model_Net_Resnet.pth",
    "1czfZ49ldTj-zmOTs58Ds_3zrTOepVTD5": "model_Net_Alex.pth",
    "16g4yipf3j06JfatYPhXXxpLcP1zMERFb": "model_Net_VGG.pth"
}
# 循環下載每個模型
for file_id, file_name in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output = os.path.join(model_dir, file_name)
    print(f"Downloading {file_name}...")
    gdown.download(url, output, quiet=False)
    print(f"{file_name} downloaded to {output}")