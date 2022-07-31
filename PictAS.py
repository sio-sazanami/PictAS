import json
from pathlib import Path
import os
import tkinter as tk
import tkinter.filedialog
import glob
from tqdm import tqdm
import shutil

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_url

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        # これを有効にしないと計算した勾配が毎回異なり再現性が担保できない
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# デバイスを選択する
device = get_device(use_gpu=True)
print("[notice] デバイス {} が選択されました".format(device))
# modelをGPUに送る
model = torchvision.models.resnet50(pretrained=True).to(device)
print("[notice] resnet50 が読み込まれました")

transform = transforms.Compose(
    [
        transforms.Resize(256),  # (256, 256)で切り抜く
        transforms.CenterCrop(224),  # 画像の中心に合わせて(224, 224)で切り抜く
        transforms.ToTensor(),  # テンソルにする
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 標準化する
    ]
)


root = tk.Tk()
root.withdraw()
print("[notice] 振り分け対象のディレクトリを選択")
target_dir = tkinter.filedialog.askdirectory(mustexist=True)
print("[notice] 振り分け対象として {} が選択されました".format(target_dir))

files = glob.glob(os.path.join(target_dir, "*"))
print("[notice] 読み込み対象数は {} です".format(len(files)))

print("[notice] コピー対象のディレクトリを選択")
copy_dir = tkinter.filedialog.askdirectory(mustexist=True)
print("[notice] コピー対象として {} が選択されました".format(copy_dir))
    
def get_classes():
    if not Path("data/imagenet_class_index.json").exists():
        # ファイルが存在しない場合はダウンロードする
        download_url("https://git.io/JebAs", "data", "imagenet_class_index.json")

    # クラス一覧を読み込む
    with open("data/imagenet_class_index.json", encoding='utf-8') as f:
        data = json.load(f)
        class_names = [x["ja"] for x in data]

    return class_names

class_names = get_classes()#インデックス番号から推定名称を検索する辞書

for filepath in tqdm(files):# C:/~~~/.JPGのような形式(絶対パス)
    fname = filepath[len(target_dir):]# ???.JPGのような形式(ファイル名のみ)
    base, ext = os.path.splitext(filepath)# パスと拡張子を分離
    if ext == '.JPG' or ext == '.jpg':
        img = Image.open(filepath)
        inputs = transform(img)# 行列データに変換
        inputs = inputs.unsqueeze(0).to(device)
        
        model.eval()# 推論モード
        outputs = model(inputs)

        batch_probs = F.softmax(outputs, dim=1)
        batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)# 推定確率が高い順に，batch_probs(推定確率), batch_indices(インデックス番号)
        
        picname = class_names[batch_indices[0][0]]
        #print(f"[notice] {base+ext} は {picname} と推測されます")
        
        copypath = copy_dir + '/' + picname
        if os.path.exists(copypath):
            shutil.copy2(filepath, copypath+'/'+fname)# 推定名称のパスにコピー
        else:
            os.mkdir(copypath)
            shutil.copy2(filepath, copypath+'/'+fname)
        