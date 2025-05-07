#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 00:05:25 2025

@author: mesut
"""
# Gerekli kütüphaneleri içe aktarma
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image
import numpy as np
import os

# --- Google Drive'ı Bağlama (Colab'da çalıştırılmalı) ---
# Modeliniz Drive'da kayıtlı olduğu için bu adım GEREKLİDİR.
# Bu hücreyi ayrı çalıştırıp gelen bağlantıya tıklayarak izinleri verin.
from google.colab import drive
drive.mount('/content/drive')
print("Google Drive başarıyla bağlandı.")

# --- 1. Model Mimarisi Tanımlama (Eğitimde Kullanılan Aynısı) ---
# Eğitilmiş modeli yüklerken model mimarisinin aynısı olmalıdır.

class ComplexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 10) # 10 sınıf

    def forward(self, x):
        x = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = self.pool3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))))

        x = torch.flatten(x, 1)

        x = self.dropout1(F.relu(self.bn7(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn8(self.fc2(x))))
        x = self.fc3(x)
        return x

# --- 2. Eğitilmiş Modeli Yükleme ---

# Modeli kaydettiğiniz Google Drive klasörünün **tam yolunu** belirtin
# drive.mount sonrası Drive'ınızın kök dizini '/content/drive/My Drive/' olur.
model_load_folder = 'trained_models_cifar10' # Model klasörünüzün yolu
model_file_name = 'cifar10_complex_cnn.pth' # Kaydettiğiniz model dosyasının adı
model_path = os.path.join(model_load_folder, model_file_name)

# Cihazı belirleme (GPU varsa GPU kullan)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Modeli oluşturma ve eğitilmiş ağırlıkları yükleme
model = ComplexNet()

# Model ağırlıklarını yüklemeden önce dosyanın varlığını kontrol edelim
if not os.path.exists(model_path):
    print(f"Hata: Model dosyası bulunamadı: {model_path}")
    print("Lütfen Google Drive'ınızın bağlı olduğundan ve model yolunun doğru olduğundan emin olun.")
else:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Modeli değerlendirme moduna al
    print(f"Model başarıyla yüklendi: {model_path}")


# --- 3. Görüntü Ön İşleme Fonksiyonu ---
# Arayüzden gelen görüntüyü modele uygun formata getirmek için
# Eğitimdeki normalize değerleri kullanılmalı

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.Resize((32, 32)), # Görüntüyü tam olarak 32x32 boyuta getir
    transforms.ToTensor(),       # PIL Image'ı PyTorch Tensor'e dönüştür (0-1 aralığına scale eder)
    normalize,                   # Normalizasyon uygula (-1 - 1 aralığına veya benzeri bir aralığa getirir)
])

# --- 4. Tahmin Fonksiyonu ---
# Gradio arayüzü bu fonksiyonu çağıracak
# Girdi: PIL Image nesnesi (Gradio tarafından sağlanır)
# Çıktı: Tahmin edilen sınıfın adı (string) veya olasılıklar (dict)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # CIFAR-10 sınıfları

def classify_image(image):
    # Görüntü None gelirse hata döndür
    if image is None:
        return "Lütfen sınıflandırmak için bir görüntü yükleyin."
    
    # Eğer görüntü RGB değilse dönüştür (bazı görüntüler tek kanallı olabilir)
    if image.mode != 'RGB':
        image = image.convert('RGB')


    # Görüntüyü ön işleme tabi tut
    # preprocess fonksiyonu PIL Image alıp Tensör döndürür
    input_tensor = preprocess(image)

    # Modeller batch input bekler, tek görüntü olduğu için batch boyutu ekle (unsqueeze(0))
    input_batch = input_tensor.unsqueeze(0)

    # Modeli kullanarak tahmin yap
    # Cihaza taşı (eğer model GPU'daysa inputu da oraya taşımalıyız)
    with torch.no_grad(): # Tahmin sırasında gradient hesaplamasına gerek yok
        output = model(input_batch.to(device))

    # Sonuçları al ve sınıf etiketini bul
    # output bir skor tensörüdür (1x10 boyutunda, her sınıf için bir skor)
    # En yüksek skora sahip sınıfı bul
    _, predicted_class_index = torch.max(output, 1) # Boyut 1 boyunca maksimum değeri ve indeksini bul

    # Tahmin edilen sınıfın ismini al
    predicted_class_label = classes[predicted_class_index.item()] # İndeksi Python int'e çevir

    # İsteğe bağlı olarak sınıflandırma olasılıklarını hesapla
    # softmax, skorları olasılıklara dönüştürür
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidences = {classes[i]: float(probabilities[i]) for i in range(10)}

    # Sonucu sadece sınıf adı olarak döndürebilirsiniz:
    # return predicted_class_label

    # Veya sonucu olasılıklarla birlikte döndürerek Gradio'nun Label bileşeninde göstermesini sağlayabilirsiniz:
    return confidences


# --- 5. Gradio Arayüzünü Tanımlama ve Başlatma ---

# Arayüz için input ve output bileşenlerini tanımla
# Giriş: Resim (`gr.Image`), type="pil" olarak ayarlamak PIL Image nesnesi olarak almak içindir
# Çıkış: `gr.Label()` olasılıkları güzel bir formatta gösterir
# veya `gr.Text()` sadece sınıf adını gösterir (yukarıdaki fonksiyonun dönüş değerine bağlı)

output_component = gr.Label(label="Tahmin Edilen Sınıf ve Olasılıkları") # Olasılıkları göstermek için bunu kullanın
# output_component = gr.Text(label="Tahmin Edilen Sınıf") # Sadece sınıf adını göstermek için (classify_image fonksiyonunu da buna göre düzenlemelisiniz)


# Arayüz nesnesini oluşturma
interface = gr.Interface(
    fn=classify_image, # Görüntü işleme ve tahmin yapacak fonksiyon
    inputs=gr.Image(type="pil", label="Sınıflandırmak İçin Bir Görüntü Yükleyin"), # Kullanıcının görüntü yükleyeceği alan
    outputs=output_component, # Sonucun gösterileceği alan
    title="CIFAR-10 Görüntü Sınıflandırıcı", # Arayüz başlığı
    description="Yüklediğiniz görüntüyü eğitilmiş model ile sınıflandırır.", # Arayüz açıklaması
    allow_flagging=False # Kullanıcıların sonuçları işaretlemesini kapatır
)

# Arayüzü başlatma
# share=True: Arayüzü dışarıdan erişilebilir yapar ve size bir link verir (Colab'da gereklidir)
# launch(debug=True) hata ayıklama bilgileri gösterir
interface.launch(share=True)
