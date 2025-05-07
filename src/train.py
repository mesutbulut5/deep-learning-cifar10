# Gerekli kütüphaneleri içe aktarma
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Cihazı Belirleme (GPU kullanımı için) ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")
if device.type == 'cuda':
    print(f"GPU Adı: {torch.cuda.get_device_name(0)}")

# --- 1. Veri Setini Yükleme ve Ön İşleme ---

# Colab'da veri seti genellikle /content/data altına iner.
data_root = './data' # Veya '/content/data' olarak da belirtebilirsiniz

# Veri setini indirmek ve yüklemek için dönüşümleri tanımlama
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# Veri setini indirme ve yükleme
# download=True: Eğer yoksa indirir, varsa indirmez.
trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, # Colab GPU'da daha büyük batch boyutu kullanabiliriz
                                          shuffle=True, num_workers=2) # num_workers Colab'da bazen 0 veya 2 iyi çalışır

testset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, # Colab GPU'da daha büyük batch boyutu kullanabiliriz
                                         shuffle=False, num_workers=2)

# CIFAR-10 sınıfları
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("CIFAR-10 veri seti başarıyla yüklendi (varsa indirilmedi).")

# --- 2. CNN Model Mimarisi Tanımlama ---

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

        # Hesaplama: 32 -> 16 -> 8 -> 4 (Pooling sonrası boyutlar)
        # Son evrişim katmanı 256 filtreye sahip
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = self.pool3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))))

        x = torch.flatten(x, 1)

        x = self.dropout1(F.relu(self.bn7(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn8(self.fc2(x))))
        x = self.fc3(x)
        return x

# Modeli oluşturma ve cihaza gönderme
net = ComplexNet().to(device)


print("Daha kapsamlı CNN modeli başarıyla tanımlandı ve cihaza yüklendi.")

# --- 3. Kayıp Fonksiyonu ve Optimizasyon ---

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# --- 4. Model Eğitimi ve Değerlendirilmesi ---

train_losses = []
test_losses = []
test_accuracy = []
test_precision = []
test_recall = []
test_f1 = []

epochs = 50 # Eğitim epoch sayısı

print("Eğitim başlıyor...")

for epoch in range(epochs):
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Eğitim kaybını daha sık yazdırma (örneğin her 100 batch'te bir)
        if i % 100 == 99:
             print(f'[{epoch + 1}, {i + 1:5d}] train loss: {running_loss / 100:.3f}')
             running_loss = 0.0


    # Her epoch sonunda test setinde değerlendirme yap
    net.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    test_loss = 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    epoch_train_loss = running_loss / len(trainloader) # Kalan running_loss varsa dahil et (eğer %100'ün katı değilse)
    if i % 100 != 99:
         # Son kısmi batch'in ortalama kaybını ekleyelim, eğer varsa
         epoch_train_loss = ( (running_loss + (epoch_train_loss * (len(trainloader) % 100))) / len(trainloader) ) if (len(trainloader) % 100 != 0) else (running_loss / len(trainloader))
         # Not: Daha doğru bir ortalama için running_loss hesaplamasını epoch döngüsü dışında tutmak daha iyidir.
         # Basitlik için, her epoch sonundaki running_loss'u sıfırlamadan biriktirip sonra ortalamasını alabilirsiniz.
         # Şu anki kodda running_loss zaten her 100 batch'te sıfırlanıyor, bu yüzden epoch sonunda kalan running_loss
         # sadece son kısmi batch'e ait olur. Yukarıdaki hesaplama yerine basitçe
         # epoch_train_loss = test_loss / len(trainloader) gibi bir yaklaşım da düşünülebilir
         # veya running_loss'u döngü dışında biriktirip epoch sonunda sıfırlamak.
         # Mevcut implementasyon test loss hesaplaması için daha uygun.
         # Eğitim loss'u için daha temiz bir yol:
         train_loss_epoch_end = 0.0
         net.train()
         with torch.no_grad(): # Eğitim loss'unu hesaplarken gradient'e ihtiyacımız yok
             for i, data in enumerate(trainloader, 0):
                  inputs, labels = data[0].to(device), data[1].to(device)
                  outputs = net(inputs)
                  loss = criterion(outputs, labels)
                  train_loss_epoch_end += loss.item()
         epoch_train_loss = train_loss_epoch_end / len(trainloader)


    epoch_test_loss = test_loss / len(testloader)
    epoch_accuracy = 100 * correct / total

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro', zero_division=0)

    train_losses.append(epoch_train_loss)
    test_losses.append(epoch_test_loss)
    test_accuracy.append(epoch_accuracy)
    test_precision.append(precision)
    test_recall.append(recall)
    test_f1.append(f1)

    print(f'Epoch {epoch + 1}/{epochs}, '
          f'Train Loss: {epoch_train_loss:.3f}, '
          f'Test Loss: {epoch_test_loss:.3f}, '
          f'Test Acc: {epoch_accuracy:.2f}%, '
          f'Test Prec: {precision:.2f}, '
          f'Test Rec: {recall:.2f}, '
          f'Test F1: {f1:.2f}')

print('Eğitim Tamamlandı')

# --- 5. Metriklerin Görselleştirilmesi ---

epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(epochs_range, train_losses, label='Eğitim Kaybı')
plt.plot(epochs_range, test_losses, label='Test Kaybı')
plt.title('Eğitim ve Test Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs_range, test_accuracy, label='Test Doğruluğu')
plt.title('Test Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk (%)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs_range, test_precision, label='Test Precision (Macro Avg)')
plt.plot(epochs_range, test_recall, label='Test Recall (Macro Avg)')
plt.plot(epochs_range, test_f1, label='Test F1 Skoru (Macro Avg)')
plt.title('Test Precision, Recall ve F1 Skoru')
plt.xlabel('Epoch')
plt.ylabel('Skor')
plt.legend()

plt.tight_layout()
plt.show()

# Karışıklık Matrisi (Confusion Matrix)
conf_matrix = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.title('Karışıklık Matrisi (Test Seti)')
plt.show()

# --- 6. Eğitilmiş Modeli Kaydetme ---

# Colab ortamında modelinizi Google Drive'ınıza kaydetmek isteyebilirsiniz
# Bunun için Google Drive'ı bağlamanız gerekir.
# from google.colab import drive
# drive.mount('/content/drive')
# model_save_path = '/content/drive/My Drive/trained_models' # Kayıt yolunu buraya ayarlayın

# Eğer Drive'a kaydetmeyecekseniz, Colab oturumu kapanınca dosyalar silinir.
model_save_path = './trained_models' # Oturum boyunca geçerli olacak kayıt yeri
os.makedirs(model_save_path, exist_ok=True)

model_file_path = os.path.join(model_save_path, 'cifar10_complex_cnn.pth')
torch.save(net.state_dict(), model_file_path)

print(f"Model başarıyla kaydedildi: {model_file_path}")
