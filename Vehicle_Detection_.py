# =============================================================================
# ARAÇ TESPİTİ PROJESİ - YOLO FORMATINDA VERİ SETİ HAZIRLAMA
# =============================================================================
# Bu kod, araç tespiti için YOLO formatında veri seti oluşturur
# CSV dosyasından bounding box bilgilerini okur ve YOLO formatına çevirir
# =============================================================================

# Gerekli kütüphaneleri import et
import os, time, random
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm.auto import tqdm
import shutil as sh

from IPython.display import Image, clear_output
import matplotlib.pyplot as plt

# =============================================================================
# VERİ SETİ PARAMETRELERİ
# =============================================================================
# Görüntü boyutları (yükseklik, genişlik, kanal sayısı)
img_h, img_w, num_channels = (380, 676, 3)

# =============================================================================
# CSV DOSYASINI YÜKLEME VE HATA KONTROLÜ
# =============================================================================
# CSV dosyasını yükle - doğru dosya adı ile
try:
    df = pd.read_csv('train_solution_bounding_boxes (1).csv')
    print("✅ CSV dosyasi başariyla yüklendi")
    print(f"📊 Toplam kayit: {len(df)}")
    print(f"�� Sütunlar: {list(df.columns)}")
except FileNotFoundError:
    print("❌ Hata: train_solution_bounding_boxes (1).csv dosyasi bulunamadi!")
    print("🔍 Mevcut CSV dosyalari:")
    
    # Mevcut dizindeki dosyaları listele
    current_files = os.listdir('.')
    csv_files = [f for f in current_files if f.endswith('.csv')]
    
    if csv_files:
        print(f"�� Bulunan CSV dosyalari: {csv_files}")
        print("�� Doğru dosya adini kullanin.")
    else:
        print("❌ Hiç CSV dosyasi bulunamadi!")
    
    exit()
except Exception as e:
    print(f"❌ CSV yükleme hatasi: {e}")
    exit()

# =============================================================================
# VERİ ÖN İŞLEME
# =============================================================================
# Sütun adını düzelt (image -> image_id)
df.rename(columns={'image':'image_id'}, inplace=True)

# Görüntü ID'lerinden dosya uzantısını kaldır
df['image_id'] = df['image_id'].apply(lambda x: x.split('.')[0])

# Bounding box koordinatlarını hesapla
# x_center: merkez x koordinatı
df['x_center'] = (df['xmin'] + df['xmax'])/2
# y_center: merkez y koordinatı  
df['y_center'] = (df['ymin'] + df['ymax'])/2
# w: genişlik
df['w'] = df['xmax'] - df['xmin']
# h: yükseklik
df['h'] = df['ymax'] - df['ymin']
# Sınıf etiketi (araç = 0)
df['classes'] = 0

# Koordinatları normalize et (0-1 aralığına)
# YOLO formatı için koordinatlar 0-1 arasında olmalı
df['x_center'] = df['x_center'] / img_w  # Genişliğe böl
df['w'] = df['w'] / img_w                # Genişliğe böl
df['y_center'] = df['y_center'] / img_h  # Yüksekliğe böl
df['h'] = df['h'] / img_h                # Yüksekliğe böl

# Veri seti bilgilerini yazdır
print("\n�� Veri önizleme:")
print(df.head())
print(f"\n📈 Veri seti bilgileri:")
print(f"  - Toplam kayit: {len(df)}")
print(f"  - Benzersiz görüntü: {df['image_id'].nunique()}")

# =============================================================================
# RASTGELE GÖRÜNTÜ TESTİ
# =============================================================================
# Rastgele bir görüntü seç ve yükle
index = list(set(df.image_id))  # Benzersiz görüntü ID'leri
image = random.choice(index)    # Rastgele bir görüntü seç
print(f"\n🎲 Rastgele seçilen görüntü ID: {image}")

# Görüntü dosya yollarını kontrol et
img_path = f'training_images/{image}.jpg'
print(f"📁 Görüntü yolu: {img_path}")

# Görüntüyü yükle
img = cv2.imread(img_path)

# Görüntü yüklenip yüklenmediğini kontrol et
if img is not None:
    print(f"✅ Görüntü başariyla yüklendi!")
    print(f"📏 Görüntü boyutlari: {img.shape}")
else:
    print("❌ Hata: Görüntü yüklenemedi!")
    print("🔍 Dosya yolu kontrol ediliyor...")
    
    # Alternatif yolları dene
    alternative_paths = [
        f'training_images/{image}.jpg',
        f'./training_images/{image}.jpg',
        f'../training_images/{image}.jpg'
    ]
    
    for path in alternative_paths:
        if os.path.exists(path):
            print(f"✅ Dosya bulundu: {path}")
            img = cv2.imread(path)
            if img is not None:
                print(f"📏 Görüntü boyutlari: {img.shape}")
                break
        else:
            print(f"❌ Dosya bulunamadi: {path}")
    
    if img is None:
        print("⚠ Hiçbir yolda görüntü bulunamadi!")
        print("📁 Mevcut dosyalar:")
        if os.path.exists('training_images'):
            files = os.listdir('training_images')[:5]  # İlk 5 dosyayı göster
            print(files)
        else:
            print("❌ training_images klasörü bulunamadı!")

# =============================================================================
# YOLO VERİ SETİ OLUŞTURMA FONKSİYONU
# =============================================================================
def create_yolo_dataset(df, index, source_dir='training_images', output_dir='yolo_dataset'):
    """
    YOLO formatinda veri seti oluşturur ve eğitim/doğrulama klasörlerine ayirir.
    
    Args:
        df (pd.DataFrame): İşlenmiş veri çerçevesi
        index (list): Görüntü ID'lerinin listesi
        source_dir (str): Kaynak görüntü klasörü
        output_dir (str): Çikti klasörü
    """
    print("\n�� YOLO veri seti oluşturuluyor...")
    print(f"�� Kaynak klasör: {source_dir}")
    print(f"📁 Çikti klasör: {output_dir}")
    
    # Çıktı klasörünü oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ {output_dir} klasörü oluşturuldu")
    
    # Cross-validation için fold ayarları
    val_ratio = 0.2  # %20 doğrulama, %80 eğitim
    
    # Doğrulama seti için rastgele indeksler seç
    val_size = int(len(index) * val_ratio)
    val_index = random.sample(index, val_size)
    
    print(f"📊 Toplam görüntü: {len(index)}")
    print(f"📊 Eğitim görüntüsü: {len(index) - val_size}")
    print(f"�� Doğrulama görüntüsü: {val_size}")
    
    # Klasör yapısını oluştur
    train_labels_dir = os.path.join(output_dir, 'labels', 'train')
    val_labels_dir = os.path.join(output_dir, 'labels', 'val')
    train_images_dir = os.path.join(output_dir, 'images', 'train')
    val_images_dir = os.path.join(output_dir, 'images', 'val')
    
    # Tüm klasörleri oluştur
    for dir_path in [train_labels_dir, val_labels_dir, train_images_dir, val_images_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("📁 Klasör yapisi oluşturuldu")
    
    # Her görüntü için işlem yap
    processed_count = 0
    error_count = 0
    
    # Her görüntü ID'si için işlem yap
    for name, mini in tqdm(df.groupby('image_id'), desc="Veri seti oluşturuluyor"):
        try:
            # Eğitim mi doğrulama mı belirle
            if name in val_index:
                labels_dir = val_labels_dir
                images_dir = val_images_dir
                split_type = "doğrulama"
            else:
                labels_dir = train_labels_dir
                images_dir = train_images_dir
                split_type = "eğitim"
            
            # YOLO formatında label dosyası oluştur
            label_file = os.path.join(labels_dir, f"{name}.txt")
            with open(label_file, 'w') as f:
                # Her araç için bounding box bilgilerini yaz
                for _, row in mini.iterrows():
                    class_id = int(row['classes'])
                    x_center = row['x_center']
                    y_center = row['y_center']
                    width = row['w']
                    height = row['h']
                    
                    # YOLO formatı: class_id x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Görüntü dosyasını kopyala
            source_image = os.path.join(source_dir, f"{name}.jpg")
            target_image = os.path.join(images_dir, f"{name}.jpg")
            
            if os.path.exists(source_image):
                sh.copy2(source_image, target_image)
                processed_count += 1
            else:
                print(f"⚠ Görüntü bulunamadi: {source_image}")
                error_count += 1
                
        except Exception as e:
            print(f"❌ Hata ({name}): {str(e)}")
            error_count += 1
    
    print(f"\n🎉 Veri seti oluşturma tamamlandi!")
    print(f"✅ Başarili: {processed_count} görüntü")
    print(f"❌ Hatali: {error_count} görüntü")
    print(f"📁 Çikti klasörü: {output_dir}")

# YOLO veri setini oluştur
create_yolo_dataset(df, index)

# =============================================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# =============================================================================
def display_sample_cars(num_samples=6):
    """
    Rastgele araç görüntülerini panelde gösterir.
    
    Args:
        num_samples (int): Gösterilecek görüntü sayisi
    """
    print(f"\n🚗 {num_samples} adet rastgele araç görüntüsü gösteriliyor...")
    
    # Rastgele görüntü ID'leri seç
    sample_images = random.sample(index, min(num_samples, len(index)))
    
    # Görüntüleri göster (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Araç Tespiti - Örnek Görüntüler', fontsize=16, fontweight='bold')
    
    for i, img_id in enumerate(sample_images):
        row = i // 3  # Satır indeksi
        col = i % 3   # Sütun indeksi
        
        # Görüntüyü yükle
        img_path = f'training_images/{img_id}.jpg'
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye çevir
            
            # Bounding box'ları çiz
            img_with_boxes = img_rgb.copy()
            car_data = df[df['image_id'] == img_id]  # Bu görüntüdeki araçları al
            
            for _, car in car_data.iterrows():
                # Normalize edilmiş koordinatları piksel koordinatlarına çevir
                x_center = car['x_center'] * img_w
                y_center = car['y_center'] * img_h
                width = car['w'] * img_w
                height = car['h'] * img_h
                
                # Bounding box köşelerini hesapla
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                # Kırmızı kutu çiz
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_with_boxes, 'CAR', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            axes[row, col].imshow(img_with_boxes)
            axes[row, col].set_title(f'ID: {img_id}\nAraç Sayisi: {len(car_data)}', 
                                   fontsize=10, fontweight='bold')
            axes[row, col].axis('off')
            
        else:
            # Görüntü bulunamadıysa hata mesajı göster
            axes[row, col].text(0.5, 0.5, f'Görüntü bulunamadi\n{img_id}', 
                              ha='center', va='center', fontsize=10)
            axes[row, col].set_title(f'ID: {img_id}', fontsize=10)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ {len(sample_images)} adet araç görüntüsü gösterildi!")

def show_car_statistics():
    """
    Araç verileri hakkinda istatistikleri gösterir.
    """
    print(f"\n�� ARAÇ TESPİTİ İSTATİSTİKLERİ")
    print("=" * 50)
    
    # Temel istatistikler
    total_cars = len(df)
    total_images = len(index)
    cars_per_image = total_cars / total_images
    
    print(f"�� Toplam araç sayisi: {total_cars}")
    print(f"📸 Toplam görüntü sayisi: {total_images}")
    print(f"📈 Görüntü başina ortalama araç: {cars_per_image:.2f}")
    
    # Araç boyutları istatistikleri
    print(f"\n📏 Araç Boyutlari:")
    print(f"  - Ortalama genişlik: {df['w'].mean():.3f} (normalize)")
    print(f"  - Ortalama yükseklik: {df['h'].mean():.3f} (normalize)")
    print(f"  - En büyük araç: {df['w'].max():.3f} x {df['h'].max():.3f}")
    print(f"  - En küçük araç: {df['w'].min():.3f} x {df['h'].min():.3f}")
    
    # En çok araç içeren görüntüler
    cars_per_img = df.groupby('image_id').size().sort_values(ascending=False)
    print(f"\n🏆 En çok araç içeren görüntüler:")
    for i, (img_id, count) in enumerate(cars_per_img.head(5).items()):
        print(f"  {i+1}. {img_id}: {count} araç")

# =============================================================================
# ANA PROGRAM ÇALIŞTIRMA
# =============================================================================
# Araç görüntülerini panelde göster
display_sample_cars(6)

# İstatistikleri göster
show_car_statistics()