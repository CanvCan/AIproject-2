# Proje 1: Generative Adversarial Networks (GAN)

## Proje Süreci
Bu projede GAN yapısını anlamak, farklı veri setleri üzerinde çalışmak ve model performansını değerlendirmek amaçlanmıştır. Aşağıdaki adımlar izlenmiştir:

### 1. Ön Çalışma - Other Model
- **Model:** İnternetten alınan bir "other model" ile GAN yapısını anlamaya çalıştım.
- **Veri Seti:** CIFAR-10 ve MNIST.
- **Sonuçlar:**
  - CIFAR-10: Başarılı (`cifar10_othermodel` klasörü).
  - MNIST: Başarılı.

### 2. GAN_model1 Geliştirilmesi
- **Teknoloji:** Keras kütüphanesi kullanılarak geliştirildi.
- **Denemeler:**
  1. **MNIST veri seti** ile eğitildi → **Başarılı** (`mnist_images1` klasörü).
  2. **Pigs (47 görüntü)** ile eğitildi → **Başarılı** (`pig_images1` klasörü).
  3. **CIFAR-10 veri seti** ile denendi:
     - İlk deneme → **Başarısız** (`cifar10_images1` klasörü).
     - Model revize edildi (katmanlar vb. ayarlandı).
     - İkinci deneme → **Başarısız** (`cifar10_images2` klasörü).
     - Model revize edildi.
     - Üçüncü deneme → **Başarısız** (`cifar10_images3` klasörü).

**Sonuç:** GAN_model1, **basit veri setlerinde başarılı** ancak **karmaşık veri setlerinde öğrenme gerçekleştiremiyor**.

### 3. GAN_model2 Geliştirilmesi
- **Teknoloji:** PyTorch kütüphanesi kullanılarak geliştirildi.
- **Denemeler:**
  1. **Pigs (47 görüntü)** ile eğitildi → **Başarılı** (`pig_images2` klasörü).
  2. **Pigs (924 görüntü)** ile eğitildi → **Başarılı** (`pig_images3` klasörü).

**Sonuç:** GAN_model2, **sağlıklı öğrenebilen bir modeldir**.

### 4. Model Dosyaları
- `GAN_model1.py` ve `GAN_model2.py` ilgili klasörlerde bulunmaktadır.

---

# Proje 2: Genetik Algoritma ile Knapsack Problemi Çözümü

## Proje Süreci
Bu projede genetik algoritma kullanarak **Knapsack (Çanta) Problemi** çözülmüştür.

### 1. Ön Çalışma
- **Proje 1 için yazılan genetik algoritma kodu incelendi.**
- **Kod, Knapsack problemine uygun hale getirildi.**
- **Doğru ağırlık ve değer ikililerinin belirlenmesi sağlandı.**

### 2. Knapsack Problemi Çözümü
- **Algoritmanın başarısı, doğal seçilim sürecine bağlıdır.**
- **Yanlış ağırlık ve değer belirlenmesi, algoritmanın etkin çalışmasını engelleyebilir.**

### 3. Kod Dosyası
- `Knapsack.java` ilgili klasörlerde bulunmaktadır.

---
