1. Proje;

İlk projede yazılımın geliştirilmesi şu sırayla oldu: ilk olarak internetten aldığım bir "other model" ile GAN yapısını anlamaya çalıştım, parametreleri denedim. Bu "other model" ile cifar10 ve mnıst veri seti üzerinde çalıştım. Daha sonra keras kütüphanesi yardımıyla ve araştırmalarım sonucu yararlandığım kaynaklarla GAN_model1'i oluşturdum. Bu modelle cifar10 ve pigs fake fotoğrafları ürettim. Daha sonra pytorch kütüphanesi yardımıyla ve araştırmalarım sonucu yararlandığım kaynaklarlarla GAN_model2'yi oluşturdum bu modeli de pigs fake resimleri üretmek için kullandım.

1. Other Model ---> cifar10 ile eğitildi. Sonuçlar cifar10_othermodel klasöründe. Ağ eğitimi başarılı oldu.

2. GAN_model1 ---> mnıst ile eğitildi. Sonuçlar mnıst_images1 klasöründe. Ağ eğitimi başarılı oldu.
3. GAN_model1 ---> pigs (47 images version) ile eğitildi. Sonuçlar pig_images 1 klasöründe. Ağ eğitimi başarılı oldu.

GAN_model1'in basit veri setlerinde başarılı olduğu görüldü, karmaşık veri seti olan cifar10 ile denemelere başlandı.

4. GAN_model1 ---> cifar10 ile eğitildi. Sonuçlar cifar10_images 1 klasöründe. Ağ eğitimi başarısız oldu. Modelde revizeye gidildi, katmanlar vb. ayarlandı.
5. GAN_model1 ---> cifar10 ile eğitildi. Sonuçlar cifar10_images 2 klasöründe. Ağ eğitimi başarısız oldu. Modelde revizeye gidildi, katmanlar vb. ayarlandı.
6. GAN_model1 ---> cifar10 ile eğitildi. Sonuçlar cifar10_images 3 klasöründe. Ağ eğitimi başarısız oldu.

Sonuç: GAN_model1; basit veri setlerinde başarılı, karmaşık veri setlerinde ise öğrenme gerçekleştiremiyor.

7. GAN_model2 ---> pigs (47 images version) ile eğitildi. Sonuçlar pig_images 2 klasöründe. Ağ eğitimi başarılı oldu.
7. GAN_model2 ---> pigs (924 images version) ile eğitildi. Sonuçlar pig_images 3 klasöründe. Ağ eğitimi başarılı oldu.

Sonuç: GAN_model2; öğrenebilen sağlıklı bir model.

GAN_model1.py ve GAN_model2.py ilgili klasörlerde.



2. Proje;

İkinci projede yazılımın geliştirilmesi şu sırayla oldu: ilk olarak proje 1 için yazdığım genetik algoritma kodunu inceledim. Daha sonra gerekli revizeler ile kodu knapsack problemini çözecek hale getirdim. Ağırlık ve değer ikililerinin doğru belirlenmesi genetik algoritma için hayati önem taşıyor. Aksi halde algoritma düzgün doğal seçilim yapamıyor.

Knapsack.java ilgili klasörlerde.
