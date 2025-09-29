# VEHICLE-DETECTION-PROJECT---PREPARATION-OF-DATA-SET-IN-YOLO-FORMAT
🚦 YOLO ile Trafik Görsellerinde Araç Tespiti

Bu projeyi yapmaya başladığımda aklımda basit ama heyecan verici bir fikir vardı: “Bir trafik fotoğrafına bakıp, orada kaç tane araç olduğunu bilgisayara saydırabilir miyim?” Günlük hayatta hepimizin karşılaştığı bir durumdur; trafik ışıklarında beklerken, köprülerde ya da otoyollarda gördüğümüz araç yoğunluğunu gözle tahmin etmeye çalışırız. Fakat bunu otomatik, hızlı ve doğru bir şekilde yapabilmek çok daha değerli olurdu. İşte tam da bu noktada bilgisayarlı görü (computer vision) ve nesne tespiti (object detection) yöntemlerinden yararlandım.

Bu projede, nesne tespitinde günümüzde en çok tercih edilen algoritmalardan biri olan YOLO (You Only Look Once)’yu kullandım. YOLO’nun en önemli avantajı, bir görüntüye tek seferde bakıp oradaki nesneleri hızlıca işaretleyebilmesi ve sınıflandırabilmesidir. Yani bir insanın “bakışta” gördüğü gibi, YOLO da tek bir adımda araçları tespit edebiliyor. Bu özelliği sayesinde trafik görsellerinde bulunan araçların sayısını çıkarmak için oldukça uygun bir yöntem oldu.

🔍 Projenin Temel Adımları

Projeyi adım adım şöyle kurguladım:

Görsellerin Hazırlanması
Öncelikle elimde bulunan 6 farklı trafik görselini seçtim. Bu görsellerin bazıları yoğun trafik sahneleri, bazıları daha az sayıda aracın bulunduğu alanlardı. Amacım, modelin farklı yoğunluklardaki görüntülerde nasıl performans göstereceğini test etmekti.

Modelin Kullanılması
Araç tespiti için YOLOv8 modelini tercih ettim. Bu model, güncel sürümler arasında hem hızlı hem de yüksek doğruluk oranına sahip. Modeli projeye entegre ettim ve görselleri teker teker bu modele verdim.

Araçların Tespit Edilmesi
Her görsel modele yüklendiğinde, YOLO araçları kutucuklarla işaretledi. Buradaki güzellik şuydu: YOLO sadece araçları işaretlemekle kalmadı, aynı zamanda her görselde kaç tane araç olduğunu da sayabildi.

Sonuçların Kaydedilmesi

Araçların işaretlenmiş görselleri, daha sonra inceleyebilmek için outputs/ klasörüne kaydedildi.

Her görselin araç sayısı terminalde yazdırıldı. Böylece hangi görselde kaç araç olduğu net bir şekilde elde edilmiş oldu.
