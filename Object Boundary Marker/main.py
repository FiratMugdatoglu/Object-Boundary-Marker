import cv2
import numpy as np

# Görüntüyü yükleme
img = cv2.imread("hand90.png")

# Gri tonlamalı görüntü oluşturma
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Kenarları algılama
edges = cv2.Canny(gray_img, 30, 200)

# Konturları bulma
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Her kontur için döngü
for contour in contours:
    # Konturun alanını hesaplama
    area = cv2.contourArea(contour)

    # Konturun dış kenar çizgilerini yaklaşıkla
    perimeter = cv2.arcLength(contour, True)

    # Konturun şeklini belirleme
    circularity = 4 * np.pi * area / (perimeter ** 2)

    # Elips benzerliği hesaplama
    ellipse = cv2.fitEllipse(contour)
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    ellipticity = major_axis / minor_axis

    # Alan bazında uzunluk genişlik oranı
    rectangularity = area / (major_axis * minor_axis)

    # Konturun merkezini bulma
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Uzunluk ve genişlik hesaplama
    x, y, w, h = cv2.boundingRect(contour)
    elongation = max(w, h) / min(w, h)

    # Sonuçları konsola yazdırma
    print("Circularity:", circularity)
    print("Ellipticity:", ellipticity)
    print("Rectangularity:", rectangularity)
    print("Elongation:", elongation)

    # Konturları çizme
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

# Sonuçları görüntüleme
cv2.imshow("Detected Shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
