import cv2
import numpy as np

# =========================
# 1. 讀取圖片
# =========================
img_path = r"C:\Users\user\Desktop\embedded_image\images\001.jpg"

data = np.fromfile(img_path, dtype=np.uint8)
img = cv2.imdecode(data, cv2.IMREAD_COLOR)

if img is None:
    print("圖片讀取失敗")
    exit()

result = img.copy()

# =========================
# 2. 灰階
# =========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# 3. 高斯模糊
# =========================
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# =========================
# 4. 二值化
# =========================
_, binary = cv2.threshold(
    blur,
    50,
    255,
    cv2.THRESH_BINARY_INV
)

# =========================
# 5. Contour
# =========================
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 畫 contour 結果
contour_result = img.copy()

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 30:
        cv2.drawContours(contour_result, [contour], -1, (255, 0, 0), 2)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(contour_result, (cx, cy), 3, (0, 0, 255), -1)
            print(f"Contour 中心=({cx}, {cy}), 面積={area:.2f}")



# =========================
# 8. 顯示結果
# =========================
cv2.imshow("Blur", blur)
cv2.imshow("Binary", binary)
cv2.imshow("Contour Result", contour_result)

cv2.waitKey(0)
cv2.destroyAllWindows()