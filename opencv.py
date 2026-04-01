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
# 瞳孔較黑，使用反向二值化
# =========================
_, binary = cv2.threshold(
    blur,
    50,
    255,
    cv2.THRESH_BINARY_INV
)

# =========================
# 5. Sobel
# =========================
sobel_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)

sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = np.uint8(np.clip(sobel, 0, 255))

# =========================
# 6. Canny
# =========================
canny = cv2.Canny(sobel, 50, 150)

# =========================
# 7. 霍夫圓轉換
# =========================
circles = cv2.HoughCircles(
    canny,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=50,
    param2=15,
    minRadius=5,
    maxRadius=30
)

# =========================
# 8. 畫圓
# =========================
if circles is not None:
    circles = np.uint16(np.around(circles))

    for c in circles[0, :]:
        x, y, r = c

        # 畫外圓
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)

        # 畫圓心
        cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

        print(f"圓心=({x}, {y}), 半徑={r}")

# =========================
# 9. 顯示結果
# =========================
cv2.imshow("Original", img)
cv2.imshow("Gray", gray)
cv2.imshow("Gaussian Blur", blur)
cv2.imshow("Binary", binary)
cv2.imshow("Sobel", sobel)
cv2.imshow("Canny", canny)
cv2.imshow("Hough Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()