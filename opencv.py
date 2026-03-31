import cv2
import numpy as np

# 圖片路徑
img_path = r"C:\Users\user\Desktop\embedded_image\images\001.jpg"

# 讀圖（Windows 穩定版）
data = np.fromfile(img_path, dtype=np.uint8)
img = cv2.imdecode(data, cv2.IMREAD_COLOR)

if img is None:
    print("圖片讀取失敗")
    exit()

# 1. 灰階
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Gaussian Blur（高斯模糊）
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# 3. Sobel 邊緣強化
sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

# 合併 X / Y 梯度
sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = np.uint8(np.clip(sobel, 0, 255))

# 4. Canny 邊緣偵測
canny = cv2.Canny(sobel, 50, 150)

# 5. 霍夫圓轉換（Hough Circle Transform）
circles = cv2.HoughCircles(
    blur,                      # 建議用 blur 或 gray
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=50,
    param2=20,
    minRadius=5,
    maxRadius=30
)

# 畫圓
result = img.copy()

if circles is not None:
    circles = np.uint16(np.around(circles))

    for c in circles[0, :]:
        x, y, r = c

        # 外圓
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)

        # 圓心
        cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

        print(f"圓心: ({x}, {y}), 半徑: {r}")

# 顯示結果
cv2.imshow("Sobel", sobel)
cv2.imshow("Canny", canny)
cv2.imshow("Hough Circle Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()