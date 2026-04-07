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
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# 2. 載入眼睛分類器
# =========================
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if eye_cascade.empty():
    print("眼睛分類器載入失敗")
    exit()

# =========================
# 3. 偵測眼睛
# =========================
eyes = eye_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print("偵測到眼睛數量:", len(eyes))

# 只取最上面且最靠左/右的兩個眼睛，比較像左右眼
eyes = sorted(eyes, key=lambda e: (e[1], e[0]))
eyes = eyes[:2]
eyes = sorted(eyes, key=lambda e: e[0])  # 左眼在前、右眼在後

pupil_centers = []

# =========================
# 4. 每個眼睛 ROI 找瞳孔
# =========================
for idx, (x, y, w, h) in enumerate(eyes, start=1):
    # 畫原始眼睛框
    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 再裁掉眼角、眉毛附近，縮小搜尋範圍
    x1 = x + int(w * 0.15)
    x2 = x + int(w * 0.85)
    y1 = y + int(h * 0.25)
    y2 = y + int(h * 0.80)

    eye_roi = gray[y1:y2, x1:x2]

    if eye_roi.size == 0:
        continue

    # 顯示實際處理區域
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # =========================
    # 5. 高斯模糊
    # =========================
    blur = cv2.GaussianBlur(eye_roi, (7, 7), 0)

    # =========================
    # 6. 二值化（瞳孔暗，所以用反向）
    # =========================
    _, binary = cv2.threshold(
        blur,
        45,
        255,
        cv2.THRESH_BINARY_INV
    )

    # =========================
    # 7. 去雜訊
    # =========================
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # =========================
    # 8. 找 Contour
    # =========================
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    best_score = -999999

    roi_h, roi_w = eye_roi.shape

    for contour in contours:
        area = cv2.contourArea(contour)

        # 面積過濾：太小或太大都不要
        if area < 10 or area > 300:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        # 圓形度，越接近1越像圓
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < 0.35:
            continue

        # contour 中心
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 距離 ROI 中心越近越好
        dist_to_center = np.sqrt((cx - roi_w / 2) ** 2 + (cy - roi_h / 2) ** 2)

        # 綜合分數
        score = circularity * 100 - dist_to_center

        if score > best_score:
            best_score = score
            best_contour = contour

    # =========================
    # 9. 畫出最佳瞳孔
    # =========================
    eye_debug = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    if best_contour is not None:
        cv2.drawContours(eye_debug, [best_contour], -1, (255, 0, 0), 1)

        M = cv2.moments(best_contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        (px, py), radius = cv2.minEnclosingCircle(best_contour)

        global_x = x1 + int(px)
        global_y = y1 + int(py)

        pupil_centers.append((global_x, global_y))

        # 畫回原圖
        cv2.circle(result, (global_x, global_y), int(radius), (0, 255, 0), 2)
        cv2.circle(result, (global_x, global_y), 2, (0, 0, 255), -1)

        print(f"眼睛 {idx} 瞳孔中心=({global_x}, {global_y}), 半徑={radius:.2f}")

    # 顯示每隻眼的中間結果
    cv2.imshow(f"Eye_{idx}_Blur", blur)
    cv2.imshow(f"Eye_{idx}_Binary", binary)
    cv2.imshow(f"Eye_{idx}_Debug", eye_debug)

# =========================
# 10. 畫左右瞳孔連線與距離
# =========================
if len(pupil_centers) == 2:
    left_pupil, right_pupil = pupil_centers

    cv2.line(result, left_pupil, right_pupil, (255, 0, 255), 2)

    dist = np.sqrt(
        (right_pupil[0] - left_pupil[0]) ** 2 +
        (right_pupil[1] - left_pupil[1]) ** 2
    )

    cv2.putText(
        result,
        f"Dist: {dist:.2f}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        2
    )

    print("左右瞳孔中心距離 =", dist)
else:
    print("未能成功找到兩個瞳孔")

# =========================
# 11. 顯示結果
# =========================
cv2.imshow("Original", img)
cv2.imshow("Final Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()