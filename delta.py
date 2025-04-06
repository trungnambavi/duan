import cv2
import numpy as np
from pymodbus.client.sync import ModbusTcpClient as mbclient

# Mở webcam
cap = cv2.VideoCapture(1)
PLC_IP = '192.168.0.1'
PLC_port = 502
client = mbclient(host=PLC_IP, port=PLC_port)  # IP PLC
UNIT = 0x1
client.connect()
dem1 = 0
dem2 = 0
dem3 = 0
dem4 = 0

# Khởi tạo các biến toàn cục cho phóng to, thu nhỏ, di chuyển
zoom_scale = 1.0  # Mức phóng to
offset_x, offset_y = 0, 0  # Dịch chuyển vùng quan sát

def adjust_view(frame, scale, offset_x, offset_y):
    """Phóng to, thu nhỏ và di chuyển vùng quan sát."""
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Tính kích thước mới theo tỷ lệ zoom
    new_w, new_h = int(w / scale), int(h / scale)

    # Giới hạn dịch chuyển để không vượt khỏi khung ảnh
    offset_x = max(min(offset_x, (w - new_w) // 2), -(w - new_w) // 2)
    offset_y = max(min(offset_y, (h - new_h) // 2), -(h - new_h) // 2)

    # Tính vùng cắt ảnh (crop region)
    start_x = max(center_x - new_w // 2 + offset_x, 0)
    start_y = max(center_y - new_h // 2 + offset_y, 0)
    end_x = min(center_x + new_w // 2 + offset_x, w)
    end_y = min(center_y + new_h // 2 + offset_y, h)

    cropped = frame[start_y:end_y, start_x:end_x]  # Cắt ảnh theo vùng quan sát

    # Thay đổi kích thước ảnh về kích thước gốc
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized

while True:
    # Đọc từng frame từ webcam
    bien1 = client.read_holding_registers(1, 1, unit=UNIT)
    ret, frame = cap.read()
    if not ret:
        break

    # Xử lý khung hình (phóng to, thu nhỏ và di chuyển)
    frame = adjust_view(frame, zoom_scale, offset_x, offset_y)

    # Chuyển ảnh từ BGR sang HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Định nghĩa khoảng màu đỏ trong không gian màu HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Tạo mặt nạ cho màu đỏ
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Lọc các đối tượng trong ảnh dựa trên mặt nạ màu đỏ
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Chuyển sang grayscale để tìm các đối tượng rõ ràng hơn
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Làm mờ ảnh để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Tìm các contour (đường viền) trong ảnh
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Duyệt qua các contour và phân loại hình dạng
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Lọc các đối tượng có diện tích nhỏ hơn 500 pixel
            # Tính toán bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Tính chu vi của contour
            perimeter = cv2.arcLength(contour, True)

            # Tính số lượng các điểm (số cạnh)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # Phân loại hình dạng dựa trên số cạnh
            if len(approx) == 3:
                shape = "Triangle"
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)  # Vẽ hình tam giác

            elif len(approx) == 4:
                aspectRatio = float(w) / h
                shape = "Square" if abs(aspectRatio - 1) <= 0.1 else "Rectangle"
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)  # Vẽ hình vuông hoặc chữ nhật

            elif len(approx) > 4:
                shape = "Circle"
                cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 2)  # Vẽ hình tròn

            # Vẽ bounding box và tên hình dạng
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if ((len(approx) == 3) & (bien1.registers[0] == 1)):
                dem1 += 1
                {
                    client.write_register(10, dem1, unit=UNIT)  # 10
                }
                dem1 = 0

            if ((len(approx) == 4) & (bien1.registers[0] == 1)):
                dem2 += 1
                {
                    client.write_register(11, dem2, unit=UNIT)  # 10
                }
                dem2 = 0

            if ((len(approx) > 4) & (bien1.registers[0] == 1)):
                dem3 += 1
                {
                    client.write_register(12, dem3, unit=UNIT)  # 10
                }
                dem3 = 0

    # Hiển thị kết quả
    cv2.imshow('Red Object Detection and Shape Classification', frame)

    # Nhấn 'q' để thoát khỏi vòng lặp
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        zoom_scale = min(zoom_scale + 0.1, 5.0)  # Giới hạn phóng to tối đa 5x
    elif key == ord('-') or key == ord('_'):
        zoom_scale = max(zoom_scale - 0.1, 1.0)  # Giới hạn thu nhỏ tối thiểu 1x
    elif key == ord('w'):
        offset_y -= 20
    elif key == ord('s'):
        offset_y += 20
    elif key == ord('a'):
        offset_x -= 20
    elif key == ord('d'):
        offset_x += 20

# Giải phóng webcam và đóng tất cả cửa sổ OpenCV
cap.release()
cv2.destroyAllWindows()
