import collections
import pandas as pd
import os

file_path = 'tt.txt'
output_file = 'city_counts.csv'

# Kiểm tra xem file có tồn tại không
if not os.path.exists(file_path):
    print(f"Không tìm thấy file {file_path}")
else:
    # 1. Đọc dữ liệu từ file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. Trích xuất danh sách các Tỉnh/Thành phố
    lines = content.split('\n')
    cities = []
    current_city = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) >= 4:
            word = parts[0]
            tag = parts[-1] 
            
            if tag == 'B-CITY':
                if current_city:
                    cities.append(" ".join(current_city))
                current_city = [word]
            elif tag == 'I-CITY':
                current_city.append(word)
            else:
                if current_city:
                    cities.append(" ".join(current_city))
                    current_city = []

    if current_city:
        cities.append(" ".join(current_city))

    # 3. Đếm số lần xuất hiện
    city_counts = collections.Counter(cities)

    # 4. Chuyển đổi dữ liệu sang DataFrame của Pandas
    # Lấy danh sách items (Tên thành phố, Số lượng) và đặt tên cột
    df = pd.DataFrame(city_counts.items(), columns=['Thành phố / Tỉnh', 'Số lượng'])

    # Sắp xếp giảm dần theo số lượng xuất hiện
    df = df.sort_values(by='Số lượng', ascending=False)

    # 5. Xuất ra file CSV
    # Sử dụng utf-8-sig để không bị lỗi font tiếng Việt khi mở bằng Excel
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"Xử lý thành công! Đã tìm thấy {len(df)} tỉnh/thành phố (unique).")
    print(f"Dữ liệu chi tiết đã được xuất ra file: {output_file}")
    
    # In thử 5 dòng đầu tiên để kiểm tra
    print("\nPreview 5 dòng đầu tiên của file CSV:")
    print(df.head())