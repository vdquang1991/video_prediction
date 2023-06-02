import os
import numpy as np
from datetime import date, timedelta

def read_data_from_file(filepath):
    with open(filepath, 'r') as f:
        s = f.readlines()
    s = s[2:]
    return s

def checknamnhuan(year):
    isnamnhuan = False
    if year%400==0:
        isnamnhuan = True
    if (year%4==0) and (year%100!=0):
        isnamnhuan = True
    return isnamnhuan

def processing_data(data, csvfile):
    index = 0
    prev_year = 0
    f = open(csvfile, 'w')
    full_new_data = []
    while index < len(data):
        year = int(data[index])
        if prev_year > 0 and year > prev_year + 1:      # Trường hợp khuyết 1 năm ở giữa
            prev_year = prev_year + 1
            if checknamnhuan(prev_year):
                new_one_year_data = list(np.zeros(shape=(366)) - 99.)   # Tạo dữ liệu gồm 366 ngày với giá trị = -99.
            else:
                new_one_year_data = list(np.zeros(shape=(365)) - 99.)   # Tạo dữ liệu gồm 365 ngày với giá trị = -99.

            # concat giá trị với days
            start_day = date(prev_year, 1, 1)
            data_combine_with_day = []
            for i in range(len(new_one_year_data)):
                current_day = start_day + timedelta(i)
                data_combine_with_day.append([current_day, new_one_year_data[i]])
                full_new_data.append([current_day, new_one_year_data[i]])

            # Write data to file
            for item in data_combine_with_day:
                f.write("%s,%10.3f\n" % (item[0], item[1]))
        else:                                       # Trường hợp các năm liên tục
            # Đọc các giá trị của 1 năm từ biến data
            data_one_year = np.zeros(shape=(31, 12)) - 99.
            for i in range(1, 32):
                row = np.asarray(data[index + i].split(), dtype=float)[1:]
                data_one_year[i-1] = row
            data_one_year = np.transpose(data_one_year)

            # Convert từ mảng 2 chiều [12, 31] sang mảng 1 chiều 366 or 365 dòng
            new_one_year_data = []
            for month in range(12):
                if month==0 or month==2 or month==4 or month==6 or month==7 or month==9 or month==11:
                    new_one_year_data += list(data_one_year[month][:31])
                elif month==3 or month==5 or month==8 or month==10:
                    new_one_year_data += list(data_one_year[month][:30])
                else:
                    if checknamnhuan(year):
                        new_one_year_data += list(data_one_year[month][:29])
                    else:
                        new_one_year_data += list(data_one_year[month][:28])

            # concat giá trị với days
            start_day = date(year, 1, 1)
            data_combine_with_day = []
            for i in range(len(new_one_year_data)):
                current_day = start_day + timedelta(i)
                data_combine_with_day.append([current_day, new_one_year_data[i]])
                full_new_data.append([current_day, new_one_year_data[i]])

            # Tăng index để đọc năm tiếp theo
            index += 32
            prev_year = year

            # Write data to file
            for item in data_combine_with_day:
                f.write("%s,%10.3f\n"%(item[0], item[1]))
    f.close()
    return full_new_data

root_folder = 'Raw_data'
dest_folder = 'csv_data'
if not os.path.exists(dest_folder):
    os.mkdir(dest_folder)
all_files = os.listdir(root_folder)
for file in all_files:
    filepath = os.path.join(root_folder, file)
    csvname = file.split('.')[0] + '.csv'
    csvfile = os.path.join(dest_folder, csvname)
    data = read_data_from_file(filepath)
    full_new_data = processing_data(data, csvfile)