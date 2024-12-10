import pandas as pd
import os

file_path = '/Users/coderschool/Documents/Projects /VietAI-x-CoderSchool/Week4/Week4_1_data.csv'
data = pd.read_csv(file_path)

# Task 1: Thống kê số lượng khách hàng theo học vấn
education_stats = data['educational_level'].value_counts()
print("Task 1: Thống kê số lượng khách hàng theo học vấn")
print(education_stats)

# Task 2: In ra 20 khách hàng có thu nhập cá nhân (annual_income) cao nhất
top_20_income = data.nlargest(20, 'annual_income')[['customer_id', 'annual_income']]
print("\nTask 2: Top 20 khách hàng có thu nhập cá nhân cao nhất")
print(top_20_income)

# Task 3: In ra những khách hàng sinh sau năm 1960 và thu nhập cá nhân trên $50,000/năm
customers_after_1960_and_income = data[(data['year_of_birth'] > 1960) & (data['annual_income'] > 50000)]
print("\nTask 3: Khách hàng sinh sau năm 1960 và thu nhập trên $50,000/năm")
print(customers_after_1960_and_income)

# Task 4: Kết hợp điều kiện của Task 2 & 3
combined_task = customers_after_1960_and_income.nlargest(20, 'annual_income')[['customer_id', 'year_of_birth', 'annual_income']]
print("\nTask 4: Kết hợp điều kiện Task 2 & Task 3")
print(combined_task)

# Task 5: In ra những khách hàng có tình trạng hôn nhân là đã kết hôn or đã ly hôn
marital_status_filter = data[data['marital_status'].isin(['Married', 'Divorced'])]
print("\nTask 5: Khách hàng đã kết hôn hoặc đã ly hôn")
print(marital_status_filter)

