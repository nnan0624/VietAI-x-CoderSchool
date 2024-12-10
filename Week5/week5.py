import pandas as pd
file_path = '/Users/coderschool/Documents/Projects /VietAI-x-CoderSchool/Week5/ds_salaries.csv'  
df = pd.read_csv(file_path)

# Task 1: Job title nào có mức lương trung bình theo USD cao nhất
highest_avg_salary_job = df.groupby('job_title')['salary_in_usd'].mean().idxmax()
highest_avg_salary = df.groupby('job_title')['salary_in_usd'].mean().max()
print(f"Task 1: Job title với mức lương trung bình cao nhất: {highest_avg_salary_job} - ${highest_avg_salary:.2f}")

# Task 2: Data Scientist mang quốc tịch nào có mức lương trung bình cao nhất
data_scientists = df[df['job_title'] == 'Data Scientist']
highest_avg_salary_nationality = data_scientists.groupby('employee_residence')['salary_in_usd'].mean().idxmax()
highest_avg_salary_value = data_scientists.groupby('employee_residence')['salary_in_usd'].mean().max()
print(f"Task 2: Quốc tịch của Data Scientist có mức lương trung bình cao nhất: {highest_avg_salary_nationality} - ${highest_avg_salary_value:.2f}")

# Task 3: So sánh lương trung bình giữa nhân sự ngoại quốc và trong nước ở USA
usa_employees = df[df['company_location'] == 'US']
foreign_employees_avg_salary = usa_employees[usa_employees['employee_residence'] != 'US']['salary_in_usd'].mean()
domestic_employees_avg_salary = usa_employees[usa_employees['employee_residence'] == 'US']['salary_in_usd'].mean()
print(f"Task 3: Lương trung bình nhân sự ngoại quốc ở USA: ${foreign_employees_avg_salary:.2f}")
print(f"Task 3: Lương trung bình nhân sự trong nước ở USA: ${domestic_employees_avg_salary:.2f}")

# Task 4: Mức lương trung bình của Data Engineer (SE level) ở các công ty nhỏ và vừa ở USA
data_engineer_se = df[
    (df['job_title'] == 'Data Engineer') &
    (df['experience_level'] == 'SE') &
    (df['company_location'] == 'US') &
    (df['company_size'].isin(['S', 'M']))
]
avg_salary_data_engineer_se = data_engineer_se['salary_in_usd'].mean()
print(f"Task 4: Mức lương trung bình của Data Engineer (SE level) ở công ty nhỏ và vừa ở USA: ${avg_salary_data_engineer_se:.2f}")

# Task 5: So sánh mức lương của Data Scientist ở USA ở các mức trình độ khác nhau
data_scientist_usa = df[(df['job_title'] == 'Data Scientist') & (df['company_location'] == 'US')]
avg_salary_by_experience = data_scientist_usa.groupby('experience_level')['salary_in_usd'].mean()
print("\nTask 5: Mức lương trung bình của Data Scientist ở USA theo trình độ:")
print(avg_salary_by_experience)

# Task 6: Tỉ lệ nhân sự ngoại quốc ở các công ty lớn tại USA
large_companies_usa = df[(df['company_size'] == 'L') & (df['company_location'] == 'US')]
foreign_percentage = (
    large_companies_usa[large_companies_usa['employee_residence'] != 'US'].shape[0] /
    large_companies_usa.shape[0]
) * 100
print(f"\nTask 6: Tỉ lệ nhân sự ngoại quốc ở các công ty lớn tại USA: {foreign_percentage:.2f}%")

# Task 7: Top 3 job title có mức lương trung bình cao nhất ở MI level dành cho nhân sự trong nước
domestic_mi = df[(df['experience_level'] == 'MI') & (df['employee_residence'] == df['company_location'])]
top_3_mi_jobs = (
    domestic_mi.groupby('job_title')['salary_in_usd']
    .mean()
    .sort_values(ascending=False)
    .head(3)
)
print("\nTask 7: Top 3 job title có mức lương trung bình cao nhất ở MI level dành cho nhân sự trong nước:")
print(top_3_mi_jobs)

# Task 8: Job nào ở Germany (DE) có mức lương trung bình thấp nhất
jobs_in_germany = df[df['company_location'] == 'DE']
lowest_avg_salary_job_germany = jobs_in_germany.groupby('job_title')['salary_in_usd'].mean().idxmin()
lowest_avg_salary_germany = jobs_in_germany.groupby('job_title')['salary_in_usd'].mean().min()
print(f"\nTask 8: Job ở Germany có mức lương trung bình thấp nhất: {lowest_avg_salary_job_germany} - ${lowest_avg_salary_germany:.2f}")

# Task 9: Job nào có khoảng cách thu nhập giữa SE và MI ít nhất
se_mi_salary_gap = df[df['experience_level'].isin(['SE', 'MI'])].groupby(['job_title', 'experience_level'])[
    'salary_in_usd'
].mean().unstack()
se_mi_salary_gap['gap'] = abs(se_mi_salary_gap['SE'] - se_mi_salary_gap['MI'])
job_smallest_gap = se_mi_salary_gap['gap'].idxmin()
smallest_gap = se_mi_salary_gap['gap'].min()
print(f"\nTask 9: Job có khoảng cách thu nhập giữa SE và MI ít nhất: {job_smallest_gap} - ${smallest_gap:.2f}")