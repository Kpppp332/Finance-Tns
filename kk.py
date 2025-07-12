# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt

# อ่านไฟล์
df = pd.read_csv("kp1.csv")
#df["ปกติได้เงินมาโรงเรียนเดือนละเท่าไหร่"] = pd.to_numeric(df["ปกติได้เงินมาโรงเรียนเดือนละเท่าไหร่"].str.replace(',', ' ', regex=False), errors='coerce').astype('Int64')   # ใช้ Int64 เพื่อรองรับ NaN ได้

#print(df.info())
# แยก features และ target
X = df.drop(['final_salary_per_week'], axis=1)
y = df['final_salary_per_week']

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# สร้างและ train model
model = RandomForestRegressor()
model.fit(X_train, y_train)
print(model.score(X_test,y_test))

# ทำนายบน test set
y_pred = model.predict(X_test)

# วาดกราฟ: ค่าจริง vs ค่าทำนาย
plt.figure(figsize=(10,6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='จริง (y_test)')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='ทำนาย (y_pred)', alpha=0.7)
plt.title('เปรียบเทียบค่าจริงกับค่าที่ทำนายได้')
plt.xlabel('ตัวอย่าง')
plt.ylabel('เงินเหลือสิ้นสัปดาห์')
plt.legend()
plt.tight_layout()
plt.show()
#print(model.predict([[4200,560,100,0,0,30,10,250,250,0]]))
# save model
#joblib.dump(model, "kp1_model.pkl")
#print("✅ Train เสร็จแล้ว และบันทึกเป็น financial_model.pkl")
#print(model.predict([[11499,1,2,61,1,426,1,0,0,1,0,1,859,2291,1,3660,426,0]]))
#print(df.columns)