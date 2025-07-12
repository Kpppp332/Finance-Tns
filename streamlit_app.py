import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# โหลดโมเดล streamlit run "c:/PKP/Robot Challenge/Ai math/for_test/streamlit_app.py"
model = joblib.load("kp1_model.pkl")

st.set_page_config(page_title="AI Financial Predictor",  layout="centered")
st.title(" AI ทำนายเงินเหลือสิ้นเดือน")
st.markdown("กรอกข้อมูลพฤติกรรมทางการเงินของคุณ แล้วให้ AI ช่วยประเมิน!")

# Input form
salary = st.number_input("เงินเดือน (บาท)", min_value=1000, max_value=100000, value=1000, step=500)
q1 = st.number_input("คุณซื้อข้าวหรือของกินในโรงอาหารรวมกี่บาทต่อสัปดาห์", value=0, step=10)
q2 = st.number_input("คุณซื้อของในมาร์ทรวมเป็นเงินกี่บาทต่อสัปดาห์", value=0, step=10)
q3 = st.number_input("คุณซื้อซื้อน้ำหรือของกินในร้านน้ำรวมเป็นเงินกี่บาทต่อสัปดาห์", value=0, step=10)
q4 = st.number_input("ในวันอาทิตย์ถ้าออกข้างนอกเสียกี่บาท (ถ้าไม่ออกใส่ 0)", value=0, step=10)
q5 = st.number_input("คุณเสียเงินในซักผ้ากี่บาทต่อสัปดาห์", value=0, step=10)
q6 = st.number_input("คุณเสียเงินให้ร้านซักรีดกี่บาทต่อสัปดาห์ (ไม่มีใส่ 0)", value=0, step=10)
q9 = st.number_input("คุณเสียเงินให้ Netflix, Spotify ต่อเดือน", value=0, step=10)
q10 = st.number_input("เติมเกม/นิยายต่อเดือน", value=0, step=10)
q11 = st.number_input("ซื้อของออนไลน์ต่อเดือน", value=0, step=10)

st.subheader("ให้คะแนนความจำเป็นแต่ละรายการ (0-10)")

importance = {}
importance['โรงอาหาร'] = st.slider("ความจำเป็น: ข้าวโรงอาหาร",value=0,max_value=10,)
importance['มาร์ท'] = st.slider("ความจำเป็น: ของจากมาร์ท",value=0,max_value=10)
importance['ร้านน้ำ'] = st.slider("ความจำเป็น: น้ำ/ขนม", value=0,max_value=10)
importance['ออกวันอาทิตย์'] = st.slider("ความจำเป็น: ใช้นอกบ้านวันอาทิตย์", value=0,max_value=10)
importance['ซักผ้า'] = st.slider("ความจำเป็น: ซักผ้า", value=0,max_value=10)
importance['ซักรีด'] = st.slider("ความจำเป็น: ซักรีด", value=0,max_value=10)
importance['Netflix'] = st.slider("ความจำเป็น: Streaming", value=0,max_value=10)
importance['เกม/นิยาย'] = st.slider("ความจำเป็น: เกม/นิยาย", value=0,max_value=10)
importance['ของออนไลน์'] = st.slider("ความจำเป็น: ซื้อของออนไลน์", value=0,max_value=10)

if st.button("ทำนายเงินเหลือสิ้นเดือน"):
    # สร้าง DataFrame สำหรับทำนาย
    input_df = pd.DataFrame([[salary,q1,q2,q3,q4,q5,q6,int(q9/4),int(q10/4),int(q11/4)]],
                            columns=['salary','q1(food)','q2(mart)','q3(drink)','q4(Sunday-pay)',
                                     'q5(washAuto)','q6(wash store)','q9(netfilx)','q10(game&coin)','q11(onlineshop)'])
    predic=model.predict(input_df)[0]
    prediction_array = predic
    margin = prediction_array * 0.05
    lower_bound = max(0, prediction_array - margin)
    upper_bound = prediction_array + margin

    st.subheader(" ผลการทำนาย: ")
    st.write(f"เงินเหลือสัปดาห์ถัดไป (ประมาณ):**{predic:,.0f} บาท** ")
    st.write(f"เงินเหลือสัปดาห์ถัดไป (ประมาณแบบช่วง): **{lower_bound:,.0f} - {upper_bound:,.0f} บาท**")
    st.info(" ถ้าเงินเหลือน้อยกว่าที่ควร อาจพิจารณาลดค่าใช้จ่ายฟุ่มเฟือยหรือลดการซื้อของออนไลน์")
    st.subheader("AI วางแผนรายจ่ายทั้งเดือน")

    # รายการใช้จ่าย
    items = list(importance.keys())
    weekly_costs = [q1, q2, q3, q4, q5, q6, q9/4, q10/4, q11/4]
    monthly_budget = salary * 1.0

    # ปรับข้อมูลรายเดือน
    monthly_costs = [c * 4 for c in weekly_costs]
    values = [importance[i] for i in items]

    # Knapsack สำหรับทั้งเดือน
    def suggest_spending_plan(items, costs, values, budget):
        n = len(items)
        dp = [[0 for _ in range(int(budget) + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for b in range(1, int(budget) + 1):
                if costs[i - 1] <= b:
                    dp[i][b] = max(dp[i - 1][b], dp[i - 1][int(b - costs[i - 1])] + values[i - 1])
                else:
                    dp[i][b] = dp[i - 1][b]

        selected = []
        b = int(budget)
        for i in range(n, 0, -1):
            if dp[i][b] != dp[i - 1][b]:
                selected.append(i - 1)
                b -= int(costs[i - 1])

        selected_items = [items[i] for i in selected]
        total_cost = sum([costs[i] for i in selected])
        return selected_items, total_cost

    suggested_items, used = suggest_spending_plan(items, monthly_costs, values, monthly_budget)

    st.write(" รายการที่แนะนำให้ 'ใช้จ่ายเหมือนเดิม' ตลอดเดือน:")
    for i in suggested_items:
        st.write(f"• {i}")
    st.write(f"รวมค่าใช้จ่าย: {used:,.0f} จากงบ {monthly_budget:,.0f} บาท")

    # เตรียมข้อมูลสำหรับกราฟ
    categories = ['q1(food)','q2(mart)','q3(drink)','q4(Sunday-pay)',
                  'q5(washAuto)','q6(wash store)','q9(netfilx)','q10(game&coin)','q11(onlineshop)']
    amounts = [q1, q2, q3, q4, q5, q6, q9/4, q10/4, q11/4]
    st.subheader(" AI เตือนล่วงหน้า: คุณจะเงินหมดเมื่อไหร่?")

  # รวมค่าใช้จ่ายต่อสัปดาห์
    weekly_expense = q1 + q2 + q3 + q4 + q5 + q6 + (q9/4) + (q10/4) + (q11/4)

  # เงินเริ่มต้น
    initial_money = salary

  # เตรียมลูปจำลอง 1 เดือน (4 สัปดาห์)
    weeks = 4
    money_left = []
    remaining = initial_money

    will_run_out = False
    run_out_week = None

    for week in range(1, weeks + 1):
        remaining -= weekly_expense
        money_left.append(remaining)

    if not will_run_out and remaining < 0:
        will_run_out = True
        run_out_week = week

# แสดงผล
    for i, m in enumerate(money_left):
        st.write(f"สัปดาห์ที่ {i+1}: เงินเหลือประมาณ **{m:,.0f} บาท**")

# แสดงข้อความเตือน
    if will_run_out:
        st.error(f" คุณจะเงินหมดใน **สัปดาห์ที่ {run_out_week}** ถ้าใช้จ่ายเท่าเดิมทุกสัปดาห์")
    else:
        st.success(" จากการคาดการณ์ คุณจะไม่เงินหมดในช่วง 4 สัปดาห์นี้")

    # ตรวจสอบว่ามีข้อมูลรวมแล้ว > 0 ค่อยวาด
    if sum(amounts) > 0:
        fig, ax = plt.subplots()
        ax.pie(amounts, labels=categories, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.warning("ไม่มีข้อมูลเพียงพอที่จะสร้างกราฟ")

