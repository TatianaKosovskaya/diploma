
import streamlit as st

header = st.container()
with header:
    st.header("Recommendation System for Clients")
    
    st.markdown('''
    Below is a possible user interface for bank customers recommending new banking products based on collaborative filtering.
                    ''')

    st.write("**Please select the product you own.**")

import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
#import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from numpy import dot
from numpy.linalg import norm
import surprise
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.prediction_algorithms.knns import KNNBasic

data_1 = pd.read_csv('train_df_1.csv')
#data_2 = pd.read_csv('train_df_2.csv')
#data_3 = pd.read_csv('train_df_3.csv')
#data_4 = pd.read_csv('train_df_4.csv')
#data_5 = pd.read_csv('train_df_5.csv')
#data_6 = pd.read_csv('train_df_6.csv')
#data_7 = pd.read_csv('train_df_7.csv')
#data_8 = pd.read_csv('train_df_8.csv')
#data_9 = pd.read_csv('train_df_9.csv')
#data_10 = pd.read_csv('train_df_10.csv')
#data = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10], axis=0, ignore_index=True)

data=data_1
data.drop('Unnamed: 0', axis= 1 , inplace= True )

col1, col2 = st.columns(2)

product_list = list(data.columns)
id = 9999999
product_list = product_list[24:]

with col1:
    arr1 = [ (1 if (st.radio(str(product_list[i]), ['Not Owns', 'Owns'], horizontal=True, index=0)) == 'Owns' else 0) for i in range(0,12)]


with col2:
    arr2 = [ (1 if (st.radio(str(product_list[i]), ['Not Owns', 'Owns'], horizontal=True, index=0)) == 'Owns' else 0) for i in range(12,24)]

#st.table(arr1)
#st.table(arr2)
click = st.button('Get Recommendations')

data.loc[-1] = ['9999999', id, '9999999', '9999999', '9999999', '9999999', '9999999', 9999999, '9999999', 9999999, '9999999', '9999999', '9999999', '9999999', '9999999', 
                '9999999', '9999999', '9999999', 9999999, 9999999, '9999999', 9999999, 9999999, '9999999'] + arr1 + arr2
data.index = data.index + 1 
data = data.sort_index()

if click:
    
    with st.spinner('Due to the calculation of the machine learning model you have to wait a few seconds for the result.'):

        if sum(arr1 + arr2) == len(product_list):

            st.success('You already have all the products.')
        
        else:
            data.rename( columns={'Derivada account':'Derivative account'}, inplace=True )
            l = list(data.columns)
            data_desc = pd.read_csv('data_desc.csv')
            data_desc['Column Name'] = l
            data_2 = data.copy()
            data_2 = data_2[data_2['Employee index'].notna()]
            data_2 = data_2[data_2['Country residence'].notna()]
            data_2 = data_2[data_2['Sex'].notna()]
            data_2 = data_2[data_2['Index'].notna()]
            data_2 = data_2[data_2['Index real'].notna()]
            data_2 = data_2[data_2['Residence index'].notna()]
            data_2 = data_2[data_2['Foreigner index'].notna()]
            data_2 = data_2[data_2['Deceased index'].notna()]
            data_2 = data_2[data_2['Province code'].notna()]
            data_2 = data_2[data_2['Province name'].notna()]
            data_2 = data_2[data_2['Activity index'].notna()]
            data_2 = data_2[data_2['Payroll'].notna()]
            data_2 = data_2[data_2['Real pensions'].notna()]
            data_2.drop(columns=['Last date as primary customer','Spouse index'], inplace=True)
            data_2['Gross income of the household'] = data_2['Gross income of the household'].fillna(134245.63)
            data_2['Customer type'] = data_2['Customer type'].fillna(1)
            data_2['Customer relation type'] = data_2['Customer relation type'].fillna('I')
            data_2['Channel index'] = data_2['Channel index'].fillna('KHE')
            data_2['Segmentation index'] = data_2['Segmentation index'].fillna('02 - PARTICULARES')
            data_2[['Age', 'Seniority (in months)']] = data_2[['Age', 'Seniority (in months)']].astype(int)
            data_2 = data_2[data_2["Age"] <= 100]
            data_2 = data_2[data_2["Age"] >= 18]
            data_2 = data_2[data_2["Seniority (in months)"] != -999999]
            # Исправление категорий столбца - indrel_1mes
            data_2['Customer type'].replace('1', 1, inplace=True)
            data_2['Customer type'].replace('1.0', 1, inplace=True)
            data_2['Customer type'].replace('2', 2, inplace=True)
            data_2['Customer type'].replace('2.0', 2, inplace=True)
            data_2['Customer type'].replace('3', 3, inplace=True)
            data_2['Customer type'].replace('3.0', 3, inplace=True)
            data_2['Customer type'].replace('4', 4, inplace=True)
            data_2['Customer type'].replace('4.0', 4, inplace=True)
            data_2['Customer type'].replace('P', 5, inplace=True)
            data_2['Customer type'].replace('None',np.nan, inplace=True)

            le = LabelEncoder()
            raw_target = data_2.iloc[:, 22:].idxmax(1)
            transformed_target = le.fit_transform(raw_target)
            data_2['service_opted'] = transformed_target
            data_2['service_opted'] = data_2['service_opted'].astype('uint8')
            names = raw_target.value_counts().index
            values = raw_target.value_counts().values
            names = [data_desc[data_desc['Column Name'] == name]['Description'].values[0] for name in names]
            user_item_matrix = pd.crosstab(index=data_2.Code, columns=le.transform(raw_target), values=1, aggfunc='sum')
            user_item_matrix.fillna(0, inplace=True)
            uim_arr = np.array(user_item_matrix)
            for row,item in tqdm(enumerate(uim_arr)):
                for column,item_value in enumerate(item):
                    uim_arr[row, column] = uim_arr[row, column] / sum(item)
            user_item_ratio_matrix = pd.DataFrame(uim_arr, columns=user_item_matrix.columns, index=user_item_matrix.index)
            user_item_ratio_stacked = user_item_ratio_matrix.stack().to_frame()
            user_item_ratio_stacked['ncodpers'] = [index[0] for index in user_item_ratio_stacked.index]
            user_item_ratio_stacked['service_opted'] = [index[1] for index in user_item_ratio_stacked.index]
            user_item_ratio_stacked.reset_index(drop=True, inplace=True)
            user_item_ratio_stacked.rename(columns={0:"service_selection_ratio"}, inplace=True)
            user_item_ratio_stacked = user_item_ratio_stacked[['ncodpers','service_opted', 'service_selection_ratio']]
            user_item_ratio_stacked.drop(user_item_ratio_stacked[user_item_ratio_stacked['service_selection_ratio']==0].index, inplace=True)
            user_item_ratio_stacked.reset_index(drop=True, inplace=True)
            reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,1), skip_lines=1)
            data = Dataset.load_from_df(user_item_ratio_stacked, reader=reader)
            trainset = data.build_full_trainset()
            svd = SVD()
            svd_results = cross_validate(algo=svd, data=data, cv=4)
            svd = SVD()
            svd.fit(trainset)
            def get_recommendation(uid,model):    
                recommendations = [(uid, sid, data_desc[data_desc['Column Name'] == le.inverse_transform([sid])[0]]['Description'].values[0], model.predict(uid,sid).est) for sid in range(24)]
                recommendations = pd.DataFrame(recommendations, columns=['uid', 'sid', 'service_name', 'pred'])
                recommendations.sort_values("pred", ascending=False, inplace=True)
                recommendations.reset_index(drop=True, inplace=True)
                return recommendations

            d = get_recommendation(9999999,svd)
            d = d['service_name']
            st.table(d)
            d = d[:5]
            d = list(d)
            st.subheader('Recommended products:')
            for i in range(len(d)):
                st.write(str(i + 1), ". ", d[i])
                
                if d[i] == 'Saving Account':
                    st.caption('A saving account is a type of banking product that allows customers to deposit and withdraw funds, while earning interest on their balance. These accounts typically have lower interest rates compared to other investment options, but they also offer more liquidity and flexibility. Saving accounts can be opened by individuals, as well as joint accounts for couples, families, or businesses. Some saving accounts may require a minimum deposit to open, and may have minimum balance requirements to avoid fees. Overall, saving accounts are a popular and convenient option for customers looking to earn some interest on their savings while still maintaining easy access to their funds.')
                if d[i] == 'Home Account':
                    st.caption('A home account is a banking product that is designed to help customers manage their household finances. This type of account can be used for paying bills, depositing paychecks, and setting aside money for home-related expenses such as mortgage payments, property taxes, and maintenance costs. Some home accounts also offer features such as budgeting tools and automatic savings plans to help customers save money and stay on top of their finances. Additionally, many banks offer incentives such as higher interest rates or waived fees for customers who maintain a certain balance in their home account. Overall, a home account is a useful tool for customers who want to keep their home-related finances organized and easily accessible.')
                if d[i] == 'Pensions':
                    st.caption('A banking product for customer pensions is designed to help individuals save money for their retirement years. Pensions are long-term savings plans that allow customers to contribute money on a regular basis, with the funds accumulating over time. These products offer various investment options, such as stocks, bonds, and mutual funds, which can help customers grow their savings over the years. Additionally, some pension products may offer tax benefits or employer contributions to help customers save even more. When a customer reaches retirement age, they can begin to withdraw their savings as a source of income in their retirement years.')
                if d[i] == 'particular Plus Account':
                    st.caption('The particular plus account is a banking product that offers customers a range of benefits and perks beyond traditional checking and savings accounts. Typically, this account requires a higher minimum balance and may charge a monthly maintenance fee, but in exchange, customers can enjoy features such as premium interest rates, waived fees on certain transactions, access to personalized financial advisors, and exclusive discounts on products and services. This type of account is designed for customers who maintain a higher balance and want to take advantage of additional benefits and services from their bank.')
                if d[i] == 'particular Account':
                    st.caption('A particular account is a type of banking product designed for individual customers, also known as personal or retail banking. This account allows customers to carry out daily transactions such as deposits, withdrawals, and transfers, as well as access to various services such as online banking, mobile banking, and ATM facilities. A particular account may also offer additional benefits such as lower fees, higher interest rates, or loyalty rewards. Some banks may require a minimum balance or charge fees for certain transactions or services, while others may offer free checking accounts with no minimum balance or fees. Overall, a particular account is a basic and essential banking product for individual customers to manage their finances and access various banking services.')
                if d[i] == 'Loans':
                    st.caption('A banking product for customer loans is a financial service that allows individuals and businesses to borrow money from a bank or other financial institution. Loans can be used for a variety of purposes, such as purchasing a home, financing a car, or starting a business. Banks typically offer a range of loan products with different terms, interest rates, and repayment schedules to meet the needs of different borrowers. Customers must qualify for a loan based on their creditworthiness and ability to repay the loan. Loans are often secured by collateral such as a car or property, or they may be unsecured, meaning they are not backed by any asset.')
                if d[i] == 'Current Accounts':
                    st.caption('A current account is a type of bank account that is used for day-to-day transactions. Customers can deposit money into the account and withdraw it when they need it. Current accounts usually come with a debit card that can be used to make purchases or withdraw cash from ATMs. Some banks also offer overdraft facilities on current accounts, allowing customers to borrow money for short periods of time. Current accounts may also come with features such as online banking, mobile banking, and the ability to set up direct debits and standing orders.')
                if d[i] == 'Medium-term deposits':
                    st.caption('Medium-term deposits are a type of banking product offered to customers who are looking for a low-risk investment option with a fixed rate of return. With a medium-term deposit account, customers can deposit a lump sum of money for a fixed period, usually between 1 to 5 years, and earn interest on their savings. The interest rate for medium-term deposits is typically higher than that of a regular savings account. Customers can choose to receive interest payments on a monthly, quarterly, or annual basis, or have the interest reinvested into the account. At the end of the term, the original deposit plus interest is returned to the customer. Medium-term deposits offer a safe and reliable way to save money and earn interest over a fixed period.')
                if d[i] == 'Funds':
                    st.caption('A banking product for customers is the option to invest in funds. Funds are professionally managed portfolios of investments that typically include a mix of stocks, bonds, and other securities. Customers can choose from a range of funds with different investment objectives, risk levels, and fees. Investing in funds can provide diversification, potentially higher returns, and access to investments that may be difficult to access on an individual basis. However, it is important to note that investing in funds carries risks, and customers should carefully consider their investment goals, risk tolerance, and financial situation before investing.')
                if d[i] == 'Securities':
                    st.caption('Banking products related to securities can include brokerage accounts, investment accounts, and trading accounts. These products allow customers to invest in various financial instruments, such as stocks, bonds, mutual funds, exchange-traded funds (ETFs), and options. The bank may also offer research and analysis tools to help customers make informed investment decisions. Securities accounts can be self-directed or managed by a financial advisor, and may involve varying levels of risk depending on the investment strategy chosen. Customers may be charged fees or commissions for trades and account maintenance.')
                if d[i] == 'Pensions':
                    st.caption('A banking product for customer pensions is designed to help individuals save money for their retirement years. Pensions are long-term savings plans that allow customers to contribute money on a regular basis, with the funds accumulating over time. These products offer various investment options, such as stocks, bonds, and mutual funds, which can help customers grow their savings over the years. Additionally, some pension products may offer tax benefits or employer contributions to help customers save even more. When a customer reaches retirement age, they can begin to withdraw their savings as a source of income in their retirement years.')
                if d[i] == 'Credit Card':
                    st.caption('A credit card is a banking product that allows customers to borrow funds up to a pre-set credit limit to make purchases or withdraw cash. Customers are required to pay back the borrowed funds along with interest charges and other fees within a billing cycle. Credit cards offer various benefits, such as cashback rewards, discounts, and travel perks. They also provide a convenient and secure way to make purchases both online and offline. Credit card issuers may charge annual fees, transaction fees, balance transfer fees, and other charges, so it is essential to read and understand the terms and conditions before applying for a credit card.')
                if d[i] == 'Guarantees':
                    st.caption('In banking, a guarantee is a promise made by the bank to a third party that a customer will fulfill their obligations. The most common type of guarantee is a letter of guarantee, which is often used to support a business transaction or project. A letter of guarantee acts as a formal assurance that the bank will cover the cost of the transaction if the customer is unable to fulfill their obligations. The bank may require the customer to provide collateral or a security deposit to cover the cost of the guarantee. Guarantees can be useful for businesses looking to secure contracts or funding, and for individuals who need to provide assurance to landlords or other creditors. However, it is important to understand the terms and conditions of the guarantee, as failure to meet the obligations can result in significant financial consequences.')
                if d[i] == 'Más particular Account':
                    st.caption('The Más particular account is a banking product offered to customers by some financial institutions. It is a premium current account that offers exclusive benefits such as discounts on products and services, personalized attention, and higher transaction limits. This type of account is usually designed for high net worth individuals or those with significant financial resources. In addition to the standard features of a current account, a Más particular account may also offer features such as concierge services, travel insurance, and access to VIP lounges. Some institutions may also offer investment and wealth management services to customers with a Más particular account.')
                if d[i] == 'Payroll Account':
                    st.caption('A payroll account is a banking product that is specifically designed for employees to receive their salaries from their employer. The account is linked to the employer payroll system, allowing for direct deposit of the employee salary into the account. It often comes with various benefits such as low fees or no fees, access to online banking services, and the ability to set up automatic bill payments or savings plans. Some payroll accounts may also offer additional features like cashback rewards or other perks. As a result, a payroll account can be a convenient and cost-effective way for employees to receive their salary and manage their finances.')
                if d[i] == 'Mortgage':
                    st.caption('A mortgage is a type of loan that is used to purchase a property, typically a home or a piece of land. The borrower, usually the homebuyer, receives a lump sum of money from the bank or lender and agrees to pay back the loan with interest over a set period of time, usually 15-30 years. The property acts as collateral for the loan, meaning that if the borrower fails to make payments, the lender can take possession of the property. Mortgages come with a variety of terms and conditions, such as fixed or adjustable interest rates, down payment requirements, and prepayment penalties. They are a popular banking product for customers looking to purchase a home or invest in real estate.')
                if d[i] == 'Taxes':
                    st.caption('Banks offer various products and services to assist customers with their tax-related matters. For instance, some banks provide tax payment services that allow customers to pay their taxes directly from their bank accounts. This service simplifies the tax payment process and ensures timely and accurate tax payments. Moreover, some banks offer tax refund anticipation loans to their customers. These loans provide customers with access to their expected tax refund in advance, which can be beneficial for those who need the funds immediately. The bank provides the loan, and when the customer tax refund arrives, the loan is repaid. Banks may also provide tax planning services to help customers manage their finances and reduce their tax liabilities. These services may include investment advice, retirement planning, and other financial planning services. Overall, banking products related to taxes can be helpful for customers to manage their finances and meet their tax obligations efficiently.')
                if d[i] == 'e-account':
                    st.caption('An e-account is a digital banking product that allows customers to access their accounts and manage their finances online. With an e-account, customers can check their account balances, view transaction history, transfer funds, pay bills, and receive electronic statements. E-accounts often offer additional features such as mobile banking apps and online customer support. This banking product is convenient for customers who prefer to manage their finances electronically and want to avoid visiting physical bank branches.')
                if d[i] == 'Long-term deposits':
                    st.caption('Long-term deposits are a type of banking product that allows customers to invest their money for a set period of time, usually ranging from one to ten years. These deposits typically offer higher interest rates than regular savings accounts, making them an attractive option for customers who want to earn more interest on their savings. However, long-term deposits usually require a minimum deposit amount and may impose penalties for early withdrawal. Customers can choose between fixed-rate deposits, where the interest rate remains the same for the entire term, or variable-rate deposits, where the interest rate may change based on market conditions. Long-term deposits are often used by customers who have long-term financial goals, such as saving for retirement or a child education.')
                if d[i] == 'Junior Account':
                    st.caption('A junior account is a type of banking product designed for children and teenagers under 18 years of age. It provides a safe and secure way for young people to save and manage their money while teaching them important financial skills. Junior accounts typically offer low or no fees, competitive interest rates, and access to online banking and mobile apps. They may also offer additional features such as parental controls, spending limits, and educational resources to help young account holders learn about budgeting and saving. Junior accounts can be opened by parents or legal guardians on behalf of the child and may require documentation such as birth certificates or social security numbers.')
                if d[i] == 'Short-term deposits':
                    st.caption('A short-term deposit account is a banking product that allows customers to earn interest on a sum of money that is deposited for a fixed period of time, usually less than one year. These accounts are ideal for customers who want to earn a higher interest rate than a traditional savings account, but do not want to commit their money to a long-term investment. Interest rates for short-term deposits are usually fixed and can vary depending on the amount of money deposited and the length of the term. Once the term has ended, customers can choose to withdraw their funds or roll them over into a new deposit account. Short-term deposit accounts are often used for short-term savings goals such as holidays, weddings, or home renovations.')
                if d[i] == 'Derivada Account':
                    st.caption('A derivative account is a banking product that enables customers to invest in various financial instruments, such as futures contracts, options, swaps, and other similar products. These products derive their value from an underlying asset or financial index. The goal of the derivative account is to provide customers with the opportunity to manage risk and maximize profits through investments in these complex financial products. Derivative accounts may be suitable for experienced investors who are looking for higher risk and higher reward opportunities. It is important to note that investing in derivatives can be risky and requires a thorough understanding of the product before investing.')
                if d[i] == 'Direct Debit':
                    st.caption('Direct Debit is a banking service that allows customers to authorize regular payments to a merchant or service provider from their bank account. This payment method is commonly used for recurring expenses such as utility bills, rent, or subscription services. Once the authorization is set up, the payment is automatically deducted from the customer account on a specified date, eliminating the need to remember to make the payment manually. Direct Debit offers convenience for customers and can help to ensure timely payments while reducing the risk of missed payments and associated late fees.')
                if d[i] == 'Payroll':
                    st.caption('A payroll service is a banking product that allows employers to process payroll for their employees efficiently. With this service, employers can pay their employees via direct deposit, make payments to the government for payroll taxes, and generate payroll reports. The service is often integrated with accounting software, which simplifies the process of keeping track of payroll expenses and generating financial statements. Payroll services can be used by businesses of all sizes, and they are especially useful for small and medium-sized enterprises that may not have dedicated human resources or accounting departments.')
                
                    
            #st.table(get_recommendation(9999999,svd))

