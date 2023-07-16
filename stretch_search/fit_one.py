from scipy.optimize import curve_fit,fsolve
import pandas as pd
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

def fitting(X,k1,k2,b1):
    dsr,theta = X
    # return np.cos((theta+(k1*(dsr-1)+b1)*np.sign(1.7-theta))/2)
    return np.cos(theta)/2+0.5 + (np.sin(theta)/2 * k1*np.sin((k2*(dsr-1)+b1)))*np.sign(theta - 1.7)

# def fitting_eq(X,k1,k2,c,target):
#     return fitting(X,k1,k2) - target

file_name = 'ibmq_kolkata_rzx_0420_3'
df = pd.read_excel('excels/'+ file_name + '.xlsx')
df_dict = dict()

qubits = df['qubit'].unique()
print(qubits)
# thetas = df['theta'].unique()
# thetas_str =[]
# for theta in thetas:
#     theta_str = str(round(theta,8))
#     thetas_str.append(theta_str)
# print(thetas)
# print(thetas_str)

d_info = {'qubit': [], 'k1': [],'k2': [], 'b1': []}
df_info = pd.DataFrame(data=d_info)

d_info2 = {'qubit': [], 'data': [],'fitted_func': []}
df_info2 = pd.DataFrame(data=d_info2)

for qubit in qubits[:4]:
    sub_df= df[(df['qubit']==qubit) ]
    dsr = np.array(sub_df['stretch_ratio'])
    theta = np.array(sub_df['theta'])
    y = np.array(sub_df['pulse-00count'])


    # try:
    popt, pcov =curve_fit(fitting,(dsr,theta),y)
    fity = fitting((dsr,theta), *popt)
    for i in range(len(y)):
        df_info2.loc[len(df_info2.index)] = [qubit,y[i],fity[i]]
    # print(popt)
    # print(pcov)
    # plt.plot(, y, 'b-', label='data')
    # plt.plot(range(len(y)), , 'r-', label='fit')
    # plt.legend()
    # plt.show()
    # except:
    #     continue

    # sys.exit(0)
    # sub_df = df.loc[ and ] 
    # print(sub_df)
    
df_info2.to_excel('excels/'+ file_name + '_draw.xlsx')