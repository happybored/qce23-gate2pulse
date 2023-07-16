from scipy.optimize import curve_fit,fsolve
import pandas as pd
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

def fitting(x,a,b,c):
    return np.cos(a*x+b)+c 

def fitting_eq(x,a,b,c,target):
    return np.cos(a*x+b)+c - target

file_name = 'ibmq_kolkata_rzx_debug01_areafix'
df = pd.read_excel('excels/'+ file_name + '.xlsx')
df_dict = dict()

qubits = df['qubit'].unique()
print(qubits)
thetas = df['theta'].unique()
thetas_str =[]
for theta in thetas:
    theta_str = str(round(theta,8))
    thetas_str.append(theta_str)
print(thetas)
print(thetas_str)

d_info = {'qubit': [],'theta': [], 'a1': [],'a2': [], 'a3': [],'cross':[]}
df_info = pd.DataFrame(data=d_info)

for qubit in qubits:
    for theta in thetas_str:
        # print(theta)
        a = df['qubit']==qubit
        b =abs(df['theta'] - float(theta))<0.01
        sub_df= df[(df['qubit']==qubit) & (abs(df['theta'] - float(theta))<0.01)]
        # print(sub_df)
        x = np.array(sub_df['stretch_ratio'])
        y = np.array(sub_df['pulse-00count'])
        y2 = np.array(sub_df['gate-00count'])
        # print(x)
        # print(y)

        try:
            popt, pcov =curve_fit(fitting,x,y)
            target = np.average(y2) 
            
            # print(target)
            x_opt = fsolve(fitting_eq,1.0,args= (popt[0], popt[1], popt[2],target))
            # print(x_opt)
            if x_opt[0] < 2 and x_opt[0]>0.1:
                df_info.loc[len(df_info.index)] = [qubit,theta,popt[0],popt[1],popt[2],x_opt[0]]
            else:
                df_info.loc[len(df_info.index)] = [qubit,theta,popt[0],popt[1],popt[2],1]
            # print(popt)
            # print(pcov)
            # plt.plot(x, y, 'b-', label='data')
            # plt.plot(x, y2, 'g-', label='target')
            # plt.plot(x, fitting(x, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt))
            # plt.legend()
            # plt.show()
    
            # sub_df = df.loc[ and ] 
            # print(sub_df)
        except:
            df_info.loc[len(df_info.index)] = [qubit,theta,pd.NA,pd.NA,pd.NA,1]
df_info.to_excel('excels/'+ file_name + '_info.xlsx')
