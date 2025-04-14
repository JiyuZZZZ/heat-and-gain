#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:27:02 2024

@author: jiyuzhang
"""
from scipy.linalg import solve_banded
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

################ 1.Inputting data
def input_data():
    """
    This is a function responsible for inputting data and it can deal with 
    situations when an inputted parameter is clearly wrong.

    Returns
    -------
    tuple
        The tuple contains all of the parameter values.
    """
    #Ask user to input the value of parameters.
    r = input("Please input the risk-free interest rate r(per annum):")
    sigma = input("Please input the stock price volatility σ(per annum):")
    T = input("Please input the option contract T:")
    K = input("Please input the strike price K of the option:")
    M = input("Please input the number of grid intervals in the price variable:")
    N = input("Please input the number of grid intervals in the time variable:")
    xmin = input("Please input the the minimum x-value in the domain:")
    xmax = input("Please input the the maximum x-value in the domain:")

    try:
        r = float(r)
    except ValueError as e:
        return("Please input a correct r value.")
    
    try:
        sigma = float(sigma)
    except ValueError as e:
        return("Please input a correct sigma value.")
        
    try:
        T = float(T)
    except ValueError as e:
        return("Please input a correct T value.")
    
    try:
        K = float(K)
    except ValueError as e:
        return("Please input a correct K value.")

    try:
        M = int(M)
    except ValueError as e:
        return("Please input a correct integer, M.")

    try:
        N = int(N)
    except ValueError as e:
        return("Please input a correct integer, N.")
    
    try:
        xmin = float(xmin)
    except ValueError as e:
        return("Please input a correct minimum x value.")
        
    try:
        xmax = float(xmax)
    except ValueError as e:
        return("Please input a correct maximum x value.")
        
    #Verify the correctness of parameters.
    for param_name, param_value in [('r', r),
                                    ('sigma', sigma), 
                                    ('T', T),
                                    ('K', K), 
                                    ('M', M),
                                    ('N', N)]:
    
        if not isinstance(param_value, (float, int)):
            if param_name in ['M', 'N']:
                return f"Error: The inputted parameter {param_name} is not an integer."
            else:
                return f"Error: The inputted parameter {param_name} is not a numeric value."
        if param_value < 0:
            return f"Error: The inputted parameter {param_name} is negative."
        if param_name == 'M' and param_value < 1:
            return "Error: The inputted parameter M should be greater than or equal to 1."
        if param_name == 'M' and param_value %1!=0:
            return "Error: The inputted parameter M should be a integer."
        if param_name == 'N' and param_value < 1:
            return "Error: The inputted parameter N should be greater than or equal to 1."
        if param_name == 'N' and param_value %1!=0:
            return "Error: The inputted parameter N should be a integer."
    
    for param_name, param_value in [('xmin', xmin), ('xmax', xmax)]:
        if not isinstance(param_value, (float, int)):
            return f"Error: The inputted parameter {param_name} is not a numeric value."
    if float(xmin) >= float(xmax):
        return "The inputted parameter minimum x value is not smaller than the maximum x value."
    return r, sigma, T, K, M, N, xmin, xmax


#############2.Thomas algorithm.
def tridiagonal_solve_thomas(alpha,beta,gamma,b):
    """
    This is a function that performs Thomas algorithm to solve matrix function.
    Parameters
    ----------
    alpha : list
        The main diagonal.
    beta : list
        The upper diagonal.
    gamma : float
        The lower diagonal.
    b : list
        right-hand side vector.
    Returns
    -------
    list
        The list contains the solution vector.
    """
    beta = beta + [0]
    gamma = [0] + gamma
    
    n = len(alpha)
    alpha_hat = [0] * n
    b_hat = [0] * n
    alpha_hat[0] = alpha[0]
    b_hat[0] = b[0]
    
    for i in range(1, n):
        alpha_hat[i] = alpha[i] - (beta[i-1]*gamma[i]/alpha_hat[i-1])
        b_hat[i] = b[i] - (b_hat[i-1]*gamma[i]/alpha_hat[i-1])
    
    x = [0] * n
    x[n-1] = b_hat[n-1]/alpha_hat[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = (b_hat[i] - beta[i] * x[i+1]) / alpha_hat[i]
        
    return x

alpha = [0.6, 0.5, 0.4]
beta = [0.3, 0.4]
gamma = [0.1, 0.2]
b = [1.2, 2.3, 1.6]

tridiagonal_solve_thomas(alpha,beta,gamma,b)


def tridiagonal_solve_banded(alpha,beta,gamma,b):
    """
    This is a function that performs banded function to solve matrix function.
    Parameters
    ----------
    alpha : list
        The main diagonal.
    beta : list
        The upper diagonal.
    gamma : list
        The lower diagonal.
    b : list
        right-hand side vector.
    Returns
    -------
    list
        The list contains the solution vector.
    """
    beta = [0] + beta
    gamma = gamma + [0]
    
    matrix = [beta,
              alpha,
              gamma]
    sol = solve_banded((1, 1), matrix, b)
    return sol

tridiagonal_solve_banded(alpha,beta,gamma,b)

# Calculate the solutions and pricing times for these two methods.
start = 2
stop = 10000

Time_thomas = []
Time_bands = []
tridiagonal_errors = []
for i in range(start, stop):
    alpha = list(np.random.rand(i))
    beta = list(np.random.rand(i-1))
    gamma = list(np.random.rand(i-1))
    b = list(np.random.rand(i))
    
    time_start = time.perf_counter()
    thomas_solution = tridiagonal_solve_thomas(alpha,beta,gamma,b)
    time_end = time.perf_counter()
    totaltime_thomas = time_end - time_start
    Time_thomas.append(totaltime_thomas)
    
    time_start = time.perf_counter()
    banded_solution = tridiagonal_solve_banded(alpha,beta,gamma,b)
    time_end = time.perf_counter()
    totaltime_banded = time_end - time_start
    Time_bands.append(totaltime_banded)
    
    errors = sum(thomas_solution - banded_solution)
    tridiagonal_errors.append(errors)

M_values = np.arange(start,stop,1)

#Plot to show the errors for different M values.
plt.figure()
plt.plot(M_values, tridiagonal_errors, markersize=2, label="tridiagonal errors")
plt.title('The errors between two algorithms')
plt.xlabel('Size of Solution Vector')
plt.ylabel('Errors')
plt.legend()
plt.show()

#Plot to show the complexity for different M values.
plt.figure()
plt.loglog(M_values, Time_thomas, markersize=2, label="Thomas algorithm")
plt.loglog(M_values, Time_bands, linestyle='--', 
           label='banded function')
plt.title('Option pricing times comparison')
plt.xlabel('Size of Solution Vector')
plt.ylabel('Time')
plt.legend()
plt.show()


###############3.Numerical solution of the Black-Scholes PDE — implicit scheme
r = 0.05
sigma = 0.1
T = 1.8
K = 673

M = 10000
N = 500
xmin = -50
xmax = 50


def black_scholes_implicit(r,sigma,T,K,M,N,xmin,xmax):
    """
    This is a function that performs Implicit scheme.
    Parameters
    ----------
    r : float
        The interest rate.
    sigma : float
        The volatility rate.
    T : float
        The option period.
    K : float
        The strike price.
    M : int
        The number of grid intervals in the price variable.
    N : int
        The number of grid intervals in the time variable.
    xmin : float
        The minimum x value.
    xmax : float
        The maximum x value. 
    Returns
    -------
    stock_price
        The stock prices.
    values
        The option values for different stock prices.
    """
    
    xs = np.linspace(xmin, xmax, M+1)
    payoffs = np.maximum(np.exp(xs) - K, 0)
 
    values = np.exp(-r * T) * payoffs
    dt = sigma ** 2 * T / (2 * N)
    dx = (xmax - xmin) / M
    lamda = dt / dx ** 2
    
    alpha = [1+2*lamda] * (M-1)
    beta = [-lamda] * (M-2)
    gamma = [-lamda] * (M-2)
    
    for nu in range(N):
        new = np.empty_like(values)
        new[0] = 0
        new[-1] = (np.exp(xmax + (nu+1) * dt) - K)*np.exp(- r * T)
        w = values[1:-1]
        w[0] = w[0] + lamda * new[0]
        w[-1] = w[-1] + lamda * new[-1]
        new[1:-1] = tridiagonal_solve_thomas(alpha,beta,gamma,w)
        
        values = new
    stock_price = np.exp(xs - (r - 0.5 * sigma ** 2) * T)
    return stock_price, values



stock_price, values = black_scholes_implicit(r,sigma,T,K,M,N,xmin,xmax)

print(values[0])
print(values[-1])

def BS_exact(S0, r, sigma, T, K): 
    '''
    This is a function that implement the pricing of a vanilla European put 
    option by Black Scholes model.

    Parameters
    ----------
    S_0 : float or int
        Initial price of the stock.
    r : float or int
        Risk_free interest rate r(per annum).
    sigma : TYPE
        The volatility of the stock price σ(per annum).
    T : float or int
        Expiry time T of the option.
    K : float or int
        Strike price K of the option.

    Returns
    -------
    put_price : float
        The price of the vanilla European put option at time 0.
    '''
    #Transform the Normal distribution into standard Normal distribution.
    d1=(np.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    
    #Calculate the option price at time 0.
    option_price = (S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return option_price

# Plot the option values for stock prices of Implicit scheme and exact prices.
BS_prices = []
for i in stock_price:
    BS_prices.append(BS_exact(i, r, sigma, T, K))

plt.plot(stock_price[:6000], BS_prices[:6000], label='exact')
plt.plot(stock_price[:6000], values[:6000], label='implicit')
plt.xlabel('Stock Price')
plt.ylabel('Option Value')
plt.title('Option values comparison for stock prices')
plt.legend()
plt.grid(True)
plt.show()

# Plot the option value errors between formula 1.19 and implicit scheme.
errors_implicit = values - BS_prices 
plt.plot(stock_price[:6000], errors_implicit[:6000])
plt.xlabel('Stock Price')
plt.ylabel('Option Value Errors')
plt.title('Errors between formula 1.19 and implicit scheme')
plt.legend()
plt.grid(True)
plt.show()

################4.Numerical solution of the Black-Scholes PDE — Crank-Nicolson scheme
def black_scholes_crank_nicolson(r,sigma,T,K,M,N,xmin,xmax):
    """
    This is a function that performs Implicit scheme.
    Parameters
    ----------
    r : float
        The interest rate.
    sigma : float
        The volatility rate.
    T : float
        The option period.
    K : float
        The strike price.
    M : int
        The number of grid intervals in the price variable.
    N : int
        The number of grid intervals in the time variable.
    xmin : float
        The minimum x value.
    xmax : float
        The maximum x value. 
    Returns
    -------
    stock_price
        The stock prices.
    values
        The option values for different stock prices.
    """
    xs = np.linspace(xmin, xmax, M+1)
    payoffs = np.maximum(np.exp(xs) - K, 0)

    values = np.exp(-r * T) * payoffs
    dt = sigma ** 2 * T / (2 * N)
    dx = (xmax - xmin) / M
    lamda = dt / dx ** 2
    
    alpha = [1+lamda] * (M-1)
    beta = [-lamda/2] * (M-2)
    gamma = [-lamda/2] * (M-2)
    
    for nu in range(N):
        lower_boundary = 0
        upper_boundary = (np.exp(xmax + (nu+1) * dt) - K)*np.exp(- r * T)
        
        w = 0.5 * lamda * values[:-2]+(1-lamda) * values[1:-1]+0.5 * lamda * values[2:]
        w[0] = w[0] + 0.5 * lamda * lower_boundary
        w[-1] = w[-1] + 0.5 * lamda * upper_boundary
        
        w_v_1 = tridiagonal_solve_thomas(alpha,beta,gamma,w)
        
        values = np.array(w_v_1)
        values = np.insert(values, 0, 0)
        values = np.append(values, upper_boundary)

    stock_price = np.exp(xs - (r - 0.5 * sigma ** 2) * T)
    return stock_price, values

stock_price_new, values_new = black_scholes_crank_nicolson(r,sigma,T,K,M,N,xmin,xmax)

# Plot the option values for stock prices of Crank-Nicolson scheme and exact prices.
plt.plot(stock_price_new[:6000], BS_prices[:6000], label='exact')
plt.plot(stock_price_new[:6000], values_new[:6000], label='Crank-Nicolson')
plt.xlabel('Stock Price')
plt.ylabel('Option Value')
plt.title('Option values comparison for stock prices')
plt.legend()
plt.grid(True)
plt.show()

# Plot the option value errors between formula 1.19 and Crank-Nicolson scheme.
errors_CN = values_new - BS_prices 
plt.plot(stock_price[:6000], errors_CN[:6000])
plt.xlabel('Stock Price')
plt.ylabel('Option Value Errors')
plt.title('Errors between formula 1.19 and Crank-Nicolson scheme')
plt.legend()
plt.grid(True)
plt.show()


##############5.Comparison of errors.
# Plot the errors for different time step lenths.
M = 50000
N_values = [2,4,8,16,32,64]
delta_t = []
for i in range(len(N_values)):
    dt = sigma ** 2 * T / (2 * N_values[i])
    delta_t.append(dt)

implicit_errors = []
CN_errors = []

exact_700 = BS_exact(700, r, sigma, T, K)

for i in N_values:
    stock_price, values = black_scholes_implicit(r,sigma,T,K,M,i,xmin,xmax)
    stock_price_new, values_new = black_scholes_crank_nicolson(r,sigma,T,K,M,i,xmin,xmax)
    
    implicit_option_price = np.interp(700, stock_price, values)
    CN_option_price = np.interp(700, stock_price_new, values_new)
    
    implicit_errors.append(exact_700-implicit_option_price)
    CN_errors.append(exact_700-CN_option_price)

plt.loglog(delta_t, implicit_errors, label='implicit')
plt.loglog(delta_t, CN_errors, label='CN')
plt.title('Empirical errors for delta t')
plt.xlabel('Values of delta t')
plt.ylabel('Option Value Errors')
plt.legend()
plt.show()

coeffs_0 = np.polyfit(np.log(delta_t), np.log(implicit_errors), 1)
slope_0 = coeffs_0[0]
print(slope_0)

coeffs_1 = np.polyfit(np.log(delta_t), np.log(CN_errors), 1)
slope_1 = coeffs_1[0]
print(slope_1)



# Plot the errors for different space step lenths.
N = 500
xmin = -50
xmax = 50
M_values = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
delta_x = []
for i in range(len(M_values)):
    dx = (xmax - xmin) / M_values[i]
    delta_x.append(dx)

implicit_errors = []
CN_errors = []
for i in M_values:
    stock_price, values = black_scholes_implicit(r,sigma,T,K,i,N,xmin,xmax)
    stock_price_new, values_new = black_scholes_crank_nicolson(r,sigma,T,K,i,N,xmin,xmax)
    
    implicit_option_price = np.interp(700, stock_price, values)
    CN_option_price = np.interp(700, stock_price_new, values_new)
    
    implicit_errors.append(implicit_option_price - exact_700)
    CN_errors.append(CN_option_price - exact_700)

plt.loglog(delta_x, implicit_errors, label='implicit')
plt.loglog(delta_x, CN_errors, label='CN')
plt.title('Empirical errors for delta x')
plt.xlabel('Values of delta x')
plt.ylabel('Option Value Errors')
plt.legend()
plt.show()

coeffs_2 = np.polyfit(np.log(delta_x), np.log(implicit_errors), 1)
slope_2 = coeffs_2[0]
print(slope_2)

coeffs_3 = np.polyfit(np.log(delta_x), np.log(CN_errors), 1)
slope_3 = coeffs_3[0]
print(slope_3)