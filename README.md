# 🧮 Black-Scholes Option Pricing via Crank–Nicolson PDE Solver

This project solves the Black-Scholes partial differential equation (PDE)  
using the **Crank–Nicolson finite difference method** to price a **European call option**.

---

## 🔧 Features

- User-defined input for:
  - Interest rate `r`, volatility `σ`, strike price `K`, maturity `T`
  - Grid resolution in time (`N`) and price (`M`)
- Stable and accurate numerical solution using Crank-Nicolson
- Efficient tridiagonal system solving via `scipy.linalg.solve_banded`
- Visualization of the option price function at \( t = 0 \)

---

## 📈 Output

The script prints the initial option value and plots the price-function relationship  
across a range of underlying asset prices.

---

## 🎓 Educational Context

This model was implemented as part of a Master's level coursework in **Computational Finance**.  
It demonstrates the application of **numerical PDE methods**, **finite difference schemes**,  
and **financial derivative modeling** in Python.
