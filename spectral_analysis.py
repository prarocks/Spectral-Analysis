# --- Spectral Analysis of Time Series Data ---
# Author: Prasanna Dahal

import numpy as np
import matplotlib.pyplot as plt

def perform_spectral_analysis(data):
    """
    Performs spectral analysis on the given time series data using Fourier series.
    
    Parameters:
        data (list or np.ndarray): Time series data (e.g., monthly runoff values) 
    
    Displays:
        - Spectral analysis table (in console)
        - Line spectrum plot
    """
    data = np.array(data)
    n = len(data)
    x_bar = np.mean(data)

    # Determine the number of harmonics
    q = n // 2 if n % 2 == 0 else (n - 1) // 2
    
    fm = np.arange(1, q + 1) / n
    am, bm = np.zeros(q), np.zeros(q)
    t = np.arange(1, n + 1)

    for m in range(1, q + 1):
        cos_ = np.cos(2 * np.pi * m * t / n)
        sin_ = np.sin(2 * np.pi * m * t / n)
        if m < q or n % 2 != 0:
            am[m - 1] = (2 / n) * np.dot(data, cos_)
            bm[m - 1] = (2 / n) * np.dot(data, sin_)
        else:
            # Nyquist frequency case (when n is even)
            am[m - 1] = (1 / n) * np.dot(data, cos_)
            bm[m - 1] = 0
    
    cm_sq = am**2 + bm**2
    phi = np.degrees(np.arctan2(-bm, am))
    contrib = cm_sq / 2
    total_var = np.sum(contrib)
    percent = 100 * contrib / total_var if total_var > 0 else np.zeros(q)

    # Print results table
    print("n =", n)
    print("Harmonic | m | fm=m/n | am | bm | cm²=am²+bm² | Øm (deg) | Var. contrib | % Var.")
    print("-" * 90)
    for i in range(q):
        print(f"{i+1:8d} | {fm[i]:.5f} | {am[i]:.6f} | {bm[i]:.6f} | {cm_sq[i]:.6f} | {phi[i]:.2f} | {contrib[i]:.6f} | {percent[i]:.2f}")

    print(f"\nTotal Variance = {total_var:.6f}\n")

    # Plot the line spectrum
    plt.figure(figsize=(10, 6))
    plt.stem(fm, cm_sq, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.xlabel('Frequency (fm = m/n)')
    plt.ylabel('cm² (am² + bm²)')
    plt.title('Line Spectrum (Variance Spectrum / Periodogram)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example data (Monthly runoff values) or Replace it with any other data
    runoff_data = [
        14.6859, 48.1636, 60.05, 41.9818, 17.5454, 7.1332,
        4.3505, 3.3805, 2.885, 2.7795, 2.7982, 3.9386
    ]
    perform_spectral_analysis(runoff_data)
