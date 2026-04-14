import numpy as np
import matplotlib.pyplot as plt
from ts2vg import NaturalVG
from scipy.optimize import curve_fit
import ruptures as rpt
import pandas as pd
import os

# === Power law ===
def power_law(x, gamma, C):
    return C * np.power(x, -gamma)

# === Script 1 function ===
def compute_degree_distribution(ts_path):
    DF = np.load(ts_path)
    NS = np.cumsum(DF)
    g = NaturalVG().build(NS, only_degrees=True)
    ks, ps = g.degree_distribution
    valid = (ps > 0) & (ks >= 10) & (ks <= 250)
    ks_valid, ps_valid = ks[valid], ps[valid]
    popt, _ = curve_fit(power_law, ks_valid, ps_valid)
    gamma, C = popt
    return ks, ps, ks_valid, gamma, C

# === Script 2 functions ===
def compute_original(ts, p, window_start, window_end):
    NS = np.cumsum(ts)
    g = NaturalVG().build(NS, only_degrees=True)
    ks, ps = g.degree_distribution
    valid = ps > 0
    ks_valid, ps_valid = ks[valid], ps[valid]

    # Exclude first point
    ks_valid, ps_valid = ks_valid[1:], ps_valid[1:]

    # Fit 1: [7, 35]
    fit_range_1 = (ks_valid >= 7) & (ks_valid <= 35)
    ks_fit_1, ps_fit_1 = ks_valid[fit_range_1], ps_valid[fit_range_1]
    gamma1, C1 = curve_fit(power_law, ks_fit_1, ps_fit_1)[0]

    # Transition detection
    x, y = np.log10(ks_valid), np.log10(ps_valid)
    data = pd.DataFrame({"x_axis": x, "y_axis": y}).values
    change_points = rpt.Pelt(model="l2").fit(data).predict(pen=p)

    fit_range_2 = (ks_valid >= change_points[window_start]) & (ks_valid <= change_points[window_end])
    ks_fit_2, ps_fit_2 = ks_valid[fit_range_2], ps_valid[fit_range_2]
    gamma2, C2 = curve_fit(power_law, ks_fit_2, ps_fit_2)[0]

    return ks, ps, (ks_fit_1, gamma1, C1), (ks_fit_2, gamma2, C2)

def compute_shuffled(ts):
    NS = np.cumsum(ts)
    g = NaturalVG().build(NS, only_degrees=True)
    ks, ps = g.degree_distribution
    valid = (ps > 0) & (ks >= 10) & (ks <= 200)
    ks_valid, ps_valid = ks[valid], ps[valid]
    gamma, C = curve_fit(power_law, ks_valid, ps_valid)[0]
    return ks, ps, ks_valid, gamma, C

# === Style ===
text_size = 8
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': text_size,
    'axes.linewidth': 1.05,
})
cmap = plt.cm.inferno
colors = ["#164979", "#E63946"]  # Dark blue and red

# === Paths ===
base_path = r"D:\T - LMO - Nícolas\NICOLAS\3. Dados\PCA Temporal Series Filtered"
files = [
    (f"{base_path}\\CW_PC_TIME_SERIES_146_0_filtered.npy", "a"),
    (f"{base_path}\\QML_PC_TIME_SERIES_286_0_filtered.npy", "b"),
    (f"{base_path}\\SML_PC_TIME_SERIES_690_0_filtered.npy", "c"),
]

p, window_start, window_end = 5, 4, -1

fig = plt.figure(figsize=(3.5, 4.6), constrained_layout=True)

gs = fig.add_gridspec(3, 1, height_ratios=[0.6, 1, 0.6])

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[2, 0])

# --- Plot a ---

ks, ps, ks_valid, gamma, C = compute_degree_distribution(files[0][0])
ax_a.scatter(ks, ps, s=7.2, facecolors='none', edgecolors=colors[0], linewidth=0.6, alpha=0.7)
ks_fit = np.logspace(np.log10(ks_valid.min()), np.log10(ks_valid.max()), 200)
ax_a.plot(ks_fit, power_law(ks_fit, gamma, C), '--', color=colors[1], lw=2)
ax_a.set_xscale("log"); ax_a.set_yscale("log")
ax_a.set_xlabel(r"$k$")
ax_a.set_ylabel(r"$p(k)$",  labelpad=12)
ax_a.text(0.25, 0.55, rf"$\gamma = {gamma:.2f}$", transform=ax_a.transAxes, fontsize=text_size, color=colors[1])
ax_a.tick_params(axis='both', which='both', direction='in', top=True, right=True)

# --- Plot b (exclude first point, 2 fits) ---

series = np.load(files[1][0])
ks, ps, (ks_fit_1, gamma1, C1), (ks_fit_2, gamma2, C2) = compute_original(series, p, window_start, window_end)
ax_b.scatter(ks, ps, s=7.2, facecolors='none', edgecolors=colors[0], linewidth=0.6)
# Fit 1
ks_fit = np.logspace(np.log10(ks_fit_1.min()), np.log10(ks_fit_1.max()), 200)
ax_b.plot(ks_fit, power_law(ks_fit, gamma1, C1), '--', color=colors[1], lw=2)
ax_b.text(0.10, 0.55, rf"$\gamma_1 = {gamma1:.2f}$", transform=ax_b.transAxes, fontsize=text_size, color=colors[1])
# Fit 2
ks_fit = np.logspace(np.log10(ks_fit_2.min()), np.log10(ks_fit_2.max()), 200)
ax_b.plot(ks_fit, power_law(ks_fit, gamma2, C2), '--', color=colors[1], lw=2)
ax_b.text(0.45, 0.25, rf"$\gamma_2 = {gamma2:.2f}$", transform=ax_b.transAxes, fontsize=text_size, color=colors[1])
ax_b.set_xscale("log"); ax_b.set_yscale("log")
ax_b.set_xlabel(r"$k$")
ax_b.set_ylabel(r"$p(k)$", labelpad=12)

ax_b.tick_params(axis='both', which='both', direction='in', top=True, right=True)
ax_b.set_xlim(1.5e0)

# --- Inset inside plot b (bottom-left) ---
ax_inset = ax_b.inset_axes([0.635, 0.56, 0.35, 0.4])  # bottom-left
shuffled = series.copy()
for _ in range(3):
    np.random.shuffle(shuffled)
ks, ps, ks_valid, gamma, C = compute_shuffled(shuffled)
ax_inset.scatter(ks, ps, s=4, facecolors='none', edgecolors=colors[0], linewidth=0.4)
ks_fit = np.logspace(np.log10(ks_valid.min()), np.log10(ks_valid.max()), 200)
ax_inset.plot(ks_fit, power_law(ks_fit, gamma, C), '--', color=colors[1], lw=1.5)
ax_inset.set_xscale("log"); ax_inset.set_yscale("log")
ax_inset.tick_params(axis='both', which='both', top=True, right=True, labelsize=text_size-4, direction='in')
ax_inset.set_xlabel(r'$k$',fontsize=text_size-4)
ax_inset.set_ylabel(r'$p(k)$',fontsize=text_size-4)
ax_inset.text(0.20, 0.4, rf"$\gamma = {gamma:.2f}$", transform=ax_inset.transAxes, fontsize=text_size-4, color=colors[1])

# --- Plot c ---
ax = ax_c
ks, ps, ks_valid, gamma, C = compute_degree_distribution(files[2][0])
ax.scatter(ks, ps, s=7.2, facecolors='none', edgecolors=colors[0], linewidth=0.6, alpha=0.7)
ks_fit = np.logspace(np.log10(ks_valid.min()), np.log10(ks_valid.max()), 200)
ax.plot(ks_fit, power_law(ks_fit, gamma, C), '--', color=colors[1], lw=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$k$") 
ax.set_ylabel(r"$p(k)$",  labelpad=12)
ax.text(0.4, 0.75, rf"$\gamma = {gamma:.2f}$", transform=ax.transAxes, fontsize=text_size, color=colors[1])
ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

ax_a.text(-0.10, 1.1, "A", transform=ax_a.transAxes, fontsize=text_size+2, fontweight='bold')
ax_b.text(-0.10, 1.1, "B", transform=ax_b.transAxes, fontsize=text_size+2, fontweight='bold')
ax_c.text(-0.10, 1.1, "C", transform=ax_c.transAxes, fontsize=text_size+2, fontweight='bold')

print("Saving figure...")
fig.savefig('FIGURES/Figure_4.pdf')
