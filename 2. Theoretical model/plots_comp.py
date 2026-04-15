
from fitting_metrics import calculate_fit_metrics
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_histosim_comparison_scatter(I1, I2, QML_data, size_sct, show, burn_frac=0.2, ymin=-10, ymax=10):
    
    metrics = calculate_fit_metrics(I1, I2, QML_data, burn_frac, ymin, ymax)
    H_sim = metrics['H_sim']
    H_qml = metrics['H_qml']

    
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

    text_size = 8
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': 'Helvetica',
        'font.size': text_size,
        'axes.linewidth': 1.05
    })
    
    ax.scatter(H_qml[:, 0], H_qml[:, 1], color='#0000FF', alpha=0.5, s=size_sct,
               edgecolors='navy', zorder=3)
    ax.scatter(H_sim[:, 0], H_sim[:, 1], color='#FF0000', alpha=0.4, s=size_sct,
               edgecolors='darkred', marker='s', zorder=3)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$P(x)$')
    ax.tick_params(axis='both', which='both', direction='in', labelsize=text_size)
    ax.set_yscale('log')
    ax.set_xlim(-10, 10)
    ax.set_ylim(1e-5, 5)

    metrics.pop('H_sim', None)
    metrics.pop('H_qml', None)

    if show == True:
        plt.show()
        return fig, metrics
    else:
        fig.savefig('FIGURES/Figure_2.pdf')
        return fig, metrics
