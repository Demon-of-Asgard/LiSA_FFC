# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
plt.style.use('plotparams.yaml')


def plot_result(self,
                result_path: str,
                store_image_to: str = "",
                chop_between: tuple = (90, 110),
                cmap=cm.plasma) -> None:

    # ::::::::::::::::::::::::::::::::::::::::::::::::::#

    result = np.load(
        result_path,
        'r',
        allow_pickle=True
    )

    o_is = result['O_i']
    o_rs = result['O_r']
    kxs = result['kx']
    kys = result['ky']
    kzs = result['kz']
    start, end = chop_between
    evother = 1

    lcolor = cmap(np.linspace(0, 1, len(kxs[start:end][::evother])))

    x0, y0 = 0.0, 0.0
    dw, dh = 0.15, 0.2
    w, h = 2, 2

    fig = plt.figure()
    axs = []

    for i in range(2):
        x, y = x0 + i*(dw + w), y0
        ax = fig.add_axes([x, y, w, h])
        axs.append(ax)

    axline_bar = fig.add_axes([x+w+dw/2, y, 0.075, h])
    axline_bar.set_xlim(0, 1)
    axline_bar.yaxis.tick_right()

    try:
        img = axs[0].contourf(kxs, kzs, o_is, cmap=cmap, levels=100)
        fig.colorbar(img, ax=axs[0], label=r'Im($\omega$)')

    except:
        pass

    chopped_o_is = o_is.T[start:end][::evother]

    kx_max, kz_max = 0, 0

    max_value = 0.0

    for i, kx in enumerate(kxs[start:end][::evother]):
        axs[1].plot(kzs, chopped_o_is[i], ls='-', lw=2,
                    color=lcolor[i], label=f'{kx:.1e}')
        axline_bar.hlines(kx, 0, 1, color=lcolor[i])

    ind = np.unravel_index(
        np.argmax(chopped_o_is, axis=None), chopped_o_is.shape)
    max_value = o_is.max()
    r, c = o_is.shape
    o_is_dup = o_is.T[:]
    max_idx = o_is_dup.reshape(-1,).argmax()
    kxmax_idx, kzmax_idx = max_idx // c, max_idx % c
    O_rmax = o_rs.T.reshape(-1,)[max_idx]
    # print(f"O_rmax: {O_rmax}")

    axs[1].text(
        0.2, max_value/2,
        s=f"$(k_x, k_z)$->({kxs[kxmax_idx]:.3f},{kzs[kzmax_idx]:.3f})",
        fontsize=20
    )

    axs[0].hlines(0, kxs.min(), kxs.max(), color='grey', ls=':')
    axs[0].set_xlabel(r'$k_x$')
    axs[0].set_ylabel(r'$k_z$')
    axs[1].set_xlabel(r'$k_z$')
    axs[1].set_ylabel(r'Im($\omega$)')

    if store_image_to != None:
        plt.savefig(
            store_image_to,
            dpi=300,
            bbox_inches='tight'
        )

        print(f"Figure saved to {store_image_to}.")
    plt.show()
