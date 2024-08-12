'''
2020 Mark Shui Hu, QuTech

www.github.com/watermarkhu/oop_surface_code
_____________________________________________

'''
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from collections import defaultdict
from scipy import optimize
import numpy as np
import math
from copy import deepcopy
from .fit import fit_thresholds, get_fit_func
from .sim import get_data, read_data
import os
import sys


def mle_binomial(errors, reps, z=1.96):
    p = errors/reps

    return p, z * (np.sqrt(np.multiply(p, 1-p)/reps) + 0.5/reps)


def plot_style(ax, title=None, xlabel=None, ylabel=None, fontsize=20, **kwargs):
    ax.grid(color='0.85', linestyle='-', linewidth=1)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    for key, arg in kwargs.items():
        func = getattr(ax, f"set_{key}")
        func(arg)
    ax.patch.set_facecolor('0.97')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def plot_settings():
    # plt.rc('font',family='serif')
    # plt.rc('text', usetex=True)
    # plt.rc('mathtext', fontset='cm')    # 'dejavusans', 'dejavuserif', 'cm', 'stix', and 'stixsans'.
    plt.rc('mathtext', fontset='cm')
    # plt.rc('mathtext', fontset='stixsans')
    # plt.rc('mathtext', fontset='custom')
    # plt.rc('font', family='STIXGeneral')
    # plt.rc('font', family='cmr10')
    # plt.rc('font', family='serif')
    plt.rc('font', family='cmss10')
    # plt.rcParams['figure.figsize'] = 8.5/2.54, 7/2.54

    # plt.rcParams["font.family"] = "Calibri"
    # plt.rcParams["font.style"] = "normal"
    # plt.rcParams["font.weight"] = "100"
    # plt.rcParams["font.stretch"] = "normal"
    # plt.rcParams["font.size"] = 12
    # plt.rcParams["lines.linewidth"] = 1
    # plt.rcParams["axes.linewidth"] = 0.3
    # plt.rcParams["grid.linewidth"] = 0.3
    # plt.rcParams.update({'figure.autolayout': True})
    pass


def get_markers():
    return ["o", "s", "v", "D", "p", "^", "h", "X", "<", "P", "*", ">", "H", "d", 4, 5, 6, 7, 8, 9, 10, 11]


def keep_rows_to_evaluate(df, show_last_results=False):

    # df = df[df['cut_off_time'].isin([0.0226081])]
    if show_last_results is not False and isinstance(show_last_results, int):
        value = 'date_and_time'
        date_time_entries = defaultdict(int)
        for row in df[value]:
            for date_time in row.split(" "):
                date_time_entries[date_time] += 1

        newest_date_time = [date_time for date_time in date_time_entries.keys()]
        newest_date_time.sort(key=int)

        date_time_keep = newest_date_time[-1*int(show_last_results):]

        rows_to_keep = []
        for row in df[value]:
            for date_time in date_time_keep:
                if date_time in row:
                    rows_to_keep.append(row)

        df = df[df[value].isin(rows_to_keep)]

    # if not cutoff_results:
    #     df = df[df['cut_off_time'] == np.inf]
    # evaluate_values = {'decoherence': [],
    #                    'fixed_lde_attempts': [2000],
    #                    'node': [],
    #                    'p_bell_success': [0.0001],
    #                    'pg': [0.01],
    #                    'pm': [0.01],
    #                    'pm_1': [None],
    #                    'pn': [0.1],
    #                    'protocol_name': ['dyn_prot_4_4_1_swap', "plain_swap"],
    #                    'pulse_duration': []}
    #
    # for key, values in evaluate_values.items():
    #     values = values + [str(i) for i in values]  # Usage: if dataframe parses the values as strings
    #
    #     if values and 'REMOVE' not in values:
    #         df = df[df[key].isin(values)]
    #     elif 'REMOVE' not in values:
    #         evaluate_values[key] = set(df[key])
    #
    # if df.empty:
    #     print("\n[ERROR] No data to show for this set of parameters!", file=sys.stderr)
    #     exit(1)

    return df


def plot_thresholds(
    file_name=None,
    plot_title="",               # Plot title
    output="",
    modified_ansatz=False,
    latts=[],
    probs=[],
    show_plot=True,             # show plotted figure
    f0=None,                   # axis object of error fit plot
    f1=None,                   # axis object of rescaled fit plot
    par=None,
    perr=None,
    lattices=None,
    ms=8,
    ymax=1,
    ymin=0.5,
    styles=[".-", "-"],             # linestyles for data and fit
    plotn=1000,                     # number of points on x axis
    time_to_failure=False,
    accuracy=3,
    pn=None,
    lde=None,
    folder=None,
    interactive_plot=False,
    linewidth=0.8,
    show_last_results=False,
    data=None,
    include_fit=True,
    include_rescaled_error_rates=True,
    fontsize=20,
    chi_squared_red=None
):

    if interactive_plot is True and sys.platform != "linux":
        mpl.use('TkAgg')

    if data is None:
        folder = "csv_files" if folder is None else folder
        ROOT = os.path.abspath(os.getcwd())
        file_path = os.path.join(ROOT, 'oopsc', 'superoperator', folder, 'threshold_sim', file_name)
        data = read_data(file_path)
        data = keep_rows_to_evaluate(data, show_last_results=show_last_results)
        output = os.path.join(ROOT, 'results', 'draft_figures', output)
        data = data[data['pn'] == pn] if pn else data
        data = data[data['p_bell_success'] == lde] if lde else data
        GHZ_accuracy = 1
        GHZ_successes = set(data.index.get_level_values('GHZ_success')) if 'GHZ_success' in data.index.names else [None]
        GHZ_successes_round = []
        for GHZ_success in GHZ_successes:
            value = round(GHZ_success, GHZ_accuracy) if GHZ_success is not None else None
            GHZ_successes_round.append(value)
        # GHZ_successes_round = [round(GHZ_success, GHZ_accuracy) for GHZ_success in GHZ_successes if GHZ_success is not None]

        GHZ_round_to_not_round = {}
        for GHZ_index, GHZ_success_round in enumerate(GHZ_successes_round):
            if GHZ_success_round not in GHZ_round_to_not_round:
                GHZ_round_to_not_round[GHZ_success_round] = [list(GHZ_successes)[GHZ_index]]
            else:
                GHZ_round_to_not_round[GHZ_success_round].append(list(GHZ_successes)[GHZ_index])
        GHZ_successes_round = list(dict.fromkeys(GHZ_successes_round))
        '''
        apply fit and get parameter
        '''
        sub_data = data
        f0_copy, f1_copy, latts_copy, probs_copy, par_copy, perr_copy = deepcopy(f0), deepcopy(f1), deepcopy(latts), \
                                                                        deepcopy(probs), deepcopy(par), deepcopy(perr)

        plot_title_copy = plot_title
        for GHZ_index, GHZ_success in enumerate(sorted(GHZ_round_to_not_round)):
            if GHZ_success is not None:
                sub_data = data[data.index.get_level_values("GHZ_success").isin(GHZ_round_to_not_round[GHZ_success])]
                # sub_data = data.xs(tuple(GHZ_successes), level='GHZ_success')
                plot_title = plot_title_copy   # + " - GHZ success: {}%".format(GHZ_success*100 if not GHZ_success > 1 else
                                                                            # 100)
                f0, f1, latts, probs, par, perr = deepcopy(f0_copy), deepcopy(f1_copy), deepcopy(latts_copy),\
                                                  deepcopy(probs_copy), deepcopy(par_copy), deepcopy(perr_copy)
            if par is None and include_fit:
                (fitL, fitp, fitN, fitt), par, perr = fit_thresholds(sub_data, modified_ansatz, latts, probs)
            else:
                fitL, fitp, fitN, fitt = get_data(sub_data, latts, probs)

            fit_func = get_fit_func(modified_ansatz)

    else:
        sub_data = data
        if par is None and include_fit:
            (fitL, fitp, fitN, fitt), par, perr = fit_thresholds(sub_data, modified_ansatz, latts, probs)
        else:
            fitL, fitp, fitN, fitt = get_data(sub_data, latts, probs)

        fit_func = get_fit_func(modified_ansatz)

    '''
    Plot and fit thresholds for a given dataset. Data is inputted as four lists for L, P, N and t.
    '''
    if f0 is None:
        plot_settings()
        f0, ax0 = plt.subplots()
        plt.xticks(fontsize=fontsize*0.8)
        plt.yticks(fontsize=fontsize*0.8)
        # plt.subplots_adjust(left=0.08, bottom=0.08, right=.98, top=.95)
        f0.set_size_inches(10, 7.5)
    else:
        ax0 = f0.axes[0]
    if include_fit and include_rescaled_error_rates:
        if f1 is None:
            f1, ax1 = plt.subplots()
        else:
            ax1 = f1.axes[0]

    LP = defaultdict(list)
    for L, P, N, T in zip(fitL, fitp, fitN, fitt):
        LP[L].append([P, N, T])

    if lattices is None:
        lattices = sorted(set(fitL))

    colors = {lati: f"C{i%10}" for i, lati in enumerate(lattices)}
    markerlist = get_markers()
    markers = {lati: markerlist[i%len(markerlist)] for i, lati in enumerate(lattices)}
    legend = []

    # X-axis precision
    if accuracy is not None:
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%.{}f'.format(len(str(accuracy)) - 2)))
        ax0.xaxis.set_ticks(np.arange(min(fitp)*100, max(fitp)*100 + accuracy, accuracy))

    for i, lati in enumerate(lattices):
        fp, fN, fs = map(list, zip(*sorted(LP[lati], key=lambda k: k[0])))
        if time_to_failure:
            ft = [100/(1-math.sqrt(si/ni)) for si, ni in zip(fs, fN)]
        else:
            ft = [si / ni for si, ni in zip(fs, fN)]
            ft_error = [mle_binomial(si, ni)[1] for si, ni in zip(fs, fN)]
        ax0.plot(
            [q * 100 for q in fp], ft, #styles[0],
            color=colors[lati],
            marker=markers[lati],                       
            ms=ms,
            fillstyle="none",
            ls="None" if include_fit else styles[1],
            linewidth=linewidth,
        )
        if not time_to_failure:
            ax0.errorbar(
                [q * 100 for q in fp], ft, ft_error, None, #styles[0],
                color=colors[lati],
                linestyle='None',
                # marker=markers[lati],
                # ms=ms,
                # fillstyle="none",
            )
        X = np.linspace(min(fp), max(fp), plotn)
        ax0.plot(
            [x * 100 for x in X],
            [fit_func((x, lati), *par) for x in X],
            color=colors[lati],
            lw=1.5,
            alpha=0.6,
            ls=styles[1],
        ) if include_fit else None

        legend.append(Line2D(
            [0],
            [0],
            ls=styles[1],
            lw=1.5 if include_fit else 1,
            alpha=0.6 if include_fit else 1,
            linewidth=linewidth,
            label="$L = {}$".format(lati),
            color=colors[lati],
            marker=markers[lati],
            ms=ms,
            fillstyle="none"
        ))
    print(perr)
    print("The LSR at threshold (pth):",fit_func((par[0], 12), *par),"+-",((fit_func((par[0]-perr, 12), *par)-fit_func((par[0], 12), *par))+(-fit_func((par[0]+perr, 12), *par)+fit_func((par[0], 12), *par)))/2)

    DS = fit_func((par[0], 20), *par) if include_fit else 0

    if include_fit:
        ax0.axvline(par[0] * 100, ls="dotted", color="k", alpha=0.6, lw=1.5)
        ax0.axvspan((par[0] - perr) * 100, (par[0] + perr) * 100, alpha=0.05, color='k', ls="None")
        text_to_annotate = r"$p_\mathrm{th} = $" + "{}%".format(str(round(100 * par[0], 4))) + r"$\pm ${:.4f}".format(perr*100) + "%"
        if chi_squared_red is not None:
            text_to_annotate += "\n" + r"$\chi_\mathrm{\nu}^2 = $" + "{:.5f}".format(chi_squared_red)
        ax0.annotate(text_to_annotate,
            # r"$p_\text[th]$ = {}%, DS = {:.5f}".format(str(round(100 * par[0], 2)), DS),
            (par[0] * 100, DS),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=fontsize*0.8,
        )

    ylabel = "Average time to failure" if time_to_failure else "Logical success rate"
    plot_style(ax0, plot_title, "$p_\mathrm{g}=p_\mathrm{m}$ (%)", ylabel, fontsize=fontsize)
    # ax0.set_ylim(ymin, ymax)
    ax0.legend(handles=legend, loc="lower left", ncol=1, prop={'size': fontsize*0.8})

    ''' Plot using the rescaled error rate'''
    if include_fit and include_rescaled_error_rates:
        for L, p, N, t in zip(fitL, fitp, fitN, fitt):
            if L in lattices:
                if modified_ansatz:
                    plt.plot(
                        (p - par[0]) * L ** (1 / par[5]),
                        t / N - par[4] * L ** (-1 / par[6]),
                        color=colors[L],
                        marker=markers[L],
                        ms=ms,
                        fillstyle="none",
                    )
                else:
                    plt.plot(
                        (p - par[0]) * L ** (1 / par[5]),
                        t / N,
                        color=colors[L],
                        marker=markers[L],
                        ms=ms,
                        fillstyle="none",
                    )
        x = np.linspace(*plt.xlim(), plotn)
        ax1.plot(x, par[1] + par[2] * x + par[3] * x ** 2, "--", color="C0", alpha=0.5)
        ax1.legend(handles=legend, loc="lower left", ncol=2)

        plot_style(ax1, "Modified curve " + plot_title, "Rescaled error rate", "Modified succcess probability", fontsize=fontsize)

    # plt.xlim([0.07, 0.12])
    # plt.ylim([0.77, 0.98])

    if show_plot:   # and GHZ_index == (len(GHZ_successes_round) - 1):
        # f0.savefig("figure_mi.png", transparent=False, format="png", bbox_inches="tight")
        plt.show()

    if output:
        if output [-4:] != ".pdf": output += ".pdf"
        f0.savefig(output, transparent=False, format="pdf", bbox_inches="tight")

    return f0, f1


class npolylogn(object):
    def func(self, N, A, B, C):
        return A*N*(np.log(N)/math.log(B))**C

    def guesses(self):
        guess = [0.01, 10, 1]
        min = (0, 1, 1)
        max = (1, 1000000, 100)
        return guess, min, max

    def show(self, *args):
        return f"n*(log_A(n))**B with A={round(args[1], 5)}, B={round(args[2], 5)}"


class linear(object):
    def func(self, N, A, B):
        return A*N+B

    def guesses(self):
        guess = [0.01, 10]
        min = (0, 0)
        max = (10, 1000)
        return guess, min, max

    def show(self, *args):
        return f"A*n with A={round(args[0], 6)}"


class nlogn(object):
    def func(self, N, A, B):
        return A*N*(np.log(N)/math.log(B))

    def guesses(self):
        guess = [0.01, 10]
        min = (0, 1)
        max = (1, 1000000)
        return guess, min, max

    def show(self, *args):
        return f"n*(log_A(n)) with A={round(args[1], 5)}"


def plot_compare(csv_names, xaxis, probs, latts, feature, plot_error, dim, xm, ms, output, fitname, **kwargs):

    if fitname == "":
        fit = None
    else:
        if fitname in globals():
            fit = globals()[fitname]()
        else:
            print("fit does not exist")
            fit=None

    markers = get_markers()

    xchoice = dict(p="p", P="p", l="L", L="L")
    ychoice = dict(p="L", P="L", l="p", L="p")
    xchoice, ychoice = xchoice[xaxis], ychoice[xaxis]
    xlabels, ylabels = (probs, latts) if xaxis == "p" else (latts, probs)
    if xlabels: xlabels = sorted(xlabels)

    linestyles = ['-', '--', ':', '-.']

    data, leg1, leg2 = [], [], []
    for i, name in enumerate(csv_names):
        ls = linestyles[i%len(linestyles)]
        leg1.append(Line2D([0], [0], ls=ls, label=name))
        data.append(read_data(name))


    if not ylabels:
        ylabels = set()
        for df in data:
            for item in df.index.get_level_values(ychoice):
                ylabels.add(round(item, 6))
        ylabels = sorted(list(ylabels))


    colors = {ind: f"C{i%10}" for i, ind in enumerate(ylabels)}

    xset = set()
    for i, df in enumerate(data):

        indices = [round(x, 6) for x in df.index.get_level_values(ychoice)]
        ls = linestyles[i%len(linestyles)]

        for j, ylabel in enumerate(ylabels):

            marker = markers[j % len(markers)]
            color = colors[ylabel]

            d = df.loc[[x == ylabel for x in indices]]
            index = [round(v, 6) for v in d.index.get_level_values(xchoice)]
            d = d.reset_index(drop=True)
            d["index"] = index
            d = d.set_index("index")

            if not xlabels:
                X = index
                xset = xset.union(set(X))
            else:
                X = [x for x in xlabels if x in d.index.values]

            column = feature if feature in df else f"{feature}_m"
            Y = [d.loc[x, column] for x in X]

            if dim != 1:
                X = [x**dim for x in X]

            # print(ylabel, X, Y)
            #
            if fit is not None:
                guess, min, max = fit.guesses()
                res = optimize.curve_fit(fit.func, X, Y, guess, bounds=[min, max])
                step = abs(int((X[-1] - X[0])/100))
                pn = np.array(range(X[0], X[-1] + step, step))
                ft = fit.func(pn, *res[0])
                plt.plot(pn, ft, ls=ls, c=color)
                plt.plot(X, Y, lw=0, c=color, marker=marker, ms=ms, fillstyle="none")
                print(f"{ychoice} = {ylabel}", fit.show(*res[0]))
            else:
                plt.plot(X, Y, ls=ls, c=color, marker=marker, ms=ms, fillstyle="none")

            if i == 0:
                leg2.append(Line2D([0], [0], ls=ls, c=color, marker=marker, ms=ms, fillstyle="none", label=f"{ychoice}={ylabel}"))

            if plot_error and f"{feature}_v" in d:
                E = list(d.loc[:, f"{feature}_v"])
                ym = [y - e for y, e in zip(Y, E)]
                yp = [y + e for y, e in zip(Y, E)]
                plt.fill_between(X, ym, yp, alpha=0.1, facecolor=color, edgecolor=color, ls=ls, lw=2)

    xnames = sorted(list(xset)) if not xlabels else xlabels
    xticks = [x**dim for x in xnames]
    xnames = [round(x*xm, 3) for x in xnames]

    plt.xticks(xticks, xnames)
    L1 = plt.legend(handles=leg1, loc="lower right")
    plt.gca().add_artist(L1)
    L2 = plt.legend(handles=leg2, loc="upper left", ncol=3)
    plt.gca().add_artist(L2)

    plot_style(plt.gca(), "Comparison of {}".format(feature), xchoice, "{} count".format(feature))
    plt.show()
