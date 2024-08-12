'''
2020 Mark Shui Hu, QuTech

www.github.com/watermarkhu/oop_surface_code
_____________________________________________

'''

import oopsc.oopsc as oopsc
from threshold_plot import plot_thresholds
from threshold_fit import fit_data
from pprint import pprint
import multiprocessing as mp
import numpy as np
import pandas as pd
import git, sys, os


def run_thresholds(
        decoder,
        lattice_type="toric",
        lattices = [],
        perror = [],
        superoperator=[],
        iters = 0,
        measurement_error=False,
        multithreading=False,
        threads=None,
        modified_ansatz=False,
        save_result=True,
        file_name="thres",
        show_plot=False,
        plot_title=None,
        folder = "./",
        P_store=1000,
        debug=False,
        **kwargs
        ):
    '''
    ############################################
    '''
    run_oopsc = oopsc.multiprocess if multithreading else oopsc.multiple

    if measurement_error:
        from oopsc.graph import graph_3D as go
    else:
        from oopsc.graph import graph_2D as go

    sys.setrecursionlimit(100000)
    r = git.Repo(os.path.dirname(__file__))
    full_name = r.git.rev_parse(r.head, short=True) + f"_{lattice_type}_{go.__name__}_{decoder.__name__}_{file_name}"
    if not plot_title:
        plot_title = full_name


    if not os.path.exists(folder):
        os.makedirs(folder)

    if kwargs.pop("subfolder"):
        os.makedirs(folder + "/data/", exist_ok=True)
        os.makedirs(folder + "/figures/", exist_ok=True)
        file_path = folder + "/data/" + full_name + ".csv"
        fig_path = folder + "/figures/" + full_name + ".pdf"
    else:
        file_path = folder + "/" + full_name + ".csv"
        fig_path = folder + "/" + full_name + ".pdf"

    progressbar = kwargs.pop("progressbar")

    data = None
    int_P = [int(p*P_store) for p in perror]
    config = oopsc.default_config(**kwargs)

    # Simulate and save results to file
    for lati in lattices:

        if multithreading:
            if threads is None:
                threads = mp.cpu_count()
            graph = [oopsc.lattice_type(lattice_type, config, decoder, go, lati) for _ in range(threads)]
        else:
            graph = oopsc.lattice_type(lattice_type, config, decoder, go, lati)

        for i, (pi, int_p) in enumerate(zip(perror, int_P)):

            print("Calculating for L = ", str(lati), "and p =", str(pi))

            superop = None
            GHZ_success = None
            if superoperator:
                pi = 0
                superop = superoperator[i]
                if "GHZ_success" in kwargs:
                    GHZ_success = kwargs["GHZ_success"][i]

            oopsc_args = dict(
                paulix=pi,
                superoperator=superop,
                GHZ_success=GHZ_success,
                lattice_type=lattice_type,
                debug=debug,
                processes=threads,
                progressbar=progressbar
            )
            if measurement_error and superoperator is None:
                oopsc_args.update(measurex=pi)
            output = run_oopsc(lati, config, iters, graph=graph, **oopsc_args)

            pprint(dict(output))
            print("")

            if data is None:
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path, header=0)
                    data = data.set_index(["L", "p"])
                else:
                    columns = list(output.keys())
                    index = pd.MultiIndex.from_product([lattices, int_P], names=["L", "p"])
                    data = pd.DataFrame(
                        np.zeros((len(lattices) * len(perror), len(columns))), index=index, columns=columns
                    )

            if data.index.isin([(lati, int_p)]).any():
                for key, value in output.items():
                    data.loc[(lati, int_p), key] += value
            else:
                for key, value in output.items():
                    data.loc[(lati, int_p), key] = value

            data = data.sort_index()
            if save_result:
                data.to_csv(file_path)

    print(data.to_string())

    par = fit_data(data, modified_ansatz)

    if show_plot:
        plot_thresholds(data, file_name, fig_path, modified_ansatz, save_result=save_result, par=par)

    if save_result:
        data.to_csv(file_path)


if __name__ == "__main__":

    import argparse


    def add_args(parser, args, group_name=None, description=None):
        if group_name:
            parser = parser.add_argument_group(group_name, description)
        for sid, lid, action, help, kwargs in args:
            parser.add_argument(sid, lid, action=action, help=help, **kwargs)

    parser = argparse.ArgumentParser(
        prog="threshold_run",
        description="run a threshold computation",
        usage='%(prog)s [-h/--help] decoder lattice_type iters -l [..] -p [..] (lattice_size)'
    )

    parser.add_argument("decoder",
        action="store",
        type=str,
        help="type of decoder - {mwpm/uf/eg}",
        metavar="d",
    )

    parser.add_argument("lattice_type",
        action="store",
        type=str,
        help="type of lattice - {toric/planar}",
        metavar="lt",
    )

    parser.add_argument("iters",
        action="store",
        type=int,
        help="number of iterations - int",
        metavar="i",
    )

    key_arguments = [
        ["-l", "--lattices", "store", "lattice sizes - verbose list int", dict(type=int, nargs='*', metavar="", required=True)],
        ["-p", "--perror", "store", "error rates - verbose list float", dict(type=float, nargs='*', metavar="", required=True)],
        ["-so", "--superoperator", "store", "Use superoperator as error input - list of superoperator filenames",
         dict(type=str, nargs='*', metavar="")],
        ["-me", "--measurement_error", "store_true", "enable measurement error (2+1D) - toggle", dict()],
        ["-mt", "--multithreading", "store_true", "use multithreading - toggle", dict()],
        ["-nt", "--threads", "store", "number of threads", dict(type=int, metavar="")],
        ["-ma", "--modified_ansatz", "store_true", "use modified ansatz - toggle", dict()],
        ["-s", "--save_result", "store_true", "save results - toggle", dict()],
        ["-sp", "--show_plot", "store_true", "show plot - toggle", dict()],
        ["-fn", "--file_name", "store", "plot filename - toggle", dict(default="thres", metavar="")],
        ["-pt", "--plot_title", "store", "plot filename - toggle", dict(default="", metavar="")],
        ["-f", "--folder", "store", "base folder path - toggle", dict(default="./", metavar="")],
        ["-sf", "--subfolder", "store_true", "store figures and data in subfolders - toggle", dict()],
        ["-pb", "--progressbar", "store_true", "enable progressbar - toggle", dict()],
        ["-dgc", "--dg_connections", "store_true", "use dg_connections pre-union processing - toggle", dict()],
        ["-dg", "--directed_graph", "store_true", "use directed graph for evengrow - toggle", dict()],
        ["-db", "--debug", "store_true", "enable debugging hearistics - toggle", dict()],
        ["-GHZ", "--GHZ_success", "store", "specify the percentage of GHZ states that are successfully created "
                                           "(works only with superoperator) - float [0-1]",
         dict(type=float, nargs='*', metavar="")]
    ]

    # from run_oopsc import add_args

    add_args(parser, key_arguments)

    args=vars(parser.parse_args())
    decoder = args.pop("decoder")


    if decoder == "mwpm":
        from oopsc.decoder import mwpm as decode
        print(f"{'_'*75}\n\ndecoder type: minimum weight perfect matching (blossom5)")
    elif decoder == "uf":
        from oopsc.decoder import uf as decode
        print(f"{'_'*75}\n\ndecoder type: unionfind")
        if args["dg_connections"]:
            print(f"{'_'*75}\n\nusing dg_connections pre-union processing")
    elif decoder == "eg":
        from oopsc.decoder import ufbb as decode
        print("{}\n\ndecoder type: unionfind evengrow with {} graph".format("_"*75,"directed" if args["directed_graph"] else "undirected"))
        if args["dg_connections"]:
            print(f"{'_'*75}\n\nusing dg_connections pre-union processing")

    run_thresholds(decode, **args)
