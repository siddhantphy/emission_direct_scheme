'''
2020 Mark Shui Hu, QuTech

www.github.com/watermarkhu/oop_surface_code
_____________________________________________

'''
import argparse
from run_oopsc import add_args, add_kwargs
from oopsc.threshold.plot import plot_thresholds
from oopsc.threshold.sim import update_result_files
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="threshold_fit",
        description="fit a threshold computation",
        usage='%(prog)s [-h/--help] file_name'
    )

    arguments = [["file_name", "store", str, "file name of csv data (without extension)", "file_name", dict()]]

    key_arguments = [
        ["-p", "--probs", "store", "p items to plot - verbose list", dict(type=float, nargs='*', metavar="")],
        ["-l", "--latts", "store", "L items to plot - verbose list", dict(type=float, nargs='*', metavar="")],
        ["-ma", "--modified_ansatz", "store_true", "use modified ansatz - toggle", dict()],
        ["-o", "--output", "store", "output file name", dict(type=str, default="", metavar="")],
        ["-pt", "--plot_title", "store", "plot filename", dict(type=str, default="", metavar="")],
        ["-ymin", "--ymin", "store", "limit yaxis min", dict(type=float, default=0.5, metavar="")],
        ["-ymax", "--ymax", "store", "limit yaxis max", dict(type=float, default=1, metavar="")],
        ["-ttf", "--time_to_failure", "store_true", "time to failure axis", dict()],
        ["-acc", "--accuracy", "store", "grid accuracy", dict(type=float, default=None, metavar="")],
        ["-pn", "--pn", "store", "Network noise", dict(type=float, default=None, metavar="")],
        ["-lde", "--lde", "store", "lde_success", dict(type=float, default=None, metavar="")],
        ["-folder", "--folder", "store", "Location of the threshold file", dict(type=str, default="", metavar="")]
    ]

    add_args(parser, arguments)
    add_kwargs(parser, key_arguments)
    args=vars(parser.parse_args())
    ROOT = os.path.abspath(os.getcwd())

    # args['folder'] = "Old/Best_prots_sIIIc_v2_SAME_STATISTICS/Thresholds_best_prots_sIIIc_v2/4.sIIIe"
    args['folder'] = "Phenom_thresholds_Siddhant_weight_0"
    prot_name = "None"
    set_name = "SetNickerson"
    decoder = "uf"
    # args['file_name'] = "recipe_" + prot_name + "_swap_" + set_name + "_toric_graph_3D_" + decoder + "_thres.csv"
    # args['file_name'] = prot_name + "_swap_" + set_name + "_toric_graph_3D_" + decoder + "_thres.csv"
    args['file_name'] = prot_name + "_toric_graph_3D_" + decoder + "_thres.csv"

    # args['folder'] = "Thresholds_known_prots"
    # prot_name = "expedient"
    # set_name = "Set3k"
    # args['file_name'] = prot_name + "_swap_" + set_name + "_toric_graph_3D_uf_thres.csv"

    folder = os.path.join(ROOT, 'oopsc', 'superoperator', args['folder'], 'threshold_sim')


    update_result_files(folder, args['file_name'].split(".csv")[0])

    args['modified_ansatz'] = True
    args['interactive_plot'] = False
    args['show_last_results'] = False
    args['include_fit'] = False
    f0, f1 = plot_thresholds(**args)
