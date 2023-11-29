# ========== parameters for training set check script ==========

# ==== general settings (meta, basic file I/O)
VERSION = 0
# TODO: implement! OVERWRITE = 0  # re-do all calculations if 1 or True
MAC_PRETTY_PASTELS = "#86ceb8 #fc9a74 #899dca #e68ac3".split() 
SAVE_PLOTS = True # summary plots and per-group plots will be saved
SHOW_PLOTS = False # (assumes SAVE_PLOTS = True) plots will be saved but not drawn on screen
PLOT_EXCLUDE_EXTREMA = True
eatom_extreme = 5000 # per atom energy extreme
fmag_extreme = 500 # force extreme
nndist_extreme = 5000 # distance extreme
species_plot_colors = MAC_PRETTY_PASTELS # this is a list of matplotlib-friendly colors, must be at least as long as the # species in your training set

# input is currently a fitsnap input script because we use scraper framework from fitsnap
# we can eventually change this to go to a specific training set folder (w/o fitsnap) instead if desired

# Example, InP JSONs
# settings_file_or_dict = "WHBe_test.in"
training_set_path = f"JSON"
settings_file_or_dict = f"InP-example.in"
user_label = "InP-example"
output_path = f"analysis_{user_label}"
report_file = f"{output_path}/brief_report.txt"

# ==== quarantine mode settings
# note that in version 0, "threshold mode" is the only quarantining method implemented
# there are potential plans for others (e.g. distributions, automated schemes)
quarantine_mode = "thresh"

# threshold mode
# user specifies thresholds for selected common training set config properties
# if a training set config's value fails the test (threshhold operator or 'threshop') of a threshold value ('threshval'), it is flagged and quarantined
# currently implemented threshold operators are "<, <=, >, >, outside"

# thresholded variables
thresh_cols = "econfig eatom fx fy fz fmag nn1_dist".split()

# settings for threshold operators <, <=, >, >= reqire one scalar real number
# note that threshops must be strings
# settings for threshold operator "outside" (or 'notin') require a list of two real numbers in ascending order 
# settings for threshold operator "between" (or 'inside' or 'in') require a list of two real numbers in ascending order
threshval_econfig = 100 # eV per config
threshop_econfig = ">"
threshval_eatom = 5 # eV per atom, elements weighed equally 
threshop_eatom = ">" 
threshval_fmag = 8 # force magnitude, eV/Angstrom
threshop_fmag = ">"
threshval_fx = threshval_fmag # eV/Angstrom for X force component
threshop_fx = threshop_fmag
threshval_fy = threshval_fmag # eV/Angstrom for Y force component
threshop_fy = threshop_fmag
threshval_fz = threshval_fmag # eV/Angstrom for Z force component
threshop_fz = threshop_fmag 
threshval_nn1_dist = [2.0, 4.5] # Angstroms # TODO implement per element pairs (self and interelemental)
threshop_nn1_dist = "outside" # 'notin' also accepted
# EXAMPLE "between"
#threshval_econfig = [-1000.0, 10.0] # Angstroms # TODO implement per element pairs (self and interelemental)
# threshop_nn1_dist = "between" # 'inside' or 'in' also accepted

# set up lists (no need to change this section unless adding threshval/threshops)
# make sure that order of variables in columns, value list, and operator list are the same!
threshval_list = [threshval_econfig, threshval_eatom, threshval_fx, threshval_fy, threshval_fz, threshval_fmag, threshval_nn1_dist]
threshop_list = [threshop_econfig, threshop_eatom, threshop_fx, threshop_fy, threshop_fz, threshop_fmag, threshop_nn1_dist]
# threshold = f"five_number_summary_{user_label}_v{VERSION}.csv"

# final summary file information
# note that "quarantine" labels are used to check whether script ran properly 
# changing the label can result in an entire re-analysis
# this is no problem unless your training check takes a long time to run
quarantine_label = f"quarantine_summary_{user_label}"
final_qcsv = f"{quarantine_label}_v{VERSION}.csv"

# ==== analysis settings
# Five number summary mode settings
# https://en.wikipedia.org/wiki/Five-number_summary
# note that in version 0, this is the only type of analysis implemented
analysis_mode = "five_number" 
tukey_factor = 1.5 # normally ~1.5
five_number_label = f"five_number_summary_{user_label}"
