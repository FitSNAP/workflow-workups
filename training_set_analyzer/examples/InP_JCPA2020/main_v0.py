import os, sys, time
import logging
# from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
import numpy as np 
import pandas as pd 
import matplotlib
# Hmmm why did I have this? for saving figs on cluster?
# matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.spatial import ConvexHull
from scipy import stats
from sklearn.neighbors import KDTree
from settings_v0 import *

# -------------------------------------------------------------------- functions
# 
def format_fs_data(data, nn_check=True):
    atom_data_list = []
    vec_to_str = lambda vec: " ".join([str(v) for v in vec.tolist()])
    group = data['Group']
    fname = data['File']
    econfig = float(data['Energy'])
    natoms = int(data['NumAtoms'])
    eatom = round(econfig/natoms, 5)
    stress1 = vec_to_str(data['Stress'][0])
    stress2 = vec_to_str(data['Stress'][1])
    stress3 = vec_to_str(data['Stress'][2])
    sum_fmag = 0
    atom_types_list = data['AtomTypes']
    elems0, nelems0 = [t.tolist() for t in np.unique(atom_types_list, return_counts=True)]
    nelems, conc = [], []
    for e in all_elems:
        if e in elems0:
            ne = nelems0[elems0.index(e)]
            nelems.append(ne)
            conc.append(round(100*ne/natoms,2))
        else:
            nelems.append(0.0)
            conc.append(0.0)

    metadata = [group,fname,natoms,econfig,eatom] + all_elems + nelems + conc + [f"{stress1}|{stress2}|{stress3}"]
    
    gen_elem_label = lambda col_name: [f"{col_name}{n+1}" for n in range(len(all_elems))]

    fs_meta = "group fname natoms econfig eatom".split() + gen_elem_label("elem") +  gen_elem_label("num_elem") +  gen_elem_label("conc_elem") + ["stress_rowform"]

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
    # Note that, in the below format, the 1st NN is actually the atom itself! 
    # https://stackoverflow.com/questions/3875062/data-type-problem-using-scipy-spatial
    coords = data["Positions"]
    kdt = KDTree(coords)
    flags_out = ''
    flags_thresh = ''
    for i in range(natoms):
        atom_id = i
        atom_elem = atom_types_list[i]
        x, y, z = data['Positions'][i]
        dd, ii = kdt.query([[x,y,z]],k=2)
        nn1_dist = dd[0,1]
        nn1_idx = ii[0,1]
        nn1_elem = atom_types_list[nn1_idx]
        fx, fy, fz = data['Forces'][i]
        fmag = np.sqrt(fx**2 + fy**2 + fz**2)
        sum_fmag += fmag
        atomdata = [atom_id,atom_elem,x,y,z,fx,fy,fz,fmag, nn1_idx, nn1_elem, nn1_dist, flags_out, flags_thresh]
        atom_data_list.append(metadata + atomdata)
    fs_atoms = "atom_id atom_elem x y z fx fy fz fmag nn1_idx nn1_elem nn1_dist flags_out flags_thresh".split()
    return atom_data_list, fs_meta, fs_atoms 

def describe_extraT(df, extra_stats):
    # https://stackoverflow.com/questions/38545828/pandas-describe-by-additional-parameters
    vanilla = df.describe().T
    esdf = df.agg(extra_stats).T
    return pd.concat([vanilla, esdf],axis=1)

def format_quarantine_df(qdf):
    # Quarantine columns: 
    # group fname natoms econfig eatom flags_out flags_thresh
    qcols = "group fname natoms econfig eatom flags_thresh flags_out".split()
    formatted_qdfs = []
    for name, group in qdf.groupby(["group","fname"]):
        ndf = group[qcols].drop_duplicates()
        ndf.loc[:,"max_fmag"] = group.fmag.max()
        ndf.loc[:,"min_nn1_dist"] = group.nn1_dist.max()
        aid_out = [aid for aid in group["atom_id"].unique()]
        aid_str = " ".join([str(aid) for aid in aid_out])
        uflags_out0 = [f.split() for f in group["flags_out"].unique()]
        uflags_out = [f for f in desc_atoms_cols if f in set([item for sublist in uflags_out0 for item in sublist])]
        uflags_str = " ".join(uflags_out)
        ndf.loc[:,"flags_out"] = uflags_str
        ndf.loc[:,"flags_natoms"] = group.shape[0]
        ndf.loc[:,"flags_atomids"] = aid_str
        reorder_cols = 'group fname flags_thresh flags_out flags_natoms flags_atomids natoms econfig eatom max_fmag min_nn1_dist '.split()
        ndf = ndf[reorder_cols]
        formatted_qdfs.append(ndf)
    
    quarantine_df = pd.concat(formatted_qdfs, ignore_index=True)
    return quarantine_df

def prettyprint_list(somelist, title="",spacer_string="  ", endprint=''):
    if title != "":
        print(title)
    [print(f"{spacer_string}{i}") for i in somelist]
    print(endprint)
    return
# ------------------------------------------------------- script

# suppress super annoying matplotlib font manager information
logging.getLogger('matplotlib.font_manager').disable = True
plt.set_loglevel(level='critical')
# pil_logger = logging.getLogger('PIL')
# pil_logger.setLevel(logging.INFO)

# TODO re-implement MPI for faster scrapes!
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# fs = FitSnap(input_file, comm=comm, arglist=["--overwrite","--verbose"])
input_file = settings_file_or_dict

desc_configs_cols = "econfig eatom fmag nn1_dist".split()
desc_configs_units = "eV eV_per_atom % % % eV_per_A A".split()
desc_atoms_cols = "fx fy fz fmag nn1_dist".split()
desc_atoms_units = "eV_per_A eV_per_A eV_per_A eV_per_A A".split()

# set up FitSNAP (no MPI communicator in this version)
fs = FitSnap(input_file, arglist=["--overwrite","--verbose"])
conf = fs.config.sections
datapath = conf["PATH"].datapath

# update the FitSNAP object's path to training set
# the input file will usually specify a directory at the same level and expects FitSNAP to be run there
# within this example folder setup, this script will need more pathing info
conf["PATH"].datapath = training_set_path

all_sections = conf.keys()
if "BISPECTRUM" in all_sections:
    model = "snap"
    all_elems = conf["BISPECTRUM"].types
elif "ACE" in all_sections:
    model = "ace"
    all_elems = conf["ACE"].types
else:
    print("!!!! descriptor types outside SNAP or ACE not implemented. exiting")
    exit()

gt0 = conf["GROUPS"].group_table
all_groups0 = list(gt0.keys())
prettyprint_list(all_groups0, title="All groups, initial:")
group_paths = [f"{output_path}/{g}" for g in all_groups0]

if os.path.exists(output_path):
    all_files = [f for f in os.listdir(output_path)]
else:
    print(f"Analysis directory '{output_path}' not found, creating new directory")

    # mkdirs in case user nests this somehow
    os.mkdir(output_path)

# create one dir per group in input file and generate metadata dict
metadata = {}
metadata["input"] = settings_file_or_dict
metadata["datapath"] = f"{os.getcwd()}/{datapath}"
metadata["group_data_not_found"] = []
for i, g in enumerate(all_groups0):
    data_group = f"{datapath}/{g}"
    if not os.path.exists(data_group):
        print('')
        print(f"! Data for group {g} not found in {datapath}, ignoring!")
        metadata["group_data_not_found"].append(g)
    else:
        if not os.path.exists(group_paths[i]): os.mkdir(group_paths[i])
        metadata[g] = {}

# update all_groups data
all_groups = [g for g in all_groups0 if g not in metadata["group_data_not_found"]]

# prettyprint debug info
prettyprint_list(metadata.keys(), title="Metadata dict. keys:")
prettyprint_list(metadata["group_data_not_found"], title="Items in group_data_not_found: ")

# start scrape and quarantine
all_dfs = []
quarantine_configs = []
for i, g in enumerate(all_groups):
    data_group = f"{datapath}/{g}"
    group_files = [f for f in os.listdir(data_group)]
    group_nfiles = len(group_files)

    # output csvs
    gallcsv = f"{output_path}/{g}/all-atoms_v{VERSION}.csv"
    gc5ncsv = f"{output_path}/{g}/{five_number_label}-{g}_configs-{g}_v{VERSION}.csv"
    ga5ncsv = f"{output_path}/{g}/{five_number_label}-{g}_atoms_v{VERSION}.csv"
    gqcsv = f"{output_path}/{g}/{quarantine_label}-{g}_v{VERSION}.csv"

    # if os.path.exists(ga5ncsv):
    #     print(f"! Group {g} already analysed!")
    #     print(f"! (v0 not perfect with this yet, but good enough.)!")
    #     print(f"! Continuing...")
    #     continue

    print("\n============================")
    print(f"Analysis: group '{g}'")
    gt1 = {g:gt0[g]}
    # if group_nfiles >= 1000:
    #     gt1["training_size"] = 0.25
    fs.config.sections["GROUPS"].group_table = gt1
    fs.scrape_configs()
    group_data = fs.scraper.all_data
    group_nconfigs = len(group_data)
    metadata[g]["nconfigs"] = group_nconfigs
    
    all_atom_data, fs_meta_cols, fs_atom_cols = [], [], []
    for d in group_data:
        new, fs_meta_cols, fs_atom_cols = format_fs_data(d, all_elems)
        fs_data_cols = fs_meta_cols + fs_atom_cols
        all_atom_data.extend(new)
    
    print(f"Creating group all_atoms dataframe")
    df0 = pd.DataFrame.from_records(all_atom_data, columns=fs_data_cols)

    group_natoms = df0.shape[0]
    metadata[g]["natoms"] = group_natoms
    group_elems = [e for e in all_elems if e in df0.atom_elem.unique()]
    metadata[g]["elems"] = group_elems
    group_nelems = len(group_elems)
    group_colors = [species_plot_colors[all_elems.index(e)] for e in group_elems] 

    print(f"Performing thresholding ")
    ## Simple thresholding
    ## other quarantine modes not implemented
    for i, thcol in enumerate(thresh_cols):
        threshhold = threshval_list[i]
        thresh_op = threshop_list[i]
        info_str = f"{thcol}{thresh_op}{threshhold} "
        if thresh_op == ">" or thresh_op == ">=" or thresh_op == "gt" or thresh_op == "gte":
            df0.loc[df0[thcol]>=threshhold,"flags_thresh"] += info_str
            qdf = df0.loc[df0[thcol]>=threshhold,:]
        elif thresh_op == "<" or thresh_op == "<=" or thresh_op == "lt" or thresh_op == "lte":
            df0.loc[df0[thcol]<=threshhold,"flags_thresh"] += info_str
            qdf = df0.loc[df0[thcol]<=threshhold,:]
        elif thresh_op == "outside" or thresh_op == "notin":
            th1, th2 = threshhold
            info_str = f"{thcol}_{thresh_op}_{th1}-{th2}A "
            # df0.loc[(df0[thcol]<=th1)|(th2<=df0[thcol]),"flags_thresh"] += info_str
            df0.loc[~df0[thcol].between(th1, th2),"flags_thresh"] += info_str
            qdf = df0.loc[(th1<=df0[thcol])&(df0[thcol]<=th2),:]
        elif thresh_op == "between" or thresh_op == "inside" or thresh_op == "in":
            th1, th2 = threshhold
            info_str = f"{thcol}_{thresh_op}_{th1}-{th2}A "
            # df0.loc[(df0[thcol]<=th1)|(th2<=df0[thcol]),"flags_thresh"] += info_str
            df0.loc[df0[thcol].between(th1, th2),"flags_thresh"] += info_str
            qdf = df0.loc[(df0[thcol]<=th1)|(df0[thcol]>=th2),:]
        else:
            print("! Threshold operator type not implemented!")
            print("! You can add new/different logic and operator tags in the 'training_check_params' companion script, see section/variable 'threshop_operators'")
            print("! Continuing for now...")
            continue

        metadata[g][f"thresh_{thcol}"] = info_str
    
    print("Collecting five-number statistics")
    ## Five-number summary (box (whisker) plot number)
    all_desc_atoms_df = []
    all_outlier_idxs = []
    extra_stats = "median skew kurt".split()
    desc_df0 = describe_extraT(df0[desc_configs_cols], extra_stats)
    desc_df0.loc[:,"elem"] = "all"
    for c in desc_configs_cols:
        desc_df0.loc[c,"units"] = desc_configs_units[desc_configs_cols.index(c)]
        desc_df0.loc[c,"IQR"] = desc_df0.loc[c,"75%"]-desc_df0.loc[c,"25%"]
        desc_df0.loc[c,"wh_hi"] = desc_df0.loc[c,"75%"] + desc_df0.loc[c,"IQR"]*tukey_factor
        desc_df0.loc[c,"wh_lo"] = desc_df0.loc[c,"25%"] - desc_df0.loc[c,"IQR"]*tukey_factor
        desc_df0.loc[c,"range"] = np.abs(desc_df0.loc[c,"max"]-desc_df0.loc[c,"min"])

        ## find outliers in original dataframe
        # list of list that maps to desc_atoms_cols (to tell what "flagged" the outlier)
        # idx_hi = df0.loc[(df0[c]>=desc_df0.loc[c,"wh_hi"]),:].index.tolist()
        # idx_lo = df0.loc[(df0[c]<=desc_df0.loc[c,"wh_lo"]),:].index.tolist()
        # outlier_idxs = idx_hi + idx_lo
        # all_outlier_idxs.extend(outlier_idxs)
        # if len(outlier_idxs) != 0:
        #     df0.loc[df0.index.isin(outlier_idxs),"flags_out"] += f"c-{c} "
    desc_df0 = desc_df0.reset_index().rename(columns={"index":"atom_prop"})
    reorder_cols = 'atom_prop elem count min 25% 50% 75% max range median mean std IQR wh_hi wh_lo skew kurt units'.split()
    desc_df0 = desc_df0[reorder_cols]
    stat_configs_df = desc_df0
    stat_configs_df.to_csv(gc5ncsv, index=False)
    # metadata[g][f"five_number_df_configs"] = desc_df0
    metadata[g][f"five_number_csv_configs"] = gc5ncsv
    
    for e in group_elems:
        desc_df1 = describe_extraT(df0.loc[df0.atom_elem==e,desc_atoms_cols], extra_stats)
        desc_df1.loc[:,"elem"] = e
        for c in desc_atoms_cols:
            desc_df1.loc[c,"units"] = desc_atoms_units[desc_atoms_cols.index(c)]
            desc_df1.loc[c,"IQR"] = desc_df1.loc[c,"75%"]-desc_df1.loc[c,"25%"]
            desc_df1.loc[c,"wh_hi"] = desc_df1.loc[c,"75%"] + desc_df1.loc[c,"IQR"]*tukey_factor
            desc_df1.loc[c,"wh_lo"] = desc_df1.loc[c,"25%"] - desc_df1.loc[c,"IQR"]*tukey_factor
            desc_df1.loc[c,"range"] = np.abs(desc_df1.loc[c,"max"]-desc_df1.loc[c,"min"])

            ## find outliers in original dataframe
            # list of list that maps to desc_atoms_cols (to tell what "flagged" the outlier)
            idx_hi = df0.loc[(df0.atom_elem==e)&(df0[c]>=desc_df1.loc[c,"wh_hi"]),:].index.tolist()
            idx_lo = df0.loc[(df0.atom_elem==e)&(df0[c]<=desc_df1.loc[c,"wh_lo"]),:].index.tolist()
            outlier_idxs = idx_hi + idx_lo
            all_outlier_idxs.extend(outlier_idxs)
            if len(outlier_idxs) != 0:
                df0.loc[(df0.atom_elem==e)&(df0.index.isin(outlier_idxs)),"flags_out"] += f"{c} "
        desc_df1 = desc_df1.reset_index().rename(columns={"index":"atom_prop"})
        reorder_cols = 'atom_prop elem count min 25% 50% 75% max range median mean std IQR wh_hi wh_lo skew kurt units'.split()
        desc_df1 = desc_df1[reorder_cols]
        all_desc_atoms_df.append(desc_df1)
    
    stat_atoms_df = pd.concat(all_desc_atoms_df, ignore_index=True)
    stat_atoms_df.to_csv(ga5ncsv, index=False)
    # metadata[g][f"five_number_df_atoms"] = desc_df0
    metadata[g][f"five_number_csv_atoms"] = ga5ncsv

    print("Quarantining configs above thresholds")
    ## collect all "problematic" configurations (threshold only for now):
    problemask = (df0["flags_thresh"]!="")
    if np.sum(problemask) != 0:
        problematique = df0.loc[problemask,:]
        gqdf = format_quarantine_df(problematique)
        gqdf.to_csv(gqcsv, index=False)
        # metadata[g][f"quarantine_df"] = gqdf
        metadata[g][f"quarantine_csv"] = gqcsv
        
        quarantine_configs.append(problematique)

    print("Writing group CSV")
    df0.to_csv(gallcsv, index=False)
    all_dfs.append(df0)
    # metadata[g][f"all_atoms_df"] = df0
    metadata[g][f"all_atoms_csv"] = gallcsv 

    ## plotting! 
    if SAVE_PLOTS:
        print("Generating plots for fmag, nn1_dist")
        for xcol in "nn1_dist fmag".split():
            omask = df0["flags_thresh"].str.contains(xcol)
            onconfigs = len(df0[omask].fname.unique())
            thresh_info = 1 # metadata[g][xcol]

            # remove any buggy extrema that can screw up plotting
            if PLOT_EXCLUDE_EXTREMA:
                extrema_mask = (df0.fmag < fmag_extreme)&(df0.nn1_dist < nndist_extreme)&(df0.eatom < eatom_extreme)
                df = df0.loc[extrema_mask,:]

            # print("CHECK", xcol, np.sum(omask), df0.shape, df1.shape, df0[omask].flags_out.tolist(),len(ofile_list) ,'\n')
            # figsize = [12,7]
            # fig, axes = plt.subplots(1,3,figsize=figsize)
            # fig, axes = plt.subplots(2,3,figsize=figsize)
            # ax1, ax2, ax3 = axes[0]
            # ax4, ax5, ax6 = axes[1]

            figsize = [14,4.5]
            fig, axes = plt.subplots(1,3,figsize=figsize)
            ax1, ax2, ax3 = axes

            sns.boxplot(df,ax=ax1,x=xcol, y="atom_elem", order=group_elems, palette=group_colors)
            sns.ecdfplot(df,ax=ax2,x=xcol, hue="atom_elem", hue_order=group_elems, palette=group_colors)
            sns.histplot(df,ax=ax3,x=xcol, hue="atom_elem", hue_order=group_elems, palette=group_colors, multiple="stack")

            # sns.boxplot(df1,ax=ax4,x=xcol, y="atom_elem", order=group_elems, palette=group_colors)
            # sns.ecdfplot(df1,ax=ax5,x=xcol, hue="atom_elem", hue_order=group_elems, palette=group_colors)
            # sns.histplot(df1,ax=ax6,x=xcol, hue="atom_elem", hue_order=group_elems, palette=group_colors, multiple="stack")
            fig.suptitle(f"Atom property '{xcol}', group {g}, {group_natoms} atoms from {group_nconfigs} files\nAtoms with {thresh_info}: {np.sum(omask)} atoms from {onconfigs} files")

            # x1, x2, x3 = ax1.get_xlim(), ax2.get_xlim(), ax3.get_xlim()
            # ax4.set_xlim(x1)
            # ax5.set_xlim(x2)
            # ax6.set_xlim(x3)
            
            # [a.legend().set_visible(False) for a in [ax1, ax2, ax3, ax4, ax6]]
            [a.legend().set_visible(False) for a in [ax1, ax2]]
            plt.tight_layout()

            png_name = f"{output_path}/{g}/atoms_distr-{xcol}_summary_v{VERSION}.png"
            fig.savefig(png_name)
            if SHOW_PLOTS: plt.show()
            # plt.close()
            # #https://stackoverflow.com/questions/28757348/how-to-clear-memory-completely-of-all-matplotlib-plots
            fig.clear()
            plt.close(fig)

moadfs = pd.concat(all_dfs,ignore_index=True)

nconfigs_all = 0
natoms_all = 0
for g in all_groups:
    nconfigs_all += metadata[g]["nconfigs"]
    natoms_all += metadata[g]["natoms"]

# create a summary plot
if SAVE_PLOTS:
        for xcol in "econfig eatom nn1_dist fmag".split():

            # remove any buggy extrema that can screw up plotting
            if PLOT_EXCLUDE_EXTREMA:
                extrema_mask = (moadfs.fmag < fmag_extreme)&(moadfs.nn1_dist < nndist_extreme)&(moadfs.eatom < eatom_extreme)
                df = moadfs.loc[extrema_mask,:]

            # print("CHECK", xcol, np.sum(omask), df0.shape, df1.shape, df0[omask].flags_out.tolist(),len(ofile_list) ,'\n')
            # figsize = [12,7]
            # fig, axes = plt.subplots(1,3,figsize=figsize)
            # fig, axes = plt.subplots(2,3,figsize=figsize)
            # ax1, ax2, ax3 = axes[0]
            # ax4, ax5, ax6 = axes[1]

            figsize = [14,4.5]
            fig, axes = plt.subplots(1,3,figsize=figsize)
            ax1, ax2, ax3 = axes

            elems_colors = species_plot_colors[:len(all_elems)]
            sns.boxplot(df,ax=ax1,x=xcol, y="atom_elem", order=all_elems, palette=elems_colors)
            sns.ecdfplot(df,ax=ax2,x=xcol, hue="atom_elem", hue_order=all_elems, palette=elems_colors)
            sns.histplot(df,ax=ax3,x=xcol, hue="atom_elem", hue_order=all_elems, palette=elems_colors, multiple="stack")

            # sns.boxplot(df1,ax=ax4,x=xcol, y="atom_elem", order=group_elems, palette=group_colors)
            # sns.ecdfplot(df1,ax=ax5,x=xcol, hue="atom_elem", hue_order=group_elems, palette=group_colors)
            # sns.histplot(df1,ax=ax6,x=xcol, hue="atom_elem", hue_order=group_elems, palette=group_colors, multiple="stack")
            fig.suptitle(f"SUMMARY, all groups ({nconfigs_all} configs, {natoms_all} atoms), atom property '{xcol}'")

            [a.legend().set_visible(False) for a in [ax1, ax2]]
            plt.tight_layout()

            png_name = f"{output_path}/all_atoms_distr-{xcol}_summary_v{VERSION}.png"
            fig.savefig(png_name)
            if SHOW_PLOTS: plt.show()
            # plt.close()
            # #https://stackoverflow.com/questions/28757348/how-to-clear-memory-completely-of-all-matplotlib-plots
            fig.clear()
            plt.close(fig)



print("\n============================")
print("Completed group analysis!")
print("Generating final summary data")

if len(quarantine_configs) > 0:
    all_qdfs = pd.concat(quarantine_configs, ignore_index=True)

    # Quarantine columns: 
    # group fname natoms econfig eatom flags_out flags_thresh
    qcols = "group fname natoms econfig eatom flags_thresh flags_out".split()
    final_qdfs = format_quarantine_df(all_qdfs)
    final_qdfs.to_csv(f"{output_path}/{final_qcsv}", index=False)

import json
with open(f"{output_path}/metadata.json", 'w') as f:
    f.write("# Metadata dictionary from training_check_v0.py\n")
    json.dump(metadata, f, indent=4)

#  Final stdout simple report
cwd = os.getcwd()

thresh_str = f"Threshold information:\n-------------------------------------------\n{'name': <16}{'value': <16}{'operator': <16}\n-------------------------------------------\n"
for i in range(len(threshval_list)):
    thresh_str += f"{thresh_cols[i]: <16}{str(threshval_list[i]): <16}{str(threshop_list[i]): <16}\n"

group_str = "Groups analyzed (with settings): \n"
for g in all_groups:
    group_weights = fs.config._original_config["GROUPS"][g]
    group_str += f"  {g} = {group_weights}\n"

report_str = f"""=======================================
Training set analysis complete!
=======================================
Settings file or dictionary: {settings_file_or_dict}
User label: {user_label}
Output path: {cwd}/{output_path}
Training set: {cwd}/{datapath}
{group_str}
Total analyzed configurations: {nconfigs_all}
Total atoms: {natoms_all}
Number of quarantined configurations: {len(quarantine_configs)}
{thresh_str}
"""

print(report_str)

print(f"Writing brief report above to file: {report_file}")
with open(report_file, "w") as f:
    f.write(report_str)

print("Script complete!\n")

exit()
