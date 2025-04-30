import os 
os.environ["POLARS_MAX_THREADS"] = '8'
os.environ["POLARS_FORCE_NEW_STREAMING"] = "1"
os.environ["OMP_NUM_THREADS"] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figs')

import polars as pl
pl.enable_string_cache()
import numpy as np
np.random.seed(1234)
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date
from dst import utils, geo

import pyarrow.parquet as pq

#####################################################
# Again collecting my pl.Expr's that I use later on #
#####################################################


# An expr to explode (outputs a list of dates) later on and then filter. 
YEAR_EXPR = pl.date_ranges(date(1985, 12, 31), date(2020, 12, 31), interval='1y').alias("year")

# To separate household types
HH_TYPE_EXPR = (pl.when(pl.col("native_hh")==True)
                       .then(1)
                       .when((pl.col("mix_non_west_share")>0))
                       .then(2)
                       .when((pl.col("mix_non_west_share")==0) & (pl.col("mix_share")>0))
                       .then(3))


# Based on the three types as defined in the paper, who are their K-nearest neighbors over time?
# How does the (unconditional) residential sorting pattern change over time?
lf = pl.scan_parquet(f'{DATA_DIR}/build/knn100.pq').set_sorted('hh_id').with_columns(
    hh_type = pl.when(pl.col("native_hh")==True).then(1).when((pl.col("mix_non_west_share")>0)).then(2).when((pl.col("mix_non_west_share")==0) & (pl.col("mix_share")>0)).then(3),
    hh_type_nn = pl.when(pl.col("native_hh_nn")==True).then(1).when((pl.col("mix_non_west_share_nn")>0)).then(2).when((pl.col("mix_non_west_share_nn")==0) & (pl.col("mix_share_nn")>0)).then(3),
).with_columns(
    year = YEAR_EXPR,
    same_type_neighbor = pl.col("hh_type")==pl.col("hh_type_nn")
)


# Did the node exist @ Dec 31st year x?
DATE_EXPR1 = pl.col("year").is_between(pl.col("bop_vfra"), pl.col("bop_vtil"))

# Did the neighboring node exist @ Dec 31st year x?
DATE_EXPR2 = pl.col("year").is_between(pl.col("bop_vfra_nn"), pl.col("bop_vtil_nn"))

# Both must be TRUE
EDGE_LINK_EXPR = (DATE_EXPR1) & (DATE_EXPR2)

# At each level of K-nearest, compute the number of same-type neighbors
ks = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]

for k in ks:
    lf_knn_adj = lf.filter(pl.col("rank_nn")<=k).explode("year").filter(EDGE_LINK_EXPR).group_by('hh_id', 'year').agg(
        pl.col("native_hh").first(),
        (pl.col("mix_non_west_share")>0).alias("mix_non_west_hh").first(),
        ((pl.col("mix_non_west_share")==0) & (pl.col("mix_share")>0)).alias("other").first(),
        pl.min_horizontal(pl.col("same_type_neighbor").sum(), pl.lit(k)).alias("same_type_neighbor")
    ).with_columns(
        knn = k,
        year = pl.col("year").dt.year().cast(pl.Int16)
    ).sort('hh_id', 'year')
    lf_knn_adj.collect().write_parquet(f'{DATA_DIR}/build/knn/knn_desc/knn{k}.pq')



### COUNTERFACTUAL ANALYSIS ###
# Quick and dirty: Based on 1990-distribution of household types, "sample" with probability for each HH_id-year pair, with probabilities given by the 1990 distribution.
df_sample_probz = utils.baseline_proba(lf)

df_sample_probz.select(pl.all().exclude("bop_vfra", "bop_vtil", "hh_type")).sort('hh_id', 'year').write_parquet(f'{DATA_DIR}/build/hh_type_counterf.pq')
lf_typez = pl.scan_parquet(f'{DATA_DIR}/build/hh_type_counterf.pq').set_sorted('hh_id')
idz = lf_typez.select(pl.col("hh_id").unique()).collect().to_series()

chunk_size = idz.len() // 10

hh_id_list = []
for i in range(0, idz.len(), chunk_size):
    chunkz = idz.slice(i, chunk_size).set_sorted()
    hh_id_list.append(chunkz)

batch_no = 0
hh_id_list = hh_id_list[batch_no::]

# Same logic as above, except now I'm doing a computationally heavy join
ks = [100, 80, 60, 40, 20, 10, 5]
for k in ks:
    batch_no = 0
    for ids in hh_id_list:
        if k<=50:
            lf = pl.scan_parquet(f'{DATA_DIR}/build/knn50.pq').set_sorted('hh_id').filter(pl.col("hh_id").is_in(ids)).filter(pl.col("bop_vfra")>=pl.date(1960, 1, 1)).filter(pl.col("bop_vfra_nn")>=pl.date(1960,1,1)).with_columns(
            hh_type = pl.when(pl.col("native_hh")==True).then(1).when((pl.col("mix_non_west_share")>0)).then(2).when((pl.col("mix_non_west_share")==0) & (pl.col("mix_share")>0)).then(3),
            hh_type_nn = pl.when(pl.col("native_hh_nn")==True).then(1).when((pl.col("mix_non_west_share_nn")>0)).then(2).when((pl.col("mix_non_west_share_nn")==0) & (pl.col("mix_share_nn")>0)).then(3),
            year = YEAR_EXPR
        )
        else:
            lf = pl.scan_parquet(f'{DATA_DIR}/build/knn100.pq').set_sorted('hh_id').filter(pl.col("hh_id").is_in(ids)).filter(pl.col("bop_vfra")>=pl.date(1960, 1, 1)).filter(pl.col("bop_vfra_nn")>=pl.date(1960,1,1)).with_columns(
            hh_type = pl.when(pl.col("native_hh")==True).then(1).when((pl.col("mix_non_west_share")>0)).then(2).when((pl.col("mix_non_west_share")==0) & (pl.col("mix_share")>0)).then(3),
            hh_type_nn = pl.when(pl.col("native_hh_nn")==True).then(1).when((pl.col("mix_non_west_share_nn")>0)).then(2).when((pl.col("mix_non_west_share_nn")==0) & (pl.col("mix_share_nn")>0)).then(3),
            year = YEAR_EXPR
        )

        lf_knn_adj = lf.filter(pl.col("rank_nn")<=k).explode("year").filter(EDGE_LINK_EXPR).with_columns(
            year = pl.col("year").dt.year().cast(pl.Int16)
        ).join(lf_typez, on = ['hh_id', 'year']).with_columns(
            same_type_neighbor_counterf = pl.col("hh_type_counterf") == pl.col("hh_type_nn"),
        ).group_by('hh_id', 'year').agg(
            pl.col("native_hh").first(),
            (pl.col("mix_non_west_share")>0).alias("mix_non_west_hh").first(),
            ((pl.col("mix_non_west_share")==0) & (pl.col("mix_share")>0)).alias("other").first(),
            pl.min_horizontal(pl.col("same_type_neighbor_counterf").sum(), pl.lit(k)).alias("same_type_neighbor")
        ).with_columns(
            knn = k,
        ).sort('hh_id', 'year')
        lf_knn_adj.collect(engine = 'streaming').write_parquet(f'{DATA_DIR}/build/knn/knn_desc/knn{k}_counterf_{batch_no}.pq')

        message = f'KNN descriptives: counterfactual analysis, K = {k}, batch {batch_no}.'
        time_ = utils.what_time_is_it()
        utils.log(message, time_)
        print(message)
        batch_no += 1

# Plotting our results [including top three quintiles of the "counterfactuals"]
# First, native households. Then non-Western households.
ks = [5, 10, 20, 40, 80, 100]

fig, axes = plt.subplots(1, 6, figsize = (15, 3), sharex = True, sharey = True, gridspec_kw={'wspace': 0.05})
original_cmap = mpl.colormaps.get_cmap('Spectral')
custom_cmap = original_cmap(np.linspace(0.25, 1, 256))
colormap = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', custom_cmap)
colors = colormap(np.linspace(0, 1, 5+1))

lines = ['-', '--', '-.', ':', '---', '.']
markers = ['o', 'v', '^', 's', 'd', 'x']


for k, axz  in zip(ks, axes):
    lf_knn = pl.scan_parquet(f'{DATA_DIR}/build/knn/knn_desc/knn_{k}.pq').filter(pl.col("year").is_between(1990, 2020)).filter(pl.col("native_hh")==True)
    lf_knn_counterf = pl.scan_parquet(f'{DATA_DIR}/build/knn/knn_desc/knn{k}_counterf_*.pq').filter(pl.col("year").is_between(1990, 2020)).filter(pl.col("native_hh")==True)
    
    df_nn_group = lf_knn.group_by('year', 'same_type_neighbor').agg(
        pl.len().alias("n")
    ).sort('year', 'same_type_neighbor').with_columns(
        n_pct = pl.col("n")/pl.col("n").sum().over('year')
    ).collect()

    df_nn_group_counterf = lf_knn_counterf.group_by('year', 'same_type_neighbor').agg(
        pl.len().alias("n")
    ).sort('year', 'same_type_neighbor').with_columns(
        n_pct = pl.col("n")/pl.col("n").sum().over('year')
    ).collect()

    step = k // 5

    knn_shares = {}
    knn_shares_counterf = {}

    years = df_nn_group.select(pl.col("year").unique()).sort('year')

    for i in range(0, k+1, step):
        if i == 0:
            y = df_nn_group.filter(pl.col("same_type_neighbor")==i).sort('year').select(pl.col("year", "n_pct"))
            y_counterf = df_nn_group_counterf.filter(pl.col("same_type_neighbor")==i).sort('year').select(pl.col("year", "n_pct"))

            y2 = years.join(y, on='year', how='left').with_columns(n_pct = pl.col("n_pct").replace({None: 0})).select(pl.col("n_pct")).to_series()
            y2_counterf = years.join(y_counterf, on='year', how='left').with_columns(n_pct = pl.col("n_pct").replace({None: 0})).select(pl.col("n_pct")).to_series()

            knn_shares[f'0%'] = y2
            knn_shares_counterf[f'0%'] = y2_counterf

        else:
            y = df_nn_group.filter(pl.col("same_type_neighbor").is_between(i-step, i, closed='right')).sort('year').select(pl.col("year", "n_pct")).with_columns(n_pct = pl.col("n_pct").sum().over("year")).filter(pl.col("year").is_first_distinct())
            y2 = years.join(y, on='year', how='left').with_columns(n_pct = pl.col("n_pct").replace({None: 0})).select(pl.col("n_pct")).to_series()

            y_counterf = df_nn_group_counterf.filter(pl.col("same_type_neighbor").is_between(i-step, i, closed='right')).sort('year').select(pl.col("year", "n_pct")).with_columns(n_pct = pl.col("n_pct").sum().over("year")).filter(pl.col("year").is_first_distinct())
            y2_counterf = years.join(y_counterf, on='year', how='left').with_columns(n_pct = pl.col("n_pct").replace({None: 0})).select(pl.col("n_pct")).to_series()

            start = int(((i - step)/k) * 100)
            end = int((i/k) * 100)
            
            knn_shares[f']{start}%, {end}%]'] = y2
            knn_shares_counterf[f']{start}%, {end}%]'] = y2_counterf

    axz.stackplot(years.to_series(), 
                  knn_shares.values(), 
                labels = knn_shares.keys(),
                colors=colors
                )
    rez_arr = np.array(list(knn_shares_counterf.values()))
    rez_cum_sum = np.cumsum(rez_arr, axis = 0)[:-1]
    counterf_legend_labs = list(knn_shares.keys())
    for i in [2, 3, 4]:
        axz.plot(years.to_series().gather_every(3),
                rez_cum_sum[i][::3],
                color = 'k',
                ls = '--',
                marker = markers[i],
                label = f'{counterf_legend_labs[i+1]} (simulated)',
                linewidth = 0.5,
                markersize = 4
                )
    axz.set_title(f'K = {k}')
    if k == ks[0]:
        axz.set_ylabel('Share of households by year')
    else:
        pass
for axxx in axes:
    for label in axxx.get_xticklabels():
        label.set_rotation(45)

axz.legend(loc = 'center right', bbox_to_anchor = (2, 0.5), fontsize=7)
plt.close()
fig.savefig(f'{FIG_DIR}/temporal_knn_native_1990_2020_w_sim.pdf', bbox_inches = 'tight')


ks = [5, 10, 20, 40, 80, 100]
fig, axes = plt.subplots(1, 6, figsize = (15, 3), sharex = True, sharey = True, gridspec_kw={'wspace': 0.05})
original_cmap = mpl.colormaps.get_cmap('Spectral')
custom_cmap = original_cmap(np.linspace(0.25, 1, 256))
colormap = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', custom_cmap)
colors = colormap(np.linspace(0, 1, 5+1))

original_cmap_bw = mpl.colormaps.get_cmap('Greys')
custom_cmap_bw = original_cmap(np.linspace(0.25, 1, 256))
colormap_bw = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap_bw', custom_cmap_bw)
colors_bw = colormap_bw(np.linspace(0, 1, 5+1))

lines = ['-', '--', '-.', ':', '---', '.']
markers = ['o', 'v', '^', 's', 'd', 'x']


for k, axz  in zip(ks, axes):
    lf_knn = pl.scan_parquet(f'{DATA_DIR}/build/knn/knn_desc/knn_{k}.pq').filter(pl.col("year").is_between(1990, 2020)).filter(pl.col("mix_non_west_hh")==True)
    lf_knn_counterf = pl.scan_parquet(f'{DATA_DIR}/build/knn/knn_desc/knn{k}_counterf*.pq').filter(pl.col("year").is_between(1990, 2020)).filter(pl.col("mix_non_west_hh")==True)
    
    df_nn_group = lf_knn.group_by('year', 'same_type_neighbor').agg(
        pl.len().alias("n")
    ).sort('year', 'same_type_neighbor').with_columns(
        n_pct = pl.col("n")/pl.col("n").sum().over('year')
    ).collect()

    df_nn_group_counterf = lf_knn_counterf.group_by('year', 'same_type_neighbor').agg(
        pl.len().alias("n")
    ).sort('year', 'same_type_neighbor').with_columns(
        n_pct = pl.col("n")/pl.col("n").sum().over('year')
    ).collect()


    if k == 5:
        step = 1

    else:
        step = k // 5

    knn_shares = {}
    knn_shares_counterf = {}

    years = df_nn_group.select(pl.col("year").unique()).sort('year')

    for i in range(0, k+1, step):
        if i == 0:
            y = df_nn_group.filter(pl.col("same_type_neighbor")==i).sort('year').select(pl.col("year", "n_pct"))
            y_counterf = df_nn_group_counterf.filter(pl.col("same_type_neighbor")==i).sort('year').select(pl.col("year", "n_pct"))

            y2 = years.join(y, on='year', how='left').with_columns(n_pct = pl.col("n_pct").replace({None: 0})).select(pl.col("n_pct")).to_series()
            y2_counterf = years.join(y_counterf, on='year', how='left').with_columns(n_pct = pl.col("n_pct").replace({None: 0})).select(pl.col("n_pct")).to_series()

            knn_shares[f'0%'] = y2
            knn_shares_counterf[f'0%'] = y2_counterf

        else:
            y = df_nn_group.filter(pl.col("same_type_neighbor").is_between(i-step, i, closed='right')).sort('year').select(pl.col("year", "n_pct")).with_columns(n_pct = pl.col("n_pct").sum().over("year")).filter(pl.col("year").is_first_distinct())
            y2 = years.join(y, on='year', how='left').with_columns(n_pct = pl.col("n_pct").replace({None: 0})).select(pl.col("n_pct")).to_series()

            y_counterf = df_nn_group_counterf.filter(pl.col("same_type_neighbor").is_between(i-step, i, closed='right')).sort('year').select(pl.col("year", "n_pct")).with_columns(n_pct = pl.col("n_pct").sum().over("year")).filter(pl.col("year").is_first_distinct())
            y2_counterf = years.join(y_counterf, on='year', how='left').with_columns(n_pct = pl.col("n_pct").replace({None: 0})).select(pl.col("n_pct")).to_series()

            start = int(((i - step)/k) * 100)
            end = int((i/k) * 100)
            
            knn_shares[f']{start}%, {end}%]'] = y2
            knn_shares_counterf[f']{start}%, {end}%]'] = y2_counterf

    axz.stackplot(years.to_series(), 
                  knn_shares.values(), 
                labels = knn_shares.keys(),
                colors=colors
                )
    rez_arr = np.array(list(knn_shares_counterf.values()))
    rez_cum_sum = np.cumsum(rez_arr, axis = 0)[:-1]
    counterf_legend_labs = list(knn_shares.keys())
    for i in [2, 3, 4]:
        axz.plot(years.to_series().gather_every(3),
                rez_cum_sum[i][::3],
                color = 'k',
                ls = '--',
                marker = markers[i],
                label = f'{counterf_legend_labs[i+1]} (simulated)',
                linewidth = 0.5,
                markersize = 4
                )
    axz.set_title(f'K = {k}')
    if k == ks[0]:
        axz.set_ylabel('Share of households by year')
    else:
        pass
for axxx in axes:
    for label in axxx.get_xticklabels():
        label.set_rotation(45)

axz.legend(loc = 'center right', bbox_to_anchor = (2, 0.5), fontsize=7)
plt.close()
fig.savefig(f'{FIG_DIR}/temporal_knn_non_west_1990_2020_w_sim.pdf', bbox_inches = 'tight')

########################################################
## LINE PLOT OF HOUSEHOLD TYPE DISTRIBUTION OVER TIME ##
########################################################
nomen_map = utils.fetch_origin_mapping()
non_west_country_cats = utils.fetch_country_cats(sub_cat = 'non-west')

lf_knn = (pl.scan_parquet(f'{DATA_DIR}/build/knn40.pq').set_sorted("hh_id").filter(pl.col("hh_id").is_first_distinct())
        .with_columns(
            hh_type = HH_TYPE_EXPR,
            year = YEAR_EXPR)
)
lf_group_hh_type = (lf_knn
).explode("year").filter(DATE_EXPR1).with_columns(
    year = pl.col("year").dt.year()
).group_by('year').agg(
    (pl.col("hh_type")==1).mean().alias("native_share"),
    (pl.col("hh_type")==2).mean().alias("non_west_share"),
    (pl.col("hh_type")==3).mean().alias("west_share")
)

df_group_hh_type = lf_group_hh_type.collect(engine = 'streaming')
df = lf.collect(engine = 'streaming')
hh_shares = {}
hh_shares['Western'] = df_group_hh_type.select(pl.col("west_share")).to_series()
hh_shares['Non-Western'] = df_group_hh_type.select(pl.col("non_west_share")).to_series()
hh_shares['Native'] = df_group_hh_type.select(pl.col("native_share")).to_series()
years = df_group_hh_type.select(pl.col("year")).to_series()

list_ = ['Western', 'Non-Western', 'Native']
colors = ['b', 'k', 'r']

fig,ax = plt.subplots(figsize = (10,6), tight_layout = True)
ax2 = ax.twinx()
lines = []
for i, label in enumerate(hh_shares.keys()):
    
    if label == 'Native':
        line = ax2.plot(
            years,
            hh_shares[label],
            label = f'{label} (right axis)',
            color = 'r',
            linewidth = 2
        )
        lines.append(line)
    else:
        line = ax.plot(years, 
                    hh_shares[label], 
                    label = label,
                    color = colors[i], 
                    linewidth = 2
                    )
        lines.append(line)
        
ax.set_ylabel('Share of non-native households by year')
ax2.set_ylabel('Share of native households by year')

ax.set_xlim(1985, 2020)

ax.set_ylim(0, 0.2)
ax2.set_ylim(0.5, 1)

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax.legend(lines + lines2, labels + labels2, loc = 'upper right')
fig.savefig(f'{FIG_DIR}/hh_dist_1985_2020.pdf', bbox_inches = 'tight')