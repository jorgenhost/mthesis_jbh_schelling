import os 
os.environ["POLARS_MAX_THREADS"] = '8'
os.environ["OMP_NUM_THREADS"] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.2.3'
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figs')

import polars as pl
import polars.selectors as cs
import numpy as np
np.random.seed(1234)
pl.enable_string_cache()
from dst import utils, geo
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
from dst.classific import educ_utils
from datetime import date
import pyarrow.parquet as pq

#####################################################
# Again collecting my pl.Expr's that I use later on #
#####################################################
YEAR_EXPR = pl.date_ranges(date(1985, 12, 31), date(2020, 12, 31), interval='1y').alias("year")
DATE_EXPR1 = pl.col("year").is_between(pl.col("bop_vfra"), pl.col("bop_vtil"))
HH_TYPE_EXPR = (pl.when(pl.col("native_hh")==True)
                       .then(1)
                       .when((pl.col("mix_non_west_share")>0))
                       .then(2)
                       .when((pl.col("mix_non_west_share")==0) & (pl.col("mix_share")>0))
                       .then(3))
HH_TYPE_EXPR_NN = (pl.when(pl.col("native_hh_nn")==True)
                       .then(1)
                       .when((pl.col("mix_non_west_share_nn")>0))
                       .then(2)
                       .when((pl.col("mix_non_west_share_nn")==0) & (pl.col("mix_share_nn")>0))
                       .then(3))

nomen_map = utils.fetch_origin_mapping()
non_west_country_cats = utils.fetch_country_cats(sub_cat = 'non-west')

lf = (pl.scan_parquet(f'{DATA_DIR}/raw/bef.pq')
      .with_columns(
          native = pl.col("opr_land") == 5100,
          non_west = pl.col("opr_land").replace_strict(nomen_map, default = None, return_dtype=pl.String).is_in(non_west_country_cats),
          year = (pl.col("aar")/100).cast(pl.Int16))
).group_by('year').agg(
    pl.col("native").mean(),
    pl.col("non_west").mean()
)

df = lf.collect(engine = 'streaming')


lf_knn = (pl.scan_parquet(f'{DATA_DIR}/build/knn40.pq').set_sorted('hh_id').with_columns(
    hh_type = HH_TYPE_EXPR,
    hh_type_nn = HH_TYPE_EXPR_NN
)
    .filter(pl.col("bop_vfra_nn")>=pl.date(1985, 1, 1))
)

lf_knn_group = lf_knn.group_by('hh_id').agg(
    pl.col("hh_type").first(),
    pl.col("kom").first(),
    pl.col("cluster_id").first(),
    ((pl.col("hh_type")==1) & (pl.col("hh_type_nn")==2) & (pl.col("bop_vfra_nn")>pl.col("bop_vfra"))).sum().alias("howdy_neighbor_native_households"),
    ((pl.col("hh_type")==2) & (pl.col("hh_type_nn")==1) & (pl.col("bop_vfra_nn")>pl.col("bop_vfra"))).sum().alias("howdy_neighbor_non_west_households"),

)

lf_knn = (pl.scan_parquet(f'{DATA_DIR}/build/knn40.pq').set_sorted('hh_id').with_columns(
    hh_type = HH_TYPE_EXPR,
    hh_type_nn = HH_TYPE_EXPR_NN
)
    .filter(pl.col("bop_vfra_nn")>=pl.date(1985, 1, 1))
)

lf_knn_group = lf_knn.group_by('hh_id').agg(
    pl.col("hh_type").first(),
    pl.col("kom").first(),
    pl.col("cluster_id").first(),
    ((pl.col("hh_type")==1) & (pl.col("hh_type_nn")==2) & (pl.col("bop_vfra_nn")>pl.col("bop_vfra"))).sum().alias("howdy_neighbor_native_households"),
    ((pl.col("hh_type")==2) & (pl.col("hh_type_nn")==1) & (pl.col("bop_vfra_nn")>pl.col("bop_vfra"))).sum().alias("howdy_neighbor_non_west_households"),
).collect(engine = 'streaming').lazy()

lf_knn_kom_dk = lf_knn_group.filter(pl.col("hh_type")==1).group_by('kom').agg(
    howdy_neighbor_scaled = pl.col("howdy_neighbor_native_households").sum()/(pl.col("hh_id").n_unique())
)


lf_knn_kom_non_west = lf_knn_group.filter(pl.col("hh_type")==2).group_by('kom').agg(
    howdy_neighbor_scaled = pl.col("howdy_neighbor_non_west_households").sum()/(pl.col("hh_id").n_unique())
)

lf_knn_cluster_dk = lf_knn_group.filter(pl.col("hh_type")==1).group_by('cluster_id').agg(
    pl.col("kom").first(),
    howdy_neighbor_scaled = pl.col("howdy_neighbor_native_households").sum()/(pl.col("hh_id").n_unique())
)
lf_knn_cluster_non_west = lf_knn_group.filter(pl.col("hh_type")==2).group_by('cluster_id').agg(
    pl.col("kom").first(),
    howdy_neighbor_scaled = pl.col("howdy_neighbor_non_west_households").sum()/(pl.col("hh_id").n_unique())
)

df_municipality_movez_dk = lf_knn_kom_dk.collect(engine = 'streaming')
df_municipality_movez_non_west = lf_knn_kom_non_west.collect(engine = 'streaming')

df_cluster_movez_dk = lf_knn_cluster_dk.collect(engine = 'streaming')
df_cluster_movez_non_west = lf_knn_cluster_non_west.collect(engine = 'streaming')



lf_hoods = pl.scan_parquet(f'{DATA_DIR}/build/geo_neighborhood.pq')
cluster_dict = lf_hoods.select(pl.col("cluster_id_500", "cluster_id_1000")).collect().to_dict()
cluster_dict = dict(zip(cluster_dict['cluster_id_500'], cluster_dict['cluster_id_1000']))

people_threshold = 500

lf_clusters = pl.scan_parquet(f'{DATA_DIR}/build/cluster_density_{people_threshold}.pq')
clusterz = lf_clusters.filter(pl.col("density_t").is_between(1_000, 25_000)).select(pl.col(f"cluster_id_{people_threshold}")).unique().collect().to_series()



gdf_munic_dk = gpd.read_parquet(f'{DATA_DIR}/raw/kommune.pq')
gdf_munic_dk['code'] = gdf_munic_dk['code'].astype('int')
gdf_munic_dk = gdf_munic_dk.merge(df_municipality_movez_dk.to_pandas(), how = 'left', left_on = 'code', right_on = 'kom')
gdf_clusters_dk = gpd.read_parquet(f'{DATA_DIR}/raw/clusters500.pq').merge(df_cluster_movez_dk.to_pandas(), left_on = 'cluster_id_500', right_on = 'cluster_id')

gdf_munic_non_west = gpd.read_parquet(f'{DATA_DIR}/raw/kommune.pq')
gdf_munic_non_west['code'] = gdf_munic_non_west['code'].astype('int')
gdf_munic_non_west = gdf_munic_non_west.merge(df_municipality_movez_non_west.to_pandas(), how = 'left', left_on = 'code', right_on = 'kom')
gdf_clusters_non_west = gpd.read_parquet(f'{DATA_DIR}/raw/clusters500.pq').merge(df_cluster_movez_non_west.to_pandas(), left_on = 'cluster_id_500', right_on = 'cluster_id')



# normalize colors on the neighborhood scale
vmin_hood_dk = gdf_clusters_dk['howdy_neighbor_scaled'].min()
vmax_hood_dk = gdf_clusters_dk['howdy_neighbor_scaled'].max()
neighborhood_norm_dk = mpl.colors.Normalize(vmin=vmin_hood_dk, vmax = vmax_hood_dk)


original_cmap = mpl.colormaps.get_cmap('plasma')
custom_cmap = original_cmap(np.linspace(0.1, 1, 256))
cm_plasma_custom = mpl.colors.LinearSegmentedColormap.from_list('plasma_custom', custom_cmap)

fig, ax = plt.subplots(figsize = (10, 6), tight_layout = True)

gdf_munic_dk.plot(
    column = 'howdy_neighbor_scaled',
    ax = ax,
    cmap = 'plasma', 
    legend = True,
    legend_kwds = {
        'label': 'Avg. diff-type move-in per HH | K = 40',
        'orientation': 'vertical',
        'shrink': .65,
        'anchor': (-1, 1)
    },
    edgecolor = 'k'
)

ax.set_axis_off()
fig.savefig(f'{FIG_DIR}/dk_howdy_neighbor_native_hh.pdf', bbox_inches = 'tight')


fig, ax = plt.subplots(figsize = (10, 6), tight_layout = True)

gdf_cph = gdf_clusters_dk[gdf_clusters_dk['kom']==101].cx[:, 6.1725*1e6:] 


gdf_cph.plot(
    column = 'howdy_neighbor_scaled',
    ax = ax,
    norm = neighborhood_norm_dk,
    cmap = cm_plasma_custom, 
    legend = True,
    legend_kwds = {
        'label': 'Avg. diff-type move-in per HH | K = 40',
        'orientation': 'vertical',
        'shrink': .65,
        'anchor': (-.5, 1)
    },
    edgecolor = 'k'
)

ax.set_axis_off()
fig.savefig(f'{FIG_DIR}/cph_howdy_neighbor.pdf', bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (10, 6), tight_layout = True)

gdf_cph = gdf_clusters_dk[gdf_clusters_dk['kom']==101].cx[:, 6.1725*1e6:] 
gdf_cph_anti = gdf_cph[~gdf_cph['cluster_id'].isin(clusterz)]


gdf_cph[gdf_cph['cluster_id'].isin(clusterz)].plot(
    column = 'howdy_neighbor_scaled',
    ax = ax,
    norm = neighborhood_norm_dk,
    cmap = cm_plasma_custom, 
    legend = True,
    legend_kwds = {
        'label': 'Avg. diff-type move-in per HH | K = 40',
        'orientation': 'vertical',
        'shrink': .65,
        'anchor': (-.5, 1)
    },
    edgecolor = 'k'
)

gdf_cph_anti.plot(
    hatch = '//',
    color = 'grey',
    alpha = .4,
    ax = ax,
    edgecolor = 'k'
)

ax.set_axis_off()
fig.savefig(f'{FIG_DIR}/cph_howdy_neighbor_sample.pdf', bbox_inches = 'tight')


fig, ax = plt.subplots(figsize = (10, 6), tight_layout = True)

gdf_aarhus = gdf_clusters_dk[gdf_clusters_dk['kom']==751]


gdf_aarhus.plot(
    column = 'howdy_neighbor_scaled',
    ax = ax,
    norm = neighborhood_norm_dk,
    cmap = cm_plasma_custom, 
    legend = True,
    legend_kwds = {
        'label': 'Avg. diff-type move-in per HH | K = 40',
        'orientation': 'vertical',
        'shrink': .65,
        'anchor': (-.4, 1)
    },
    edgecolor = 'k'
)

ax.set_axis_off()
fig.savefig(f'{FIG_DIR}/aarhus_howdy_neighbor.pdf', bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (10, 6), tight_layout = True)

gdf_aarhus = gdf_clusters_dk[gdf_clusters_dk['kom']==751] 
gdf_aarhus_anti = gdf_aarhus[~gdf_aarhus['cluster_id'].isin(clusterz)]


gdf_aarhus[gdf_aarhus['cluster_id'].isin(clusterz)].plot(
    column = 'howdy_neighbor_scaled',
    ax = ax,
    norm = neighborhood_norm_dk,
    cmap = cm_plasma_custom, 
    legend = True,
    legend_kwds = {
        'label': 'Avg. diff-type move-in per HH | K = 40',
        'orientation': 'vertical',
        'shrink': .65,
        'anchor': (-.4, 1)
    },
    edgecolor = 'k'
)

gdf_aarhus_anti.plot(
    hatch = '//',
    color = 'grey',
    alpha = .4,
    ax = ax,
    edgecolor = 'k'
)

ax.set_axis_off()
fig.savefig(f'{FIG_DIR}/aarhus_howdy_neighbor_sample.pdf', bbox_inches = 'tight')


fig, ax = plt.subplots(figsize = (10, 6), tight_layout = True)

gdf_greve = gdf_clusters_dk[gdf_clusters_dk['kom']==253]

gdf_greve.plot(
    column = 'howdy_neighbor_scaled',
    ax = ax,
    norm = neighborhood_norm_dk,
    cmap = cm_plasma_custom, 
    legend = True,
    legend_kwds = {
        'label': 'Avg. diff-type move-in per HH | K = 40',
        'orientation': 'vertical',
        'shrink': .65,
        'anchor': (-.5, 1)
    },
    edgecolor = 'k'
)

ax.set_axis_off()
fig.savefig(f'{FIG_DIR}/greve_howdy_neighbor.pdf', bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (10, 6), tight_layout = True)

gdf_greve_anti = gdf_greve[~gdf_greve['cluster_id'].isin(clusterz)]
gdf_greve[gdf_greve['cluster_id'].isin(clusterz)].plot(
    column = 'howdy_neighbor_scaled',
    ax = ax,
    cmap = cm_plasma_custom, 
    norm = neighborhood_norm_dk,
    legend = True,
    legend_kwds = {
        'label': 'Avg. diff-type move-in per HH | K = 40',
        'orientation': 'vertical',
        'shrink': .65,
        'anchor': (-.5, 1)
    },
    edgecolor = 'k'
)

gdf_greve_anti.plot(
    hatch = '//',
    color = 'grey',
    alpha = .4,
    ax = ax,
    edgecolor = 'k'
)

ax.set_axis_off()
fig.savefig(f'{FIG_DIR}/greve_howdy_neighbor_sample.pdf', bbox_inches = 'tight')