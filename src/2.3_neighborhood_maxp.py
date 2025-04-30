import os 
os.environ["POLARS_MAX_THREADS"] = '8'
os.environ["POLARS_FORCE_NEW_STREAMING"] = "1"
os.environ["OMP_NUM_THREADS"] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import sys
sys.path.append('../src')

from spopt.region import MaxPHeuristic
import libpysal
import polars as pl
pl.enable_string_cache()
from dst import utils, geo
import geopandas as gpd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figs')

# Map local neighborhoods to address_map_ids
df_hh = pl.scan_parquet(f'{DATA_DIR}/build/geo_hh.pq').group_by("address_map_id").agg(
    pl.col("etrs89_east").first(),
    pl.col("etrs89_north").first()
).collect()

gdf_hh = gpd.GeoDataFrame(df_hh.to_pandas(), geometry=gpd.points_from_xy(df_hh['etrs89_east'], df_hh['etrs89_north']), crs='EPSG:25832')

# read clusters from nabolagsatlas.dk
gdf_clusters = gpd.read_parquet(f'{DATA_DIR}/raw/clusters.pq')

gdf_hh = gdf_hh.sjoin(gdf_clusters, how='left', predicate='within')

# our sequence dataset with hh_id
lf_seq = pl.scan_parquet(f'{DATA_DIR}/build/geo_hh.pq')
df_hh_cluster = pl.from_pandas(gdf_hh.drop(['geometry', 'munic_code', 'id_munic'], axis=1)).select(pl.all().exclude("index_right"))

lf_hh = df_hh_cluster.lazy()
df_clusters = df_hh_cluster.select(pl.col("cluster_id").unique())

# Determine the population density within these clusters (I require at least 500)
years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]

for year in years:
    lf_seq_adj = lf_seq.filter(pl.date(year, 12, 31).is_between(pl.col("bop_vfra"), pl.col("bop_vtil")))
    
    pop_count = lf_hh.join(lf_seq_adj, on='address_map_id').group_by('cluster_id').agg(
    pl.col("person_id").n_unique().alias(f'cluster_id_count_{year}')
    ).collect()

    df_clusters = df_clusters.join(pop_count, on='cluster_id')

gdf_clusters = gdf_clusters.merge(df_clusters.to_pandas(), how='left', on='cluster_id')

# Use MaxP-algo to delinearize fixed neighborhoods
message = f'Spatially optimize neighborhood cells such that they contain around x amount people'
time_ = utils.what_time_is_it()
utils.log(message, time_)
print(message)

list_of_cols = [f'cluster_id_count_{year}' for year in years]
threshold = 500
w = libpysal.weights.Queen.from_dataframe(gdf_clusters)
gdf_clusters['min_ids'] = gdf_clusters[list_of_cols].min(axis=1)
gdf_clusters = gdf_clusters.drop(w.islands)
w = libpysal.weights.Queen.from_dataframe(gdf_clusters)
threshold_name = "min_ids"
model = MaxPHeuristic(gdf_clusters, w, [], threshold_name, threshold, top_n=8, max_iterations_construction=500)
model.solve()
message = f'Model solved. {model.p} neighborhoods created for min {threshold} people (single method).'
time_ = utils.what_time_is_it()
utils.log(message, time_)
print(message)

gdf_clusters[f'cluster_id_{threshold}'] = model.labels_
gdf_clusters.to_parquet(f'{DATA_DIR}/build/geo_local_neighborhood{threshold}.pq')

gdf_clustersadj1 = gdf_clusters[['geometry', f'cluster_id_{threshold}']].dissolve(by=f'cluster_id_{threshold}', as_index=False).reset_index(drop=True)
gdf_clustersadj1.to_parquet(f'{DATA_DIR}/raw/clusters{threshold}.pq')

gdf_hh = gdf_hh.drop('index_right', axis=1).sjoin(gdf_clustersadj1, how='left', predicate='within')

pl.from_pandas(gdf_hh[['address_map_id', 'etrs89_east', 'etrs89_north', 'cluster_id','cluster_id_500']]).with_columns(
    pl.col("cluster_id").cast(pl.Int16).alias("cluster_id"),
    pl.col("cluster_id_500").cast(pl.Int16).alias("cluster_id_500"),
).write_parquet(f'{DATA_DIR}/build/geo_neighborhood.pq')
