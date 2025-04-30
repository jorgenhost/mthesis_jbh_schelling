import os 
os.environ["POLARS_MAX_THREADS"] = '8'
os.environ["POLARS_FORCE_NEW_STREAMING"] = "1"
os.environ["OMP_NUM_THREADS"] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figs')

sys.path.append(PROJECT_ROOT)

import time
import glob
import pyarrow.parquet as pq
import polars as pl
pl.enable_string_cache()
from dst import utils, geo

# pl.Expr's I use when I compute my dataset as an undirected weighted graph
OVERLAP_EXPR = (pl.min_horizontal("bop_vtil", "bop_vtil_nn")-pl.max_horizontal("bop_vfra", "bop_vfra_nn")).dt.total_days() # You only need a single logical statement in order to determine overlap between two intervals :)

list_of_non_west_cnt = utils.fetch_country_cats(sub_cat = 'non-west')
NATIVE_SUM_EXPR = ((pl.col("person_id").is_first_distinct()) & (pl.col("opr_land") == "DNK")).sum().over("hh_id")
NON_NATIVE_SUM_EXPR = ((pl.col("person_id").is_first_distinct()) & (pl.col("opr_land") != "DNK")).sum().over("hh_id")
NON_WEST_SUM_EXPR = ((pl.col("person_id").is_first_distinct()) & (pl.col("opr_land").is_in(list_of_non_west_cnt))).sum().over("hh_id")

DURATION_EXPR = (pl.col("bop_vtil")-pl.col("bop_vfra")).dt.total_days()
HH_SUM = pl.col("person_id").n_unique().over("hh_id")
MIX_SHARE_EXPR = NON_NATIVE_SUM_EXPR / HH_SUM
MIX_NON_WEST_SHARE_EXPR = NON_WEST_SUM_EXPR / HH_SUM

kd_tree = geo.load_kd_tree()

k = 50
l_norm = 2

# Fetch neighborhood identifier
lf_n_hood = pl.scan_parquet(f'{DATA_DIR}/build/geo_neighborhood.pq').select(pl.col("address_map_id"), pl.col("cluster_id_500").alias("cluster_id"))

# Fetch residential data with individual chars + hh identifier
# Make hh vars: native, non-west etc hh
lf = (pl.scan_parquet(f'{DATA_DIR}/build/geo_hh.pq')
        .with_columns(
            native_hh = (pl.col("opr_land") == "DNK").all().over("hh_id"),
            mix_hh = (pl.col("opr_land") != "DNK").any().over("hh_id"),
            non_west_all_hh = (pl.col("non_west_roots") == True).all().over("hh_id"),
            menapt_all_hh = (pl.col("menapt_roots") == True).all().over("hh_id"),
            start_ = pl.col("bop_vfra").min().over("hh_id")
            )
        .with_columns(
            mix_share = pl.when(pl.col("mix_hh") == True)
            .then(MIX_SHARE_EXPR)
            .otherwise(None).alias("mix_share"),
            mix_non_west_share = pl.when(pl.col("mix_hh") == True)
            .then(MIX_NON_WEST_SHARE_EXPR)
            .otherwise(None).alias("mix_non_west_share"),
            first_non_west = pl.when(pl.col("bop_vfra")==pl.col("start_")).then(pl.col("non_west_roots")).otherwise(False),
            first_menapt = pl.when(pl.col("bop_vfra") == pl.col("start_")).then(pl.col("menapt_roots")).otherwise(False),
            hh_size = pl.col("person_id").n_unique().over("hh_id").alias('hh_size')).select(pl.all().exclude("start_"))
        .join(lf_n_hood, on='address_map_id', how='left')
)

# Group to hh level, preserving essential columns
lf_hh = (lf.group_by('hh_id')
        .agg(
            pl.col("address_map_id").first(),
            pl.col("kom").first(),
            pl.col("etrs89_east", "etrs89_north","etage", "sidedoer").first(),
            pl.col("cluster_id").first(),
            pl.col("person_id").unique(),
            pl.col("bop_vfra").min(),
            pl.col("bop_vtil").max(),
            pl.col("hh_size").first(),
            pl.col("menapt_all_hh").first(),
            pl.col("native_hh").first(),
            pl.col("mix_hh").first(),
            pl.col("mix_share", "mix_non_west_share").first(),
            pl.col("first_non_west", "first_menapt").first())
        .sort('hh_id')
        .select(pl.all().exclude("koen_hh", "person_id"))
)

df_hh = lf_hh.collect()

# "4D" coordinates of each address
# If etage / sidedoer is None, replace with 0.
# Feed it to the KDTree and query K-nearest
# Save (nested) dataset
coords = df_hh.select(pl.col("etrs89_east", "etrs89_north"), 
                    pl.col("etage").replace({None: 0}).cast(pl.Float64), 
                    pl.col("sidedoer").replace({None: 0}).cast(pl.Float64))

df_out = geo.query_knn(kd_tree, coords=coords, df = df_hh, k=k).sort('hh_id').select(pl.all().exclude("etrs89_east", "etrs89_north","etage", "sidedoer"))

df_out.write_parquet(f'{DATA_DIR}/build/knn{k}_prep.pq')

# time.sleep(10)

lf_knn = pl.scan_parquet(f'{DATA_DIR}/build/knn{k}_prep.pq').sort('hh_id')
lf_nn = lf_hh.select(pl.all().exclude("etrs89_east", "etrs89_north","etage", "sidedoer")).rename(lambda col_name: f'{col_name}_nn').collect().lazy()

# Now merge the whole thing from nested to long format
# This is a overlapping self-join, see pl.Expr above
# Do this in batches to not crash the whole thing

total_rows = lf_hh.select(pl.len()).collect().item()
batch_size = 4_000_000
batch_no = 0
start_row = batch_no * batch_size

for i in range(start_row, total_rows, batch_size):
        lf_batch = (lf_knn.slice(i, batch_size).select(pl.all())
                    .explode("query_dist", "query_adr_idx", "rank_nn")
                    .filter(pl.col("rank_nn")>1)
                    .with_columns(rank_nn = pl.col("rank_nn")-1)
                    .join(lf_nn,
                        left_on = "query_adr_idx",
                        right_on = "address_map_id_nn",
                        )
                    .filter(OVERLAP_EXPR > 0)
                    .with_columns(rank_nn = pl.col("rank_nn").rank("dense").over("hh_id"))
        )
        lf_batch.sort('hh_id', 'rank_nn').sink_parquet(f'{DATA_DIR}/build/knn/batches/knn{k}_{batch_no}.pq', engine = 'streaming')
        message = f'Success! KNN, batch_no {batch_no}'
        time_ = utils.what_time_is_it()
        utils.log(message, time_)
        print(message)
        batch_no += 1
        time.sleep(2)

# "Concat" the whole thing to a single dataset
# Use pyarrow for this; by far the most efficient way
# All you need to have is consistent data schema (which we do as per the batching method)
message = f'Part 2: Dump knn dataset to single .pq-file.'
time_ = utils.what_time_is_it()
utils.log(message, time_)
print(message)

filez = glob.glob(f'{DATA_DIR}/build/knn/batches/knn{k}_*.pq')
total_files = len(filez)
filez = [f'{DATA_DIR}/build/knn/batches/knn{k}_{i}.pq' for i in range(total_files)]
schema = pq.ParquetFile(filez[0]).schema_arrow

output_file = f'{DATA_DIR}/build/knn{k}.pq'

utils.pyarrow_pq_list(output_file, filez)

message = f'Dump to single .pq-file complete.'
time_ = utils.what_time_is_it()
utils.log(message, time_)
print(message)