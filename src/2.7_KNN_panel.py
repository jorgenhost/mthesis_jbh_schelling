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
MISC_DIR = os.path.join(PROJECT_ROOT, 'misc')

import polars as pl
import geopandas as gpd
from datetime import date
import glob
pl.enable_string_cache()
import time
from dst import utils, geo
from dst.classific import educ_utils
import pyarrow.parquet as pq

#####################################################
# Again collecting my pl.Expr's that I use later on #
#####################################################
# Does the node exist?
DATE_OVERLAP1 = (pl.min_horizontal("bop_vtil", "window_end")-pl.max_horizontal("bop_vfra", "window_start")).dt.total_days()
# Does the neighbor node exist?
DATE_OVERLAP2 = (pl.min_horizontal("bop_vtil_nn", "window_end")-pl.max_horizontal("bop_vfra_nn", "window_start")).dt.total_days()

# Based on above Booleans, I can go from an undirected weighted graph to a panel dataset
PANEL_OVERLAP_EXPR = (DATE_OVERLAP1 > 0) & (DATE_OVERLAP2 > 0)

OVERLAP_EXPR = (pl.min_horizontal("bop_vtil", "bop_vtil_nn")-pl.max_horizontal("bop_vfra", "bop_vfra_nn")).dt.total_days()

START_EXPR = pl.date_ranges(date(1984, 1, 1), date(2021, 12, 31), interval='1q').alias("window_start")
END_EXPR = START_EXPR.list.eval(pl.element().dt.offset_by('1q').dt.offset_by('-1d')).alias("window_end")

DURATION_EXPR = (pl.col("bop_vtil")-pl.col("bop_vfra")).dt.total_days()
T_EXPR = END_EXPR.list.eval(pl.element().rank().cast(pl.UInt8))
TIME_TRIM_EXPR = ((pl.col("window_start") >= pl.date(1985, 1, 1)) & (pl.col("window_end") <= pl.date(2020, 12, 31)))

MOVE_OUT_EXPR = pl.col("bop_vtil") <= pl.col("window_start").dt.offset_by('2y')

AGE_EXPR = pl.col("year")-pl.col("foed_dag").dt.year()
EQUI_EXPR =  (pl.when(AGE_EXPR.is_between(15, 999).is_first_distinct()).then(1).when(AGE_EXPR.is_between(15, 999)).then(0.5).when(AGE_EXPR.is_between(0, 14)).then(0.3)).over("hh_id", "year")

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


lf_knn = pl.scan_parquet(f'{DATA_DIR}/build/knn40.pq').set_sorted("hh_id")

lf_sample = pl.scan_parquet(f'{DATA_DIR}/build/knn40.pq').set_sorted('hh_id').filter(DURATION_EXPR >= 365).with_columns(
    treat_first_non_west = ((pl.col("first_non_west_nn")==True) & (pl.col("rank_nn").is_between(1,3)) & (pl.col("bop_vfra_nn")>pl.col("bop_vfra"))).any().over("hh_id"),
    control_first_non_west =  ((pl.col("first_non_west_nn")==True) & (pl.col("rank_nn").is_between(4,40)) & (pl.col("bop_vfra_nn")>pl.col("bop_vfra"))).any().over("hh_id")
).group_by('hh_id').agg(
    pl.col("native_hh").first(),
    (pl.col("mix_non_west_share")>0).first().alias("mix_non_west_hh"),
    pl.col("treat_first_non_west", "control_first_non_west").first()
).collect(new_streaming=True).lazy()
hh_id_sample = lf_sample.filter((pl.col("treat_first_non_west")==True) | (pl.col("control_first_non_west")==True)).filter(pl.col("native_hh")==True).select(pl.col("hh_id").unique()).collect(new_streaming = True).to_series()
hh_id_sample.to_frame().write_parquet(f'{DATA_DIR}/build/hh_sample_native.pq')

lf_knn = pl.scan_parquet(f'{DATA_DIR}/build/knn40.pq').set_sorted("hh_id")
hh_id_sample = lf_knn.select(pl.col("hh_id")).unique().collect(engine='streaming').to_series()

chunk_size = hh_id_sample.len() // 30

hh_id_list = []
for i in range(0, hh_id_sample.len(), chunk_size):
    chunkz = hh_id_sample.slice(i, chunk_size).set_sorted()
    hh_id_list.append(chunkz)

batch_no = 0
hh_id_list = hh_id_list[batch_no::]

for ids in hh_id_list:

    lf_knn_panel = lf_knn.filter(pl.col("hh_id").is_in(ids))
    lf_knn_panel = (lf_knn_panel.with_columns(
                        window_start = START_EXPR,
                        window_end = END_EXPR,
                        t = T_EXPR,
                        hh_type = HH_TYPE_EXPR,
                        hh_type_nn = HH_TYPE_EXPR_NN)
                        .filter(pl.col("hh_type")!=3)
                        .filter(pl.col("hh_type_nn")!=3)
                        .explode("window_start", "window_end", "t")
                        .filter(PANEL_OVERLAP_EXPR)
                        .with_columns(
                            N_diff_type_nearest = ((pl.col("rank_nn").is_between(1,3)) & (pl.col("hh_type")!=pl.col("hh_type_nn"))).sum().over("hh_id", "t"),
                            N_diff_type_near = ((pl.col("rank_nn").is_between(4, 6)) & (pl.col("hh_type")!=pl.col("hh_type_nn"))).sum().over("hh_id", "t"),
                            N_diff_type_far_7_10 = ((pl.col("rank_nn").is_between(7,10)) & (pl.col("hh_type")!=pl.col("hh_type_nn"))).sum().over("hh_id", "t"),                            
                            N_diff_type_far_11_20 =((pl.col("rank_nn").is_between(11,20)) & (pl.col("hh_type")!=pl.col("hh_type_nn"))).sum().over("hh_id", "t"),
                            N_diff_type_far_21_30 =((pl.col("rank_nn").is_between(21,30)) & (pl.col("hh_type")!=pl.col("hh_type_nn"))).sum().over("hh_id", "t"),
                            N_diff_type_far_31_40 = ((pl.col("rank_nn").is_between(31,40)) & (pl.col("hh_type")!=pl.col("hh_type_nn"))).sum().over("hh_id", "t"))
                        .filter(pl.struct(pl.col("hh_id", "t")).is_first_distinct())
                        .with_columns(
                                move = pl.when(MOVE_OUT_EXPR == True).then(100).otherwise(0),
                                year = pl.col("window_start").dt.year()
                        )
                        .select(
                            pl.col("hh_id", "t", "move", "cluster_id", "kom", 'year', "bop_vfra", "bop_vtil", "window_start", "window_end", "hh_type"),
                            pl.col("address_map_id"),
                            pl.col("^N_diff.*$")).sort('hh_id', 't')
    )
    lf_knn_panel.sink_parquet(f'{DATA_DIR}/build/knn/batches_panel/knn40_all_{batch_no}.pq', engine = 'streaming')
    message = f'Panel dataset, batch {batch_no} saved.'
    print(message)
    batch_no += 1

# Calc difference between time periods 
lf = pl.scan_parquet(f'{DATA_DIR}/build/knn/batches_panel/knn40_all_*.pq').sort('hh_id', 't').with_columns(
                            I_nearest = (pl.col("N_diff_type_nearest").diff()>=1).cast(pl.Int8),
                            I_near = (pl.col("N_diff_type_near").diff()>=1).cast(pl.Int8),
                            I_close_10 = (pl.col("N_diff_type_far_7_10").diff()>=1).cast(pl.Int8),
                            I_close_20 = (pl.col("N_diff_type_far_11_20").diff()>=1).cast(pl.Int8),
                            I_close_30 = (pl.col("N_diff_type_far_21_30").diff()>=1).cast(pl.Int8),
                            I_close_40 = (pl.col("N_diff_type_far_31_40").diff()>=1).cast(pl.Int8)).filter(TIME_TRIM_EXPR)

lf.sink_parquet(f'{DATA_DIR}/input/knn40_panel.pq', engine = 'streaming')

# Based on the graph dataset, who moved in and when?
# Save this (to later on condition on neighbor type)
lf_movez = (lf_knn
            .with_columns(
            hh_type = HH_TYPE_EXPR,
            hh_type_nn = HH_TYPE_EXPR_NN)
            .filter(pl.col("bop_vfra_nn")>pl.col("bop_vfra"))
            .select(pl.col("hh_id", "hh_type", "hh_type_nn", "cluster_id", "hh_id_nn", 
                           "cluster_id_nn", "first_non_west_nn", "query_dist", "rank_nn",
                             "bop_vfra_nn", "bop_vtil_nn"))
            .with_columns(
                        window_start = START_EXPR,
                        window_end = END_EXPR,
                        t = T_EXPR)
            .explode("window_start", "window_end", "t")
            .filter(pl.col("bop_vfra_nn").is_between("window_start", "window_end"))
            .select(pl.col("hh_id", "t"), pl.all().exclude("hh_id", "t", "window_start", "window_end")).sort('hh_id', 't')
)
lf_movez.sink_parquet(f'{DATA_DIR}/build/hh_move_ins.pq', engine = 'streaming')

###########################################
### Covariates from administrative data ###
###########################################

cpi = pl.read_excel(f'{MISC_DIR}/pris8.xlsx').with_columns(year = pl.col("year").cast(pl.Int16)).lazy()
lf_geo_hh = pl.scan_parquet(f'{DATA_DIR}/build/geo_hh.pq')
audd_cats = educ_utils.fetch_audd_cat()
audd_mapping, audd_order = educ_utils.map_audd_cats(audd_cats)
audd_mapping_contin = educ_utils.map_audd_cats_cont()

DATE_EXPR = pl.date_ranges(date(1985, 12, 31), date(2020, 12, 31), interval='1y').alias("date")

lf_bef = pl.scan_parquet(f'{DATA_DIR}/raw/bef.pq').filter(pl.struct(pl.col("person_id", "foed_dag")).is_first_distinct()).select(pl.col("person_id", "foed_dag"))

lf_ind = (pl.scan_parquet(f'{DATA_DIR}/raw/ind.pq')
            .with_columns(
              year = pl.col("aar"),
              person_inc = pl.col("perindkialt_13"),
              emp = pl.col("beskst13").cast(pl.Int8))
).join(cpi, on = 'year').with_columns(real_person_inc = (pl.col("person_inc")/pl.col("index")).cast(pl.Int64)).select(pl.col("person_id", "emp", "real_person_inc", "year"))

lf_uddf = (pl.scan_parquet(f'{DATA_DIR}/raw/uddf.pq').with_columns(DATE_EXPR).explode("date")
           .filter(pl.col("date").is_between("hf_vfra", "hf_vtil"))
           .with_columns(year = pl.col("date").dt.year()).with_columns(hfaudd_cat =  pl.col("hfaudd").replace_strict(audd_mapping, return_dtype=pl.Enum(audd_order), default='<9g'))
           .select(pl.col("person_id", "year", "hfaudd", "hfaudd_cat")).with_columns(hfaudd_cat_phys = pl.col("hfaudd_cat").to_physical())
).collect(engine = 'streaming').lazy()

edu_classification = lf_uddf.select(pl.col("hfaudd_cat_phys", "hfaudd_cat")).filter(pl.struct(pl.col("hfaudd_cat", "hfaudd_cat_phys")).is_first_distinct()).with_columns(hfaudd_cat_years = pl.col("hfaudd_cat").replace_strict(audd_mapping_contin, return_dtype = pl.Int8))

year_old = [year for year in range(1985, 1997)]
year_new = [year for year in range(1997, 2020+1)]

rez = []

for year in year_old:
    lf = pl.scan_parquet(f'{DATA_DIR}/raw/super_raw/ind_{year}.pq').select(pl.col("person_id").cast(pl.Int32), pl.col("form"), pl.col("aar").alias("year").cast(pl.Int16))
    rez.append(lf) 

for year in year_new:
    lf = pl.scan_parquet(f'{DATA_DIR}/raw/super_raw/ind_{year}.pq').select(pl.col("person_id").cast(pl.Int32), pl.col("formrest_ny05"), pl.col("aar").alias("year").cast(pl.Int16))
    rez.append(lf) 


lf_concat = pl.concat(rez, how = 'diagonal').with_columns(net_wealth = pl.when(pl.col("year")<1997).then(pl.col("form")).otherwise(pl.col("formrest_ny05"))).join(cpi, on = 'year').with_columns(real_net_wealth = (pl.col("net_wealth")/pl.col("index")).cast(pl.Int64)).select(pl.col("person_id", "year", "real_net_wealth"))

lf_net_wealth = lf_concat.collect(engine = 'streaming').lazy()


lf_hh = (pl.scan_parquet(f'{DATA_DIR}/build/geo_hh.pq').select(pl.col("person_id", "bop_vfra", "bop_vtil", "hh_id"))
            .with_columns(start = pl.col("bop_vfra").dt.year(),
                       end = pl.col("bop_vtil").dt.year()
                       )
            .with_columns(year = pl.int_ranges(pl.col("bop_vfra").dt.year(), pl.col("bop_vtil").dt.year(), dtype=pl.Int16))
            .join(lf_bef, on = ['person_id']).explode("year").filter(pl.col("year").is_between(1985, 2020))
            .with_columns(equi_factor = EQUI_EXPR)
)
lf_covars_person = (lf_hh.select(pl.col("person_id", "hh_id", "equi_factor", "year"))
            .join(lf_ind, on = ['person_id', 'year'], how = 'left')
            .join(lf_net_wealth, on = ['person_id', 'year'], how = 'left')
            .join(lf_uddf, on = ['person_id', 'year'], how = 'left')
            .join(lf_bef, on = 'person_id')
            .with_columns(age = pl.col("year")-pl.col("foed_dag").dt.year())
)

lf_covars_hh = (lf_covars_person
            .group_by('hh_id', 'year').agg(
                pl.col("equi_factor").sum().alias("hh_equi_factor"),
                pl.col("real_person_inc").sum().alias("real_hh_inc"),
                pl.col("real_net_wealth").sum().alias("real_hh_net_wealth"),
                pl.col("age").unique().alias("ages"),
                pl.col("emp").unique().alias("hh_emp"),
                pl.col("hfaudd_cat").to_physical().max().alias("hh_highest_edu_phys")
                )
            .with_columns(
                equi_hh_real_inc = (pl.col("real_hh_inc")/pl.col("hh_equi_factor")).cast(pl.Int32),
                equi_hh_real_net_wealth  = (pl.col("real_hh_net_wealth")/pl.col("hh_equi_factor")).cast(pl.Int64))
            .join(edu_classification, left_on = 'hh_highest_edu_phys', right_on = 'hfaudd_cat_phys', how = 'left', nulls_equal = True)
            .rename({'hfaudd_cat': 'hh_highest_educ'})
            .select(pl.all().exclude("hh_highest_edu_phys"))
)


lf_covars_hh.sink_parquet(f'{DATA_DIR}/build/hh_covars.pq', engine = 'streaming')
