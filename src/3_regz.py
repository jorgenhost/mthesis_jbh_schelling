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
MISC_DIR = os.path.join(PROJECT_ROOT, 'misc')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figs')
TABS_DIR = os.path.join(PROJECT_ROOT, 'tabs')

sys.path.append(PROJECT_ROOT)

import polars as pl
import polars.selectors as cs
import geopandas as gpd
from ryp import r, to_py, to_r
r('library("modelsummary")')
r('library("gt")')
r('library("car")')
r('library("fixest")')
r('library("broom")')

pl.enable_string_cache()
from dst import utils
from datetime import date


START_EXPR = pl.date_ranges(date(1984, 1, 1), date(2021, 12, 31), interval='1q').alias("window_start")
END_EXPR = START_EXPR.list.eval(pl.element().dt.offset_by('1q').dt.offset_by('-1d')).alias("window_end")

T_EXPR = END_EXPR.list.eval(pl.element().rank().cast(pl.UInt8))

# How long has the household stayed at their residence?
TENURE_EXPR = (pl.col("window_end")-pl.col("bop_vfra")).dt.total_days()


lf_covars = (pl.scan_parquet(f'{DATA_DIR}/build/hh_covars.pq').with_columns(hh_employed = pl.col("hh_employed").cast(pl.Int8))
                .select(pl.all().exclude("hh_equi_factor", "real_hh_inc", "ages"))
)


lf_hoods = pl.scan_parquet(f'{DATA_DIR}/build/geo_neighborhood.pq')
cluster_dict = lf_hoods.select(pl.col("cluster_id_500", "cluster_id_1000")).collect().to_dict()
cluster_dict = dict(zip(cluster_dict['cluster_id_500'], cluster_dict['cluster_id_1000']))

people_threshold = 500

lf_clusters = pl.scan_parquet(f'{DATA_DIR}/build/cluster_density_{people_threshold}.pq')
clusterz = lf_clusters.filter(pl.col("density_t").is_between(1_000, 25_000)).select(pl.col(f"cluster_id_{people_threshold}")).unique().collect().to_series()

lf_hh_size = pl.scan_parquet(f'{DATA_DIR}/build/hh_size_qt.pq')

################
### MOVE-INS ###
################

# Natives & non-Western neighbors
lf_movez_dk = pl.scan_parquet(f'{DATA_DIR}/build/hh_move_ins.pq').filter(pl.col("hh_type")==1).filter(pl.col("hh_type_nn")==2).group_by('hh_id', 't').agg(
    pl.struct(pl.col("hh_id_nn", "rank_nn", "query_dist")).unique(maintain_order = True).alias('movez')
).with_columns(
    nearest_movez = pl.col("movez").list.eval(pl.element().filter(pl.element().struct.field("rank_nn").is_between(1,3))),
    close_movez = pl.col("movez").list.eval(pl.element().filter(pl.element().struct.field("rank_nn")>=4))
)


df_movez_native_households = lf_movez_dk.with_columns(
    nearest_movez_dist = pl.col("nearest_movez").list.eval(pl.element().struct.field("query_dist").mean()),
    close_movez_dist = pl.col("close_movez").list.eval(pl.element().struct.field("query_dist").mean())
).select(pl.col("hh_id", "t", "nearest_movez_dist", "close_movez_dist")).explode(cs.ends_with("dist")).collect(engine = 'streaming')


# Non-Western hh's & native neighbors
lf_movez_non_west = pl.scan_parquet(f'{DATA_DIR}/build/hh_move_ins.pq').filter(pl.col("hh_type")==2).filter(pl.col("hh_type_nn")==1).group_by('hh_id', 't').agg(
    pl.struct(pl.col("hh_id_nn", "rank_nn", "query_dist")).unique(maintain_order = True).alias('movez')
).with_columns(
    nearest_movez = pl.col("movez").list.eval(pl.element().filter(pl.element().struct.field("rank_nn").is_between(1,3))),
    close_movez = pl.col("movez").list.eval(pl.element().filter(pl.element().struct.field("rank_nn")>=4))
)

df_movez_non_west = lf_movez_non_west.with_columns(
    nearest_movez_dist = pl.col("nearest_movez").list.eval(pl.element().struct.field("query_dist").mean()),
    close_movez_dist = pl.col("close_movez").list.eval(pl.element().struct.field("query_dist").mean())
).select(pl.col("hh_id", "t", "nearest_movez_dist", "close_movez_dist")).explode(cs.ends_with("dist")).collect(engine = 'streaming')

# Same-type neighbors - [test of identifying assumpetions can be included here], native-native pairs
lf_movez_dk_dk = pl.scan_parquet(f'{DATA_DIR}/build/hh_move_ins.pq').filter(pl.col("hh_type")==1).filter(pl.col("hh_type_nn")==1).group_by('hh_id', 't').agg(
    pl.struct(pl.col("hh_id_nn", "rank_nn", "query_dist")).unique(maintain_order = True).alias('movez')
).with_columns(
    nearest_movez = pl.col("movez").list.eval(pl.element().filter(pl.element().struct.field("rank_nn").is_between(1,3))),
    close_movez = pl.col("movez").list.eval(pl.element().filter(pl.element().struct.field("rank_nn")>=4))
)

df_movez_dk_dk = lf_movez_dk_dk.with_columns(
    nearest_movez_dist = pl.col("nearest_movez").list.eval(pl.element().struct.field("query_dist").mean()),
    close_movez_dist = pl.col("close_movez").list.eval(pl.element().struct.field("query_dist").mean())
).select(pl.col("hh_id", "t", "nearest_movez_dist", "close_movez_dist")).explode(cs.ends_with("dist")).collect(engine = 'streaming')


# Different type neighbors concatened (to merge on main panel dataset)
all_movez = pl.concat([df_movez_native_households, df_movez_non_west])

# Same type neighbors concatenated (to merge on main panel dataset), natives
all_movez_same_type_native = pl.concat([df_movez_native_households, df_movez_dk_dk])

#######################################
## MERGE COVARIATES TO PANEL DATASET ##
#######################################

lf_panel = pl.scan_parquet(f'{DATA_DIR}/input/knn40_all_panel.pq')

lf_distances = (pl.scan_parquet(f'{DATA_DIR}/build/knn40.pq').select(pl.col("hh_id", "query_dist", "rank_nn"))
                .set_sorted("hh_id").group_by('hh_id').agg(
                    nearest_25 = pl.when(pl.col("rank_nn").is_between(1,3))
                    .then(pl.col("query_dist")<=25)
                    .when(pl.col("rank_nn").is_between(4, 20))
                    .then(pl.col("query_dist")<=100)
                    .when(pl.col("rank_nn")>20).then(pl.col("query_dist")<=400).otherwise(False).all(),
                    nearest_100 = pl.when(pl.col("rank_nn").is_between(1,3))
                    .then(pl.col("query_dist")<=100)
                    .when(pl.col("rank_nn")>=4).then(pl.col("query_dist")<=1_000).otherwise(False).all(),
).collect(engine = 'streaming').lazy()
)

hh_25 = lf_distances.filter(pl.col("nearest_25")==True).select(pl.col("hh_id").unique()).collect().to_series()
hh_100 = lf_distances.filter(pl.col("nearest_100")==True).select(pl.col("hh_id").unique()).collect().to_series()


# Add filters/constraints to panel dataset specified in the paper
lf_panel_unrestrict = (lf_panel
        .filter(pl.col("cluster_id").is_in(clusterz))
        .join(lf_clusters, left_on = ['cluster_id', 't'], right_on=['cluster_id_500', 't'])
        .join(lf_covars, on = ['hh_id', 'year'], how = 'left')
        .filter(pl.col("hh_oldest").is_between(30, 60))
        .filter(pl.col("equi_hh_real_inc").is_between(200_000, 1_000_000))
        .filter(pl.col("equi_hh_real_net_wealth").is_between(-200_000, 750_000))                                
        .join(lf_hh_size, on = ['hh_id', 't'], how = 'left') # we know hh size at the quarterly level, so why not merge that
        .join(all_movez.lazy(), on = ['hh_id', 't'], how = 'left')
        .join(all_movez_same_type_native.lazy(), on = ['hh_id', 't'], how = 'left', suffix = '_same')
        .with_columns(
            inc = (pl.when(pl.col("equi_hh_real_inc").is_between(200_000, 400_000)).then(1)
                    .when(pl.col("equi_hh_real_inc").is_between(400_001, 600_000)).then(2)
                    .when(pl.col("equi_hh_real_inc").is_between(600_001, 800_000)).then(3)
                    .when(pl.col("equi_hh_real_inc").is_between(800_001, 1_000_000)).then(4)
            ),
            wealth = (pl.when(pl.col("equi_hh_real_net_wealth").is_between(-200_000, 0)).then(1)
                      .when(pl.col("equi_hh_real_net_wealth").is_between(1, 200_000)).then(2)
                      .when(pl.col("equi_hh_real_net_wealth").is_between(200_001, 400_000)).then(3)
                      .when(pl.col("equi_hh_real_net_wealth").is_between(400_001, 600_000)).then(4)
                      .when(pl.col("equi_hh_real_net_wealth").is_between(600_001, 750_000)).then(5)
            ),
            age = (pl.when(pl.col("hh_oldest").is_between(30, 40)).then(1)
                    .when(pl.col("hh_oldest").is_between(41, 50)).then(2)
                    .when(pl.col("hh_oldest").is_between(51, 60)).then(3)
                ),
            tenure = (pl.when(TENURE_EXPR.is_between(0, 365, closed = 'left')).then(1)
                    .when(TENURE_EXPR.is_between(365, 365*2, closed = 'left')).then(2)
                    .when(TENURE_EXPR.is_between(365*2, 365*4, closed = 'left')).then(3)
                    .when(TENURE_EXPR.is_between(365*4, 365*6, closed = 'left')).then(4)
                    .when(TENURE_EXPR >= 365*6).then(5)
            ),
            equi_hh_real_inc_1000 = pl.col("equi_hh_real_inc") / 1_000,
            equi_hh_real_net_wealth_1000 = pl.col("equi_hh_real_net_wealth") / 1_000,
            tenure_total = TENURE_EXPR,
            hh_highest_educ_cat = pl.col("hh_highest_educ").to_physical()
        ).with_columns(
            splitter = pl.when(pl.col("I_nearest")==1)
            .then(pl.lit("Nearest"))
            .when(pl.max_horizontal(pl.col("I_near"), cs.starts_with('I_close'))==1)
            .then(pl.lit("Close")).cast(pl.Categorical),
            equi_hh_real_inc_1000_median_cluster_t = pl.col("equi_hh_real_inc_1000").median().over("cluster_id", "t"),
            equi_hh_real_net_wealth_1000_median_cluster_t = pl.col("equi_hh_real_net_wealth_1000").median().over("cluster_id", "t"),
            dist_to_neighbor = pl.when(pl.col("I_nearest")==1)
            .then(pl.col("nearest_movez_dist"))
            .when(pl.max_horizontal(pl.col("I_near"), cs.starts_with("I_close"))==1)
            .then(pl.col("close_movez_dist"))
            .otherwise(None)
)
      .sort('hh_id', 't')
)

# Collect results
df_panel_unrestrict = lf_panel_unrestrict.collect(engine = 'streaming')

# Split between native and non-Western
df_panel_native = df_panel_unrestrict.filter(pl.col("hh_id").is_in(hh_25)).filter(pl.col("hh_type")==1)   
df_panel_non_west = df_panel_unrestrict.filter(pl.col("hh_id").is_in(hh_25)).filter(pl.col("hh_type")==2)        

df_panel_reg_native = df_panel_native.filter(pl.max_horizontal('^I_.*$')==1).drop_nulls((pl.col("inc", "age", "tenure", "hh_employed", "hh_highest_educ", "hh_size"))) 
df_panel_reg_native_w_same_type = df_panel_native.with_columns(I_nearest_same = pl.when(pl.col("nearest_movez_dist_same").is_not_null()).then(1).otherwise(0)).filter(pl.max_horizontal('^I_.*$')==1).drop_nulls((pl.col("inc", "age", "tenure", "hh_employed", "hh_highest_educ", "hh_size"))) 

df_panel_reg_non_west = df_panel_non_west.filter(pl.max_horizontal('^I_.*$')==1).drop_nulls((pl.col("inc", "age", "tenure", "hh_employed", "hh_highest_educ", "hh_size"))) 

df_panel_reg_100_native = (df_panel_unrestrict
                           .filter(pl.max_horizontal('^I_.*$')==1)
                           .filter(pl.col("hh_id").is_in(hh_100))
                           .filter(pl.col("hh_type")==1).with_columns(
                            I_control = (pl.max_horizontal('I_near', 'I_close_10', 'I_close_20', 'I_close_30', 'I_close_40')),
                            I_nearest_0 = pl.when(pl.col("I_nearest")==1)
                            .then(pl.col("nearest_movez_dist").is_between(0, 12.5))
                            .otherwise(0),
                            I_nearest_25 = pl.when(pl.col("I_nearest")==1)
                            .then(pl.col("nearest_movez_dist").is_between(12.5, 25))
                            .otherwise(0),
                            I_nearest_100 = pl.when(pl.col("I_nearest")==1)
                            .then(pl.col("nearest_movez_dist").is_between(25, 100))
                            .otherwise(0)
                        )
)


df_panel_reg_100_non_west = (df_panel_unrestrict
                           .filter(pl.max_horizontal('^I_.*$')==1)
                           .filter(pl.col("hh_id").is_in(hh_100))
                           .filter(pl.col("hh_type")==2)
                           .with_columns(
                            I_control = (pl.max_horizontal('I_near', 'I_close_10', 'I_close_20', 'I_close_30', 'I_close_40')),
                            I_nearest_0 = pl.when(pl.col("I_nearest")==1)
                            .then(pl.col("nearest_movez_dist").is_between(0, 12.5))
                            .otherwise(0),
                            I_nearest_25 = pl.when(pl.col("I_nearest")==1)
                            .then(pl.col("nearest_movez_dist").is_between(12.5, 25))
                            .otherwise(0),
                            I_nearest_100 = pl.when(pl.col("I_nearest")==1)
                            .then(pl.col("nearest_movez_dist").is_between(25, 100))
                            .otherwise(0)
                        )
)



to_r(df_panel_native, 'df_panel_native')
to_r(df_panel_reg_native_w_same_type, 'df_panel_reg_native_w_same_type')
to_r(df_panel_non_west, 'df_panel_non_west')
to_r(df_panel_reg_non_west, 'df_panel_reg_non_west')
to_r(df_panel_reg_native, 'df_panel_reg_native')
to_r(df_panel_reg_100_native, 'df_panel_reg_100_native')
to_r(df_panel_reg_100_non_west, 'df_panel_reg_100_non_west')

#################
## MAKE TABLES ##
#################

## SUMMARY TABLES ##

r(f'''
  sum_tab_native = datasummary(move + equi_hh_real_inc_1000 + equi_hh_real_net_wealth_1000 + hh_employed 
  + hh_highest_educ + dist_to_neighbor  + hh_size + hh_oldest + density_t + native_share_t + non_west_share_t 
  + equi_hh_real_inc_1000_median_cluster_t + equi_hh_real_net_wealth_1000_median_cluster_t 
  ~ (mean*Arguments(na.rm=TRUE) + sd*Arguments(na.rm=TRUE)) + splitter*(mean*Arguments(na.rm=TRUE) + sd*Arguments(na.rm=TRUE)), 
  data = df_panel_native,
  output = 'data.frame'
  )

  sum_tab_non_west = datasummary(move + equi_hh_real_inc_1000 + equi_hh_real_net_wealth_1000 + hh_employed 
  + hh_highest_educ + dist_to_neighbor  + hh_size + hh_oldest + density_t + native_share_t + non_west_share_t 
  + equi_hh_real_inc_1000_median_cluster_t + equi_hh_real_net_wealth_1000_median_cluster_t 
  ~ (mean*Arguments(na.rm=TRUE) + sd*Arguments(na.rm=TRUE)) + splitter*(mean*Arguments(na.rm=TRUE) + sd*Arguments(na.rm=TRUE)), 
  data = df_panel_non_west,
  output = 'data.frame'
  )
  
  ''')

sum_tab_native = (to_py('sum_tab_native', index = False)
      .select(" ", pl.col("mean").alias("All / mean"), pl.col("sd").alias("All / sd"), "Nearest / mean", "Nearest / sd", "Close / mean", "Close / sd")
      .with_columns(pl.when(cs.ends_with('sd') != "").then(cs.ends_with('sd').map_elements(lambda x: f'({x})', return_dtype=pl.String)),
                    pl.col(" ").replace_strict({
          'move': 'Move within 2 years',
          'equi_hh_real_inc_1000': 'Real inc. (1000s) DKK',
          'equi_hh_real_net_wealth_1000': 'Real net wealth (1000s) DKK',
          'hh_employed': 'Employed',
          'hh_highest_educ': 'Years of education',
          'dist_to_neighbor': 'Distance to neighbor',
          'hh_size': 'Household size',
          'hh_oldest': 'Oldest household member',
          'density_t': 'Population density',
          'native_share_t': 'Native share',
          'non_west_share_t': 'Non-Western share',
          'equi_hh_real_inc_1000_median_cluster_t': 'Real income (median)',
          'equi_hh_real_net_wealth_1000_median_cluster_t': 'Real net wealth (median)',
          'Observations': 'N'
      }, default = "", return_dtype=pl.String))
)

val_count = df_panel_native.select(pl.col("splitter").value_counts())

all_count = df_panel_native.select(pl.len()).item()
nearest_count = val_count.unnest("splitter").filter(pl.col("splitter")=="Nearest").select(pl.col("count")).item()
close_count = val_count.unnest("splitter").filter(pl.col("splitter")=="Close").select(pl.col("count")).item()
last_row = pl.DataFrame({' ': 'N',
                         'All / mean': f'{all_count:,.0f}',
                         'All / sd': "",
                         'Nearest / mean': f'{nearest_count:,.0f}',
                         'Nearest / sd': "",
                         'Close / mean': f'{close_count:,.0f}',
                         'Close / sd': ""})
sum_tab_native = pl.concat([sum_tab_native,last_row])

sum_tab_non_west = (to_py('sum_tab_non_west', index = False)
      .select(" ", pl.col("mean").alias("All / mean"), pl.col("sd").alias("All / sd"), "Nearest / mean", "Nearest / sd", "Close / mean", "Close / sd")
      .with_columns(pl.when(cs.ends_with('sd') != "").then(cs.ends_with('sd').map_elements(lambda x: f'({x})', return_dtype=pl.String)),
                    pl.col(" ").replace_strict({
          'move': 'Move within 2 years',
          'equi_hh_real_inc_1000': 'Real inc. (1000s) DKK',
          'equi_hh_real_net_wealth_1000': 'Real net wealth (1000s) DKK',
          'hh_employed': 'Employed',
          'hh_highest_educ': 'Years of education',
          'dist_to_neighbor': 'Distance to neighbor',
          'hh_size': 'Household size',
          'hh_oldest': 'Oldest household member',
          'density_t': 'Population density',
          'native_share_t': 'Native share',
          'non_west_share_t': 'Non-Western share',
          'equi_hh_real_inc_1000_median_cluster_t': 'Real income (median)',
          'equi_hh_real_net_wealth_1000_median_cluster_t': 'Real net wealth (median)',
          'Observations': 'N'
      }, default = "", return_dtype=pl.String))
)

val_count = df_panel_non_west.select(pl.col("splitter").value_counts())

all_count = df_panel_non_west.select(pl.len()).item()
nearest_count = val_count.unnest("splitter").filter(pl.col("splitter")=="Nearest").select(pl.col("count")).item()
close_count = val_count.unnest("splitter").filter(pl.col("splitter")=="Close").select(pl.col("count")).item()
last_row = pl.DataFrame({' ': 'N',
                         'All / mean': f'{all_count:,.0f}',
                         'All / sd': "",
                         'Nearest / mean': f'{nearest_count:,.0f}',
                         'Nearest / sd': "",
                         'Close / mean': f'{close_count:,.0f}',
                         'Close / sd': ""})
sum_tab_non_west = pl.concat([sum_tab_non_west,last_row])

sum_tab_merged = sum_tab_native.join(sum_tab_non_west, on = ' ', suffix = 'nw')

## descriptives
to_r(sum_tab_native, 'sum_tab_native')
to_r(sum_tab_non_west, 'sum_tab_non_west')
to_r(sum_tab_merged, 'sum_tab_merged')

tab_name_desc_native = f'{TABS_DIR}/descriptives_native.tex'
tab_name_desc_native_R = f'"{tab_name_desc_native}"'

tab_name_desc_non_west = f'{TABS_DIR}/descriptives_non_west.tex'
tab_name_desc_non_west_R = f'"{tab_name_desc_non_west}"'

tab_name_desc_merged = f'{TABS_DIR}/descriptives_merged.tex'
tab_name_desc_merged_R = f'"{tab_name_desc_merged}"'

r(f'''

sum_tab_native_gt = gt(sum_tab_native)
  
sum_tab_native_gt %>%
    cols_label(.list = setNames(rep("", ncol(sum_tab_native)), names(sum_tab_native))) %>%
    tab_row_group(label = "Neighborhood characteristics", rows = 9:13) %>%
    tab_row_group(label = "Household characteristics", rows = 1:8) %>%
    tab_spanner(label = "All", columns = 2:3) %>%
    tab_spanner(label = "Nearest", columns = 4:5) %>%
    tab_spanner(label = "Close", columns = 6:7) %>%
    gtsave(filename = {tab_name_desc_native_R})

sum_tab_non_west_gt = gt(sum_tab_non_west)

sum_tab_non_west_gt %>%
    cols_label(.list = setNames(rep("", ncol(sum_tab_non_west)), names(sum_tab_non_west))) %>%
    tab_row_group(label = "Neighborhood characteristics", rows = 9:13) %>%
    tab_row_group(label = "Household characteristics", rows = 1:8) %>%
    tab_spanner(label = "All", columns = 2:3) %>%
    tab_spanner(label = "Nearest", columns = 4:5) %>%
    tab_spanner(label = "Close", columns = 6:7) %>%    
    gtsave(filename = {tab_name_desc_non_west_R})

sum_tab_merged_gt = gt(sum_tab_merged)
  
sum_tab_merged_gt %>%
    cols_label(.list = setNames(rep("", ncol(sum_tab_merged)), names(sum_tab_merged))) %>%
    tab_row_group(label = "Neighborhood characteristics", rows = 9:13) %>%
    tab_row_group(label = "Household characteristics", rows = 1:8) %>%
    tab_spanner(label = "All1", columns = 2:3) %>%
    tab_spanner(label = "Nearest1", columns = 4:5) %>%
    tab_spanner(label = "Close1", columns = 6:7) %>%
    tab_spanner(label = "All", columns = 8:9) %>%
    tab_spanner(label = "Nearest", columns = 10:11) %>%
    tab_spanner(label = "Close", columns = 12:13) %>%
    tab_spanner(label = "Native households", columns = 2:7) %>%
    tab_spanner(label = "Non-Western households", columns = 8:13) %>%
    gtsave(filename = {tab_name_desc_merged_R})

''')

utils.remove_tab_env(f'{tab_name_desc_native}')
utils.remove_tab_env(f'{tab_name_desc_non_west}')
utils.remove_tab_env(f'{tab_name_desc_merged}')


# model regressions (native)
r('''
  mod1 = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)
  mod2 = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(inc, ref = "1") | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)
  mod3 = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(inc, ref = "1") + i(tenure, ref = "1") | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)
  mod4 = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)
  mod5 = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(wealth, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)
  mod6 = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(wealth, ref = "1") + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)
  ''')

# model regressions (non-west)

r('''
  mod1_non_west = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)
  mod2_non_west = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(inc, ref = "1") | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)
  mod3_non_west = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(inc, ref = "1") + i(tenure, ref = "1") | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)
  mod4_non_west = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)
  mod5_non_west = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(wealth, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)
  mod6_non_west = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(wealth, ref = "1") + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)
    ''')

# Model regressions (100m)
r('''
  mod4_100 = feols(move ~ I_nearest_0 + I_nearest_25 + I_nearest_100 + I_control + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_reg_100_native, vcov = ~cluster_id)
  mod4_100_non_west = feols(move ~ I_nearest_0 + I_nearest_25 + I_nearest_100 + I_control + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_reg_100_non_west, vcov = ~cluster_id)
''')

# model balance regressions

r('''
  mod1_balance = feols(equi_hh_real_inc_1000 ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)
  mod2_balance = feols(equi_hh_real_net_wealth_1000 ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)
  mod3_balance = feols(hh_oldest ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)
  mod4_balance = feols(tenure_total ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)
  mod5_balance = feols(hh_employed ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)  
  mod6_balance = feols(hh_highest_educ ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)
  mod7_balance = feols(hh_size ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_native, vcov = ~cluster_id)
  ''')

# model_non_west balance regressions

r('''
  mod1_balance_non_west = feols(equi_hh_real_inc_1000 ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)
  mod2_balance_non_west = feols(equi_hh_real_net_wealth_1000 ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)
  mod3_balance_non_west = feols(hh_oldest ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)
  mod4_balance_non_west = feols(tenure_total ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)
  mod5_balance_non_west = feols(hh_employed ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)  
  mod6_balance_non_west = feols(hh_highest_educ ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)
  mod7_balance_non_west = feols(hh_size ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 | cluster_id^t, data = df_panel_reg_non_west, vcov = ~cluster_id)
  ''')

# difference in coefs function, fetches std errs 
r('''
  get_coef_diff = function(model, coef_1_name, coef_2_name, note) {
    coef_1 = coef(model)[coef_1_name]
    coef_2 = coef(model)[coef_2_name]

    estimate = coef_1 - coef_2

    test = linearHypothesis(model, paste0(coef_1_name, " - ", coef_2_name, " = 0"))
    
    se = sqrt(vcov(model)[coef_1_name, coef_1_name] + vcov(model)[coef_2_name, coef_2_name] - 2 * vcov(model)[coef_1_name, coef_2_name])

    p_value = test$Pr[2]

    statistic = estimate/se

    result = data.frame(
      term = note,
      estimate = estimate,
      std.error = se,
      statistic = statistic, 
      p.value = p_value
    )

    return(result)
  }
  glance_custom.fixest = function(x, ...) {
    y = insight::get_response(x)
    y_mean = sprintf("%.2f", mean(y, na.rm = TRUE))

    model_data = insight::get_data(x)
    clusterz = "cluster_id"
    num_hoods = length(unique(model_data[[clusterz]]))

    data.table::data.table(`Mean of dependent variable` = y_mean,
                          `Number of neighborhoods` = num_hoods)
  }
''')

# Run difference in coefs test
# Main results

diff_control = 'I_near'

diffs = [('I_nearest', diff_control),
         ('I_near', diff_control),
         ('I_close_10', diff_control),
         ('I_close_20', diff_control),
         ('I_close_30', diff_control),
         ('I_close_40', diff_control)]

modz = {}

mods = ['mod1', 'mod2', 'mod3', 'mod4', 'mod5', 'mod6'
        ]
modelz = []

for mod in mods:
    diffz = []
    for coef_1, coef_2 in diffs:
        if coef_1 == coef_2:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ' ,')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ' ,')
            diff = pl.DataFrame({'term': fr'New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$ (omit.)',
                          'estimate': 0.0,
                          'std.error': 0.0,
                          'statistic': None,
                          'p.value': None})
            diffz.append(diff)
        else:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ',')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ',')
            r(f'diff = get_coef_diff({mod}, "{coef_1}", "{coef_2}", "New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$")')
            diff = to_py('diff', index = False)
            diffz.append(diff)
        
    start = pl.concat(diffz)
    model = (to_py(f'tidy({mod})', index = False).with_columns(
        term = pl.col("term").replace({ 
            'inc::1': 'Income 200,000 DKK - 400,000 DKK',
            'inc::2': 'Income 400,001 DKK - 600,000 DKK',
            'inc::3': 'Income 600,001 DKK - 800,000 DKK',
            'inc::4': 'Income 800,001 DKK - 1,000,000 DKK',
            'inc::5': 'Income >1,000,001 DKK',
            'age::2': 'Oldest in HH, age 41-50',
            'age::3': 'Oldest in HH, age 51-60',
            'tenure::2': r'Tenure $[1-2[$ years',
            'tenure::3': r'Tenure $[2-4[$ years',
            'tenure::4': r'Tenure $[4-6[$ years',
            'tenure::5': r'Tenure $\geq 6$ years',
            'wealth::2': 'Wealth 1 - 200,000 DKK',
            'wealth::3': 'Wealth 200,001 - 400,000 DKK',
            'wealth::4': 'Wealth 400,001 - 600,000 DKK',
            'wealth::5': 'Wealth 600,001 - 750,000 DKK'
            }))
    )
    model_raw=model.with_columns(
        term = pl.col("term").replace({
            'I_nearest': '$k_{nearest}$',
            'I_near': '$k_{near}$',
            'I_close_10': '$k_{close, 10}$',
            'I_close_20': '$k_{close, 20}$',
            'I_close_30': '$k_{close, 30}$',
            'I_close_40': '$k_{close, 40}$'})
    )

    model_diff = model.filter(pl.col("term").str.starts_with('I_').not_())

    model_out = pl.concat([start, model_diff])
    to_r(model_out, f'{mod}_adj_tidy', format = 'data.frame')
    to_r(model_raw, f'{mod}_adj_tidy_raw', format = 'data.frame')
    modelz.append(model_out)
    modz[f'{mod}'] = diffz


# Run difference in coefs test (non_west)
# Main results

diff_control = 'I_near'

diffs = [('I_nearest', diff_control),
         ('I_near', diff_control),
         ('I_close_10', diff_control),
         ('I_close_20', diff_control),
         ('I_close_30', diff_control),
         ('I_close_40', diff_control)]

modz_non_west = {}

mods_non_west = ['mod1_non_west', 'mod2_non_west', 'mod3_non_west', 'mod4_non_west', 'mod5_non_west', 'mod6_non_west'
        ]
modelz_non_west = []

for mod_non_west in mods_non_west:
    diffz = []
    for coef_1, coef_2 in diffs:
        if coef_1 == coef_2:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ' ,')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ' ,')
            diff = pl.DataFrame({'term': fr'New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$ (omit.)',
                          'estimate': 0.0,
                          'std.error': 0.0,
                          'statistic': None,
                          'p.value': None})
            diffz.append(diff)
        else:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ',')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ',')
            r(f'diff = get_coef_diff({mod_non_west}, "{coef_1}", "{coef_2}", "New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$")')
            diff = to_py('diff', index = False)
            diffz.append(diff)
        
    start = pl.concat(diffz)
    model_non_west = (to_py(f'tidy({mod_non_west})', index = False).with_columns(
        term = pl.col("term").replace({ 
            'inc::1': 'Income 200,000 DKK - 400,000 DKK',
            'inc::2': 'Income 400,001 DKK - 600,000 DKK',
            'inc::3': 'Income 600,001 DKK - 800,000 DKK',
            'inc::4': 'Income 800,001 DKK - 1,000,000 DKK',
            'inc::5': 'Income >1,000,001 DKK',
            'age::2': 'Oldest in HH, age 41-50',
            'age::3': 'Oldest in HH, age 51-60',
            'tenure::2': r'Tenure $[1-2[$ years',
            'tenure::3': r'Tenure $[2-4[$ years',
            'tenure::4': r'Tenure $[4-6[$ years',
            'tenure::5': r'Tenure $\geq 6$ years',
            'wealth::2': 'Wealth 1 - 200,000 DKK',
            'wealth::3': 'Wealth 200,001 - 400,000 DKK',
            'wealth::4': 'Wealth 400,001 - 600,000 DKK',
            'wealth::5': 'Wealth 600,001 - 750,000 DKK'}))
    )
    model_raw_non_west=model_non_west.with_columns(
        term = pl.col("term").replace({
            'I_nearest': '$k_{nearest}$',
            'I_near': '$k_{near}$',
            'I_close_10': '$k_{close, 10}$',
            'I_close_20': '$k_{close, 20}$',
            'I_close_30': '$k_{close, 30}$',
            'I_close_40': '$k_{close, 40}$'})
    )

    model_diff_non_west = model_non_west.filter(pl.col("term").str.starts_with('I_').not_())

    model_out_non_west = pl.concat([start, model_diff_non_west])
    to_r(model_out_non_west, f'{mod_non_west}_adj_tidy', format = 'data.frame')
    to_r(model_raw_non_west, f'{mod_non_west}_adj_tidy_raw', format = 'data.frame')

    modelz_non_west.append(model_out_non_west)
    modz_non_west[f'{mod_non_west}'] = diffz


    diff_control = 'I_control'

diffs = [('I_nearest_0', diff_control),
         ('I_nearest_25', diff_control),
         ('I_nearest_100', diff_control)]

modz = ['mod4_100', 'mod4_100_non_west']


for mod in modz:
    diffz = []
    for coef_1, coef_2 in diffs:
        if coef_1 == coef_2:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ', ')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ', ')
            diff = pl.DataFrame({'term': fr'New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$ (omit.)',
                            'estimate': 0.0,
                            'std.error': 0.0,
                            'statistic': None,
                            'p.value': None})
            diffz.append(diff)
        else:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ',')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ',')
            r(f'diff = get_coef_diff({mod}, "{coef_1}", "{coef_2}", "New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$")')
            diff = to_py('diff', index = False)
            diffz.append(diff)
        
    start = pl.concat(diffz)
    model = (to_py(f'tidy({mod})', index = False).with_columns(
        term = pl.col("term").replace({ 
            'inc::1': 'Income 200,000 DKK - 400,000 DKK',
            'inc::2': 'Income 400,001 DKK - 600,000 DKK',
            'inc::3': 'Income 600,001 DKK - 800,000 DKK',
            'inc::4': 'Income 800,001 DKK - 1,000,000 DKK',
            'inc::5': 'Income >1,000,001 DKK',
            'age::2': 'Oldest in HH, age 41-50',
            'age::3': 'Oldest in HH, age 51-60',
            'tenure::2': r'Tenure $[1-2[$ years',
            'tenure::3': r'Tenure $[2-4[$ years',
            'tenure::4': r'Tenure $[4-6[$ years',
            'tenure::5': r'Tenure $\geq 6$ years'}))
    )
    model_raw=model.with_columns(
        term = pl.col("term").replace({
            'I_nearest': '$k_{nearest}$',
            'I_near': '$k_{near}$',
            'I_close_10': '$k_{close, 10}$',
            'I_close_20': '$k_{close, 20}$',
            'I_close_30': '$k_{close, 30}$',
            'I_close_40': '$k_{close, 40}$'})
    )

    model_diff = model.filter(pl.col("term").str.starts_with('I_').not_())
    model_glance = to_py(f'glance({mod})', index=False).with_columns(nobs = pl.col("nobs").map_elements(lambda x: f'{x:,.0f}', return_dtype = pl.String))

    model_out = pl.concat([start, model_diff])
    to_r(model_out, f'{mod}_adj_tidy', format = 'data.frame')
    to_r(model_raw, f'{mod}_adj_tidy_raw', format = 'data.frame')
    to_r(model_glance, f'{mod}_adj_glance', format = 'data.frame')


# Run difference in coefs test
# Balance tests
diff_control = 'I_near'

diffs = [('I_nearest', diff_control),
         ('I_near', diff_control),
         ('I_close_10', diff_control),
         ('I_close_20', diff_control),
         ('I_close_30', diff_control),
         ('I_close_40', diff_control)]

modz = {}

mods = ['mod1_balance', 'mod2_balance', 'mod3_balance', 'mod4_balance', 'mod5_balance', 'mod6_balance', 'mod7_balance']
modelz = []

for mod in mods:
    diffz = []
    for coef_1, coef_2 in diffs:
        if coef_1 == coef_2:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ' ,')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ' ,')
            diff = pl.DataFrame({'term': fr'New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$ (omit.)',
                          'estimate': 0.0,
                          'std.error': 0.0,
                          'statistic': None,
                          'p.value': None})
            diffz.append(diff)
        else:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ',')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ',')
            r(f'diff = get_coef_diff({mod}, "{coef_1}", "{coef_2}", "New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$")')
            diff = to_py('diff', index = False)
            diffz.append(diff)
        
    start = pl.concat(diffz)
    model = to_py(f'tidy({mod})', index = False).filter(pl.col("term").str.starts_with('I_').not_())
    model_out = pl.concat([start, model])
    to_r(model_out, f'{mod}_adj_tidy', format = 'data.frame')

    modelz.append(model_out)
    modz[f'{mod}'] = diffz


# Run difference in coefs test (non_west)
# Balance tests
diff_control = 'I_near'

diffs = [('I_nearest', diff_control),
         ('I_near', diff_control),
         ('I_close_10', diff_control),
         ('I_close_20', diff_control),
         ('I_close_30', diff_control),
         ('I_close_40', diff_control)]

modz_non_west = {}

mods_non_west = ['mod1_balance_non_west', 'mod2_balance_non_west', 'mod3_balance_non_west', 'mod4_balance_non_west', 'mod5_balance_non_west', 'mod6_balance_non_west', 'mod7_balance_non_west']
modelz_non_west = []

for mod_non_west in mods_non_west:
    diffz = []
    for coef_1, coef_2 in diffs:
        if coef_1 == coef_2:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ' ,')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ' ,')
            diff = pl.DataFrame({'term': fr'New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$ (omit.)',
                          'estimate': 0.0,
                          'std.error': 0.0,
                          'statistic': None,
                          'p.value': None})
            diffz.append(diff)
        else:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ',')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ',')
            r(f'diff = get_coef_diff({mod_non_west}, "{coef_1}", "{coef_2}", "New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$")')
            diff = to_py('diff', index = False)
            diffz.append(diff)
        
    start = pl.concat(diffz)
    model_non_west = to_py(f'tidy({mod_non_west})', index = False).filter(pl.col("term").str.starts_with('I_').not_())

    model_out_non_west = pl.concat([start, model_non_west])
    to_r(model_out_non_west, f'{mod_non_west}_adj_tidy', format = 'data.frame')
    modelz_non_west.append(model_out_non_west)
    modz_non_west[f'{mod_non_west}'] = diffz

# Run difference in coefs test
# Balance tests
diff_control = 'I_near'

diffs = [('I_nearest', diff_control),
         ('I_near', diff_control),
         ('I_close_10', diff_control),
         ('I_close_20', diff_control),
         ('I_close_30', diff_control),
         ('I_close_40', diff_control)]

modz_non_west = {}

mods_non_west = ['mod1_balance_non_west', 'mod2_balance_non_west', 'mod3_balance_non_west', 'mod4_balance_non_west', 'mod5_balance_non_west', 'mod6_balance_non_west', 'mod7_balance_non_west']
modelz_non_west = []

for mod_non_west in mods_non_west:
    diffz = []
    for coef_1, coef_2 in diffs:
        if coef_1 == coef_2:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ' ,')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ' ,')
            diff = pl.DataFrame({'term': fr'New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$ (omit.)',
                          'estimate': 0.0,
                          'std.error': 0.0,
                          'statistic': None,
                          'p.value': None})
            diffz.append(diff)
        else:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ',')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ',')
            r(f'diff = get_coef_diff({mod_non_west}, "{coef_1}", "{coef_2}", "New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$")')
            diff = to_py('diff', index = False)
            diffz.append(diff)
        
    start = pl.concat(diffz)
    model_non_west = to_py(f'tidy({mod_non_west})', index = False).filter(pl.col("term").str.starts_with('I_').not_())

    model_out_non_west = pl.concat([start, model_non_west])
    to_r(model_out_non_west, f'{mod_non_west}_adj_tidy', format = 'data.frame')
    modelz_non_west.append(model_out_non_west)
    modz_non_west[f'{mod_non_west}'] = diffz

#######################
## REGRESSION TABLES ##
#######################

## main results
tab_name_main_results = f'{TABS_DIR}/main_results.tex'
tab_name_main_results_R = f'"{tab_name_main_results}"'
tab_name_main_results_full = f'{TABS_DIR}/main_results_full.tex'
tab_name_main_results_full_R = f'"{tab_name_main_results_full}"'

r(f'''
  mod1_list = modelsummary(mod1, output = "modelsummary_list")
  mod1_list$tidy = mod1_adj_tidy

  colnames(mod1_list$glance)[which(names(mod1_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod1_list$glance)[which(names(mod1_list$glance) == "nobs")] = "N"

  mod2_list = modelsummary(mod2, output = "modelsummary_list")
  mod2_list$tidy = mod2_adj_tidy

  colnames(mod2_list$glance)[which(names(mod2_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod2_list$glance)[which(names(mod2_list$glance) == "nobs")] = "N"

  mod3_list = modelsummary(mod3, output = "modelsummary_list")
  mod3_list$tidy = mod3_adj_tidy

  colnames(mod3_list$glance)[which(names(mod3_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod3_list$glance)[which(names(mod3_list$glance) == "nobs")] = "N"

  mod4_list = modelsummary(mod4, output = "modelsummary_list")
  mod4_list$tidy = mod4_adj_tidy
  
  colnames(mod4_list$glance)[which(names(mod4_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod4_list$glance)[which(names(mod4_list$glance) == "nobs")] = "N"

  mod5_list = modelsummary(mod5, output = "modelsummary_list")
  mod5_list$tidy = mod5_adj_tidy
  
  colnames(mod5_list$glance)[which(names(mod5_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod5_list$glance)[which(names(mod5_list$glance) == "nobs")] = "N"

  mod6_list = modelsummary(mod6, output = "modelsummary_list")
  mod6_list$tidy = mod6_adj_tidy
  
  colnames(mod6_list$glance)[which(names(mod6_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod6_list$glance)[which(names(mod6_list$glance) == "nobs")] = "N"
    
  tab = modelsummary(list(mod1_list, mod2_list, mod3_list, mod4_list, mod5_list, mod6_list), 
  coef_omit = -1,
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  add_rows = data.frame(
  term = c("Income", "Wealth", "Tenure", "Age"), 
  model1 = c("", "", "", ""), 
  model2 = c("X", "", "", ""), 
  model3 = c("X", "", "X", ""),
  model4 = c("X", "", "X", "X"),
  model5 = c("", "X", "X", "X"),
  model6 = c("X", "X", "X", "X")),
  output = 'gt'
  )

  tab_full = modelsummary(list(
                        mod1_list, mod2_list, mod3_list, mod4_list, mod5_list, mod6_list
  ), 
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  output = 'gt',
  add_rows = data.frame(
  term = c("Income", "Wealth", "Tenure", "Age"), 
  model1 = c("", "", "", ""), 
  model2 = c("X", "", "", ""), 
  model3 = c("X", "", "X", ""),
  model4 = c("X", "", "X", "X"),
  model5 = c("", "X", "X", "X"),
  model6 = c("X", "X", "X", "X"))
  )

  tab_full %>%
    tab_spanner(label = "<=25m", columns = 2:5) %>%
    tab_spanner(label = "Move within 2 years (=100)", columns = 2:5) %>%
    gt::gtsave(filename = {tab_name_main_results_full_R})

  tab %>%
    tab_spanner(label = "Move within 2 years (=100)", columns = 2:5) %>%
    gt::gtsave(filename = {tab_name_main_results_R})
  ''')
utils.remove_tab_env(f'{tab_name_main_results}')
utils.remove_tab_env(f'{tab_name_main_results_full}')



# same but with non-west

## main results
tab_name_main_results_non_west = f'{TABS_DIR}/main_results_non_west.tex'
tab_name_main_results_non_west_R = f'"{tab_name_main_results_non_west}"'
## main results, full version with diff dists to nearest neighbor
tab_name_main_results_full_non_west = f'{TABS_DIR}/main_results_full_non_west.tex'

tab_name_main_results_full_non_west_R = f'"{tab_name_main_results_full_non_west}"'

r(f'''
  mod1_non_west_list = modelsummary(mod1_non_west, output = "modelsummary_list")
  mod1_non_west_list$tidy = mod1_non_west_adj_tidy

  colnames(mod1_non_west_list$glance)[which(names(mod1_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod1_non_west_list$glance)[which(names(mod1_non_west_list$glance) == "nobs")] = "N"

  mod2_non_west_list = modelsummary(mod2_non_west, output = "modelsummary_list")
  mod2_non_west_list$tidy = mod2_non_west_adj_tidy

  colnames(mod2_non_west_list$glance)[which(names(mod2_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod2_non_west_list$glance)[which(names(mod2_non_west_list$glance) == "nobs")] = "N"

  mod3_non_west_list = modelsummary(mod3_non_west, output = "modelsummary_list")
  mod3_non_west_list$tidy = mod3_non_west_adj_tidy

  colnames(mod3_non_west_list$glance)[which(names(mod3_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod3_non_west_list$glance)[which(names(mod3_non_west_list$glance) == "nobs")] = "N"

  mod4_non_west_list = modelsummary(mod4_non_west, output = "modelsummary_list")
  mod4_non_west_list$tidy = mod4_non_west_adj_tidy

  colnames(mod4_non_west_list$glance)[which(names(mod4_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod4_non_west_list$glance)[which(names(mod4_non_west_list$glance) == "nobs")] = "N"
  
  mod5_non_west_list = modelsummary(mod5_non_west, output = "modelsummary_list")
  mod5_non_west_list$tidy = mod5_non_west_adj_tidy

  colnames(mod5_non_west_list$glance)[which(names(mod5_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod5_non_west_list$glance)[which(names(mod5_non_west_list$glance) == "nobs")] = "N"
  
  mod6_non_west_list = modelsummary(mod6_non_west, output = "modelsummary_list")
  mod6_non_west_list$tidy = mod6_non_west_adj_tidy

  colnames(mod6_non_west_list$glance)[which(names(mod6_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod6_non_west_list$glance)[which(names(mod6_non_west_list$glance) == "nobs")] = "N"
  
  tab = modelsummary(list(mod1_non_west_list, mod2_non_west_list, mod3_non_west_list, mod4_non_west_list, mod5_non_west_list, mod6_non_west_list), 
  coef_omit = -1,
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  add_rows = data.frame(
  term = c("Income", "Wealth", "Tenure", "Age"), 
  model1 = c("", "", "", ""), 
  model2 = c("X", "", "", ""), 
  model3 = c("X", "", "X", ""),
  model4 = c("X", "", "X", "X"),
  model5 = c("", "X", "X", "X"),
  model6 = c("X", "X", "X", "X")),
  output = 'gt'
  )

  tab_full_non_west = modelsummary(list(mod1_non_west_list, mod2_non_west_list, mod3_non_west_list, mod4_non_west_list, mod5_non_west_list, mod6_non_west_list), 
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  add_rows = data.frame(
  term = c("Income", "Wealth", "Tenure", "Age"), 
  model1 = c("", "", "", ""), 
  model2 = c("X", "", "", ""), 
  model3 = c("X", "", "X", ""),
  model4 = c("X", "", "X", "X"),
  model5 = c("", "X", "X", "X"),
  model6 = c("X", "X", "X", "X")),
  output = 'gt'
  )

  tab_full_non_west %>%
    tab_spanner(label = "<=25m", columns = 2:5) %>%
    tab_spanner(label = "Move within 2 years (=100)", columns = 2:7) %>%
    gt::gtsave(filename = {tab_name_main_results_full_non_west_R})

  tab %>%
    tab_spanner(label = "Move within 2 years (=100)", columns = 2:7) %>%
    gt::gtsave(filename = {tab_name_main_results_non_west_R})
  ''')
utils.remove_tab_env(f'{tab_name_main_results_non_west}')
utils.remove_tab_env(f'{tab_name_main_results_full_non_west}')


## main results, distance <= 100m
tab_name_main_results_full_100m = f'{TABS_DIR}/main_results_full_100m.tex'
tab_name_main_results_full_100m_R = f'"{tab_name_main_results_full_100m}"'

tab_name_main_results_100m = f'{TABS_DIR}/main_results_100m.tex'
tab_name_main_results_100m_R = f'"{tab_name_main_results_100m}"'

r(f'''
  
  mod4_100_list = modelsummary(mod4_100, output = "modelsummary_list")
  mod4_100_list$tidy = mod4_100_adj_tidy
  
  colnames(mod4_100_list$glance)[which(names(mod4_100_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod4_100_list$glance)[which(names(mod4_100_list$glance) == "nobs")] = "N"

  mod4_100_non_west_list = modelsummary(mod4_100_non_west, output = "modelsummary_list")
  mod4_100_non_west_list$tidy = mod4_100_non_west_adj_tidy
  
  colnames(mod4_100_non_west_list$glance)[which(names(mod4_100_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod4_100_non_west_list$glance)[which(names(mod4_100_non_west_list$glance) == "nobs")] = "N"

  tab = modelsummary(list(
                        mod4_100_list, mod4_100_non_west_list
  ), 
  coef_omit = -1:-3,
  gof_omit = "^(?!N|Mean|Neighb)",

  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  output = 'gt',
  add_rows = data.frame(
  term = c("Income", "Tenure", "Age"), 
  model1 = c("X", "X", "X"),
  model2 = c("X", "X", "X"))
  )

  tab %>%
    tab_spanner(label = "Native", columns = 2) %>%
    tab_spanner(label = "Non-Western", columns = 3) %>%
    tab_spanner(label = "Move within 2 years (=100), distance <= 100m", columns = 2:3) %>%
    gt::gtsave(filename = {tab_name_main_results_100m_R})

  
  tab_full = modelsummary(list(
                        mod4_100_list, mod4_100_non_west_list
  ), 
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  output = 'gt',
  add_rows = data.frame(
  term = c("Income", "Tenure", "Age"), 
  model1 = c("X", "X", "X"),
  model2 = c("X", "X", "X"))
    )

  tab_full %>%
    tab_spanner(label = "Native", columns = 2) %>%
    tab_spanner(label = "Non-Western", columns = 3) %>%
    tab_spanner(label = "Move within 2 years (=100), distance <= 100m", columns = 2:3) %>%
    gt::gtsave(filename = {tab_name_main_results_full_100m_R})
  ''')
utils.remove_tab_env(f'{tab_name_main_results_100m}')
utils.remove_tab_env(f'{tab_name_main_results_full_100m}')


## main results, full version with diff dists to nearest neighbor
tab_name_main_results_full_raw = f'{TABS_DIR}/main_results_full_raw.tex'

tab_name_main_results_full_raw_R = f'"{tab_name_main_results_full_raw}"'

r(f'''
  mod1_list = modelsummary(mod1, output = "modelsummary_list")
  mod1_list$tidy = mod1_adj_tidy_raw
  colnames(mod1_list$glance)[which(names(mod1_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod1_list$glance)[which(names(mod1_list$glance) == "nobs")] = "N"

  mod2_list = modelsummary(mod2, output = "modelsummary_list")
  mod2_list$tidy = mod2_adj_tidy_raw
  colnames(mod2_list$glance)[which(names(mod2_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod2_list$glance)[which(names(mod2_list$glance) == "nobs")] = "N"

  mod3_list = modelsummary(mod3, output = "modelsummary_list")
  mod3_list$tidy = mod3_adj_tidy_raw

  colnames(mod3_list$glance)[which(names(mod3_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod3_list$glance)[which(names(mod3_list$glance) == "nobs")] = "N"

  mod4_list = modelsummary(mod4, output = "modelsummary_list")
  mod4_list$tidy = mod4_adj_tidy_raw
  
  colnames(mod4_list$glance)[which(names(mod4_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod4_list$glance)[which(names(mod4_list$glance) == "nobs")] = "N"

  tab_full = modelsummary(list(
                        mod1_list, mod2_list, mod3_list, mod4_list
  ), 
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  output = 'gt',
  add_rows = data.frame(
  term = c("Income", "Tenure", "Age"), 
  model1 = c("", "", ""), 
  model2 = c("X", "", ""), 
  model3 = c("X", "X", ""),
  model4 = c("X", "X", "X"))
  )

  tab_full %>%
    tab_spanner(label = "<=25m", columns = 2:5) %>%
    tab_spanner(label = "Move within 2 years (=100)", columns = 2:5)
  gt::gtsave(tab_full, filename = {tab_name_main_results_full_raw_R})
  ''')
utils.remove_tab_env(f'{tab_name_main_results_full_raw}')


## main results, full version with diff dists to nearest neighbor
tab_name_main_results_full_raw_non_west = f'{TABS_DIR}/main_results_full_non_west_raw.tex'

tab_name_main_results_full_raw_non_west_R = f'"{tab_name_main_results_full_raw_non_west}"'

r(f'''
  mod1_non_west_list = modelsummary(mod1_non_west, output = "modelsummary_list")
  mod1_non_west_list$tidy = mod1_non_west_adj_tidy_raw
  colnames(mod1_non_west_list$glance)[which(names(mod1_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod1_non_west_list$glance)[which(names(mod1_non_west_list$glance) == "nobs")] = "N"

  mod2_non_west_list = modelsummary(mod2_non_west, output = "modelsummary_list")
  mod2_non_west_list$tidy = mod2_non_west_adj_tidy_raw
  colnames(mod2_non_west_list$glance)[which(names(mod2_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod2_non_west_list$glance)[which(names(mod2_non_west_list$glance) == "nobs")] = "N"

  mod3_non_west_list = modelsummary(mod3_non_west, output = "modelsummary_list")
  mod3_non_west_list$tidy = mod3_non_west_adj_tidy_raw
  colnames(mod3_non_west_list$glance)[which(names(mod3_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod3_non_west_list$glance)[which(names(mod3_non_west_list$glance) == "nobs")] = "N"

  mod4_non_west_list = modelsummary(mod4_non_west, output = "modelsummary_list")
  mod4_non_west_list$tidy = mod4_non_west_adj_tidy_raw
  colnames(mod4_non_west_list$glance)[which(names(mod4_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod4_non_west_list$glance)[which(names(mod4_non_west_list$glance) == "nobs")] = "N"

  tab_full = modelsummary(list(
                        mod1_non_west_list, mod2_non_west_list, mod3_non_west_list, mod4_non_west_list
  ), 
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  output = 'gt',
  add_rows = data.frame(
  term = c("Income", "Tenure", "Age"), 
  model1 = c("", "", ""), 
  model2 = c("X", "", ""), 
  model3 = c("X", "X", ""),
  model4 = c("X", "X", "X"))
  )

  tab_full %>%
    tab_spanner(label = "<=25m", columns = 2:5) %>%
    tab_spanner(label = "Move within 2 years (=100)", columns = 2:5)
  gt::gtsave(tab_full, filename = {tab_name_main_results_full_raw_non_west_R})
  ''')
utils.remove_tab_env(f'{tab_name_main_results_full_raw_non_west}')


## Balance test
tab_name_balance = f'{TABS_DIR}/balance_test.tex'
tab_name_balance_R = f'"{tab_name_balance}"'

tab_name_balance_full = f'{TABS_DIR}/balance_test_full.tex'
tab_name_balance_full_R = f'"{tab_name_balance_full}"'

r(f'''
  mod1_balance_list = modelsummary(mod1_balance, output = "modelsummary_list")
  mod1_balance_list$tidy = mod1_balance_adj_tidy
  colnames(mod1_balance_list$glance)[which(names(mod1_balance_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod1_balance_list$glance)[which(names(mod1_balance_list$glance) == "nobs")] = "N"

  mod2_balance_list = modelsummary(mod2_balance, output = "modelsummary_list")
  mod2_balance_list$tidy = mod2_balance_adj_tidy
  colnames(mod2_balance_list$glance)[which(names(mod2_balance_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod2_balance_list$glance)[which(names(mod2_balance_list$glance) == "nobs")] = "N"  

  mod3_balance_list = modelsummary(mod3_balance, output = "modelsummary_list")
  mod3_balance_list$tidy = mod3_balance_adj_tidy
  colnames(mod3_balance_list$glance)[which(names(mod3_balance_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod3_balance_list$glance)[which(names(mod3_balance_list$glance) == "nobs")] = "N"  


  mod4_balance_list = modelsummary(mod4_balance, output = "modelsummary_list")
  mod4_balance_list$tidy = mod4_balance_adj_tidy
  colnames(mod4_balance_list$glance)[which(names(mod4_balance_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod4_balance_list$glance)[which(names(mod4_balance_list$glance) == "nobs")] = "N"  


  mod5_balance_list = modelsummary(mod5_balance, output = "modelsummary_list")
  mod5_balance_list$tidy = mod5_balance_adj_tidy
  colnames(mod5_balance_list$glance)[which(names(mod5_balance_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod5_balance_list$glance)[which(names(mod5_balance_list$glance) == "nobs")] = "N"  

  mod6_balance_list = modelsummary(mod6_balance, output = "modelsummary_list")
  mod6_balance_list$tidy = mod6_balance_adj_tidy
  colnames(mod6_balance_list$glance)[which(names(mod6_balance_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod6_balance_list$glance)[which(names(mod6_balance_list$glance) == "nobs")] = "N"  

  mod7_balance_list = modelsummary(mod7_balance, output = "modelsummary_list")
  mod7_balance_list$tidy = mod7_balance_adj_tidy
  colnames(mod7_balance_list$glance)[which(names(mod7_balance_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod7_balance_list$glance)[which(names(mod7_balance_list$glance) == "nobs")] = "N"  

  
  modelsummary(list("Income (1,000)" = mod1_balance_list, 
                    "Net wealth (1,000)" = mod2_balance_list, 
                    "Oldest HH member (years)" = mod3_balance_list, 
                    "Tenure (days)" = mod4_balance_list, 
                    "Employed" = mod5_balance_list, 
                    "Educ. length (years)" = mod6_balance_list, 
                    "HH size" = mod7_balance_list), 
  coef_omit = -1,
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  output = {tab_name_balance_R}
  )
  
  modelsummary(list("Income (1,000)" = mod1_balance_list, 
                    "Net wealth (1,000)" = mod2_balance_list, 
                    "Oldest HH member (years)" = mod3_balance_list, 
                    "Tenure (days)" = mod4_balance_list, 
                    "Employed" = mod5_balance_list, 
                    "Educ. length (years)" = mod6_balance_list, 
                    "HH size" = mod7_balance_list), 
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  output = {tab_name_balance_full_R}
  )
  ''')

utils.remove_tab_env(f'{tab_name_balance}')
utils.remove_tab_env(f'{tab_name_balance_full}')

## Balance test (non_west)
tab_name_non_west_balance = f'{TABS_DIR}/balance_test_non_west.tex'
tab_name_non_west_balance_R = f'"{tab_name_non_west_balance}"'

tab_name_non_west_balance_full = f'{TABS_DIR}/balance_test_full_non_west.tex'
tab_name_non_west_balance_full_R = f'"{tab_name_non_west_balance_full}"'

r(f'''
  mod1_balance_non_west_list = modelsummary(mod1_balance_non_west, output = "modelsummary_list")
  mod1_balance_non_west_list$tidy = mod1_balance_non_west_adj_tidy
  colnames(mod1_balance_non_west_list$glance)[which(names(mod1_balance_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod1_balance_non_west_list$glance)[which(names(mod1_balance_non_west_list$glance) == "nobs")] = "N"

  mod2_balance_non_west_list = modelsummary(mod2_balance_non_west, output = "modelsummary_list")
  mod2_balance_non_west_list$tidy = mod2_balance_non_west_adj_tidy
  colnames(mod2_balance_non_west_list$glance)[which(names(mod2_balance_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod2_balance_non_west_list$glance)[which(names(mod2_balance_non_west_list$glance) == "nobs")] = "N"  

  mod3_balance_non_west_list = modelsummary(mod3_balance_non_west, output = "modelsummary_list")
  mod3_balance_non_west_list$tidy = mod3_balance_non_west_adj_tidy
  colnames(mod3_balance_non_west_list$glance)[which(names(mod3_balance_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod3_balance_non_west_list$glance)[which(names(mod3_balance_non_west_list$glance) == "nobs")] = "N"  


  mod4_balance_non_west_list = modelsummary(mod4_balance_non_west, output = "modelsummary_list")
  mod4_balance_non_west_list$tidy = mod4_balance_non_west_adj_tidy
  colnames(mod4_balance_non_west_list$glance)[which(names(mod4_balance_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod4_balance_non_west_list$glance)[which(names(mod4_balance_non_west_list$glance) == "nobs")] = "N"  


  mod5_balance_non_west_list = modelsummary(mod5_balance_non_west, output = "modelsummary_list")
  mod5_balance_non_west_list$tidy = mod5_balance_non_west_adj_tidy
  colnames(mod5_balance_non_west_list$glance)[which(names(mod5_balance_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod5_balance_non_west_list$glance)[which(names(mod5_balance_non_west_list$glance) == "nobs")] = "N"  

  mod6_balance_non_west_list = modelsummary(mod6_balance_non_west, output = "modelsummary_list")
  mod6_balance_non_west_list$tidy = mod6_balance_non_west_adj_tidy
  colnames(mod6_balance_non_west_list$glance)[which(names(mod6_balance_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod6_balance_non_west_list$glance)[which(names(mod6_balance_non_west_list$glance) == "nobs")] = "N"  

  mod7_balance_non_west_list = modelsummary(mod7_balance_non_west, output = "modelsummary_list")
  mod7_balance_non_west_list$tidy = mod7_balance_non_west_adj_tidy
  colnames(mod7_balance_non_west_list$glance)[which(names(mod7_balance_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod7_balance_non_west_list$glance)[which(names(mod7_balance_non_west_list$glance) == "nobs")] = "N"  

  
  modelsummary(list("Income (1,000)" = mod1_balance_non_west_list, 
                    "Net wealth (1,000)" = mod2_balance_non_west_list, 
                    "Oldest HH member (years)" = mod3_balance_non_west_list, 
                    "Tenure (days)" = mod4_balance_non_west_list, 
                    "Employed" = mod5_balance_non_west_list, 
                    "Educ. length (years)" = mod6_balance_non_west_list, 
                    "HH size" = mod7_balance_non_west_list), 
  coef_omit = -1,
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  output = {tab_name_non_west_balance_R}
  )
  
  modelsummary(list("Income (1,000)" = mod1_balance_non_west_list, 
                    "Net wealth (1,000)" = mod2_balance_non_west_list, 
                    "Oldest HH member (years)" = mod3_balance_non_west_list, 
                    "Tenure (days)" = mod4_balance_non_west_list, 
                    "Employed" = mod5_balance_non_west_list, 
                    "Educ. length (years)" = mod6_balance_non_west_list, 
                    "HH size" = mod7_balance_non_west_list), 
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  output = {tab_name_non_west_balance_full_R}
  )
  ''')

utils.remove_tab_env(f'{tab_name_non_west_balance}')
utils.remove_tab_env(f'{tab_name_non_west_balance_full}')

# LOW SES: unemp / earns between [100_000, 200_000] / educ less than or equal to HS
# HIGH SES: emp / ears more than median inc (~350_000-400_000) / educ over HS
year = pl.DataFrame().with_columns(
    START_EXPR,
    END_EXPR,
    t = T_EXPR
).explode(pl.all()).with_columns(
    year = pl.col("window_start").dt.year()
).select(pl.col("t", "year"))
lf_ses_class = lf_covars.with_columns(
    ses_class = pl.when(
        (pl.col("hh_employed") == 0) & (pl.col("equi_hh_real_inc") < 200_000) | (pl.col("hh_highest_educ") <= 11)
    ).then(pl.lit("low"))
    .when((pl.col("hh_employed")== 1) & (pl.col("equi_hh_real_inc") >= 600_000) | (pl.col("hh_highest_educ") >= 18))
    .then(pl.lit("high"))
).filter(pl.col("ses_class").is_not_null()).select(pl.col("hh_id", "year", "ses_class"))

lf_movez_dk_SES_nearest = (lf_movez_dk
                           .select(pl.col("hh_id", "t"), pl.col("nearest_movez"))
                           .explode(cs.ends_with('movez')).unnest(cs.ends_with('movez')).filter(pl.col("hh_id_nn").is_not_null())
                           .join(year.lazy(), on = 't')
                            .join(lf_ses_class, left_on = ['hh_id_nn', 'year'], right_on = ['hh_id', 'year'])
                            .rename({'ses_class': 'ses_class_nn'})
                           ).group_by('hh_id', 't').agg(
                               (pl.col("ses_class_nn")=="low").any().alias("any_low_ses_nn_nearest"),
                               (pl.col("ses_class_nn")=="low").all().alias("all_low_ses_nn_nearest"),
                               (pl.col("ses_class_nn")=="high").any().alias("any_high_ses_nn_nearest"),
                               (pl.col("ses_class_nn")=="high").all().alias("all_high_ses_nn_nearest")
                           )
lf_movez_dk_SES_close = (lf_movez_dk
                         .select(pl.col("hh_id", "t"), pl.col("close_movez"))
                         .explode(cs.ends_with('movez')).unnest(cs.ends_with('movez')).filter(pl.col("hh_id_nn").is_not_null())
                         .join(year.lazy(), on = 't')
                         .join(lf_ses_class, left_on = ['hh_id_nn', 'year'], right_on = ['hh_id', 'year'])
                         .rename({'ses_class': 'ses_class_nn'})
                           ).group_by('hh_id', 't').agg(
                               (pl.col("ses_class_nn")=="low").any().alias("any_low_ses_nn_close"),
                               (pl.col("ses_class_nn")=="low").all().alias("all_low_ses_nn_close"),
                               (pl.col("ses_class_nn")=="high").any().alias("any_high_ses_nn_close"),
                               (pl.col("ses_class_nn")=="high").all().alias("all_high_ses_nn_close")
                           )

lf_movez_non_west_SES_nearest = (lf_movez_non_west
                           .select(pl.col("hh_id", "t"), pl.col("nearest_movez"))
                           .explode(cs.ends_with('movez')).unnest(cs.ends_with('movez')).filter(pl.col("hh_id_nn").is_not_null())
                           .join(year.lazy(), on = 't')
                            .join(lf_ses_class, left_on = ['hh_id_nn', 'year'], right_on = ['hh_id', 'year'])
                            .rename({'ses_class': 'ses_class_nn'})
                           ).group_by('hh_id', 't').agg(
                               (pl.col("ses_class_nn")=="low").any().alias("any_low_ses_nn_nearest"),
                               (pl.col("ses_class_nn")=="low").all().alias("all_low_ses_nn_nearest"),
                               (pl.col("ses_class_nn")=="high").any().alias("any_high_ses_nn_nearest"),
                               (pl.col("ses_class_nn")=="high").all().alias("all_high_ses_nn_nearest")
                           )
lf_movez_non_west_SES_close = (lf_movez_non_west
                         .select(pl.col("hh_id", "t"), pl.col("close_movez"))
                         .explode(cs.ends_with('movez')).unnest(cs.ends_with('movez')).filter(pl.col("hh_id_nn").is_not_null())
                         .join(year.lazy(), on = 't')
                         .join(lf_ses_class, left_on = ['hh_id_nn', 'year'], right_on = ['hh_id', 'year'])
                         .rename({'ses_class': 'ses_class_nn'})
                           ).group_by('hh_id', 't').agg(
                               (pl.col("ses_class_nn")=="low").any().alias("any_low_ses_nn_close"),
                               (pl.col("ses_class_nn")=="low").all().alias("all_low_ses_nn_close"),
                               (pl.col("ses_class_nn")=="high").any().alias("any_high_ses_nn_close"),
                               (pl.col("ses_class_nn")=="high").all().alias("all_high_ses_nn_close")
                           )


# Heterogeneity
# low - low / low - high / high - high / high - low SES
lf_panel_unrestrict_SES = (lf_panel
        .filter(pl.col("cluster_id").is_in(clusterz))
        .join(lf_clusters, left_on = ['cluster_id', 't'], right_on=['cluster_id_500', 't'])
        .join(lf_ses_class, on = ['hh_id', 'year'], how = 'left')
        .join(lf_covars, on = ['hh_id', 'year'], how = 'left')
        .join(lf_hh_size, on = ['hh_id', 't'], how = 'left') # we know hh size at the quarterly level, so why not merge that
        .filter(pl.col("hh_oldest").is_between(30, 60))
        .filter(pl.col("equi_hh_real_inc").is_between(100_000, 1_000_000))
        .filter(pl.col("equi_hh_real_net_wealth").is_between(-200_000, 750_000))                                
        .join(lf_movez_dk_SES_nearest,  on = ['hh_id', 't'] , how = 'left')
        .join(lf_movez_dk_SES_close, on = ['hh_id', 't'] , how = 'left')
        .with_columns(
            inc = (pl.when(pl.col("equi_hh_real_inc").is_between(200_000, 400_000)).then(1)
                    .when(pl.col("equi_hh_real_inc").is_between(400_001, 600_000)).then(2)
                    .when(pl.col("equi_hh_real_inc").is_between(600_001, 800_000)).then(3)
                    .when(pl.col("equi_hh_real_inc").is_between(800_001, 1_000_000)).then(4)
                    .when(pl.col("equi_hh_real_inc")>=1_000_001).then(5)
            ),
            age = (pl.when(pl.col("hh_oldest").is_between(30, 40)).then(1)
                    .when(pl.col("hh_oldest").is_between(41, 50)).then(2)
                    .when(pl.col("hh_oldest").is_between(51, 60)).then(3)
                ),
            tenure = (pl.when(TENURE_EXPR.is_between(0, 365, closed = 'left')).then(1)
                    .when(TENURE_EXPR.is_between(365, 365*2, closed = 'left')).then(2)
                    .when(TENURE_EXPR.is_between(365*2, 365*4, closed = 'left')).then(3)
                    .when(TENURE_EXPR.is_between(365*4, 365*6, closed = 'left')).then(4)
                    .when(TENURE_EXPR >= 365*6).then(5)
            ),
            equi_hh_real_inc_1000 = pl.col("equi_hh_real_inc") / 1_000,
            equi_hh_real_net_wealth_1000 = pl.col("equi_hh_real_net_wealth") / 1_000,
            tenure_total = TENURE_EXPR,
            hh_highest_educ_cat = pl.col("hh_highest_educ").to_physical()
        )
      .sort('hh_id', 't')
)

df_panel_SES = lf_panel_unrestrict_SES.collect(engine = 'streaming')

lf_panel_unrestrict_SES_non_west = (lf_panel
        .filter(pl.col("cluster_id").is_in(clusterz))
        .join(lf_clusters, left_on = ['cluster_id', 't'], right_on=['cluster_id_500', 't'])
        .join(lf_ses_class, on = ['hh_id', 'year'], how = 'left')
        .join(lf_covars, on = ['hh_id', 'year'], how = 'left')
        .join(lf_hh_size, on = ['hh_id', 't'], how = 'left') # we know hh size at the quarterly level, so why not merge that
        .filter(pl.col("hh_oldest").is_between(30, 60))
        .filter(pl.col("equi_hh_real_inc").is_between(100_000, 1_000_000))
        .filter(pl.col("equi_hh_real_net_wealth").is_between(-200_000, 750_000))                                
        .join(lf_movez_non_west_SES_nearest,  on = ['hh_id', 't'] , how = 'left')
        .join(lf_movez_non_west_SES_close, on = ['hh_id', 't'] , how = 'left')
        .with_columns(
            inc = (pl.when(pl.col("equi_hh_real_inc").is_between(200_000, 400_000)).then(1)
                    .when(pl.col("equi_hh_real_inc").is_between(400_001, 600_000)).then(2)
                    .when(pl.col("equi_hh_real_inc").is_between(600_001, 800_000)).then(3)
                    .when(pl.col("equi_hh_real_inc").is_between(800_001, 1_000_000)).then(4)
            ),
            age = (pl.when(pl.col("hh_oldest").is_between(30, 40)).then(1)
                    .when(pl.col("hh_oldest").is_between(41, 50)).then(2)
                    .when(pl.col("hh_oldest").is_between(51, 60)).then(3)
                ),
            tenure = (pl.when(TENURE_EXPR.is_between(0, 365, closed = 'left')).then(1)
                    .when(TENURE_EXPR.is_between(365, 365*2, closed = 'left')).then(2)
                    .when(TENURE_EXPR.is_between(365*2, 365*4, closed = 'left')).then(3)
                    .when(TENURE_EXPR.is_between(365*4, 365*6, closed = 'left')).then(4)
                    .when(TENURE_EXPR >= 365*6).then(5)
            ),
            equi_hh_real_inc_1000 = pl.col("equi_hh_real_inc") / 1_000,
            equi_hh_real_net_wealth_1000 = pl.col("equi_hh_real_net_wealth") / 1_000,
            tenure_total = TENURE_EXPR,
            hh_highest_educ_cat = pl.col("hh_highest_educ").to_physical()
        )
      .sort('hh_id', 't')
)

df_panel_SES_non_west = lf_panel_unrestrict_SES_non_west.collect(engine = 'streaming')
# LOW SES: unemp / earns between [100_000, 200_000] / educ less than or equal to HS
# HIGH SES: emp / ears more than median inc (~350_000-400_000) / educ over HS

df_panel_ses_low = df_panel_SES.filter(pl.col("ses_class")=="low").filter(pl.col("hh_type")==1).filter(pl.max_horizontal(cs.starts_with('I_'))==1)
df_panel_ses_high = df_panel_SES.filter(pl.col("ses_class")=="high").filter(pl.col("hh_type")==1).filter(pl.max_horizontal(cs.starts_with('I_'))==1)

df_panel_ses_low_low = (df_panel_SES
                        .filter(pl.col("ses_class")=="low")
                        .filter(pl.col("hh_type")==1)
                        .filter((pl.col("any_low_ses_nn_nearest")==True) | (pl.col("any_low_ses_nn_close")==True))
                        .filter(pl.max_horizontal(cs.starts_with('I_'))==1)
                        )
df_panel_ses_low_high = (df_panel_SES
                        .filter(pl.col("ses_class")=="low")
                        .filter(pl.col("hh_type")==1)
                        .filter((pl.col("any_high_ses_nn_nearest")==True) | (pl.col("any_high_ses_nn_close")==True))
                        .filter(pl.max_horizontal(cs.starts_with('I_'))==1)
                        )

df_panel_ses_high_high = (df_panel_SES
                        .filter(pl.col("ses_class")=="high")
                        .filter(pl.col("hh_type")==1)
                        .filter((pl.col("any_high_ses_nn_nearest")==True) | (pl.col("any_high_ses_nn_close")==True))
                        .filter(pl.max_horizontal(cs.starts_with('I_'))==1)
                        )
df_panel_ses_high_low = (df_panel_SES
                        .filter(pl.col("ses_class")=="high")
                        .filter(pl.col("hh_type")==1)
                        .filter((pl.col("any_low_ses_nn_nearest")==True) | (pl.col("any_low_ses_nn_close")==True))
                        .filter(pl.max_horizontal(cs.starts_with('I_'))==1)
                        )

df_panel_ses_low_non_west = df_panel_SES_non_west.filter(pl.col("ses_class")=="low").filter(pl.col("hh_type")==2).filter(pl.max_horizontal(cs.starts_with('I_'))==1)
df_panel_ses_high_non_west = df_panel_SES_non_west.filter(pl.col("ses_class")=="high").filter(pl.col("hh_type")==2).filter(pl.max_horizontal(cs.starts_with('I_'))==1)

df_panel_ses_low_low_non_west = (df_panel_SES_non_west
                        .filter(pl.col("ses_class")=="low")
                        .filter(pl.col("hh_type")==2)
                        .filter((pl.col("any_low_ses_nn_nearest")==True) | (pl.col("any_low_ses_nn_close")==True))
                        .filter(pl.max_horizontal(cs.starts_with('I_'))==1)
                        )
df_panel_ses_low_high_non_west = (df_panel_SES_non_west
                        .filter(pl.col("ses_class")=="low")
                        .filter(pl.col("hh_type")==2)
                        .filter((pl.col("any_high_ses_nn_nearest")==True) | (pl.col("any_high_ses_nn_close")==True))
                        .filter(pl.max_horizontal(cs.starts_with('I_'))==1)
                        )

df_panel_ses_high_high_non_west = (df_panel_SES_non_west
                        .filter(pl.col("ses_class")=="high")
                        .filter(pl.col("hh_type")==2)
                        .filter((pl.col("any_high_ses_nn_nearest")==True) | (pl.col("any_high_ses_nn_close")==True))
                        .filter(pl.max_horizontal(cs.starts_with('I_'))==1)
                        )
df_panel_ses_high_low_non_west = (df_panel_SES_non_west
                        .filter(pl.col("ses_class")=="high")
                        .filter(pl.col("hh_type")==2)
                        .filter((pl.col("any_low_ses_nn_nearest")==True) | (pl.col("any_low_ses_nn_close")==True))
                        .filter(pl.max_horizontal(cs.starts_with('I_'))==1)
                        )

to_r(df_panel_ses_low, 'df_panel_ses_low')
to_r(df_panel_ses_high, 'df_panel_ses_high')
to_r(df_panel_ses_low_low, 'df_panel_ses_low_low')


to_r(df_panel_ses_low_high, 'df_panel_ses_low_high')
to_r(df_panel_ses_high_high, 'df_panel_ses_high_high')
to_r(df_panel_ses_high_low, 'df_panel_ses_high_low')

to_r(df_panel_ses_low_non_west, 'df_panel_ses_low_non_west')
to_r(df_panel_ses_high_non_west, 'df_panel_ses_high_non_west')
to_r(df_panel_ses_low_low_non_west, 'df_panel_ses_low_low_non_west')


to_r(df_panel_ses_low_high_non_west, 'df_panel_ses_low_high_non_west')
to_r(df_panel_ses_high_high_non_west, 'df_panel_ses_high_high_non_west')
to_r(df_panel_ses_high_low_non_west, 'df_panel_ses_high_low_non_west')

# model regressions by SES, native

r('''
  mod1_ses = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_ses_low, vcov = ~cluster_id)
  mod2_ses = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_ses_high, vcov = ~cluster_id)
  mod3_ses = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_ses_low_low, vcov = ~cluster_id)
  mod4_ses = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_ses_low_high, vcov = ~cluster_id)
  mod5_ses = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_ses_high_high, vcov = ~cluster_id)
  mod6_ses = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_ses_high_low, vcov = ~cluster_id)
  ''')

# model regressions by SES, non-Western

r('''
  mod1_ses_non_west = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_ses_low_non_west, vcov = ~cluster_id)
  mod2_ses_non_west = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_ses_high_non_west, vcov = ~cluster_id)
  mod3_ses_non_west = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_ses_low_low_non_west, vcov = ~cluster_id)
  mod4_ses_non_west = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_ses_low_high_non_west, vcov = ~cluster_id)
  mod5_ses_non_west = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_ses_high_high_non_west, vcov = ~cluster_id)
  mod6_ses_non_west = feols(move ~ I_nearest + I_near + I_close_10 + I_close_20 + I_close_30 + I_close_40 + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_ses_high_low_non_west, vcov = ~cluster_id)
  ''')

diff_control = 'I_near'

diffs = [('I_nearest', diff_control),
         ('I_near', diff_control),
         ('I_close_10', diff_control),
         ('I_close_20', diff_control),
         ('I_close_30', diff_control),
         ('I_close_40', diff_control)]

modz = {}

mods = ['mod1_ses', 'mod2_ses', 'mod3_ses', 'mod4_ses', 'mod5_ses', 'mod6_ses',
        'mod1_ses_non_west', 'mod2_ses_non_west', 'mod3_ses_non_west', 'mod4_ses_non_west', 'mod5_ses_non_west', 'mod6_ses_non_west'
        ]
modelz = []

for mod in mods:
    diffz = []
    for coef_1, coef_2 in diffs:
        if coef_1 == coef_2:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ' ,')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ' ,')
            diff = pl.DataFrame({'term': fr'New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$ (omit.)',
                          'estimate': 0.0,
                          'std.error': 0.0,
                          'statistic': None,
                          'p.value': None})
            diffz.append(diff)
        else:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ',')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ',')
            r(f'diff = get_coef_diff({mod}, "{coef_1}", "{coef_2}", "New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$")')
            diff = to_py('diff', index = False)
            diffz.append(diff)
        
    start = pl.concat(diffz)
    model = (to_py(f'tidy({mod})', index = False).with_columns(
        term = pl.col("term").replace({ 
            'inc::1': 'Income 200,000 DKK - 400,000 DKK',
            'inc::2': 'Income 400,001 DKK - 600,000 DKK',
            'inc::3': 'Income 600,001 DKK - 800,000 DKK',
            'inc::4': 'Income 800,001 DKK - 1,000,000 DKK',
            'inc::5': 'Income >1,000,001 DKK',
            'age::2': 'Oldest in HH, age 41-50',
            'age::3': 'Oldest in HH, age 51-60',
            'tenure::2': r'Tenure $[1-2[$ years',
            'tenure::3': r'Tenure $[2-4[$ years',
            'tenure::4': r'Tenure $[4-6[$ years',
            'tenure::5': r'Tenure $\geq$ years'}))
    )
    model_raw=model.with_columns(
        term = pl.col("term").replace({
            'I_nearest': '$k_{nearest}$',
            'I_near': '$k_{near}$',
            'I_close_10': '$k_{close, 10}$',
            'I_close_20': '$k_{close, 20}$',
            'I_close_30': '$k_{close, 30}$',
            'I_close_40': '$k_{close, 40}$'})
    )

    model_diff = model.filter(pl.col("term").str.starts_with('I_').not_())

    model_out = pl.concat([start, model_diff])
    to_r(model_out, f'{mod}_adj_tidy', format = 'data.frame')
    to_r(model_raw, f'{mod}_adj_tidy_raw', format = 'data.frame')
    modelz.append(model_out)
    modz[f'{mod}'] = diffz

## main results
tab_name_main_results_ses = f'{TABS_DIR}/main_results_ses.tex'
tab_name_main_results_ses_R = f'"{tab_name_main_results_ses}"'
tab_name_main_results_ses_full = f'{TABS_DIR}/main_results_ses_full.tex'
tab_name_main_results_ses_full_R = f'"{tab_name_main_results_ses_full}"'
r(f'''
 
  mod1_ses_list = modelsummary(mod1_ses, output = "modelsummary_list")
  mod1_ses_list$tidy = mod1_ses_adj_tidy
  colnames(mod1_ses_list$glance)[which(names(mod1_ses_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod1_ses_list$glance)[which(names(mod1_ses_list$glance) == "nobs")] = "N"

  mod2_ses_list = modelsummary(mod2_ses, output = "modelsummary_list")
  mod2_ses_list$tidy = mod2_ses_adj_tidy
  colnames(mod2_ses_list$glance)[which(names(mod2_ses_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod2_ses_list$glance)[which(names(mod2_ses_list$glance) == "nobs")] = "N"

  mod3_ses_list = modelsummary(mod3_ses, output = "modelsummary_list")
  mod3_ses_list$tidy = mod3_ses_adj_tidy

  colnames(mod3_ses_list$glance)[which(names(mod3_ses_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod3_ses_list$glance)[which(names(mod3_ses_list$glance) == "nobs")] = "N"

  mod4_ses_list = modelsummary(mod4_ses, output = "modelsummary_list")
  mod4_ses_list$tidy = mod4_ses_adj_tidy
  
  colnames(mod4_ses_list$glance)[which(names(mod4_ses_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod4_ses_list$glance)[which(names(mod4_ses_list$glance) == "nobs")] = "N"

  mod5_ses_list = modelsummary(mod5_ses, output = "modelsummary_list")
  mod5_ses_list$tidy = mod5_ses_adj_tidy
  
  colnames(mod5_ses_list$glance)[which(names(mod5_ses_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod5_ses_list$glance)[which(names(mod5_ses_list$glance) == "nobs")] = "N"  

  mod6_ses_list = modelsummary(mod6_ses, output = "modelsummary_list")
  mod6_ses_list$tidy = mod6_ses_adj_tidy
  
  colnames(mod6_ses_list$glance)[which(names(mod6_ses_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod6_ses_list$glance)[which(names(mod6_ses_list$glance) == "nobs")] = "N"  

  tab = modelsummary(list(mod1_ses_list, mod2_ses_list, mod3_ses_list, mod4_ses_list, mod5_ses_list, mod6_ses_list), 
  coef_omit = -1,
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  add_rows = data.frame(
  term = c("Income", "Tenure", "Age"), 
  model1 = c("", "X", "X"),
  model2 = c("X", "X", "X"),
  model3 = c("", "X", "X"),
  model4 = c("", "X", "X"),
  model5 = c("X", "X", "X"),
  model6 = c("X", "X", "X")),
  output = 'gt'
  )

  tab %>%
    tab_spanner(label = "SES: Low", columns = 2) %>%
    tab_spanner(label = "SES: High", columns = 3) %>%
    tab_spanner(label = "SES: Low v Low", columns = 4) %>%
    tab_spanner(label = "SES: Low v High", columns = 5) %>%
    tab_spanner(label = "SES: High v High", columns = 6) %>%
    tab_spanner(label = "SES: High v Low", columns = 7) %>%
    tab_spanner(label = "Move within 2 years (=100)", columns = 2:7) %>%
    gt::gtsave(filename = {tab_name_main_results_ses_R})


  tab_full = modelsummary(list(mod1_ses_list, mod2_ses_list, mod3_ses_list, mod4_ses_list, mod5_ses_list, mod6_ses_list), 
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  add_rows = data.frame(
  term = c("Income", "Tenure", "Age"), 
  model1 = c("", "X", "X"),
  model2 = c("X", "X", "X"),
  model3 = c("", "X", "X"),
  model4 = c("", "X", "X"),
  model5 = c("X", "X", "X"),
  model6 = c("X", "X", "X")),
  output = 'gt'
  )

  tab_full %>%
    tab_spanner(label = "SES: Low", columns = 2) %>%
    tab_spanner(label = "SES: High", columns = 3) %>%
    tab_spanner(label = "SES: Low v Low", columns = 4) %>%
    tab_spanner(label = "SES: Low v High", columns = 5) %>%
    tab_spanner(label = "SES: High v High", columns = 6) %>%
    tab_spanner(label = "SES: High v Low", columns = 7) %>%
    tab_spanner(label = "Move within 2 years (=100)", columns = 2:7) %>%
    gt::gtsave(filename = {tab_name_main_results_ses_full_R}) 

  
  ''')
utils.remove_tab_env(f'{tab_name_main_results_ses}')
utils.remove_tab_env(f'{tab_name_main_results_ses_full}')

## main results
tab_name_main_results_ses_non_west = f'{TABS_DIR}/main_results_ses_non_west.tex'
tab_name_main_results_ses_non_west_R = f'"{tab_name_main_results_ses_non_west}"'
tab_name_main_results_ses_non_west_full = f'{TABS_DIR}/main_results_ses_non_west_full.tex'
tab_name_main_results_ses_non_west_full_R = f'"{tab_name_main_results_ses_non_west_full}"'
r(f'''
 
  mod1_ses_non_west_list = modelsummary(mod1_ses_non_west, output = "modelsummary_list")
  mod1_ses_non_west_list$tidy = mod1_ses_non_west_adj_tidy
  colnames(mod1_ses_non_west_list$glance)[which(names(mod1_ses_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod1_ses_non_west_list$glance)[which(names(mod1_ses_non_west_list$glance) == "nobs")] = "N"

  mod2_ses_non_west_list = modelsummary(mod2_ses_non_west, output = "modelsummary_list")
  mod2_ses_non_west_list$tidy = mod2_ses_non_west_adj_tidy
  colnames(mod2_ses_non_west_list$glance)[which(names(mod2_ses_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod2_ses_non_west_list$glance)[which(names(mod2_ses_non_west_list$glance) == "nobs")] = "N"

  mod3_ses_non_west_list = modelsummary(mod3_ses_non_west, output = "modelsummary_list")
  mod3_ses_non_west_list$tidy = mod3_ses_non_west_adj_tidy

  colnames(mod3_ses_non_west_list$glance)[which(names(mod3_ses_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod3_ses_non_west_list$glance)[which(names(mod3_ses_non_west_list$glance) == "nobs")] = "N"

  mod4_ses_non_west_list = modelsummary(mod4_ses_non_west, output = "modelsummary_list")
  mod4_ses_non_west_list$tidy = mod4_ses_non_west_adj_tidy
  
  colnames(mod4_ses_non_west_list$glance)[which(names(mod4_ses_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod4_ses_non_west_list$glance)[which(names(mod4_ses_non_west_list$glance) == "nobs")] = "N"

  mod5_ses_non_west_list = modelsummary(mod5_ses_non_west, output = "modelsummary_list")
  mod5_ses_non_west_list$tidy = mod5_ses_non_west_adj_tidy
  
  colnames(mod5_ses_non_west_list$glance)[which(names(mod5_ses_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod5_ses_non_west_list$glance)[which(names(mod5_ses_non_west_list$glance) == "nobs")] = "N"  

  mod6_ses_non_west_list = modelsummary(mod6_ses_non_west, output = "modelsummary_list")
  mod6_ses_non_west_list$tidy = mod6_ses_non_west_adj_tidy
  
  colnames(mod6_ses_non_west_list$glance)[which(names(mod6_ses_non_west_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod6_ses_non_west_list$glance)[which(names(mod6_ses_non_west_list$glance) == "nobs")] = "N"  

  tab = modelsummary(list(mod1_ses_non_west_list, mod2_ses_non_west_list, mod3_ses_non_west_list, mod4_ses_non_west_list, mod5_ses_non_west_list, mod6_ses_non_west_list), 
  coef_omit = -1,
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  add_rows = data.frame(
  term = c("Income", "Tenure", "Age"), 
  model1 = c("", "X", "X"),
  model2 = c("X", "X", "X"),
  model3 = c("", "X", "X"),
  model4 = c("", "X", "X"),
  model5 = c("X", "X", "X"),
  model6 = c("X", "X", "X")),
  output = 'gt'
  )

  tab %>%
    tab_spanner(label = "SES: Low", columns = 2) %>%
    tab_spanner(label = "SES: High", columns = 3) %>%
    tab_spanner(label = "SES: Low v Low", columns = 4) %>%
    tab_spanner(label = "SES: Low v High", columns = 5) %>%
    tab_spanner(label = "SES: High v High", columns = 6) %>%
    tab_spanner(label = "SES: High v Low", columns = 7) %>%
    tab_spanner(label = "Move within 2 years (=100)", columns = 2:7) %>%
    gt::gtsave(filename = {tab_name_main_results_ses_non_west_R})


  tab_full = modelsummary(list(mod1_ses_non_west_list, mod2_ses_non_west_list, mod3_ses_non_west_list, mod4_ses_non_west_list, mod5_ses_non_west_list, mod6_ses_non_west_list), 
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  add_rows = data.frame(
  term = c("Income", "Tenure", "Age"), 
  model1 = c("", "X", "X"),
  model2 = c("X", "X", "X"),
  model3 = c("", "X", "X"),
  model4 = c("", "X", "X"),
  model5 = c("X", "X", "X"),
  model6 = c("X", "X", "X")),
  output = 'gt'
  )

  tab_full %>%
    tab_spanner(label = "SES: Low", columns = 2) %>%
    tab_spanner(label = "SES: High", columns = 3) %>%
    tab_spanner(label = "SES: Low v Low", columns = 4) %>%
    tab_spanner(label = "SES: Low v High", columns = 5) %>%
    tab_spanner(label = "SES: High v High", columns = 6) %>%
    tab_spanner(label = "SES: High v Low", columns = 7) %>%
    tab_spanner(label = "Move within 2 years (=100)", columns = 2:7) %>%
    gt::gtsave(filename = {tab_name_main_results_ses_non_west_full_R}) 

  
  ''')
utils.remove_tab_env(f'{tab_name_main_results_ses_non_west}')
utils.remove_tab_env(f'{tab_name_main_results_ses_non_west_full}')


##################################################################
## APPENDIX: NEW NEIGHBOR AT [0, 12.5], [12.5, 25] or [25, 100] ##
##################################################################

df_panel_reg_native_robust = (df_panel_native
                              .filter(pl.max_horizontal('^I_.*$')==1)
                              .with_columns(
                                  I_control = (pl.max_horizontal('I_near', 'I_close_10', 'I_close_20', "I_close_30", "I_close_40"))
                              )
                              .drop_nulls((pl.col("inc", "age", "tenure", "hh_employed", "hh_highest_educ", "hh_size"))))

df_panel_reg_non_west_robust = (df_panel_non_west
                              .filter(pl.max_horizontal('^I_.*$')==1)
                              .with_columns(
                                  I_control = (pl.max_horizontal('I_near', 'I_close_10', 'I_close_20', "I_close_30", "I_close_40"))
                              )
                              .drop_nulls((pl.col("inc", "age", "tenure", "hh_employed", "hh_highest_educ", "hh_size"))))


to_r(df_panel_reg_native_robust, 'df_panel_reg_native_robust')
to_r(df_panel_reg_non_west_robust, 'df_panel_reg_non_west_robust')

# Run difference in coefs test
# Main results

r('''  
  mod4_robust = feols(move ~ I_nearest + I_control + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_reg_native_robust, vcov = ~cluster_id)
  mod4_non_west_robust = feols(move ~ I_nearest + I_control + i(inc, ref = "1") + i(tenure, ref = "1") + i(age, ref = "1") | cluster_id^t, data = df_panel_reg_non_west_robust, vcov = ~cluster_id)

''')

diff_control = 'I_control'

diffs = [('I_nearest', diff_control),
         ('I_control', diff_control),
         ]

modz = {}

mods = ['mod4_robust', 'mod4_non_west_robust'
        ]
modelz = []

for mod in mods:
    diffz = []
    for coef_1, coef_2 in diffs:
        if coef_1 == coef_2:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ' ,')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ' ,')
            diff = pl.DataFrame({'term': fr'New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$ (omit.)',
                          'estimate': 0.0,
                          'std.error': 0.0,
                          'statistic': None,
                          'p.value': None})
            diffz.append(diff)
        else:
            suffix_1 = coef_1[coef_1.index('_'):][1:].replace('_', ',')
            suffix_2 = coef_2[coef_2.index('_'):][1:].replace('_', ',')
            r(f'diff = get_coef_diff({mod}, "{coef_1}", "{coef_2}", "New diff neighbor $k_{{{suffix_1}}}$ v $k_{{{suffix_2}}}$")')
            diff = to_py('diff', index = False)
            diffz.append(diff)
        
    start = pl.concat(diffz)
    model = (to_py(f'tidy({mod})', index = False).with_columns(
        term = pl.col("term").replace({ 
            'inc::1': 'Income 200,000 DKK - 400,000 DKK',
            'inc::2': 'Income 400,001 DKK - 600,000 DKK',
            'inc::3': 'Income 600,001 DKK - 800,000 DKK',
            'inc::4': 'Income 800,001 DKK - 1,000,000 DKK',
            'inc::5': 'Income >1,000,001 DKK',
            'age::2': 'Oldest in HH, age 41-50',
            'age::3': 'Oldest in HH, age 51-60',
            'tenure::2': r'Tenure $[1-2[$ years',
            'tenure::3': r'Tenure $[2-4[$ years',
            'tenure::4': r'Tenure $[4-6[$ years',
            'tenure::5': r'Tenure $\geq$ years'}))
    )
    model_raw=model.with_columns(
        term = pl.col("term").replace({
            'I_nearest': '$k_{nearest}$',
            'I_near': '$k_{near}$',
            'I_close_10': '$k_{close, 10}$',
            'I_close_20': '$k_{close, 20}$',
            'I_close_30': '$k_{close, 30}$',
            'I_close_40': '$k_{close, 40}$'})
    )

    model_diff = model.filter(pl.col("term").str.starts_with('I_').not_())

    model_out = pl.concat([start, model_diff])
    to_r(model_out, f'{mod}_adj_tidy', format = 'data.frame')
    to_r(model_raw, f'{mod}_adj_tidy_raw', format = 'data.frame')
    modelz.append(model_out)
    modz[f'{mod}'] = diffz

## main results
tab_name_main_results_robust_I_control = f'{TABS_DIR}/main_results_robust_I_control.tex'
tab_name_main_results_robust_I_control_R = f'"{tab_name_main_results_robust_I_control}"'
tab_name_main_results_robust_I_control_full = f'{TABS_DIR}/main_results_robust_I_control_full.tex'
tab_name_main_results_robust_I_control_full_R = f'"{tab_name_main_results_robust_I_control_full}"'


tab_name_main_results_robust_I_control_non_west = f'{TABS_DIR}/main_results_robust_I_control_non_west.tex'
tab_name_main_results_robust_I_control_non_west_R = f'"{tab_name_main_results_robust_I_control_non_west}"'
tab_name_main_results_robust_I_control_non_west_full = f'{TABS_DIR}/main_results_robust_I_control_non_west_full.tex'
tab_name_main_results_robust_I_control_non_west_full_R = f'"{tab_name_main_results_robust_I_control_non_west_full}"'

r(f'''
 
  mod4_robust_list = modelsummary(mod4_robust, output = "modelsummary_list")
  mod4_robust_list$tidy = mod4_robust_adj_tidy
  
  colnames(mod4_robust_list$glance)[which(names(mod4_robust_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod4_robust_list$glance)[which(names(mod4_robust_list$glance) == "nobs")] = "N"
  
  mod4_non_west_robust_list = modelsummary(mod4_non_west_robust, output = "modelsummary_list")
  mod4_non_west_robust_list$tidy = mod4_non_west_robust_adj_tidy
  
  colnames(mod4_non_west_robust_list$glance)[which(names(mod4_non_west_robust_list$glance) == "FE: cluster_id^t")] = "Neighborhood-by-quarter FE"
  colnames(mod4_non_west_robust_list$glance)[which(names(mod4_non_west_robust_list$glance) == "nobs")] = "N"

  tab = modelsummary(list(mod4_robust_list, mod4_non_west_robust_list), 
  coef_omit = -1,
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  add_rows = data.frame(
  term = c("Income", "Tenure", "Age"), 
  model4_1 = c("X", "X", "X"),
  model4_2 = c("X", "X", "X")),
  output = 'gt'
  )

  tab %>%
    tab_spanner(label = "Move within 2 years (=100)", columns = 2:3) %>%
    tab_spanner(label = "Native", columns = 2) %>%
    tab_spanner(label = "Non-Western", columns = 3) %>%
    gt::gtsave(filename = {tab_name_main_results_robust_I_control_R})


  tab_full = modelsummary(list(mod4_robust_list, mod4_non_west_robust_list), 
  gof_omit = "^(?!N|Mean|Neighb)",
  stars = c('*' = 0.05, '**' = 0.01, '***' = 0.001),
  add_rows = data.frame(
  term = c("Income", "Tenure", "Age"), 
  model4_1 = c("X", "X", "X"),
  model4_2 = c("X", "X", "X")),
  output = 'gt'
  )

  tab_full %>%
    tab_spanner(label = "Move within 2 years (=100)", columns = 2:3) %>%
    tab_spanner(label = "Native", columns = 2) %>%
    tab_spanner(label = "Non-Western", columns = 3) %>%
    gt::gtsave(filename = {tab_name_main_results_robust_I_control_full_R})

  ''')
utils.remove_tab_env(f'{tab_name_main_results_robust_I_control}')
utils.remove_tab_env(f'{tab_name_main_results_robust_I_control_full}')