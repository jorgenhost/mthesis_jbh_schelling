import os 
os.environ["POLARS_MAX_THREADS"] = '8'
os.environ["POLARS_FORCE_NEW_STREAMING"] = "1"
import sys
import time
sys.path.append('../src')
from dst import utils, geo

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

import polars as pl
import polars_grouper as plg
import polars.selectors as cs

# This script:
# 1) determines households (people living at same adress that temporally and geographically overlap)
# 2) constructs a single KDTree of Danish addresses in 4 dimensions: x, y, etage & sidedoer. 



OVERLAP_EXPR = (pl.min_horizontal("bop_vtil", "bop_vtil_hh")-pl.max_horizontal("bop_vfra", "bop_vfra_hh")).dt.total_days()
EDGE_WEIGHT_EXPR = ((OVERLAP_EXPR / ((pl.col("bop_vtil")-pl.col("bop_vfra")).dt.total_days()))).sqrt()
PARENT_CHILD_EXPR = pl.max_horizontal('child_link', 'parent_link')
PARTNER_EXPR = pl.max_horizontal('partner_typ1_linkA', 'partner_typ1_linkB', 'partner_typ2_linkA', 'partner_typ2_linkB')

# If seq is unique -> only one lived there. If seq is not unique, more than one lived in that household. Drop "themselves".
HH_EXPR = pl.when(pl.col("seq").is_unique()).then(True).when((pl.col("seq").is_unique().not_()) & (pl.col("seq")!=pl.col("seq_hh"))).then(True).otherwise(False)



lf_bop_full = pl.scan_parquet(f'{DATA_DIR}/build/geo_bop.pq').with_columns(
    bop_vtil = pl.when(pl.col("bop_vtil").dt.year()==2040).then(pl.date(2024, 12, 31)).otherwise(pl.col("bop_vtil"))
)

lf_overlap = lf_bop_full.select(pl.all().exclude("ages", "years", "etrs89_east", "id", "etrs89_north", "etage", "sidedoer", "vejkode", "husnr"))
col_names = lf_overlap.collect_schema().names()
col_names = {name: f'{name}_hh' for name in col_names}
lf_overlap=lf_overlap.rename(col_names)

lf_hh = (lf_bop_full
            .join(lf_overlap,
                        left_on = "address_map_id",
                            right_on="address_map_id_hh")
                            .filter(OVERLAP_EXPR > 0)                                   

).sort('seq')            
lf_hh.sink_parquet(f'{DATA_DIR}/build/knn/knn_hh.pq', engine = 'streaming')

message = f'Part 3: Add household identifier via graph methods'
time_ = utils.what_time_is_it()
utils.log(message, time_)
print(message)

time.sleep(20)

# Not so sure what to call it here, maybe a "subgraph" approach?
# This is to ensure all, statically identified, individuals in a HH overlap
# So join on the hh_id to make all possible combinations, then filter for overlap 
# Solve the graph / network "statically"
lf_hh_static = pl.scan_parquet(f'{DATA_DIR}/build/knn/knn_hh.pq').filter(pl.col("bop_vtil")>=pl.date(1960, 1, 1)).filter(pl.col("bop_vtil_hh")>=pl.date(1960, 1, 1)).filter(pl.col("bop_vfra")<=pl.date(2020, 12, 31)).filter(pl.col("bop_vfra_hh")<=pl.date(2020, 12,31)).select(pl.all().exclude( "vejkode", "husnr", "id", "koen")).filter(HH_EXPR).with_columns(
    hh_id_static = plg.graph_solver(pl.col("seq"), pl.col("seq_hh")).cast(pl.UInt32),
).with_columns(
    hh_size_static = pl.col("person_id").n_unique().over("hh_id_static")
).filter(pl.col("seq").is_first_distinct()).select(pl.all().exclude("^.*hh$")).collect().lazy()

# If statically solved households are less than or equal to two people, then accept the static id
# These static id's run from 1,...,N in increments of 1
lf_ok = lf_hh_static.filter(pl.col("hh_size_static")<=2).collect().lazy()

# The other part, we try to optimize the partition
lf_hh = lf_hh_static.filter(pl.col("hh_size_static")>2)

lf_bef = pl.scan_parquet(f'{DATA_DIR}/raw/bef.pq')

# Grab unique family_ids by start_date and person_id
lf_fam_id = lf_bef.with_columns(familie_id = pl.col("familie_id").cast(pl.Int32)).group_by('person_id', 'bop_vfra').agg(
    pl.col("familie_id").unique()
)

# Merge these to statically solved graph (as pl.List(pl.Int32))
lf_hh_fam = lf_hh.join(lf_fam_id, on=['person_id', 'bop_vfra'], how='left')

# Explode and get combinations of static household_id and family id
# These constitute household compositions
lf_famz = lf_hh_fam.explode("familie_id")

# Self-join to find combinations of household compositions
# Compute edge weights (later scaled by number of people in household) for optimizing the best household "split"
lf_famz_adj = lf_famz.join(lf_famz, on = ['address_map_id', 'familie_id'], how = 'inner', suffix='_hh', join_nulls=True).filter(OVERLAP_EXPR>=0).with_columns(
    weight_asym = EDGE_WEIGHT_EXPR
).filter(pl.col("familie_id").is_not_null()).with_columns(
    hh_id_temp = pl.struct(pl.col("address_map_id", "hh_id_static", "familie_id")).hash().rank("dense")
).with_columns(
    hh_size_temp = pl.col("person_id").n_unique().over("hh_id_temp")
).collect().lazy()

### EDGE WEIGHTS ####
# Parent-child links
lf_fam = lf_bef.group_by('person_id').agg(
    pl.col("far_pid").unique(),
    pl.col("mor_pid").unique()
)
lf_dad = lf_fam.select(pl.col("person_id", "far_pid")).explode("far_pid").with_columns(parent_id = pl.col("far_pid")).select(pl.all().exclude("far_pid")).filter(pl.col("parent_id").is_not_null())
lf_mom = lf_fam.select(pl.col("person_id", "mor_pid")).explode("mor_pid").with_columns(parent_id = pl.col("mor_pid")).select(pl.all().exclude("mor_pid")).filter(pl.col("parent_id").is_not_null())
lf_parents = pl.concat([lf_dad, lf_mom])

# Partner links
lf_partners = lf_bef.group_by('person_id').agg(
    pl.col("aegte_pid").unique(),
    pl.col("e_faelle_pid").unique()
)
lf_partner1 = lf_partners.select(pl.col("person_id", "aegte_pid")).explode("aegte_pid").with_columns(partner_id1 = pl.col("aegte_pid")).select(pl.all().exclude("aegte_pid")).filter(pl.col("partner_id1").is_not_null())
lf_partner2 = lf_partners.select(pl.col("person_id", "e_faelle_pid")).explode("e_faelle_pid").with_columns(partner_id2 = pl.col("e_faelle_pid")).select(pl.all().exclude("e_faelle_pid")).filter(pl.col("partner_id2").is_not_null())

# Derive edge weights
lf_right = lf_famz_adj.select(pl.col("seq", "seq_hh", "weight_asym")).rename(lambda x: f'{x}_right')
lf_hh_adj_weight = (lf_famz_adj.join(lf_right, left_on = ['seq', 'seq_hh'], right_on = ['seq_hh_right', 'seq_right'])
                    .join(lf_parents.with_columns(child_link = 1), left_on = ['person_id', 'person_id_hh'], right_on = ['person_id', 'parent_id'], how='left')
                    .join(lf_parents.with_columns(parent_link = 1), left_on =  ['person_id', 'person_id_hh'], right_on = ['parent_id', 'person_id'], how='left')
                    .join(lf_partner1.with_columns(partner_typ1_linkA = 1), left_on = ['person_id', 'person_id_hh'], right_on = ['person_id', 'partner_id1'], how = 'left')
                    .join(lf_partner1.with_columns(partner_typ1_linkB = 1), left_on = ['person_id', 'person_id_hh'], right_on = ['partner_id1', 'person_id'], how = 'left')
                    .join(lf_partner2.with_columns(partner_typ2_linkA = 1), left_on = ['person_id', 'person_id_hh'], right_on = ['person_id', 'partner_id2'], how = 'left')
                    .join(lf_partner2.with_columns(partner_typ2_linkB = 1), left_on = ['person_id', 'person_id_hh'], right_on = ['partner_id2', 'person_id'], how = 'left')
                    .with_columns(cs.numeric().fill_null(0))
                    .with_columns(
                        weight = pl.min_horizontal('weight_asym', 'weight_asym_right') + pl.when((PARTNER_EXPR) & (PARENT_CHILD_EXPR) == 1).then(pl.lit(0)).otherwise(PARTNER_EXPR + PARENT_CHILD_EXPR))
)

# Best partition by sequence id
partitionz = lf_hh_adj_weight.filter(HH_EXPR).with_columns(
    part_weight = pl.col("weight").sum().over("hh_id_temp")**(1/(pl.col("hh_size_temp").over("hh_id_temp")))
).with_columns(
    is_max_part = pl.col("part_weight").max().over("seq", "hh_id_static") == pl.col("part_weight").max().over("hh_id_temp")
).filter(pl.col("is_max_part")==True).filter(pl.col("seq").is_first_distinct()).with_columns(
    hh_id_adj = pl.struct(pl.col("hh_id_static", "hh_id_temp", "familie_id")).hash()
).select(pl.col("seq", "hh_id_adj"))

# Grab the max of the static ids
# add to hashed values, just so that if some hashed and non-hashed values intersect at first, they don't after adding this
max_ = lf_ok.select(pl.col("hh_id_static").rank("dense").max()).collect().item()

# VERY IMPORTANT!!!
# Need to "re-scale" the address_map_id index to account for potentially "lost" addresses
lf_hh = lf_hh.join(partitionz, on = 'seq').filter(pl.col("hh_id_adj").is_not_null())

lf_out = pl.concat([lf_ok, lf_hh], how = 'diagonal').with_columns(hh_id = pl.when(pl.col("hh_size_static")<=2).then(pl.col("hh_id_static")).otherwise(pl.col("hh_id_adj")+max_)).with_columns(hh_id = pl.col("hh_id").rank("dense")).with_columns(address_map_id = pl.col("address_map_id").rank("dense")-1).select(pl.all().exclude("hh_id_adj"))
df = lf_out.collect()
df.write_parquet(f'{DATA_DIR}/build/geo_hh.pq')

#################################
### COMPUTE NETWORK / KD-TREE ###
#################################

lf = pl.scan_parquet(f'{DATA_DIR}/build/geo_hh.pq')
lf_hh = lf.group_by("hh_id").agg(
    pl.col("bop_vfra").min(), pl.col("bop_vtil").max(),
    pl.col("person_id").n_unique().alias("hh_size"),
    pl.col("etrs89_east", "etrs89_north", "etage", "sidedoer").first(),
    pl.col("address_map_id").first()
).sort('hh_id')

df_hh = lf_hh.collect()

# KD-tree is computed over unique addresses map
# This is to populate those addresses later on
lf_adr = lf.group_by("address_map_id").agg(
    pl.col("etrs89_east", "etrs89_north", "etage", "sidedoer").first()
).sort('address_map_id')

df_adr = lf_adr.collect()

address_map_id = df_adr.select(pl.col("address_map_id"))

coords = df_adr.select(pl.col("etrs89_east", "etrs89_north"), pl.col("etage").replace({None: 0}).cast(pl.Float64), pl.col("sidedoer").replace({None: 0}).cast(pl.Float64))
kd_tree = geo.construct_kd_tree(coords=coords)

geo.save_obj(kd_tree, f'{DATA_DIR}/build/knn/kd_tree.pickle')

message = f'Constructed KD-tree'
time_ = utils.what_time_is_it()
utils.log(message, time_)
print(message)
