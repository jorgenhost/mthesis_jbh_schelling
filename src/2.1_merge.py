import os 
os.environ["POLARS_FORCE_NEW_STREAMING"] = '1'
os.environ["POLARS_MAX_THREADS"] = '8'
import polars as pl
pl.enable_string_cache()
from dst import utils

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MISC_DIR = os.path.join(PROJECT_ROOT, 'misc')


# This script: 
# 1) Cleans addresses and assigns these a unique id. Letters are assigned a numerical value: A is 0, B is 1 etc 
# 2) Adds individual characteristics from registers (country of origin, birthday)

message = 'Cleaning coordinate dataset (each row is a "sequence").'
time_ = utils.what_time_is_it()
utils.log(message, time_)
print(message)

list_floor_endings = ['A', 'B', 'C', 'D', 'E', 'F',
             'G', 'H', 'I', 'J', 'K',
             'L', 'M', 'N', 'O', 'P', 'Q', 'R',
             'S', 'T', 'U', 'V', 'W', 'X',
             'Y', 'Z']

lf_bop = (pl.scan_parquet(f'{DATA_DIR}/raw/bopael_koord.pq').filter(pl.col("bop_vtil")>=pl.date(1985, 1, 1))
            .filter(pl.col("etrs89_east").is_not_null()).filter(pl.col("etrs89_north").is_not_null())
            .with_columns(bop_vtil = pl.when(pl.col("bop_vtil").dt.year()==9999)
                        .then(pl.date(2040, pl.col("bop_vtil").dt.month(), pl.col("bop_vtil").dt.day()))
                        .otherwise(pl.col("bop_vtil")))
            .sort('bop_vfra', "etrs89_east", "etrs89_north")
            .with_columns(duration = pl.col("bop_vtil")-pl.col("bop_vfra"))
            .filter(pl.col("duration").dt.total_days() > 90)
            .with_columns_seq(pl.col("etage").replace({
                'ST': "0",
                "SV": None,
                'KL': "-1",
                "K2": "-1",
                'MZ': None,
                'KV': '-1',
                'PT': None}))
            .with_columns_seq(etage=pl.col("etage").str.extract_all(r"\d+").list.explode().cast(pl.Int8))
            .with_columns_seq(etage=pl.when(pl.col("etage")>35).then((pl.col("etage")/10).cast(pl.Int8)).otherwise(pl.col("etage")))
            .with_columns(
                kom = pl.col("kom").cast(pl.Int16),
                address_map_id = pl.struct(pl.col("etrs89_east", "etrs89_north", "etage", "sidedoer")).hash().rank("dense").sub(1).cast(pl.Int32)).with_row_index('seq')
)


# Country of origin classificatio
list_of_non_west_cnt = utils.fetch_country_cats(sub_cat = 'non-west')
list_of_west_cnt = utils.fetch_country_cats(sub_cat = 'west')
list_of_menapt_cnt = utils.fetch_country_cats(sub_cat = 'menapt')
list_of_countries = utils.fetch_country_cats()

# Mapping of classfication
origin = utils.fetch_origin_mapping()
lf_bef = (pl.scan_parquet(f'{DATA_DIR}/raw/bef.pq')
            .select(pl.col("person_id", 'foed_dag', 'ie_type', 'koen', 'opr_land')).group_by('person_id').agg(pl.col("*").last())
            .with_columns(opr_land = pl.col("opr_land").replace_strict(origin, return_dtype=pl.String, default=None))
            .filter(pl.col("opr_land").is_not_null())
            .with_columns(
                west_roots = pl.col("opr_land").is_in(list_of_west_cnt),
                non_west_roots = pl.col("opr_land").is_in(list_of_non_west_cnt),
                menapt_roots = pl.col('opr_land').is_in(list_of_menapt_cnt)
                          )
)                     

move_in_age_expr = ((pl.col("bop_vfra")-pl.col("foed_dag")).dt.total_days()/365.25).floor()
move_out_age_expr = ((pl.col("bop_vtil")-pl.col("foed_dag")).dt.total_days()/365.25).floor()+1

lf_bop_full = (lf_bop
            .join(lf_bef, on=['person_id'], how='left')
            .filter(pl.col("foed_dag").is_not_null()).select(pl.all().exclude('kvalitet', 'duration'))
)

df_bop_full = (lf_bop_full
            .sort('seq').select(pl.col("seq", "address_map_id"), pl.all().exclude("seq", "address_map_id"))
            .with_columns(
                sidedoer = pl.col("sidedoer").str.strip_chars())
            .with_columns(
                sidedoer_num = pl.col("sidedoer").str.extract_all(r"\d+").list.head(1).explode(),
                sidedoer_char = pl.when(pl.col("sidedoer").str.contains('TV')).then(pl.lit('TV'))
                    .when(pl.col("sidedoer").str.contains('MF')).then(pl.lit('MF'))
                    .when(pl.col("sidedoer").str.contains('TH')).then(pl.lit('TH'))
                    .when(pl.col("sidedoer").str.contains('ST')).then(pl.lit('ST'))
                    .otherwise(pl.col("sidedoer").str.extract_all(r'[A-Za-z]').list.head(1).explode()))
                .with_columns(
                    sidedoer_char = pl.col("sidedoer_char").replace({
                        "A" : 0,
                        "B" : 1, 
                        "C" : 2,
                        "D" : 3,
                        "E" : 4,
                        "F" : 5,
                        "G" : 6,
                        "H" : 7,
                        "I" : 8,
                        "J" : 9,
                        "K" : 10,
                        "L" : 11,
                        "M" : 12,
                        "N" : 13,
                        "O" : 14,
                        "P" : 15,
                        "Q" : 16,
                        "R" : 17,
                        "S" : 18,
                        "T" : 19,
                        "U" : 20,
                        "V" : 21,
                        "X" : 22,
                        "Y" : 23,
                        "Z" : 24,
                        "W" : 25,
                        "ST" : 0,
                        "MF" : 0,
                        "TV" : -1,
                        "TH" : 1
                    }).cast(pl.Int8),
                    sidedoer_num = pl.col("sidedoer_num").cast(pl.Int16)
                ).with_columns(
                    sidedoer = pl.when(pl.col("sidedoer_num").is_null() & pl.col("sidedoer_char").is_null()).then(None)
                    .otherwise(pl.col("sidedoer_num").replace({None: 0}) + pl.col("sidedoer_char").replace({None: 0}))
                    ).select(pl.all().exclude("sidedoer_num", "sidedoer_char"))
                .collect(new_streaming=True)
)

utils.write_pq(df_bop_full, f'{DATA_DIR}/build/geo_bop.pq')
message = f'Cleaning done. Created network w/ addresses map ids.'
time_ = utils.what_time_is_it()
utils.log(message, time_)
print(message)
