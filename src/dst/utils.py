import polars as pl
import os
from polars import selectors as cs
import oracledb
import random
from tenacity import retry, stop_after_attempt, wait_fixed
import time
import sys
import pyarrow.parquet as pq

DATA_DIR = 'K:/Specialemappe_xw7/data'
MISC_DIR = 'K:/Specialemappe_xw7/misc'
TOP_DIR = 'K:/Specialemappe_xw7'
import sys
sys.path.append(f'{TOP_DIR}/src')

DIST_EXPR = pl.when((pl.col("rank_nn").is_between(1, 3) & (pl.col("query_dist")<=51)).over("hh_id")).then(True).when((pl.col("rank_nn")>=4) & (pl.col("query_dist")<=200)).then(True).otherwise(False)
DURATION_EXPR = (pl.col("bop_vtil")-pl.col("bop_vfra")).dt.total_days()


def con_db(user: str, password: str, dsn: str) -> oracledb.Connection:
    return oracledb.connect(user=user, password=password, dsn=dsn)

def parse_admin_data(dataset_name: str, query: str, year: int, 
                     con: oracledb.Connection, **kwargs) -> pl.DataFrame:
    print(f'Parsing {dataset_name}_{year}...')
    
    df = (pl.read_database(query=query, connection=con, **kwargs)
        .select(pl.all().name.to_lowercase())
        .with_columns(
        cs.datetime().cast(pl.Date)
                    )
        )   
    return df

def what_time_is_it():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def fetch_family_mapping():

    # FM-Mark variable in BEF
    fam_mapping = {
        1: 'both_pars',
        2: 'mom_new_couple',
        3: 'single_mom',
        4: 'dad_new_couple',
        5: 'single_dad',
        6: 'lives_alone'
    }

    return fam_mapping

def fetch_cpi_index():
    df = pl.read_excel(f'{MISC_DIR}/pris8.xlsx', schema_overrides={'year': pl.Int16})
    mapping = {}

    mapping.update([(key,value) for key,value in df.iter_rows()])

    return mapping

def fetch_origin_mapping():
    df_immi_class = pl.read_csv(f'{MISC_DIR}/nomenclat_cpr_country.csv', separator=";").select(pl.col("CPR Myndighedskode", "ISO Alpha-3"))
    mapping = {}

    mapping.update([(key, value) for key,value in df_immi_class.iter_rows()])

    # Change some codes from XXX to meaningful abbrev. See ../misc/nomenclat_cpr_country.csv
    mapping[5299] = 'AFRIKA'

    mapping[5499] = 'ASIEN'

    mapping[5001] = 'UNKNOWN'

    mapping[5199] = 'EUROPA'

    mapping[5223] = 'GAZA'
    
    mapping[5105] = 'ISL'

    mapping[5999] = 'UNKNOWN'

    mapping[5487] = 'MIDDLE-EAST'

    mapping[5397] = 'NORTH-AMERICA'

    mapping[5103] = 'STATELESS'

    mapping[5398] = 'SYD/MELLEM-AMERIKA'

    mapping[5393] = 'WEST-BANK'

    mapping[5599] = 'PACIFIC ISLANDS'

    mapping[5437] = 'EAST-JERUSALEM'

    mapping[5102] = 'UNKNOWN'

    return mapping

def fetch_country_cats(sub_cat = None):
    list_of_countries_non_west = [
        'AFG', 'ALB', 'DZA', 'AGO',
        'ATG', 'ARG', 'ARM', 'AZE',
        'BHS', 'BHR', 'BGD', 'BRB',
        'BLR', 'BLZ', 'BEN', 'BTN',
        'BOL', 'BIH', 'BWA', 'BRA', 
        'BRN', 'BFA', 'BDI', 'KHM',
        'CMR', 'CAF', 'CHL', 'CHL',
        'COL', 'COM', 'COM', 'COG',
        'COD', 'COK', 'CRI', 'CUB',
        'DJI', 'DMA', 'DOM', 'ECU',
        'EGY', 'SLV', 'CIV', 'ERI',
        'SWZ', 'ETH', 'EUROPA', 'FJI',
        'PHL', 'ARE', 'GAB', 'GMB',
        'GAZA', 'GEO', 'GHA', 'GRD',
        'GTM', 'GIN', 'GNB', 'GUY',
        'HTI', 'HND', 'IND', 'IDN',
        'IRQ', 'IRN', 'ISR', 'JAM', 
        'JPN', 'JOR', 'YUG', 'CPV',
        'KAZ', 'KEN', 'CHN', 'KGZ',
        'KIR', 'PRK', 'KOR', 'XKX',
        'KWT', 'LAO', 'LSO', 'LBN',
        'LBR', 'LBY', 'MDG', 'MWI',
        'MYS', 'MDV', 'MMR', 'MLI', 'MAR',
        'MHL', 'MRT', 'MUS', 'MIDDLE-EAST',
        'MEX', 'MDA', 'MNG', 'MNE', 
        'MOZ', 'MNR', 'NAM', 'NRU',
        'NPL', 'NIC', 'NIC', 'NER',
        'NGA', 'MKD', 'OMN', 'PAK', 
        'PSE', 'PAN', 'PNG', 'PRY',
        'PER', 'QAT', 'RUS', 'RWA',
        'STP', 'SAU', 'SEN', 'SRB',
        'SCG', 'SYC', 'SLE', 'SGP',
        'SOM', 'SUN', 'LKA', 'SDN', 
        'ZAF', 'SSD', 'SYR', 'TJK', 'SUR',
        'TWN', 'TZA', 'TCD', 'THA',
        'TGO', 'TTO', 'TON', 'TUN',
        'TMK', 'TUV', 'TUR', 'UGA', 'TLS', 'KNA', 'SLB',
        'UKR', 'URY', 'UZB', 'VUT',
        'VCT', 'LCA', 'VGB', 'WSM',
        'VEN', 'WEST-BANK', 'VNM', 'YEM',
        'ZMB', 'ZWE', 'GNQ', 'STATELESS', 'AFRIKA', 'PACIFIC ISLANDS', 'SYD/MELLEM-AMERIKA', 'ASIEN'
    ]
        
    if sub_cat == None:
        list_of_countries = [value for key, value in fetch_origin_mapping().items()]
        list_of_countries = list(dict.fromkeys(list_of_countries))
    elif sub_cat == 'non-west':
        list_of_countries = list_of_countries_non_west
    elif sub_cat == 'west':
        list_of_countries = [
            'AND', 'AUS', 'BEL', 'BGR', 'DNK',
            'CAN', 'CYP', 'EST', 'FIN',
            'FRA', 'GRC', 'IRL', 'ISL',
            'ITA', 'HRV', 'LVA', 'LIE',
            'LTU', 'LUX', 'MLT', 'MCO',
            'NLD', 'NZL', 'NOR', 'GBR',
            'POL', 'PRT', 'ROU', 'SMR',
            'CHE', 'SVK', 'SVN', 'ESP',
            'SWE', 'CZE', 'CSK', 'DEU',
            'HUN', 'USA', 'VAT', 'AUT',
            'NORTH-AMERICA'
        ]
    elif sub_cat == 'menapt':
        list_of_countries = [
            'AFG', 'BHR', 'SYR', 'KWT',
            'LBY', 'SAU', 'LBN', 'SOM',
            'IRQ', 'QAT', 'SDN', 'DJI',
            'JOR', 'DZA', 'ARE', 'TUN',
            'EGY', 'MAR', 'IRN', 'YEM',
            'MRT', 'OMN', 'PAK', 'PSE',
            'GAZA', 'WEST-BANK', 'EAST-JERUSALEM',
            'TUR'
        ]

    return list_of_countries

def make_coord_mapping(df: pl.DataFrame, col_list: list[str]) -> dict:
    '''
    Make mapping between key (first element of list) and coordinates.
    '''
    mapping_east = {}
    mapping_north = {}

    for key, coord1, coord2 in df.select(pl.col(col_list)).iter_rows():
        mapping_east.update({key: coord1})
        mapping_north.update({key: coord2})
    return mapping_east, mapping_north

def baseline_proba(lf: pl.LazyFrame, year: int = 1990):
    df = lf.filter(pl.col("hh_id").is_first_distinct()).filter(pl.date(year, 1, 1).is_between("bop_vfra", "bop_vtil")).filter(pl.date(year, 1, 1).is_between("bop_vfra_nn", "bop_vtil_nn")).with_columns(
        hh_type = pl.when(pl.col("native_hh")==True).then(1).when((pl.col("mix_non_west_share")>0)).then(2).when((pl.col("mix_non_west_share")==0) & (pl.col("mix_share")>0)).then(3),
        year = year
    ).select(pl.col("hh_id", "hh_type")).collect(new_streaming=True)

    return df.select(pl.col("hh_type").value_counts(normalize = True)).unnest("hh_type").sort("hh_type")

def log(message: str, time: str | None):

    if os.path.isfile(f'{TOP_DIR}/log.txt'):
        log=open(f'{TOP_DIR}/log.txt', 'a')
    else:
        log = open(f'{TOP_DIR}/log.txt', "w")

        header = ['TIME', 'MESSAGE']

        log.write('---'.join(header) + '\n')

    with open(f'{TOP_DIR}/log.txt', 'a') as log:
        log.write(f'({time}): {message}\n')


def remove_tab_env(file_path: str):
    import re
    with open(file_path, 'r') as file:
        content = file.read() 
    content = re.sub(r'\\begin{table\}(\[.*?\])?', '', content)
    content = re.sub(r'\\centering', '', content, count = 1)
    content = re.sub(r'\\end{table\}', '', content)
    content = re.sub(r'\n\s*\n', '\n', content)
    content = re.sub(r'\\end{longtable\}', r'\\end{tabular}', content)
    content = re.sub(r'\\begin{longtable\}(\[.*?\])?', r'\\begin{tabular}\1', content)

    with open(file_path, 'w') as file:
        file.write(content.strip())

@retry(stop = stop_after_attempt(5), wait = wait_fixed(5), reraise=True)
def write_pq(df: pl.DataFrame, path: str, **kwargs):
    return df.write_parquet(f'{path}', **kwargs)

@retry(stop = stop_after_attempt(5), wait = wait_fixed(5), reraise=True)
def sink_pq(lf: pl.LazyFrame, path: str, **kwargs):
    return lf.sink_parquet(f'{path}', **kwargs)

@retry(stop = stop_after_attempt(5), wait = wait_fixed(5), reraise=True)
def pyarrow_pq_list(output_file: str, file_list: list[str]):
    with pq.ParquetWriter(output_file, pq.ParquetFile(file_list[0]).schema_arrow) as writer:
        for file in file_list:
            parquet_file = pq.ParquetFile(file)

            writer.write_table(parquet_file.read())

@retry(stop = stop_after_attempt(5), wait = wait_fixed(5), reraise=True)
def pyarrow_pq(output_file: str, file: str | pq.ParquetFile):
    with pq.ParquetWriter(output_file, pq.ParquetFile(file).schema_arrow) as writer:
        for batch in file.iter_batches():
            parquet_file = pq.ParquetFile(file)

            writer.write_table(parquet_file.read())

# @retry(stop = stop_after_attempt(5), wait = wait_fixed(5), reraise=True)
# def pyarrow_pq_sample(output_file: str, file: pq.ParquetFile, k_sample = 40):
#     with pq.ParquetWriter(output_file, file.schema_arrow) as writer:
#         for batch in file.iter_batches(batch_size = 5_000_000):
#             table = pl.from_arrow(batch).filter(pl.col("hh_id").is_in(hh_id_sample)).filter(pl.col("rank_nn")<=k_sample).to_arrow()
#             writer.write_table(table)


@retry(stop = stop_after_attempt(5), wait = wait_fixed(5), reraise=True)
def do_something_unreliable():
    if random.randint(1, 10) >= 1:
        raise IOError("Broken sauce")
    else:
        return "nice"
