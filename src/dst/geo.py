import polars as pl
from scipy.spatial import KDTree
from pathlib import Path
import os, sys
import pickle
from dst import utils 

DATA_DIR = 'K:/Specialemappe_xw7/data'
MISC_DIR = 'K:/Specialemappe_xw7/misc'
TOP_DIR = 'K:/Specialemappe_xw7/'
sys.path.append(f'{TOP_DIR}/src')


list_of_non_west_cnt = utils.fetch_country_cats(sub_cat = 'non-west')

OVERLAP_EXPR = (pl.min_horizontal("bop_vtil", "bop_vtil_nn")-pl.max_horizontal("bop_vfra", "bop_vfra_nn")).dt.total_days()

NATIVE_SUM_EXPR = ((pl.col("person_id").is_first_distinct()) & (pl.col("opr_land") == "DNK")).sum().over("hh_id")
NON_NATIVE_SUM_EXPR = ((pl.col("person_id").is_first_distinct()) & (pl.col("opr_land") != "DNK")).sum().over("hh_id")
NON_WEST_SUM_EXPR = ((pl.col("person_id").is_first_distinct()) & (pl.col("opr_land").is_in(list_of_non_west_cnt))).sum().over("hh_id")

NON_WEST_SUM_EXPR_NN = ((pl.col("person_id").is_first_distinct()) & (pl.col("opr_land").is_in(list_of_non_west_cnt))).sum().over("hh_id_nn")


HH_SUM = pl.col("person_id").n_unique().over("hh_id")
HH_SUM_NN = pl.col("person_id").n_unique().over("hh_id_nn")

MIX_SHARE_EXPR = NON_NATIVE_SUM_EXPR / HH_SUM
MIX_NON_WEST_SHARE_EXPR = NON_WEST_SUM_EXPR / HH_SUM
MIX_NON_WEST_SHARE_EXPR_NN = NON_WEST_SUM_EXPR_NN / HH_SUM_NN 

def construct_kd_tree(coords: pl.DataFrame, **kwargs) -> KDTree:
    '''
    Construct KDTree. Make sure it is made of the unique addresses.
    '''
    
    kd_tree = KDTree(coords, **kwargs)
    return kd_tree

def save_obj(obj: object, path: str | Path) -> None:
    with open(f'{path}', 'wb') as f:
        pickle.dump(obj, f)

def load_kd_tree(path: str = f'{DATA_DIR}/build/knn/kd_tree.pickle') -> KDTree:
    if os.path.exists(path):
        with open(path, "rb") as f:
            kd_tree = pickle.load(f)
    else:
        print(f'{path} not correct or KDtree does not exist in dir.')
    
    return kd_tree

def query_knn(kd_tree: KDTree, coords: pl.DataFrame, df: pl.DataFrame, k=50, **kwargs) -> pl.DataFrame:
    '''
    Assumes that a missing obs under "etage" is because you live in a house. If that is so, then k-nearest is computed as etage := 0.
    '''
    query_dist, query_adr_idx = kd_tree.query(coords, workers=16, k=k+1, **kwargs)

    dist = pl.Series('query_dist', values=query_dist, dtype=pl.List(pl.Float64))

    adr_idx = pl.Series('query_adr_idx', values=query_adr_idx, dtype=pl.List(pl.UInt32))

    rank_nn = dist.list.eval(pl.element().rank("ordinal")).alias("rank_nn")

    df_out = (df.hstack([dist, adr_idx, rank_nn])
                .with_columns(distance_self = (pl.col("query_adr_idx").list.first())==pl.col("address_map_id"))
                .with_columns(self_idx_list = pl.col("address_map_id").cast(pl.List(pl.UInt32)))
    )

    df_out = df_out.with_columns(
        query_adr_idx = pl.when(pl.col("distance_self")==True).then(pl.col("query_adr_idx")).otherwise(pl.col("query_adr_idx").list.set_difference("self_idx_list"))
    )
    df_out = (df_out.with_columns(
        query_adr_idx = pl.col("self_idx_list").list.set_union("query_adr_idx"))
    )
    # For those where their own position was not in neighbor list, slice the last element
    df_out = df_out.with_columns(
        query_adr_idx = pl.when(pl.col("query_adr_idx").list.len()>k+1).then(pl.col("query_adr_idx").list.head(k+1)).otherwise(pl.col("query_adr_idx"))
    ).select(pl.all().exclude("distance_self", "self_idx_list"))
    return df_out


def query_knn_t(kd_tree: KDTree, coords: pl.DataFrame, df: pl.DataFrame, k=50, **kwargs) -> pl.DataFrame:
    '''
    Assumes that a missing obs under "etage" is because you live in a house. If that is so, then k-nearest is computed as etage := 0.
    '''
    query_dist, query_adr_idx = kd_tree.query(coords, workers=16, k=k+1, **kwargs)

    dist = pl.Series('query_dist', values=query_dist, dtype=pl.List(pl.Float64))

    adr_idx = pl.Series('query_adr_idx', values=query_adr_idx, dtype=pl.List(pl.UInt32))

    rank_nn = dist.list.eval(pl.element().rank("ordinal")).alias("rank_nn")

    df_out = (df.hstack([dist, adr_idx, rank_nn])
                .with_columns(distance_self = (pl.col("query_adr_idx").list.first())==pl.col("address_map_id_t"))
                .with_columns(self_idx_list = pl.col("address_map_id_t").cast(pl.List(pl.UInt32)))
    )

    df_out = df_out.with_columns(
        query_adr_idx = pl.when(pl.col("distance_self")==True).then(pl.col("query_adr_idx")).otherwise(pl.col("query_adr_idx").list.set_difference("self_idx_list"))
    )
    df_out = (df_out.with_columns(
        query_adr_idx = pl.col("self_idx_list").list.set_union("query_adr_idx"))
    )
    # For those where their own position was not in neighbor list, slice the last element
    df_out = df_out.with_columns(
        query_adr_idx = pl.when(pl.col("query_adr_idx").list.len()>k+1).then(pl.col("query_adr_idx").list.head(k+1)).otherwise(pl.col("query_adr_idx"))
    ).select(pl.all().exclude("distance_self", "self_idx_list"))
    return df_out


def knn_by_year(kd_tree: KDTree, year: int , k: int) -> pl.DataFrame:

    date_expr = (pl.date(year, 12, 31).is_between(pl.col("bop_vfra"), pl.col("bop_vtil")))

    lf = (pl.scan_parquet(f'{DATA_DIR}/build/geo_hh.pq').filter(pl.col("bop_vtil").dt.year()>=1985).filter(date_expr)
        .select(pl.col("seq", "person_id", "address_map_id", "etrs89_east", "etrs89_north", "etage", "sidedoer",
                    "bop_vfra", "bop_vtil", "opr_land", "west_roots", "non_west_roots", "menapt_roots",
                    "non_west_roots_hh", "menapt_roots_hh", "hh_id"))
        .with_columns(
            native_hh = (pl.col("opr_land") == "DNK").all().over("hh_id"),
            mix_hh = (pl.col("opr_land") != "DNK").any().over("hh_id"),
            non_west_all_hh = (pl.col("non_west_roots") == True).all().over("hh_id"),
            menapt_all_hh = (pl.col("menapt_roots") == True).all().over("hh_id")
            )
        .with_columns(
            mix_share = pl.when(pl.col("mix_hh") == True)
            .then(MIX_SHARE_EXPR)
            .otherwise(None).alias("mix_share"),
            mix_non_west_share = pl.when(pl.col("mix_hh") == True)
            .then(MIX_NON_WEST_SHARE_EXPR)
            .otherwise(None).alias("mix_non_west_share"),
            hh_size = pl.col("person_id").n_unique().over("hh_id").alias('hh_size'))
        .filter(pl.col("hh_size")<=40)
    )

    # Group to hh level, preserving essential columns
    lf_hh = (lf.group_by('hh_id')
        .agg(
            pl.col("address_map_id").first(),
            pl.col("etrs89_east", "etrs89_north", "etage", "sidedoer").first(),
            pl.col("bop_vfra").min(),
            pl.col("bop_vtil").max(),
            pl.col("non_west_all_hh").first(),
            pl.col("hh_size").first(),
            pl.col("menapt_all_hh").first(),
            pl.col("native_hh").first(),
            pl.col("mix_hh").first(),
            pl.col("mix_share", "mix_non_west_share").first())
        .sort('hh_id')
        .select(pl.all())
    )

    # "4D" coordinates of each address
    # If etage / sidedoer is None, replace with 0.
    # Feed it to the KDTree and query K-nearest
    # Save (nested) dataset
    df_hh = lf_hh.collect()
    coords = df_hh.select(pl.col("etrs89_east", "etrs89_north"), 
                        pl.col("etage").replace({None: 0}).cast(pl.Float64), 
                        pl.col("sidedoer").replace({None: 0}).cast(pl.Float64))


    df_knn = query_knn(kd_tree = kd_tree, coords = coords, df=df_hh, k = k)

    return df_hh.lazy(), df_knn.select(pl.all().exclude("etrs89_east", "etrs89_north", "etage", "sidedoer")).lazy()