from sklearn.cluster import DBSCAN
from pandas.core.frame import DataFrame
from needed import SqlManager, create_folder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import davies_bouldin_score


columns = ['num_reactions',
           'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows',
           'num_hahas', 'num_sads', 'num_angrys']


def db_scan_plots(df):
    create_folder("outs\\MainDBSCAN")
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            print(columns[i], "   ", columns[j])
            plt.scatter(df[columns[i]], df[columns[j]], c=df["cluster"])
            plt.xlabel = columns[i]
            plt.ylabel = columns[j]
            plt.savefig("outs\\MainDBSCAN\\{}---{}.png".format(columns[i], columns[j]))
            plt.close()


def db_scan_each_2_columns(df):
    plt.close()
    create_folder("outs\\DBSCAN_each2columns")

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            print(columns[i], "  ", columns[j])
            samples = df[[columns[i], columns[j]]].copy()
            db_scan = DBSCAN()
            db_scan.fit(samples)
            samples["cluster"] = db_scan.labels_
            plt.scatter(samples[columns[i]], samples[columns[j]], c=samples["cluster"])
            plt.xlabel = columns[i]
            plt.ylabel = columns[j]
            plt.savefig("outs\\DBSCAN_each2columns\\{}---{}.png".format(columns[i], columns[j]))
            plt.close()


def best_db_scan(df, sql_manager):
    db_scan = DBSCAN()
    db_scan.fit(df)
    df["cluster"] = db_scan.labels_
    df.to_sql(name="clusters_DBSCAN", con=sql_manager.conn, if_exists="replace")
    return df


def reduce_dimension_DBSCAN(samples, sql_manager):
    db_scan = DBSCAN(eps=0.85)
    db_scan.fit(samples)
    datas=samples.copy()
    samples["cluster"] = db_scan.labels_
    samples.to_sql(name="reduce_dimension_DBSCAN_cluster", con=sql_manager.conn, if_exists="replace")
    plt.scatter(samples[0], samples[1], c=samples["cluster"])
    plt.savefig("outs\\reduce_dimension_DBSCAN.png")
    plt.close()
    print("davies_bouldin_score=",davies_bouldin_score(datas,samples["cluster"]))


if __name__ == '__main__':
    sql_manager = SqlManager("information.sqlite")
    df = pd.read_sql_query(sql="select * from information ", con=sql_manager.conn)
    samples = DataFrame(PCA(n_components=2).fit_transform(df))
    reduce_dimension_DBSCAN(samples=samples, sql_manager=sql_manager)
    df = best_db_scan(df=df, sql_manager=sql_manager)
    db_scan_plots(df)
    db_scan_each_2_columns(df=df)
