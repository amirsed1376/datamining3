from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from needed import SqlManager, create_folder
import pandas as pd
from pandas.core.frame import DataFrame
from matplotlib import pyplot as plt
from  scipy.cluster import hierarchy
from sklearn.metrics import davies_bouldin_score


columns = ['status_type', 'num_reactions',
           'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows',
           'num_hahas', 'num_sads', 'num_angrys']


def agglomerative_plots(df):
    create_folder("outs\\agglomerative")
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            print(columns[i], "   ", columns[j])
            plt.scatter(df[columns[i]], df[columns[j]], c=df["cluster"])
            plt.xlabel = columns[i]
            plt.ylabel = columns[j]
            plt.savefig("outs\\agglomerative\\{}---{}.png".format(columns[i], columns[j]))
            plt.close()


def agglomerative_each_2_columns(df, k):
    plt.close()
    create_folder("outs\\agglomerative_each2columns")

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            print(columns[i], "  ", columns[j])
            samples = df[[columns[i], columns[j]]].copy()
            agglomerative = AgglomerativeClustering(n_clusters=k)
            agglomerative.fit(samples)
            samples["cluster"] = agglomerative.labels_
            plt.scatter(samples[columns[i]], samples[columns[j]], c=samples["cluster"])
            plt.xlabel = columns[i]
            plt.ylabel = columns[j]
            plt.savefig("outs\\agglomerative_each2columns\\{}---{}.png".format(columns[i], columns[j]))
            plt.close()


def best_agglomerative(df, k, sql_manager):
    agglomerative = AgglomerativeClustering(n_clusters=k)
    agglomerative.fit(df)
    df["cluster"] = agglomerative.labels_
    df.to_sql(name="agglomerative_clusters", con=sql_manager.conn, if_exists="replace")
    return df


def reduce_dimension_agglomerative(samples, k, sql_manager):
    agglomerative = AgglomerativeClustering(n_clusters=k)
    agglomerative.fit(samples)
    datas=samples.copy()
    samples["cluster"] = agglomerative.labels_
    samples.to_sql(name="agglomerative_reduce_dimension_clusters", con=sql_manager.conn, if_exists="replace")

    plt.scatter(samples[0], samples[1], c=samples["cluster"])
    plt.savefig("outs\\reduce_dimension_agglomerative.png")
    plt.close()
    print("davies_bouldin_score=",davies_bouldin_score(datas,samples["cluster"]))


def hierarchy_plot(df):
    dendogram = hierarchy.dendrogram(Z=hierarchy.linkage(df, method="ward"))
    plt.savefig("outs\\hierarchy.png")
    plt.close()

if __name__ == '__main__':
    sql_manager = SqlManager("information.sqlite")
    df = pd.read_sql_query(sql="select * from information ", con=sql_manager.conn)
    hierarchy_plot(df)
    samples = DataFrame(PCA(n_components=2).fit_transform(df))
    reduce_dimension_agglomerative(samples=samples, k=4, sql_manager=sql_manager)
    df = best_agglomerative(df=df, k=4, sql_manager=sql_manager)
    agglomerative_plots(df)
    agglomerative_each_2_columns(df=df, k=4)
