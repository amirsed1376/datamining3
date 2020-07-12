from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from needed import SqlManager, create_folder
import pandas as pd
from pandas.core.frame import DataFrame
from matplotlib import pyplot as plt
from sklearn.metrics import davies_bouldin_score

columns = ['status_type', 'num_reactions',
           'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows',
           'num_hahas', 'num_sads', 'num_angrys']


def k_means_plots(df, centers):
    create_folder("outs\\MainKMeans")
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            print(columns[i], "   ", columns[j])
            plt.scatter(df[columns[i]], df[columns[j]], c=df["cluster"])
            x_centers = [x[i] for x in centers]
            y_centers = [y[j] for y in centers]
            plt.scatter(x_centers, y_centers, c="r", marker="+", s=200)
            plt.xlabel = columns[i]
            plt.ylabel = columns[j]
            plt.savefig("outs\\MainKMeans\\{}---{}.png".format(columns[i], columns[j]))
            plt.close()


def k_means_each_2_columns(df, k):
    plt.close()
    create_folder("outs\\KMeans_each2columns")

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            print(columns[i], "  ", columns[j])
            samples = df[[columns[i], columns[j]]].copy()
            k_means = KMeans(n_clusters=k, random_state=5)
            k_means.fit(samples)
            samples["cluster"] = k_means.labels_
            plt.scatter(samples[columns[i]], samples[columns[j]], c=samples["cluster"])
            centers = k_means.cluster_centers_
            x_centers = [x[0] for x in centers]
            y_centers = [y[1] for y in centers]
            plt.scatter(x_centers, y_centers, c="r", marker="+", s=200)
            plt.xlabel = columns[i]
            plt.ylabel = columns[j]
            plt.savefig("outs\\KMeans_each2columns\\{}---{}.png".format(columns[i], columns[j]))
            plt.close()


def elbow_inertia(df, address_png):
    inertias = []
    for k in range(1, 10):
        k_means = KMeans(n_clusters=k)
        k_means.fit(df)
        inertias.append(k_means.inertia_)
    plt.plot(range(1, 10), inertias, "-o")
    plt.savefig(address_png)
    plt.close()


def best_k_means(df, k, sql_manager):
    k_means = KMeans(n_clusters=k, random_state=5)
    k_means.fit(df)
    df["cluster"] = k_means.labels_
    df.to_sql(name="KMeans_clusters", con=sql_manager.conn, if_exists="replace")
    center_df = DataFrame(k_means.cluster_centers_, columns=columns)
    center_df.to_sql(name="KMeans_centers", con=sql_manager.conn, if_exists="replace")
    return df, k_means.cluster_centers_


def reduce_dimension_k_means(samples, k, sql_manager):
    k_means = KMeans(n_clusters=k, random_state=5)
    k_means.fit(samples)
    datas=samples.copy()
    samples["cluster"] = k_means.labels_
    samples.to_sql(name="KMeans_reduce_dimension_clusters", con=sql_manager.conn, if_exists="replace")
    center_df = DataFrame(k_means.cluster_centers_)
    center_df.to_sql(name="KMeans_reduce_dimension_centers", con=sql_manager.conn, if_exists="replace")

    plt.scatter(samples[0], samples[1], c=samples["cluster"])
    centers = k_means.cluster_centers_
    x_centers = [x[0] for x in centers]
    y_centers = [y[1] for y in centers]
    plt.scatter(x_centers, y_centers, c="r", marker="+", s=200)
    plt.savefig("outs\\reduce_dimension_k_means.png")
    plt.close()
    print("davies_bouldin_score=",davies_bouldin_score(datas,samples["cluster"]))



if __name__ == '__main__':
    sql_manager = SqlManager("information.sqlite")
    df = pd.read_sql_query(sql="select * from information ", con=sql_manager.conn)
    elbow_inertia(df, "outs\\elbow.png")
    samples = DataFrame(PCA(n_components=2).fit_transform(df))
    elbow_inertia(samples, "outs\\PCA_elbow.png")
    reduce_dimension_k_means(samples=samples, k=4, sql_manager=sql_manager)
    df, centers = best_k_means(df=df, k=4, sql_manager=sql_manager)
    k_means_plots(df, centers)
    k_means_each_2_columns(df=df, k=4)
