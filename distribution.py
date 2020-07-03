import pandas as pd
from pandas.core.frame import DataFrame
from needed import SqlManager, create_folder
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import preprocessing
import numpy as np
from math import sqrt


def combine_features_tuples(label_encode_features):
    features = tuple(zip(*label_encode_features))
    return features


def label_encode(column):
    """
    become nominal value to number value
    :param column: each column
    :return: label encoded
    """
    sql_manager = SqlManager("information.sqlite")
    column_value = sql_manager.crs.execute(
        'select  {} from information '.format(column)).fetchall()

    labels = [x[0] for x in list(column_value)]
    if type(labels[0]) == int:
        label_encoded = labels
    else:
        le = preprocessing.LabelEncoder()
        label_encoded = le.fit_transform(labels)
    return label_encoded


def pie_plots(columns_name):
    for col in columns_name:
        result = sql_manager.crs.execute(
            ("select distinct {},count({}) from information group by {}".format(col, col, col))).fetchall()
        counts = [x[1] for x in result]
        attr = [x[0] for x in result]
        fig1, ax1 = plt.subplots()
        ax1.pie(counts, labels=attr, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        create_folder("outs\\pie_plots")
        plt.savefig("outs\\pie_plots\\{}.png".format(col))
        plt.close()


def boxes(columns_name, df):
    for col in columns_name:
        df[col].plot.box()
        create_folder("outs\\boxes")
        plt.savefig("outs\\boxes\\{}.png".format(col))
        plt.close()


def diff(df,cols):
    for col in cols:
        df[col].diff().hist()
        create_folder("outs\\diff_hists")
        plt.savefig("outs\\diff_hists\\{}.png".format(col))
        plt.close()


def cumsum(df):
    df.cumsum().plot()
    plt.savefig("outs\\cumsum.png")
    plt.close()


def corr(df):
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    plt.close()

# def euclidean(df:DataFrame):
#     distances = []
#     for row in df.iterrows():
#         result=0
#         for col in df.columns:
#             result+= row[1][col]**2
#         distances.append(sqrt(result))
#
#     df["distance"]=distances
#     print(df)
#
#     # df.plot.scatter(x="distance" , y=1)
#     # plt.show()
#
#
# def calculate_distance(**kwargs):
#     print(kwargs)


if __name__ == '__main__':

    sql_manager = SqlManager("information.sqlite")
    try:
        sql_manager.crs.execute("delete from encoding_guide")
        sql_manager.conn.commit()
    except:
        pass
    label_encode_features = []
    columns_name = ['num_reactions',
                    'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows',
                    'num_hahas', 'num_sads', 'num_angrys']
    for column in columns_name:
        encode_labels = label_encode(column)
        label_encode_features.append(encode_labels)

    data = combine_features_tuples(label_encode_features)
    df = DataFrame(data, columns=columns_name)
    cumsum(df=df)
    diff(df=df , cols=columns_name)
    boxes(columns_name=columns_name, df=df)
    pie_plots(columns_name=columns_name)
    corr(df=df)
