from pandas.core.frame import DataFrame
from needed import SqlManager, create_folder
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors


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


def diff(df, cols):
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


def hist2d(df, columns):
    create_folder("outs\\hist2d")
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            plt.hist2d(df[columns[i]], df[columns[j]], (50, 50), cmin=1 )
            plt.colorbar()
            plt.savefig("outs\\hist2d\\{}--{}.png".format(columns[i], columns[j]))
            plt.close()

def density(df, columns):
    create_folder("outs\\density")
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            g = sns.jointplot(x=columns[i], y=columns[j], data=df, kind="kde")
            # g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
            # g.ax_joint.collections[0].set_alpha(0)
            plt.savefig("outs\\density\\{}--{}.png".format(columns[i], columns[j]))
            plt.close()


def point_point_plot(df, columns):
    create_folder("outs\\point2point")
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            g = sns.jointplot(x=columns[i], y=columns[j], data=df)
            plt.savefig("outs\\point2point\\{}--{}.png".format(columns[i], columns[j]))
            plt.close()


def hex_bin(df, columns):
    create_folder("outs\\hex_bin")
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            g = sns.jointplot(x=columns[i], y=columns[j], data=df, kind="hex")
            plt.savefig("outs\\hex_bin\\{}--{}.png".format(columns[i], columns[j]))
            plt.close()


if __name__ == '__main__':
    sql_manager = SqlManager("information.sqlite")
    try:
        sql_manager.crs.execute("delete from encoding_guide")
        sql_manager.conn.commit()
    except:
        pass

    columns_name = ['status_type', 'num_reactions',
                    'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows',
                    'num_hahas', 'num_sads', 'num_angrys']

    df = pd.read_sql_query("select * from information", con=sql_manager.conn)
    hist2d(df, columns_name)
    # density(df, columns=columns_name)
    # point_point_plot(df, columns_name)
    # hex_bin(df, columns_name)
    # cumsum(df=df)
    # diff(df=df, cols=columns_name)
    # boxes(columns_name=columns_name, df=df)
    # pie_plots(columns_name=columns_name)
    # corr(df=df)
