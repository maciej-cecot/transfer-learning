import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from bokeh.io import output_notebook
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import HoverTool
from ipywidgets import interact


COLOR_PALETTE = sns.color_palette("hls", 10)

def scatterplot_vis(df, **kwargs):
    plot_width = kwargs.get('plot_width',500)
    plot_height = kwargs.get('plot_height',500)
    size = kwargs.get('size',8)
    
    hover = HoverTool(
        tooltips="""
        <div>
            <div>
                
                <img
                    src="@img_filepath" height="64" alt="@img_filepath" width="64"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            
        </div>
        """
    )

    p = figure(plot_width=plot_width, plot_height=plot_height, 
               toolbar_location = 'right',
               tools='pan,box_zoom,wheel_zoom,reset,resize')
    p.add_tools(hover)
    df['label_color'] = df['label'].apply(lambda x: COLOR_PALETTE.as_hex()[int(x)])
    source = ColumnDataSource(df)
    circles = p.circle('x', 'y', size=size, source=source)
    circles.glyph.fill_color = 'label_color'
    output_notebook()
    show(p)

def tsne_fit(img_features, img_labels, img_paths):
    tsne_features = TSNE(n_components=2, learning_rate=100).fit_transform(img_features)
    scaler = MinMaxScaler()
    tsne_features = scaler.fit_transform(tsne_features)
    df = pd.DataFrame(tsne_features, columns=['x','y'])
    df['img_filepath'] = img_paths
    df['label'] = img_labels
    return df

def plot_tsne_embeddings(features, labels, n):
    x_data = features[:n]
    y_data = labels[:n]
    tsne_features = TSNE(n_components=2, learning_rate=100).fit_transform(x_data)
    plt.figure(figsize=(12,8))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y_data, cmap=plt.cm.get_cmap("jet",10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.show()
