# Packages / libraries
from flask import Flask, render_template, request
from flask_menu import Menu, register_menu
from flask_wtf.csrf import CSRFProtect
from flask_wtf import csrf
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import os #provides functions for interacting with the operating system
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, confusion_matrix, accuracy_score, classification_report, log_loss
from math import sqrt
from sklearn.metrics import silhouette_samples, silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.cluster import KMeans, k_means
from barchart import count_clusters
from clsuterByAge import stacked_bar_chart
from clusterPerJob import stacked_bar_chart_job
from clusterByMarital import stacked_bar_chart_marital
from clusterByEducation import stacked_bar_chart_education
from clusterByDefault import stacked_bar_chart_default
from clusterByHousing import stacked_bar_chart_housing
from clusterByLoanStatus import stacked_bar_chart_y
from clusterTree import create_cluster_tree
from parallelCoordinatesPlot import parallel_coordinates_plot
# To install sklearn type "pip install numpy scipy scikit-learn" to the anaconda terminal

# To change scientific numbers to float
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Increases the size of sns plots
sns.set(rc={'figure.figsize':(8,6)})

app = Flask(__name__)
secret_key = os.urandom(16).hex()
app.config['SECRET_KEY'] = secret_key
csrf = CSRFProtect(app)

class MyForm(FlaskForm):
    input_number = 0
    input_number = StringField('Input Number', validators=[DataRequired()])
    submit = SubmitField('Submit')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = MyForm()
    if form.validate_on_submit():
        input_number = form.input_number.data
        print(f"Input Number: {input_number}")
        
    # Specify the path to the CSV file
    csv_file_path = '/GUI/DataSets/data_with_clusters.csv'

    raw_csv_file_path = '/GUI/DataSets/raw_data_with_clusters.csv'

    raw_df = pd.read_csv(raw_csv_file_path)


    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    df = df.drop(columns=['Unnamed: 0'])

    # Bar chart for clusters
    plot_count_clusters = count_clusters(raw_df)

    # Stacked Bar Chart for Age by Clsuters
    plot_count_clusters_by_age = stacked_bar_chart(raw_df)

    # Stacked Bar Chart for Job by Clusters
    plot_count_clusters_by_job = stacked_bar_chart_job(raw_df)

    # Stacked Bar Chart for Marital by Clusters
    plot_count_clusters_by_marital = stacked_bar_chart_marital(raw_df)

    # Stacked Bar Chart for Education by Clusters
    plot_count_clusters_by_education = stacked_bar_chart_education(raw_df)

    # Stacked Bar Chart for Default by Clusters
    plot_count_clusters_by_default = stacked_bar_chart_default(raw_df)

    # Stacked Bar Chart for Housing by Clusters
    plot_count_clusters_by_housing = stacked_bar_chart_housing(raw_df)

    # Stacked Bar Chart for LoanStatus by Clusters
    plot_count_clusters_by_y = stacked_bar_chart_y(raw_df)

    # Cluster Trees
    plot_cluster_tree = create_cluster_tree(raw_df)

    #Parallel Coordinates
    plot_parallel = parallel_coordinates_plot(df)

    return render_template('index.html', 
    plot_parallel_html=plot_parallel, 
    plot_count_clusters_html=plot_count_clusters, 
    plot_count_cluster_by_age_html=plot_count_clusters_by_age, 
    plot_count_clusters_by_job_html = plot_count_clusters_by_job,
    plot_count_clusters_by_marital_html = plot_count_clusters_by_marital,
    plot_count_clusters_by_education_html = plot_count_clusters_by_education,
    plot_count_clusters_by_default_html = plot_count_clusters_by_default,
    plot_count_clusters_by_housing_html = plot_count_clusters_by_housing,
    plot_count_clusters_by_y_html = plot_count_clusters_by_y,
    plot_cluster_tree_html = plot_cluster_tree,
    form=form
    )


if __name__ == '__main__':
    app.run(debug=True)