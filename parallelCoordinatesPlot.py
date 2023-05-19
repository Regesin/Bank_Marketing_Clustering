from flask import Flask, render_template
from flask_menu import Menu, register_menu
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
import mpld3
from matplotlib.backends.backend_agg import FigureCanvasAgg


def parallel_coordinates_plot(df):
    # create parallel coordinates plot
    prallel_fig = px.parallel_coordinates(df, color='cluster')

    # Convert the plot to HTML
    plot_parallel_plotly = prallel_fig.to_html(full_html=False)

    return plot_parallel_plotly