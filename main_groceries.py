import streamlit as st
import pandas as pd
import pickle
import re
import io
import base64
import csv
import time
import urllib
import string
import requests
import matplotlib
import shutil
import plotly.graph_objects as go
import plotly.express as px
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from io import BytesIO
from matplotlib.colors import Normalize
from numpy.random import rand
import matplotlib.cm as cm
from streamlit import caching
from datetime import datetime
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
my_cmap = cm.get_cmap('jet')
my_norm = Normalize(vmin=0, vmax=8)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def get_image_download_link(img):
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}" download="result.jpg">Download JPG Data</a>'
	return href

def plot_diagram(x_label, y_label, rules, title):
    fig = px.scatter(rules, x=x_label, y=y_label,
    title=title)
    return st.plotly_chart(fig)

def input_choice_flow():
    input_choices_activities = ["Upload Transactions File"] 

    input_choice = st.sidebar.selectbox("Select Choices", input_choices_activities)

    metric_activities1 = ["support", "confidence", "lift", "leverage", "conviction"]
    metric_activities = ["support", "confidence"]
    metric_choice = st.sidebar.selectbox("Select Metric", metric_activities)

    if input_choice == "Upload Transactions File":
        caching.clear_cache()
        if st.button('Download Sample File'):
            df = pd.read_csv("Groceries_dataset.csv")
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  
            href = f'<a href="data:file/csv;base64,{b64}" download="sample.csv">Download csv file</a>'
            st.markdown(href, unsafe_allow_html=True)

        data = st.file_uploader("Upload File", type=["csv"])

        if data is not None:

            df = pd.read_csv(data)

            fig = go.Figure(data=[go.Table(header=dict(values=['Transaction_id', 'Date', 'Item']),
                    cells=dict(values=[df.iloc[:,0],df.iloc[:, 1], df.iloc[:,2]]))
                        ])
            st.plotly_chart(fig)
            df = df.set_index(['date'])

            # if st.button("Process Data"):

            st.info("Data Status: ")
            st.success("Data Shape: "+str(df.shape))

            describe = df.describe()
            describe_detail = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%","Max"]

            fig_describe = go.Figure(data=[go.Table(header=dict(values=['Info','Transaction']),
                cells=dict(values=[describe_detail,round(describe, 2)]))
                    ])
            st.plotly_chart(fig_describe)

            transactions = [a[1]['item'].tolist() for a in list(df.groupby(['transaction_id','date']))]
            te=TransactionEncoder()
            te_data=te.fit(transactions).transform(transactions)
            transactions = pd.DataFrame(te_data, columns=te.columns_)
            pf = transactions.describe()
            f = pf.iloc[0]-pf.iloc[3]
            a = f.tolist()
            b = list(f.index)
            item = pd.DataFrame([[a[r],b[r]]for r in range(len(a))], columns=['Count','Item'])
            item = item.sort_values(['Count'], ascending=False).head(50)

            min_support_slider, x,min_threshold_slider = st.beta_columns(3)

            with min_support_slider:
                support_slider = st.slider("min_support (Apriori):",0.001, 0.5, step=0.001)
            with x:
                pass
            with min_threshold_slider:
                threshold_slider = st.slider("min_threshold (Association Rules):",0.001, 0.5, step=0.001)

            min_support = support_slider
            frequent_items= apriori(transactions, use_colnames=True, min_support=min_support    )
            frequent_items['Number of Items'] = frequent_items['itemsets'].apply(lambda x:len(x))

            min_threshold = threshold_slider

            if metric_choice == "support":
                rules = association_rules(frequent_items, metric="support", min_threshold=min_threshold)
            elif metric_choice == "confidence":
                rules = association_rules(frequent_items, metric="confidence", min_threshold=min_threshold)

            rules.antecedents = rules.antecedents.apply(lambda x: next(iter(x)))
            rules.consequents = rules.consequents.apply(lambda x: next(iter(x)))
            
            rules = round(rules, 3)
            
            st.success("Apriori Rules: ")

            st.info("Itemsets: ")

            filter_itemsets = st.selectbox("Select Iterations:", ["All",2,3,4,5,6,7,8,9,10], key=1)

            if filter_itemsets == "All":
                frequent_n = frequent_items
            else:
                frequent_n = frequent_items[frequent_items['Number of Items'] == filter_itemsets]

            items = [list(x) for x in set(frequent_n['itemsets'])]
            items_fre = [x for x in frequent_n['support']]
            items_tuples = list(zip(items, items_fre))
            items_df = pd.DataFrame(items_tuples, columns=['Items', 'Support'])
            items_df['Number of Items'] = items_df['Items'].apply(lambda x : len(x))
            sorted_items = items_df.sort_values(by=['Support'], ascending=False)
            fig_item = go.Figure(data=[go.Table(header=dict(values=['Itemsets', 'Support', 'Number of Items']),cells=dict(values=[sorted_items['Items'], round(sorted_items['Support'], 3), sorted_items['Number of Items']]))])
            st.plotly_chart(fig_item)

            st.info("Support: ")

            bahan_filter_support = list(set(rules['antecedents']))
            bahan_filter_support.insert(0, "All")
            filter_box_support = st.selectbox("Select Item to Filter:", bahan_filter_support, key = 2)

            if filter_box_support == "All":       
                fig_rules = go.Figure(data=[go.Table(header=dict(values=['Before','After','Support','Confidence']),
                cells=dict(values=[rules["antecedents"], rules["consequents"], rules["support"], rules["confidence"]]))
                    ])
                st.plotly_chart(fig_rules)
            else:
                rules1 = rules[rules["antecedents"] == filter_box_support]
                fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','Support','Confidence']),
                    cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                        ])
                fig_item.update_layout(width=800)
                st.plotly_chart(fig_item)

            
            st.info("Association Rules: ")

            bahan_filter = list(set(rules['antecedents']))
            bahan_filter.insert(0, "All")
            filter_box1 , filter_box2 = st.beta_columns(2)
            with filter_box1:
                filter_box = st.selectbox("Select Before Item to Filter:", bahan_filter, key=3)
            with filter_box2:
                filter_box_after = st.selectbox("Select After Item to Filter:", bahan_filter, key=4)

            filter_box3, filter_box4 = st.beta_columns(2)
            with filter_box3:
                filter_box_len = st.selectbox("Select Before Number of Item:", ["All",2,3,4,5,6,7,8,9,10], key=5)
            with filter_box4:
                filter_box_after_len = st.selectbox("Select After Number of Item:", ["All",2,3,4,5,6,7,8,9,10], key=6)

            if metric_choice == "support":
                rules_baru = association_rules(frequent_items, metric="support", min_threshold=min_threshold)
            elif metric_choice == "confidence":
                rules_baru = association_rules(frequent_items, metric="confidence", min_threshold=min_threshold)

            rules1 = rules_baru.copy()
            rules1 = round(rules1, 3)
            rules1['antecedents'] = rules1['antecedents'].apply(lambda x: list(x))
            rules1['consequents'] = rules1['consequents'].apply(lambda x: list(x))
            rules1['antecedents_length'] = rules1['antecedents'].apply(lambda x: len(x))
            rules1['consequents_length'] = rules1['consequents'].apply(lambda x: len(x))


            if filter_box == "All" and filter_box_after == "All":
                if filter_box_len == "All" and filter_box_after_len == "All":
                    fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                        cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                            ])
                    fig_item.update_layout(width=800)
                    st.plotly_chart(fig_item)

                elif filter_box_len != "All" and filter_box_after_len == "All":
                    rules1 = rules1[(rules1['antecedents_length'] == filter_box_len)]
                    fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                        cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                            ])
                    fig_item.update_layout(width=800)
                    st.plotly_chart(fig_item)

                elif filter_box_len == "All" and filter_box_after_len != "All":
                    rules1 = rules1[(rules1['consequents_length'] == filter_box_after_len)]
                    fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                        cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                            ])
                    fig_item.update_layout(width=800)
                    st.plotly_chart(fig_item)

                elif filter_box_len != "All" and filter_box_after_len != "All":
                    rules1 = rules1[(rules1['antecedents_length'] == filter_box_len) & (rules1['consequents_length'] == filter_box_after_len) & (rules1['antecedents'] == filter_box) & (rules1['consequents'] == filter_box_after)]
                    fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                        cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                            ])
                    fig_item.update_layout(width=800)
                    st.plotly_chart(fig_item)

            elif filter_box != "All" and filter_box_after == "All":
                if filter_box_len == "All" and filter_box_after_len == "All":
                    rules1 = rules1[[filter_box in x for x in rules1['antecedents']]]
                    fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                        cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                            ])
                    fig_item.update_layout(width=800)
                    st.plotly_chart(fig_item)

                elif filter_box_len != "All" and filter_box_after_len == "All":
                    rules1 = rules1[(rules1['antecedents_length'] == filter_box_len) & ([filter_box in x for x in rules1['antecedents']])]
                    fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                        cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                            ])
                    fig_item.update_layout(width=800)
                    st.plotly_chart(fig_item)

                elif filter_box_len == "All" and filter_box_after_len != "All":
                    rules1 = rules1[(rules1['consequents_length'] == filter_box_len) & ([filter_box in x for x in rules1['antecedents']])]
                    fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                        cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                            ])
                    fig_item.update_layout(width=800)
                    st.plotly_chart(fig_item)           

                elif filter_box_len != "All" and filter_box_after_len != "All":
                    rules1 = rules1[(rules1['antecedents_length'] == filter_box_len) & (rules1['consequents_length'] == filter_box_len) & ([filter_box in x for x in rules1['antecedents']])]
                    fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                        cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                            ])
                    fig_item.update_layout(width=800)
                    st.plotly_chart(fig_item)                             


            elif filter_box == "All" and filter_box_after != "All":
                if filter_box_len == "All" and filter_box_after_len == "All":
                    rules1 = rules1[[filter_box_after in x for x in rules1['consequents']]]
                    fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                        cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                            ])
                    fig_item.update_layout(width=800)
                    st.plotly_chart(fig_item)

                elif filter_box_len != "All" and filter_box_after_len == "All":
                    rules1 = rules1[([filter_box_after in x for x in rules1['consequents']]) & (rules1['antecedents_length'] == filter_box_len)]
                    fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                        cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                            ])
                    fig_item.update_layout(width=800)
                    st.plotly_chart(fig_item)

                elif filter_box_len == "All" and filter_box_after_len != "All":
                    rules1 = rules1[([filter_box_after in x for x in rules1['consequents']]) & (rules1['consequents_length'] == filter_box_after_len)]
                    fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                        cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                            ])
                    fig_item.update_layout(width=800)
                    st.plotly_chart(fig_item)

                elif filter_box_len != "All" and filter_box_after_len != "All":
                    rules1 = rules1[([filter_box_after in x for x in rules1['consequents']]) & (rules1['antecedents_length'] == filter_box_len) & (rules1['consequents_length'] == filter_box_after_len)]
                    fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                        cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                            ])
                    fig_item.update_layout(width=800)
                    st.plotly_chart(fig_item)

            elif filter_box != "All" and filter_box_after != "All":
                rules1 = rules1[(rules1['antecedents'].apply((lambda x: filter_box in x))) & (rules1['consequents'].apply((lambda x: filter_box_after in x)))]
                if len(rules.index) > 0:
                    if filter_box_len == "All" and filter_box_after_len == "All":
                        fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                            cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                                ])
                        fig_item.update_layout(width=800)
                        st.plotly_chart(fig_item)
                    elif filter_box_len != "All" and filter_box_after_len == "All":
                        rules1 = rules1[(rules1['antecedents_length'] == filter_box_len) & (rules1['antecedents'].apply((lambda x: filter_box in x))) & (rules1['consequents'].apply((lambda x: filter_box_after in x)))]
                        fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                            cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                                ])
                        fig_item.update_layout(width=800)
                        st.plotly_chart(fig_item)

                    elif filter_box_len == "All" and filter_box_after_len != "All":
                        rules1 = rules1[(rules1['consequents_length'] == filter_box_after_len) & (rules1['antecedents'].apply((lambda x: filter_box in x))) & (rules1['consequents'].apply((lambda x: filter_box_after in x)))]
                        fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                            cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))])
                        fig_item.update_layout(width=800)
                        st.plotly_chart(fig_item)

                    elif filter_box_len != "All" and filter_box_after_len != "All":
                        rules1 = rules1[(rules1['consequents_length'] == filter_box_after_len) & (rules1['antecedents_length'] == filter_box_len) & (rules1['antecedents'].apply((lambda x: filter_box in x))) & (rules1['consequents'].apply((lambda x: filter_box_after in x)))]
                        fig_item = go.Figure(data=[go.Table(header=dict(values=['Before','After','support','confidence']),
                            cells=dict(values=[rules1["antecedents"], rules1["consequents"], rules1["support"], rules1["confidence"]]))
                                ])
                        fig_item.update_layout(width=800)
                        st.plotly_chart(fig_item)             

                else:
                    st.warning("Data tidak ditemukan")
            
            asso_tuples = list(zip(rules["antecedents"], rules["consequents"], rules["support"], rules["confidence"]))
            asso_df = pd.DataFrame(asso_tuples, columns=['Before','After','support','confidence'])


            st.success("Visualizations: ")

            visual_x, visual_y = st.beta_columns(2)

            with visual_x:
                visual_xaxis= st.selectbox("Select Metric for X", metric_activities1, key=1)

            with visual_y:
                visual_yaxis= st.selectbox("Select Metric for Y", metric_activities1, key=2)

            plot_diagram(visual_xaxis, visual_yaxis, rules, "{} V/S {}".format(visual_xaxis, visual_yaxis))

            st.success("Download Results: ")

            apriori_csv, asso_csv = st.beta_columns(2)

            with apriori_csv:
                frequent_items_csv = frequent_items.copy()
                frequent_items_csv['itemsets'] = frequent_items_csv['itemsets'].apply(lambda x : list(x))
                tmp_download_link_apriori = download_link(frequent_items_csv, 'apriori.csv', 'Click here to download Apriori!')
                st.markdown(tmp_download_link_apriori, unsafe_allow_html=True)

            with asso_csv:
                rules_csv = rules.copy()
                rules_csv = rules_csv[['antecedents', 'consequents', 'support', 'confidence']]
                tmp_download_link_asso = download_link(rules_csv, 'association.csv', 'Click here to download Association Rules!')
                st.markdown(tmp_download_link_asso, unsafe_allow_html=True)

def main():
    caching.clear_cache()
    st.title("Market Basket Analysis Demo")
    activities = ["Show Instructions","Market Basket Analysis"]
    choice = st.sidebar.selectbox("Activities", activities)

    if choice == "Show Instructions":
        filename = 'instruct1.md'
        try:
            with open(filename) as input:
                st.subheader(input.read())
        except FileNotFoundError:
            st.error('File not found')
        st.sidebar.success('To continue select Market Basket Analysis in activities.') 

    elif choice == "Market Basket Analysis":
        st.subheader("Market Basket Analysis")
        input_choice_flow()

main()