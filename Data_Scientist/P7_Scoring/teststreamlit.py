import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from preprocess import *
from lgbm_shap import *

import os


path = Path(__file__).parent
path_data = path.joinpath('Data')
# st.write(os.getcwd(), path)

st.set_page_config(page_title='PRET A DEPENSER - Scoring Client', layout='wide')


@st.cache(allow_output_mutation =True)
def load_data():

    train_df = pd.read_csv(path/'proc_train_df.csv', index_col=0)
    test_df = pd.read_csv(path/'test_preds.csv', index_col=0).reset_index(drop=True)
    contribs_df = np.array(pd.read_csv(path/'test_contribs.csv', index_col=0))
    app_test_df = pd.read_csv(path_data/'application_test.csv')
    app_train_df = pd.read_csv(path_data/'application_train.csv')

    return train_df, test_df, contribs_df, app_train_df, app_test_df


train, test, contribs, app_train, app_test = load_data()


with st.sidebar:
    image = Image.open(path/'place_marche_logo.png')
    st.image(image)
    st.header("Scoring Client")

    app_id = st.selectbox('Please select application ID', test['SK_ID_CURR'])
    index = test.loc[test['SK_ID_CURR'] == app_id].index[0]

tab1, tab2, tab3 = st.tabs(["SCORING   ", "PERSONAL   ", "INCOME & EMPLOYMENT"])

with tab1:

    pred_value = test.loc[test['SK_ID_CURR'] == app_id, 'PREDS']
    st.header(f"Application Status")
    if float(pred_value) < 0.3:
        st.success(body="Approved"+ "✅")
    else:
        st.error(body="Rejected"+"\U0000274C")

    st.markdown('This section provides information about the client default'
                    ' score and the current credit application\n___')

    col1, col2, col3 = st.columns([2, 1, 1])
    # building gauge
    with col1:

        st.subheader("Default Risk")

        # gauge steps parameters
        cols = ["#267302", "#65A603", "#65A603", "#F29F05", "#F28705",
                "#F27405", "#F25C05", "#F24405", "#F21D1D", "#BF0413"]
        ranges = [[i / 10, (i + 1) / 10] for i in list(range(0, 10))]
        steps = [dict(zip(['range', 'color', 'thickness', 'line'],
                          [range_size, col, 0.66, {'color': "white", "width": 2}]))
                 for range_size, col in zip(ranges, cols)]

        fig = go.Figure(go.Indicator(
           domain={'row': 0, 'column': 0},
           value=float(pred_value),
           number={"font": {"color": "#404040"}},
           mode="gauge+number+delta",
           # title={'text': "Default Risk", 'font': {'size': 50}, 'align': 'center'},
           delta={'reference': 0.3, 'decreasing': {'color': '#3D9970'}, 'increasing': {'color': '#FF4136'}},
           gauge={'axis': {'range': [None, 1]},
                  'bgcolor': '#F2F2F2',
                  #  'shape':'bullet',
                  'bar': {'color': "#404040", 'thickness': 0.41},
                  'borderwidth': 0,
                  'steps': steps,
                  'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.86, 'value': 0.3}}))

        fig.update_layout(
                     margin={'t': 0, 'b': 0},
                     )

        st.plotly_chart(fig, use_container_width=True, sharing="streamlit")
    # display credit amount
    with col2:
        st.subheader("credit amount")

        credit_value = test.loc[test['SK_ID_CURR'] == app_id, 'APPLI_AMT_CREDIT']
        median_credit = train['APPLI_AMT_CREDIT'].median()

        fig3 = go.Figure(go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=float(credit_value),
            number={"font": {"color": "#404040"}},
            mode="number+delta",
            # title={'text': "Credit amount", 'font': {'size': 50}, 'align': 'center'}
            delta={'reference': median_credit, 'decreasing': {'color': '#3D9970'}, 'increasing': {'color': '#FF4136'}}
                         ))

        fig3.update_layout(
            # title={"y": 1, 'yanchor': 'top'},
            margin={'t': 90, 'b': 0},
        )

        st.plotly_chart(fig3, use_container_width=True, sharing="streamlit")
    # diplay Annuity amount
    with col3:
        st.subheader("Annuity")
        annuity_value = test.loc[test['SK_ID_CURR'] == app_id, 'APPLI_AMT_ANNUITY']
        median_annuity = train['APPLI_AMT_ANNUITY'].median()

        fig4 = go.Figure(go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=float(annuity_value),
            number={"font": {"color": "#404040"}},
            mode="number+delta",
            # title={'text': "Credit amount", 'font': {'size': 50}, 'align': 'center'},
            delta={'reference': median_annuity, 'decreasing': {'color': '#3D9970'}, 'increasing': {'color': '#FF4136'}}
        ))

        fig4.update_layout(
           #     title={"y": 1, 'yanchor': 'top'},
           margin={'t': 90, 'b': 0},
        )

        st.plotly_chart(fig4, use_container_width=True, sharing="streamlit")

    st.markdown('\n___')
    st.subheader("Scoring Analysis")
    st.markdown('This section provides explanation about the client default score\n___')

    col4, col5 = st.columns([1, 1])

    with col4:
        feats = list(train.columns[2:])
        shap_values, exp_values, feat_values, feat_names = shap_viz_prep(contribs, feats, test)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('score_cmap', colors=cols, N=100)
        fig, ax = plt.subplots(figsize=(5,10))
        st.pyplot(shap.decision_plot(exp_values[index],
                                     shap_values[index],
                                     feat_values[index],
                                     feature_names=feat_names,
                                     link='logit',
                                     plot_color=cmap,
                                     auto_size_plot=False,
                                     show=False),
                  clear_figure=True,
                  use_container_width=True)
    with col5:
        image = Image.open(path / 'shap_summary.png')
        st.image(image)



    st.markdown('\n___')
    st.subheader("Scoring in depth analysis")
    st.markdown('Get additionnal insight about the client default score\n___')

    col4, col5 = st.columns([1, 1])

    with col4:
        st.subheader("Main variables distributions")

        exp_variables = st.multiselect('Select variables for scoring explanation',
                                       feats)

        var_df = train[feats + ["TARGET"]]

        def plot_main_variables(variable):
            # if variable in exp_variables:
            var_value = float(test.loc[test['SK_ID_CURR'] == app_id, variable])
            fig_source, ax = plt.subplots()
            for location in ['top', 'right']:
                    ax.spines[location].set_visible(False)
            ax.axvline(x=var_value, ymin=0, ymax=1, color="black")
            sns.kdeplot(
                    data=var_df, x=variable, hue="TARGET",
                    fill=True, common_norm=False, palette=["#267302", "#BF0413"],
                    alpha=.5, linewidth=0, ax=ax
            )
            ax.legend(labels=[app_id, 1, 0], frameon=0, loc="best")

            return fig_source

        for var in exp_variables:
            st.pyplot(plot_main_variables(var), clear_figure=True)

    with col5:
        st.subheader("Main variables correlations")
        exp_variables2 = st.multiselect('Select two variables for scoring explanation',
                                        feats)
        @st.cache
        def plot_bivariate(var0, var1):

            df = train[[var0, var1, "TARGET"]].copy()
            df.loc[:, "TARGET"] = df.loc[:, "TARGET"].astype('object')
            df_app_id = test.loc[test["SK_ID_CURR"] == app_id, [var0, var1]]

            fig_biv = px.scatter(df.sample(2500), x=var0, y=var1, color="TARGET",
                                 opacity=0.3,
                                 color_discrete_map={0: "#267302", 1: "#BF0413"},
                                 trendline='ols'
                             )
            fig_biv.update_traces(marker={"symbol": 0, "size": 6})

            fig_biv.add_trace(
                go.Scatter(
                    x=df_app_id[var0],
                    y=df_app_id[var1],
                    hovertemplate=f"{var0}:" + " %{x}<br>"
                                  + f"{var1}:" + ": %{y}<br>"
                                  + '<b>%{text}</b>',
                    text=["Client"],
                    marker={"color": "darkorange", "size": 15, "symbol": 0},
                    name=app_id)
            )

            fig_biv.update_layout({
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
            })
            return fig_biv

        if len(exp_variables2) != 2:
            st.markdown("Please select exactly two variables to see the plot")
        else:
            variable0 = exp_variables2[0]
            variable1 = exp_variables2[1]
            fig_b = plot_bivariate(variable0, variable1)

            st.plotly_chart(fig_b, use_container_width=False, sharing="streamlit")


with tab2:
    st.header("Background & Personnal Information")
    st.markdown('This section provides information about the client personnal information\n___')

    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    # display customer age
    age = -test.loc[test['SK_ID_CURR'] == app_id, "APPLI_DAYS_BIRTH"].values[0]
    years = age/365
    months = int((age % 365)/30)

    birth_df = train[["APPLI_DAYS_BIRTH", "TARGET"]].copy()

    birth_df["APPLI_YEARS_BIRTH"] = -birth_df["APPLI_DAYS_BIRTH"] / 365
    median_age = birth_df["APPLI_YEARS_BIRTH"].median()
    age_delta = round(years-median_age, 2)

    col1.metric(label="Age", value=f"{int(years)} years and {months} months", delta=age_delta)

    # display customer NAME_EDUCATION_TYPE & NAME_HOUSING_TYPE
    ed_fam_df = app_test[['SK_ID_CURR', "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "CNT_CHILDREN"]].copy()
    education = ed_fam_df.loc[ed_fam_df['SK_ID_CURR'] == app_id, "NAME_EDUCATION_TYPE"].values[0]

    col4.metric(label="Education", value=f"{education}", delta=None)
    # col4.metric(label="housing", value=f"{housing}", delta=None)

    # display customer FAMILY_STATUS
    fam_status = ed_fam_df.loc[ed_fam_df['SK_ID_CURR'] == app_id, "NAME_FAMILY_STATUS"].values[0]
    col2.metric(label="Family Status", value=f"{fam_status}", delta=None)

    # display customer CNT_CHILDREN
    children = int(ed_fam_df.loc[ed_fam_df['SK_ID_CURR'] == app_id, "CNT_CHILDREN"].values[0])
    col3.metric(label="Children", value=f"{children}", delta=None)

    st.markdown('\n___')


    fig = px.histogram(birth_df, x="APPLI_YEARS_BIRTH", color="TARGET",
                       title="Client Age Distribution",
                       marginal="box",  # or violin, rug
                       hover_data=["APPLI_YEARS_BIRTH", 'TARGET'],
                       opacity=0.5,
                       color_discrete_sequence=["#BF0413", "#267302"],
                       histnorm='probability density',
                       nbins=50,
                       labels={"APPLI_YEARS_BIRTH": 'Age', "TARGET": 'Default Status'})
    fig.update_layout({
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
    })
    fig.add_vline(x=years, line_width=3,  line_color="black", opacity=0.8)
    st.plotly_chart(fig, use_container_width=False, sharing="streamlit")

    ed_df = app_train[["NAME_EDUCATION_TYPE", "TARGET"]].copy()
    fig_ed = px.histogram(ed_df, y="NAME_EDUCATION_TYPE",
                          color="TARGET",
                          title="Client Education",
                          hover_data=["NAME_EDUCATION_TYPE", 'TARGET'],
                          opacity=0.5,
                          color_discrete_map={0: "#267302", 1: "#BF0413"},
                          category_orders={"NAME_EDUCATION_TYPE": ['Secondary / secondary special',
                                                                   'Higher education', 'Incomplete higher',
                                                                   'Lower secondary', 'Academic degree']},
                          barmode='group',
                          histnorm='percent',
                          nbins=50,
                          labels={"NAME_EDUCATION_TYPE": 'Education', "TARGET": 'Default Status'})
    fig_ed.update_layout({
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
    })
    st.plotly_chart(fig_ed, use_container_width=False, sharing="streamlit")


with tab3:
    st.header("Income and employement history")
    st.markdown('This section provides information about the client income and employment history\n_____')

    # client income
    income = test.loc[test['SK_ID_CURR'] == app_id, "APPLI_AMT_INCOME_TOTAL"].values[0]

    income_df = train.loc[train["APPLI_AMT_INCOME_TOTAL"] < 350000, ["APPLI_AMT_INCOME_TOTAL", "TARGET"]]

    fig5 = px.histogram(income_df, x="APPLI_AMT_INCOME_TOTAL", color="TARGET",
                        title="Income Distribution for client earning less than 350K",
                        marginal="box",  # or violin, rug
                        hover_data=["APPLI_AMT_INCOME_TOTAL", 'TARGET'],
                        opacity=0.5,
                        color_discrete_sequence=["#BF0413", "#267302"],
                        histnorm='percent',
                        nbins=100,
                        labels={"APPLI_AMT_INCOME_TOTAL": 'Income', "TARGET": 'Default Status'},
                        barmode='group',
                        width=1300,
                        height=600)
    fig5.update_layout({
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
            })

    fig5.add_vrect(x0=int(income), x1=5000+income, line_width=1,
                   # line_color="black",
                   fillcolor="grey",
                   opacity=0.1)

    fig5.add_annotation(
        x=int(income)+2500,
        y=15,
        xref="x",
        yref="y",
        text=app_id,
        showarrow=True,
        font=dict(
            # family="Courier New, monospace",
            size=12,
            color="black"
        ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=0,
        ay=-30,
        hovertext='Client income range'
    )
    st.plotly_chart(fig5, use_container_width=True, sharing="streamlit")
