import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

train = pd.read_csv('~/PycharmProjects/Openclassrooms/Data_Scientist/P7_Scoring/proc_train_df.csv')
test = pd.read_csv('~/PycharmProjects/Openclassrooms/Data_Scientist/P7_Scoring/test_preds.csv')
app_test = pd.read_csv('~/PycharmProjects/Openclassrooms/Data_Scientist/P7_Scoring/Data/application_test.csv')
app_train = pd.read_csv("~/PycharmProjects/Openclassrooms/Data_Scientist/P7_Scoring/Data/application_train.csv")

st.set_page_config(page_title='PRET A DEPENSER - Scoring Client', layout='wide')


with st.sidebar:
    image = Image.open('C:/Users/carol/PycharmProjects/Openclassrooms/Data_Scientist/P7_Scoring/place_marche_logo.png')

    st.image(image)

    st.header("Scoring Client")

    app_id = st.selectbox('Please select application ID', test['SK_ID_CURR'])


tab1, tab2, tab3 = st.tabs(["SCORING   ", "PERSONAL   ", "INCOME & EMPLOYMENT"])

with tab1:
    st.header("Application Scoring")
    st.markdown('This section provides information about the client default'
                ' score and the current credit application\n___')

    col1, col2, col3 = st.columns([2, 1, 1])
    # building gauge
    with col1:

        st.subheader("Default Risk")

        pred_value = test.loc[test['SK_ID_CURR'] == app_id, 'PREDS']
        # gauge steps parameters
        cols = "#267302", "#65A603", "#65A603", "#F29F05", "#F28705",\
               "#F27405", "#F25C05", "#F24405", "#F21D1D", "#BF0413"
        ranges = [[i / 10, (i + 1) / 10] for i in list(range(0, 10))]
        steps = [dict(zip(['range','color','thickness', 'line'],
                          [range_size, col, 0.66, {'color': "white", "width": 2}]))
                 for range_size,col in zip(ranges, cols)]

        fig2 = go.Figure(go.Indicator(
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
                  'steps':
                      steps
                  ,

                  'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.86, 'value': 0.3}}))

        fig2.update_layout(
                     margin={'t': 0, 'b': 0},
                     )

        st.plotly_chart(fig2, use_container_width=True, sharing="streamlit")
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

    with col5:
        st.subheader("Score explanation")
        var_list = ["APPLI_EXT_SOURCE_1", "APPLI_EXT_SOURCE_2",
                    "APPLI_EXT_SOURCE_3", "APPLI_PAYMENT_RATE", "BUREAU_DAYS_CREDIT_MAX"]

        exp_variables = st.multiselect('Select variables for scoring explanation',
                                       var_list)

        var_df = train[var_list+["TARGET"]]

        def plot_ext_sources(source_num):
            if source_num in exp_variables:
                var_value = float(test.loc[test['SK_ID_CURR'] == app_id, source_num])
                fig_source, ax = plt.subplots()
                for location in ['top', 'right']:
                    ax.spines[location].set_visible(False)
                ax.axvline(x=var_value, ymin=0, ymax=1, color="black")
                sns.kdeplot(
                    data=var_df, x=source_num, hue="TARGET",
                    fill=True, common_norm=False, palette=["#267302", "#BF0413"],
                    alpha=.5, linewidth=0, ax=ax
                )
                ax.legend(labels=[app_id, 1, 0], frameon=0, loc="best")

                return st.pyplot(fig_source, clear_figure=True)

        for var in var_list:
            plot_ext_sources(var)


with tab2:
    st.header("Background & Personnal Information")
    st.markdown('This section provides information about the client personnal information\n___')

    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    # display customer age
    age = -test.loc[test['SK_ID_CURR'] == app_id, "APPLI_DAYS_BIRTH"].values[0]
    years = age/365
    months = int((age % 365)/30)

    train["APPLI_YEARS_BIRTH"] = -train["APPLI_DAYS_BIRTH"] / 365
    median_age = train["APPLI_YEARS_BIRTH"].median()
    age_delta = round(years-median_age, 2)

    col1.metric(label="Age", value=f"{int(years)} years and {months} months", delta=age_delta)

    # display customer NAME_EDUCATION_TYPE & NAME_HOUSING_TYPE
    education = app_test.loc[app_test['SK_ID_CURR'] == app_id, "NAME_EDUCATION_TYPE"].values[0]
    # housing = app_test.loc[app_test['SK_ID_CURR'] == app_id, "NAME_HOUSING_TYPE"][0]

    col4.metric(label="Education", value=f"{education}", delta=None)
    # col4.metric(label="housing", value=f"{housing}", delta=None)

    # display customer FAMILY_STATUS
    fam_status = app_test.loc[app_test['SK_ID_CURR'] == app_id, "NAME_FAMILY_STATUS"].values[0]
    col2.metric(label="Family Status", value=f"{fam_status}", delta=None)

    # display customer CNT_CHILDREN
    children = int(app_test.loc[app_test['SK_ID_CURR'] == app_id, "CNT_CHILDREN"].values[0])
    col3.metric(label="Children", value=f"{children}", delta=None)

    st.markdown('\n___')

    birth_df = train[["APPLI_YEARS_BIRTH", "TARGET"]]
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

    ed_df = app_train[["NAME_EDUCATION_TYPE", "TARGET"]]
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
