# -*- coding: utf-8 -*-
"""
Created on January 20 2023

@author: Khuong LE NGUYEN / University of Canberra
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd


@st.experimental_memo
# function to convert to superscript
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans("".join(normal), "".join(super_s))
    return x.translate(res)


# function to convert to subscript
def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans("".join(normal), "".join(sub_s))
    return x.translate(res)


# loading the saved models

XGBoost_6Var = pickle.load(open("XGBoost_6inputs.sav", "rb"))
XGBoost_4Var = pickle.load(open("XGBoost_745.sav", "rb"))
CatBoost_6Var = pickle.load(open("CatBoost_6inputs.sav", "rb"))
CatBoost_4Var = pickle.load(open("CatBoost_745.sav", "rb"))
GBRT_6Var = pickle.load(open("GBRT_6inputs.sav", "rb"))
GBRT_4Var = pickle.load(open("GBRT_745.sav", "rb"))

# sidebar for navigation
# Icons: https://icons.getbootstrap.com/

with st.sidebar:
    selected = option_menu(
        "Flat Slabs - Punching Shear Prediction",
        [
            "Project Description",
            "6 Input features",
            "4 Input features",
            "Main Results",
        ],
        icons=[
            "server",
            "activity",
            "activity",
            "palette-fill",
        ],
        default_index=0,
    )
    st.write(
        "Contact: K.Le-Nguyen \n\n University of Transport Technology, Vietnam \n\nUniversity of Canberra, Australia \n\nkhuongln@utt.edu.vn | khuong.lenguyen@canberra.edu.au"
    )
    # st.sidebar.markdown('<a href="mailto:khuong.lenguyen@canberra.edu.au">Contact us!</a>', unsafe_allow_html=True)

# Project Description
if selected == "Project Description":
    # page title
    st.title("Project Description")
    st.image(
        "Process_Description.png",
        caption="Research process",
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )
    st.write(
        """This study investigates the application of ensemble learning models to predict the punching shear strength (PSS) of flat slabs without transverse reinforcement, utilising two databases with differing numbers of input variables. The first database contains 522 samples with six input variables, while the second database includes 745 samples with four essential input variables. The ensemble learning models – Random Forest, AdaBoost, Light GBM, GBRT, CatBoost, and XGBoost - are evaluated using a Bayesian optimisation process combined with 10-fold cross-validation to determine the best-performing model. Results indicate that the CatBoost model has the highest performance, achieving an R2 score of 0.97 for both databases. The highest-performing model is evaluated against design codes and empirical equations in the literature, demonstrating superior prediction accuracy. Model performances are further investigated through Monte Carlo simulations with 2700 simulations. Finally, a user-interface application is developed for estimating the punching shear of the slab concrete, representing a significant development in predicting the PSS of flat slabs in construction structures.
    """
    )
    st.markdown(
        "**However, it is important to use realistic values for inputs, as using unrealistic values may result in poor predictions. Users are therefore suggested to use the realistic component of input features.**"
    )
    # st.subheader("Research Flowchart")
    # st.image(
    #     "Process_Description.png",
    #     caption="Research Flowchart",
    #     width=None,
    #     use_column_width=None,
    #     clamp=False,
    #     channels="RGB",
    #     output_format="auto",
    # )
    # st.subheader("Data Decription")
    # st.markdown(
    #     "There are four input features categories, namely, geometric dimensions, reinforcement ar-rangements, material properties, and applied axial load. The detailed input features are the height _hw_, length _lw_, web thickness _tw_, flange length _bf_, flange thickness _tf_, concrete compres-sive strength _fck_, vertical web reinforcement ratio _ρv_ and strength _fyv_, horizontal web rein-forcement ratio _ρh_ and strength _fyh_, longitudinal reinforcement ratio _ρL_ and strength _fyL_, and, finally, the applied axial load _P_. The output is simply the shear strength _Vn_."
    # )

    pd.options.display.float_format = "{:,.2f}".format
    df = pd.read_csv("data_Slab522.csv")
    df = df.applymap("{0:.2f}".format)
    # page title
    st.title("Original Database with 6 input features")
    st.dataframe(df)

# Prediction Page with 6 inputs
if selected == "6 Input features":

    # page title
    st.title("Prediction from data with 6 input features")

    # getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        depth = st.slider(
            "Effective flexural depth: d(mm)",
            min_value=30,
            max_value=500,
            value=120,
        )

    with col2:
        shear_span = st.slider(
            "Shear span: a(mm)",
            min_value=40,
            max_value=2500,
            value=1000,
        )

    with col1:
        column_width = st.slider(
            "Column width: c(mm)",
            min_value=40,
            max_value=750,
            value=250,
        )

    with col2:
        ro = st.slider(
            "Reinforcement ratio : ro (%)",
            min_value=0.2,
            max_value=4.0,
            value=1.2,
        )

    with col1:
        fy = st.slider(
            "Yield strength of reinforcement: fy(MPa)",
            min_value=250,
            max_value=750,
            value=400,
        )

    with col2:
        fck = st.slider(
            "Concrete compressive strength: fck(MPa)",
            min_value=10,
            max_value=110,
            value=35,
        )

    # code for Prediction
    # diab_diagnosis = ''
    ccstrength = ""

    # creating a button for Prediction
    st.subheader("Punching Shear Strength of RC Slab defined by 6 input features")

    input_data = np.array([depth, shear_span, column_width, ro, fy, fck]).reshape(1, -1)
    ccsCatBoost = CatBoost_6Var.predict(input_data)
    ccsXGB = XGBoost_6Var.predict(input_data)
    ccsGBRT = GBRT_6Var.predict(input_data)

    # Calculate the average prediction value
    avg_ccs = (ccsCatBoost + ccsXGB) / 2.0

    str1 = "Model CatBoost: {} MPa \n".format(np.round(ccsCatBoost, 2))
    str2 = "Model XGBoost: {} MPa \n".format(np.round(ccsXGB, 2))
    # str3 = "Model GBRT: {} MPa \n".format(np.round(ccsGBRT, 2))
    str3 = "Average Prediction: {} MPa \n".format(np.round(avg_ccs, 2))
    str4 = "Please note that for accurate predictions, it is crucial to use realistic input values."

    if st.button("Prediction by ML"):
        st.success(str1 + "\n" + str2 + "\n" + str3 + "\n" + str4)

# Prediction Page with 4 inputs
if selected == "4 Input features":

    # page title
    st.title("Prediction from data with 4 input features")

    # getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        depth = st.slider(
            "Effective flexural depth: d(mm)",
            min_value=30,
            max_value=500,
            value=120,
        )

    with col2:
        column_width = st.slider(
            "Column width: c(mm)",
            min_value=40,
            max_value=750,
            value=200,
        )

    with col1:
        ro = st.slider(
            "Reinforcement ratio : ro (%)",
            min_value=0.2,
            max_value=5.0,
            value=1.2,
        )

    with col2:
        fck = st.slider(
            "Concrete compressive strength: fck(MPa)",
            min_value=10,
            max_value=110,
            value=35,
        )

    ccstrength = ""

    # creating a button for Prediction
    st.subheader("Punching Shear Strength of RC Slab defined by 4 input features")

    input_data = np.array([depth, column_width, ro, fck]).reshape(1, -1)
    ccsCatBoost = CatBoost_4Var.predict(input_data)
    ccsXGB = XGBoost_4Var.predict(input_data)
    # ccsGBRT = GBRT_4Var.predict(input_data)

    # Calculate the average prediction value
    avg_ccs = (ccsCatBoost + ccsXGB) / 2.0

    str1 = "Model CatBoost: {} MPa \n".format(np.round(ccsCatBoost, 2))
    str2 = "Model XGBoost: {} MPa \n".format(np.round(ccsXGB, 2))
    # str3 = "Model GBRT: {} MPa \n".format(np.round(ccsGBRT, 2))
    str3 = "Average Prediction: {} MPa \n".format(np.round(avg_ccs, 2))
    str4 = "Please note that for accurate predictions, it is crucial to use realistic input values."

    if st.button("Prediction by ML"):
        st.success(str1 + "\n" + str2 + "\n" + str3 + "\n" + str4)

# Main Results
if selected == "Main Results":

    # page title
    st.title("Main Results")
    st.image(
        "CorrelationMatrix.png",
        caption="Correlation matrix of the variables",
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )
    st.image(
        "Performance_Default Hyperparameters.png",
        caption="Performance of the models with default hyperparameters",
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )

    st.image(
        "Performance_Compared.png",
        caption="Comparison of the performance of the models with default and optimal hyperparameters",
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )

    st.image(
        "Histograms.png",
        caption="Histograms for the ratio of the punching shear strength between the model predictions and the experimental data",
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )
