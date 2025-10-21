import streamlit as st
import adaptive_fractionation_overlap as af
import numpy as np
from adaptive_fractionation_overlap.constants import (
    SLOPE, 
    INTERCEPT, 
    DEFAULT_MIN_DOSE, 
    DEFAULT_MAX_DOSE, 
    DEFAULT_MEAN_DOSE,
    DEFAULT_DOSE_STEPS, 
    DEFAULT_NUMBER_OF_FRACTIONS,
    DEFAULT_ALPHA,
    DEFAULT_BETA
)

st.set_page_config(layout="wide")
st.title('Overlap Adaptive Fractionation Interface')
st.markdown('## info \n This web app is supposed to be used as user interface to compute the optimal dose to be delivered in PTV-OAR-overlap adaptive fractionation if you have any questions please contact [yoel.perezhaas@usz.ch](mailto:yoel.perezhaas@usz.ch)')

@st.cache_data
def convert_df(df):
    """converts a dataframe to csv

    Args:
        df (pd.DataFrame): dataframe to be converted

    Returns:
        str: csv as a string
    """
    return df.to_csv(index=False).encode('utf-8')

st.header('User Input')
function = st.radio('Type of adaptive fractionation', ['actual fraction calculation','precompute plan','full plan calculation'])


left, right = st.columns(2)  
with left:
    fractions = st.text_input('total number of fractions', str(DEFAULT_NUMBER_OF_FRACTIONS), help = 'insert the number of total fractions in the treatment')
    overlaps_str = st.text_input('observed overlap volumes in cc separated by spaces', help = 'insert ALL observed overlaps for the patient. for a full plan at least (number of fractions - 1) volumes are required')
    actual_fraction = st.text_input('number of actual fraction', disabled = (function =='full plan calculation'), help = 'the actual fraction number is only needed for the actual fraction calculation')
with right:
    minimum_dose = st.text_input('minimum dose', str(DEFAULT_MIN_DOSE), help = 'insert the minimum dose in Gy')
    maximum_dose = st.text_input('maximum dose', str(DEFAULT_MAX_DOSE), help = 'insert the maximum dose in Gy')
    mean_dose = st.text_input('mean dose to be delivered over all fractions', str(DEFAULT_MEAN_DOSE), help = 'insert mean dose in Gy')
    dose_steps = st.text_input('difference between the deliverable doses', str(DEFAULT_DOSE_STEPS), help= 'e.g. 0.5 leads to the dose steps of 7.5,8.0,8.5,... any other dose is not allowed')
    accumulated_dose = st.text_input('accumulated physical dose in previous fractions', disabled = (function =='full plan calculation'), help = 'the accumulated dose is only needed in the actual fraction calculation set to 0 if actual fraction is 1')

st.header('Results')


if st.button('compute optimal dose', help = 'takes the given inputs from above to compute the optimal dose'):
    overlaps_str = overlaps_str.split()
    overlaps = [float(i) for i in overlaps_str]
    if function == 'actual fraction calculation':
        [policies, policies_overlap, volume_space, physical_dose, penalty_added, values, dose_space, probabilities, final_penalty] = af.adaptive_fractionation_core(fraction = int(actual_fraction),volumes = np.array(overlaps), accumulated_dose = float(accumulated_dose), number_of_fractions = int(fractions), min_dose = float(minimum_dose), max_dose = float(maximum_dose), mean_dose = float(mean_dose), dose_steps = float(dose_steps))
        left2, right2 = st.columns(2)
        with left2:
            actual_value = 'Goal can not be reached' if final_penalty <= -100000000000 else str(np.round(final_penalty,1)) + 'ccGy'
            st.metric(label="optimal dose for actual fraction", value= str(physical_dose) + 'Gy', delta = (physical_dose - float(mean_dose)))
            st.metric(label="expected final penalty from this fraction", value = actual_value)
            if final_penalty <= -100000000000:
                st.write('the minimal dose is delivered if we overdose, the maximal dose is delivered if we underdose')
                st.markdown('by taking this approach and delivering the minimum/maximum dose in each fraction we miss the goal by:')
                st.metric(label= '', value = str(float(accumulated_dose) + float(physical_dose)*(int(fractions) - int(actual_fraction) + 1) - float(mean_dose) * int(fractions)))
        with right2:
            st.pyplot(af.actual_policy_plotter(policies_overlap,volume_space,probabilities))
        with st.expander('see Analytics'):
            st.header('Analytics')
            if int(actual_fraction) != int(fractions):
                figure = af.analytic_plotting(int(actual_fraction),int(fractions),values, volume_space, dose_space)    
                st.pyplot(figure)
                st.write('The figures above show the value function for each future fraction. These functions help to identify whether a potential mistake has been made in the calculation.')
    elif function == 'precompute plan':
        with st.spinner('computing plans. This might take up to 2-3 minutes'):
            volume_x_dose, volumes_to_check, predicted_policies = af.precompute_plan(fraction = int(actual_fraction), volumes = np.array(overlaps), accumulated_dose = float(accumulated_dose), number_of_fractions = int(fractions), min_dose = float(minimum_dose), max_dose = float(maximum_dose), mean_dose = float(mean_dose), dose_steps = float(dose_steps))
        csv = convert_df(volume_x_dose)
        left2, right2 = st.columns(2)  
        with left2:
            st.dataframe(data = volume_x_dose,height = 600, hide_index = True)
            st.download_button(
            "Download table",
            csv,
            "precomputed_plans.csv",
            "text/csv",
            key='data')
        with right2:
            st.pyplot(af.actual_policy_plotter(predicted_policies,volumes_to_check))   
    else:
        [physical_doses, accumulated_doses, total_penalty] = af.adaptfx_full(volumes = np.array(overlaps), number_of_fractions = int(fractions), min_dose = float(minimum_dose), max_dose = float(maximum_dose), mean_dose = float(mean_dose), dose_steps = float(dose_steps))
        cols = st.columns(int(fractions))
        for i, col in enumerate(cols):
            with col:
                st.metric(label=f"**overlap**", value=f"{overlaps[-(int(fractions) - i)]}cc")
                st.metric(label=f"**fraction {i + 1}**", value=f"{physical_doses[i]}Gy", delta=np.round(physical_doses[i] - float(mean_dose),1))
        st.header('Plan summary')
        st.markdown('The adaptive plan achieved a total penalty of:')
        st.metric(label = "penalty", value = str(total_penalty) + 'ccGy', delta = np.round(total_penalty + af.penalty_calc_single(float(mean_dose),6,np.array(overlaps[-int(fractions):]), intercept = INTERCEPT, slope = SLOPE).sum(),2))
        st.markdown('The arrow shows the comparison to standard fractionation, i.e. (number of fractions x mean dose). A green arrow shows an improvement.')

