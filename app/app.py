import re
import io
import numpy
import scipy.optimize
import pandas as pd
import streamlit as st
from itertools import compress
import matplotlib.pyplot as plt

header = st.container()

with header:
    st.title("Variant Neutralization")
    st.caption(
        """
        This app plots the fraction of all variants
        of a viral protein that escape binding or neutralization by
        a polyclonal antibody mix like those found in sera for a variant
        with user specified substitutions in the viral protein.
        """
        )
 
## == Download the Data == ##

# Mutation Escape 
mut_escape_df = pd.read_csv("data/RBD_mut_escape_df_two_epitopes.csv")
# Epitope Activity 
activity_wt_df = pd.read_csv("data/RBD_activity_wt_df_two_epitopes.csv")

# Escape data must contain these columns
assert all(col in mut_escape_df.columns for col in ['epitope', 'mutation', 'escape'])

# Activity data must contain these columns
assert all(col in activity_wt_df.columns for col in ['epitope', 'activity'])

# Epitopes must be the same between the datasets 
assert all(activity_wt_df.epitope.unique() == mut_escape_df.epitope.unique())

# Get the epitopes in the data 
epitopes = activity_wt_df.epitope.unique().tolist()

with header:
    epitopes_to_plot = st.multiselect("Select Epitopes to plot:", options = activity_wt_df.epitope.unique(), default = activity_wt_df.epitope.unique())
    st.write(f"Currently showing epitopes: {', '.join(epitopes_to_plot)}")

header.write("---")

# Mutations 
mutations = mut_escape_df.mutation.unique().tolist()


## == Functions to Run App == ##

def parse_variants(variant_string): 
    """
    TODO: 
        - check the string is in proper format.
        - check that none of the mutations are at the same position.
        - check that all mutations are actually present in the protein.
    """
    if not variant_string.strip(): 
        return [] 
    else: 
        variant = variant_string.strip().split(" ")
        # Check if the mutations in the variant are valid
        if not all(mutation in mutations for mutation in variant):
            invalid_mutations = list(compress(variant, [mutation not in mutations for mutation in variant]))
            raise Exception(f"{' '.join(invalid_mutations)} not in valid mutations.")
        # Make sure that the same positon isn't repeated 
        if len({re.findall(r'\d+', mutation)[0] for mutation in variant}) != len(variant):
            raise Exception("There cannot be more than one subsitution at the same position in a single variant.")
            
        return variant


def make_sigmoid(
    variants,
    escape_df,
    activity_df,
    min_c=1e-5,
    max_c=1e5
):
    """
    TODO: 
        - make sigmoid over a range of concentrations.
        - user sets the range of concentraions. 
        - returns the wt sigmoid when variant string is empty.
    """
    
    # Get the epitopes from the activity df
    epitopes = activity_df['epitope'].tolist()
    
    # Get a geometrically spaced list of concentrations
    c = numpy.geomspace(min_c, max_c, num=1000)
    
    # Get the escape scores against individual mutations in the variant 
    beta = (
        escape_df[['epitope', 'mutation', 'escape']]
        .pivot_table(index = "mutation", columns = "epitope", values = "escape")
        .loc[variants]
    ) 
    
    # Calculate phi, the total binding activity of antibodies to epitope 
    phi = (
        activity_df
        .merge(beta
              .sum()
              .to_frame(name="beta_sum")
              .reset_index()
             )
        .assign(phi = lambda row: -row['activity'] + row['beta_sum'])
        .assign(exp_phi = lambda row: numpy.exp(-row.phi))
    )
    
    # Calculate the fraction unbound per epitope
    fraction_unbound_list = []
    for epitope in epitopes:
        Ue_vc = 1.0 / (1.0 + c * phi.query(f"epitope == '{epitope}'").exp_phi.item())
        Ue_vcs = pd.DataFrame(Ue_vc, columns = ['Ue_vc'])
        Ue_vcs['epitope'] = epitope
        Ue_vcs['concentration'] = c
        fraction_unbound_list.append(Ue_vcs)

    return pd.concat(fraction_unbound_list)


def plot_sigmoid(variant_df,
                 wt_df,
                 epitope_colors,
                 epitopes_to_plot,
                 title="",
                 by_epitope=True,
                 plot_wt=True):
    """
    TDOD: 
        - plots the sigmoid using matplotlib to look like a neut plot.
        - each epitope is given a user specified color
    """
    
    # Axis lables
    xlabel = 'concentration'
    ylabel = 'fraction infectivity'

    # Initialize plot object
    fig, ax = plt.subplots()
    fig.set_size_inches((4, 3))
    ylowerbound = -0.05
    yupperbound = 1.05
    ax.autoscale(True, 'both')
    
    # Plot multiple epitopes or combined escape
    if by_epitope: 


        for epitope, epitope_df in variant_df.groupby("epitope"):
            if epitope in epitopes_to_plot:
                epitope_df.plot(ax=ax,
                                x='concentration',
                                y='Ue_vc',
                                label=epitope,
                                linestyle='-',
                                linewidth=1,
                                color=epitope_colors[epitope])
        ax.legend(title = 'Epitope')
        
        if plot_wt:
            for epitope, epitope_df in wt_df.groupby("epitope"):
                if epitope in epitopes_to_plot:
                    ax.plot('concentration',
                            'Ue_vc',
                            data=epitope_df,
                            label = epitope,
                            linestyle='--',
                            linewidth=1,
                            color=epitope_colors[epitope])     

    else:
        ax.plot('concentration',
                'p_vc',
                data=(
                    variant_df
                    .groupby('concentration')
                    .prod()
                    .reset_index()
                    .rename(columns={"Ue_vc": "p_vc"})
                ),
                linestyle='-',
                linewidth=1,
                color="#000000")
        
        if plot_wt:
            ax.plot('concentration',
                    'p_vc',
                    data=(
                        wt_df
                        .groupby('concentration')
                        .prod()
                        .reset_index()
                        .rename(columns={"Ue_vc": "p_vc"})
                    ),
                    linestyle='--',
                    linewidth=1,
                    color="#000000")        

    ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params('both', labelsize=12, length=5, width=1)
    ax.minorticks_off()
    ax.set_title(title)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(ymin, ylowerbound), max(ymax, yupperbound))
    
    return fig, ax


def calculate_icxx(escape_df,
                   activity_df,
                   variants=[],
                   x=0.5,
                   min_c=1e-5,
                   max_c=1e5): 
    """
    TODO:
        - calculates the ICXX given a value between 0 and 1. 
    """
    # Get the escape scores against individual mutations in the variant 
    beta = (
        escape_df[['epitope', 'mutation', 'escape']]
        .pivot_table(index = "mutation", columns = "epitope", values = "escape")
        .loc[variants]
    ) 
    
    # Calculate phi, the total binding activity of antibodies to epitope 
    phi = (
        activity_df
        .merge(beta
              .sum()
              .to_frame(name="beta_sum")
              .reset_index()
             )
        .assign(phi = lambda row: -row['activity'] + row['beta_sum'])
        .assign(exp_phi = lambda row: numpy.exp(-row.phi))
    )
    
    exp_phi_e = phi.exp_phi.to_numpy()
    
    def _func(c):
        pv = numpy.prod(1.0 / (1.0 + c * exp_phi_e))
        return 1 - x - pv

    if _func(min_c) > 0:
        ic = min_c
    elif _func(max_c) < 0:
        ic = max_c
    else:
        sol = scipy.optimize.root_scalar(
            _func, x0=1, bracket=(min_c, max_c), method="brenth"
        )
        ic = sol.root
    if not sol.converged:
        raise ValueError(f"root finding failed:\n{sol}")

    return ic

## == User Input == ##

variant_string = st.text_input("Enter Substitution in Variant:", placeholder = "Space delimited list of substitutions (i.e. N501Y K417N)")

if variant_string:
    variant = parse_variants(variant_string)
    st.write(f"Substitution in Variant: {variant_string}")
else:
    st.info("Please enter the substitutions in your variant.")
    variant = parse_variants(variant_string)

xlims = st.select_slider(
    "Select min and max concentration to display",
    options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
    value=(1e-3, 1e3))
    

# Epitope colors -- default are these colors
epitope_colors = {
    'class 1': '#1f77b4',
    'class 2': '#ff7f0e',
    'class 3': '#2ca02c'
}

## ==== Plot  ==== ## 

# Wildtype fraction neutralized
wt_df = make_sigmoid(
    variants = [],
    escape_df = mut_escape_df,
    activity_df = activity_wt_df,
    min_c = xlims[0],
    max_c = xlims[1])


# Variant fraction neutralized
variant_df = make_sigmoid(
    variants = variant,
    escape_df = mut_escape_df,
    activity_df = activity_wt_df,
    min_c = xlims[0],
    max_c = xlims[1])

by_epitope = plot_sigmoid(
    variant_df,
    wt_df, 
    epitope_colors,
    epitopes_to_plot,
    title="Fraction Unbound by Epitope $U_e(v,c)$",
    by_epitope=True,
    plot_wt=True)

total = plot_sigmoid(
    variant_df,
    wt_df, 
    epitope_colors,
    epitopes_to_plot,
    title="Total Escape Fraction $p_v(c)$",
    by_epitope=False,
    plot_wt=True)


col1, col2 = st.columns(2)

with col1:
    st.pyplot(by_epitope[0])
    if variant_string:
        by_epitope_fn = f'{"_".join(parse_variants(variant_string))}_by_epitope.svg'
    else: 
        by_epitope_fn = 'by_epitope.svg'
    by_epitope_img = io.BytesIO()
    by_epitope[0].savefig(by_epitope_img, format='svg', bbox_inches='tight')
    by_epitope_btn = st.download_button(
        label="Download image",
        data=by_epitope_img,
        file_name=by_epitope_fn,
        mime="image/svg"
    )

with col2:
    st.pyplot(total[0])
    if variant_string:
        total_fn = f'{"_".join(parse_variants(variant_string))}_total.svgv'
    else: 
        total_fn = 'total.svg'    
    total_img = io.BytesIO()
    total[0].savefig(total_img, format='svg', bbox_inches='tight')
    total_btn = st.download_button(
        label="Download image",
        data=total_img,
        file_name=total_fn,
        mime="image/svg"
    )

st.write("---")
col3, col4, col5 = st.columns(3)


with col3:
    # Determine the ICXX value
    ic_xx = st.number_input("Select the Inhibitory Concentration", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    variant_icxx = calculate_icxx(escape_df = mut_escape_df,
                    activity_df = activity_wt_df,
                    variants=variant,
                    x=ic_xx,
                    min_c=1e-5,
                    max_c=1e5)

    wt_icxx = calculate_icxx(escape_df = mut_escape_df,
                    activity_df = activity_wt_df,
                    variants=[],
                    x=ic_xx,
                    min_c=1e-5,
                    max_c=1e5)


with col5:
    # Ticker for difference in ICXX
    st.metric(f"IC{ic_xx*100:.0f}", f"{variant_icxx:.4f}", delta=f"{(variant_icxx - wt_icxx):.2f}", delta_color="inverse")


