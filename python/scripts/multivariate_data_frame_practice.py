import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path

def gen_random_data_frame(n_row):
    """Use autoDocstring to generate this

    Args:
        n_row (Integer): number of rows of randomly generated variable to set

    Returns:
        pd.DataFrame: a DataFrame with random variables
    """
    rng = np.random.default_rng()
    x = rng.normal(loc=4, scale=1.4, size=n_row)
    y = rng.normal(loc=100, scale=20, size=n_row)
    z = rng.uniform(low=20, high=42, size=n_row)
    dep = (3* x + y + z) * rng.uniform(low=0.2, high=1.2, size=n_row)

    # we can pass in the data frame as a 2D numpy array
    df = pd.DataFrame(np.array([x,y,z, dep]).T, columns=['x','y','z', 'dep'])
    print(df)

    # we can also create the table from a dictionary, like the following
    dat_dict = {'x':x, 'y':y, 'z':z, 'dep':dep}
    df2 = pd.DataFrame(data=dat_dict)
    print(df)

    return df

def plot_multiple_regressions(frame, formula: str):
    form_split_list = formula.split('~')
    dep_var = form_split_list[0].strip()

    indep_vars = [ind.strip() for ind in form_split_list[1].split('+')]

    fig, axs = plt.subplots(ncols=len(indep_vars))
    fig.suptitle("Linear Regressions")

    for i, var in enumerate(indep_vars):
        sns.regplot(frame, x=var, y=dep_var, ax=axs[i])
    
    plt.show()


def perform_multivariate_linear_regression(frame, formula):
    """Perform a multivariate linear regression on dataframe given

    Args:
        frame (pd.DataFrame): a multivariate set of data
        formla(str): a model that can be interpreted by statsmodels
    """
    plot_multiple_regressions(frame,formula=formula)

    print(frame.corr())

    # Show the correlations as a nice plot
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show() # it is good practice to instead save this to a file in a function

    # Show the correlations as a pairplot
    sns.pairplot(df)
    plt.show()

    fit_res = smf.ols(formula=formula, data=df).fit()
    print(fit_res.summary())

# df = gen_random_data_frame(300)

df = pd.read_excel(Path('/home/clintc/projects/lab-lessons/example-data/multiple_variables_example.xlsx'))
df = df.rename(columns={'age.yr':'age_yr','height.cm':'height_cm'}) # R naming conventions can cause problems in python
df = df.drop(columns=['obs']) # We're not interested in observation number

perform_multivariate_linear_regression(df, formula='biomass ~ age_yr + height_cm + nutrient1 + nutrient2')

