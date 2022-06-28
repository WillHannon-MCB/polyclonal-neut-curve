# Polyclonal Neutralization Curves 

This small application takes a hypothetical polyclonal antibody mix and simulates 'neutralization curves' for a user specified SARS-CoV-2 variant. The hypothetical polyclonal antibody mix is described [here](https://jbloomlab.github.io/polyclonal/visualize_RBD.html). The biophysical model that these simulations are based on is described [here](https://jbloomlab.github.io/polyclonal/biophysical_model.html).

## Running the App

To run this application locally, simply clone the repo locally.

```
git clone https://github.com/WillHannon-MCB/polyclonal-neut-curve
cd polyclonal-neut-curve
```

Then, make a [conda](https://docs.conda.io/en/latest/) environment from the [`environment.yml`](./environment.yml) file.

```
conda env create --file environment.yml
conda activate polyclonal-neut 
```

Finally, run the application from the command line using [Streamlit](https://streamlit.io/).

```
streamlit run app/app.py
```