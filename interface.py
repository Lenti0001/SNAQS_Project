#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 17:31:46 2025

@author: lenti
"""
from dash import Dash, html, dcc, Input, Output, callback, State
from dash.exceptions import PreventUpdate

import pandas as pd
import numpy as np
from astropy.io import fits

import plotly.express as px
import plotly.graph_objects as go

import os
import dill

from main_script import SNAQS

from helper_functions import find_decimal_point as f_n

#df = pd.read_csv("Full_run_export_SDSS.csv")
#df = pd.read_csv("Full_run_export_NOT_GTC.csv")
#path = "/home/lenti/Skrivebord/SNAQS_MSc_Project/Data/GaiaNGP/scripts/spectra/"
#path = "/home/lenti/NOT_Spectra/NOT_GTC_spectra/"

app = Dash()

df = pd.DataFrame({"LOF_val":[]})
#fig = px.scatter(df, x="RA", y="Dec", color="Type", labels={"RA": "Right Ascension [A.U.]", "Dec": "Declination [A.U.]"}, hover_data=["Object_name", "GAIA_ID", "Subtype", "z"], height=1200)
fig = px.scatter(x=None, y=None)
fig2 = px.line(x=None, y=None)
fig3 = px.line(x=None, y=None)
batch = None

app.layout = html.Div(children=[
    html.H1(children='SNAQS Tool Web Interface', style={"textAlign": "center"}),

    html.Div(children='''
        Smaller N.O.T Astrometric Quasar Survey (SNAQS) Classification/Analysis Tool WebUI edition - by: Florent I. Mustafaj
    ''', style={"textAlign": "center"}),
    html.Br(),
    html.Div(children=[
        '''Please provide a path to the spectra for classification/analysis''',
        dcc.Input(id='spectra-path', type='text', value=''),
        html.Br(),
        '''Please provide full path and filename to export/output data from classification''',
        html.Br(),
        '''(Only relevant if loading from file, else leave empty):''',
        html.Br(),
        dcc.Input(id="export-data-path", type="text", value=""),
        dcc.RadioItems(id="load-or-classif", options=["Load from file", "Classification menu"], value="Load from file"),
        html.Div(id="main-menu"),
        dcc.Interval(id="console-update-interval", interval=5000),
        ], style={"width": "19%", "float": "left", "display": "inline-block"}),
    html.Div(children=[
        dcc.Dropdown(
            ["g-r (SDSS) vs. J-K (UKIDSS)", "g-r (SDSS) vs. u-g (SDSS)", "g-J (SDSS/UKIDSS) vs. J-K (UKIDSS)", "W1-W2 (WISE) vs. W2-W3 (WISE)", "W1-W2 (WISE) vs. J-K (UKIDSS)", "g-r (SDSS) vs. G (GAIA)", "u-g (SDSS) vs. G (GAIA)", "J-K (UKIDSS) vs. G (GAIA)", "Right Ascension vs. Declination"],
            "Right Ascension vs. Declination",
            id='main_plot_selector',
            ),
        html.Br(),
        ####### Buttons that control plotting for the main plot #######
        html.Div(children='''
        Redshift range (from minimum z to maximum z)
        '''),
        dcc.Input(id='input-zmin-state', type='text', value=''),
        dcc.Input(id='input-zmax-state', type='text', value=''),
        html.Div(children='''
        Magnitude of errorbar range for x-axis (Entire range is typically 0-99)
        '''),
        dcc.Input(id='input-min-err-y-state', type='text', value='0'),
        dcc.Input(id='input-max-err-y-state', type='text', value='2'),
        html.Div(children='''
        Magnitude of errorbar range for y-axis (Entire range is typically 0-99)
        '''),
        dcc.Input(id='input-min-err-x-state', type='text', value='0'),
        dcc.Input(id='input-max-err-x-state', type='text', value='2'),
        html.Div(children=f'''
        Range of local outlier factor
        '''),
        dcc.Input(id='input-min-LOF', type='text', value=f'{df["LOF_val"].min()}'),
        dcc.Input(id='input-max-LOF', type='text', value=f'{df["LOF_val"].max()}'),
        html.Br(),
        html.Button(id='submit-states', n_clicks=0, children='Reload plots'),
        dcc.Graph(
            id='Main-graph',
            figure=fig,
        ),
        html.Br(),
        html.Div(id="classification-progress")
        ], style={"width": "40%", "float": "left", "display": "inline-block"}),
    html.Div(children=[
        dcc.Dropdown(
            ["Redshift histogram", "Subtype histogram", "Reddening histogram"],
            "Redshift histogram",
            id='analysis_plot_selector',
            ),
        dcc.Graph(
            id='Analysis-graph',
            figure=fig3
        ),
        html.H4(children='Object Spectrum', style={"textAlign": "center"}),
        html.Br(),
        html.H4('Select a best-fit  template to plot alongside the flux (Note: only works if pipeline has been completed on this interface)', style={"textAlign": "center"}),
        html.Br(),
        dcc.RadioItems(id="template-selection", options=["None", "xPCA", "CompoM LMC", "CompoM SMC", "CompoM MW", "Stellar classification"], value="None", inline=True, style={"textAlign": "center"}),
        dcc.Graph(
            id='Object-spectrum',
            figure=fig2
        )
        ], style={"height": 700, "width": "40%", "display": "inline-block"})])

@callback(
    Output("main-menu", "children"),
    Input("load-or-classif", "value"),
    Input("spectra-path", "value")
    )
def update_main_menu(state, spectra_path):
    elements = []
    if state=="Classification menu":
        elements = [
            html.H3(children='Classification menu', style={"textAlign": "center"}),
            html.H4(children='General parameters'),
            html.Br(),
            '''Do the files include any SDSS spectra?''',
            dcc.RadioItems(id="SDSS-toggle", options=["Yes", "No"], value="No"),
            '''RA range for the targets (SNAQS range is 190-210):''',
            html.Br(),
            dcc.Input(id='input-RA-min', type='text', value='190'),
            dcc.Input(id='input-RA-max', type='text', value='210'),
            html.Br(),
            '''Dec range for the targets (SNAQS range is 22-36):''',
            html.Br(),
            dcc.Input(id='input-dec-min', type='text', value='22'),
            dcc.Input(id='input-dec-max', type='text', value='36'),
            html.Br(),
            '''Name of the survey photometry file''',
            html.Br(),
            dcc.Input(id='survey-photo-name', type='text', value='Surveyphotometry.dat'),
            html.Br(),
            '''Name of the SDSS datafile''',
            html.Br(),
            dcc.Input(id='SDSS-datafile', type='text', value='AllSDSS.dat'),
            html.Br(),
            html.H4(children='Pipeline settings'),
            html.Br(),
            '''Do you wish to generate plots of spectra, contained in the Outputs/ folder? (Non-UI plots)''',
            html.Br(),
            dcc.RadioItems(id="generate-plots-toggle", options=["Yes", "No"], value="Yes"),
            html.Br(),
            '''Should the classification run?''',
            html.Br(),
            dcc.RadioItems(id="classification-toggle", options=["Yes", "No"], value="Yes"),
            html.Br(),
            '''Should some analytical plots be generated outside of the webUI?''',
            html.Br(),
            dcc.RadioItems(id="analysis-toggle", options=["Yes", "No"], value="No"),
            html.Br(),
            '''Do you wish to run local outlier factor detection on the spectra?''',
            html.Br(),
            dcc.RadioItems(id="LOF-toggle", options=["Yes", "No"], value="Yes"),
            html.Br(),
            '''Save & Export path (Standard: empty)''',
            html.Br(),
            dcc.Input(id="export-path", type="text", value=""),
            html.Br(),
            '''Export data filename (without file extension)''',
            html.Br(),
            dcc.Input(id="export-filename", type="text", value="Full_run_export"),
            html.Br(),
            '''(Optional) SNAQS object filename''',
            html.Br(),
            dcc.Input(id="SNAQS-object-filename-save", type="text", value=""),
            html.Br(),
            html.H4(children='Local Outlier Factor detection settings (only valid if LOF is selected)'),
            html.Br(),
            '''LOF detection number of wavepoints & number of neighbors''',
            html.Br(),
            dcc.Input(id='LOF-wavepoints', type='text', value='2000'),
            dcc.Input(id='LOF-neighbors', type='text', value='15'),
            html.Br(),
            '''Should LOF attempt to fit & normalise with a continuum model first?''',
            html.Br(),
            dcc.RadioItems(id="LOF-continuum-toggle", options=["Yes", "No"], value="Yes"),
            html.Br(),
            html.Button(id='submit-pipeline', n_clicks=0, children='Run the pipeline!'),
            html.Button(id="SNAQS-object-save", n_clicks=0, children="Save SNAQS object", disabled=True),
            html.Br(),
            html.Div(id="save-object-output")

            ]
    elif state=="Load from file":
        elements = [
            html.H3(children="Load from output file", style={"textAlign": "center"}),
            html.Br(),
            '''Note: If loading from file, make sure the export/output data from classification and spectra path correspond to the same spectra!''',
            html.Br(),
            html.Br(),
            '''You can also load the saved SNAQS object from file here instead of the export data (type the path and filename):''',
            html.Br(),
            dcc.Input(id="SNAQS-object-filename", type="text", value=""),
            html.Br(),
            html.Button(id='SNAQS-object-load', n_clicks=0, children='Load SNAQS object'),
            html.Br(),
            html.Div(id="load-object-output")
            ]
    return elements

###### Stuff related to running pipeline
@callback(
    Output("submit-pipeline", "disabled"),
    Input("submit-pipeline", "n_clicks"),
    State("spectra-path", "value"),
    State("SDSS-toggle", "value"),
    State("input-RA-min", "value"),
    State("input-RA-max", "value"),
    State("input-dec-min", "value"),
    State("input-dec-max", "value"),
    State("survey-photo-name", "value"),
    State("SDSS-datafile", "value"),
    State("generate-plots-toggle", "value"),
    State("classification-toggle", "value"),
    State("analysis-toggle", "value"),
    State("LOF-toggle", "value"),
    State("export-path", "value"),
    State("export-filename", "value"),
    State("LOF-wavepoints", "value"),
    State("LOF-neighbors", "value"),
    State("LOF-continuum-toggle", "value")
    )
def trigger_pipeline(n_clicks, spectra_path, SDSS_toggle, RA_min, RA_max, dec_min, dec_max, survey_photo_name, SDSS_filename, generate_plots, classify, analyse, LOF_toggle, export_path, export_filename, LOF_wavepoints, LOF_neighbours, LOF_continuum_toggle):
    global batch

    if SDSS_toggle=="Yes":
        SDSS_toggle=True
    else:
        SDSS_toggle=False

    if generate_plots=="Yes":
        generate_plots=True
    else:
        generate_plots=False

    if classify=="Yes":
        classify=True
    else:
        classify=False

    if analyse=="Yes":
        analyse=True
    else:
        analyse=False

    if LOF_toggle=="Yes":
        LOF_toggle=True
    else:
        LOF_toggle=False

    if LOF_continuum_toggle=="Yes":
        LOF_continuum_toggle=True
    else:
        LOF_continuum_toggle=False

    if n_clicks>0:
        batch = SNAQS(spectra_path, SDSS=SDSS_toggle, RA_range=[int(RA_min), int(RA_max)], DEC_range=[int(dec_min), int(dec_max)], survey_photo_filename=survey_photo_name, SDSS_dat_filename=SDSS_filename, webui=True)
        batch.full_run(generate_plots=generate_plots, local_outlier_detection=LOF_toggle, wave_points=int(LOF_wavepoints), n_neighbors=int(LOF_neighbours), fit_continuum=LOF_continuum_toggle, classification=classify, analysis=analyse, export_path=export_path, filename=export_filename)
        return True
    else:
        return False

@callback(
    Output("classification-progress", "children"),
    Input("console-update-interval", "n_intervals")
    )
def update_classification_progress(num):
    if batch==None:
        elements = []
        raise PreventUpdate
    else:
        elements = [html.H4("Classification progress (updates every 5s)", style={"textAlign": "center"}), html.Br(), f'''{batch.inline_text}''', html.Br(), f'''Progress: {batch.progress_text}''']
    return elements
##################



####### SNAQS Object saving/loading
@callback(
    Output("SNAQS-object-save", "disabled"),
    Input("console-update-interval", "n_intervals"),
    State("SNAQS-object-save", "disabled")
    )
def update_SNAQS_save(num, button_state):
    button_state
    if batch!=None:
        if batch.inline_text=="Classification complete":
            if button_state==True:
                return False
            elif button_state==False:
                raise PreventUpdate
        else:
            if button_state==True:
                raise PreventUpdate
            elif button_state==False:
                return True
    else:
        if button_state==True:
            raise PreventUpdate
        elif button_state==False:
            return True

@callback(
    Output("save-object-output", "children"),
    Input("SNAQS-object-save", "n_clicks"),
    State("SNAQS-object-filename-save", "value")
    )
def save_SNAQS_object(n_clicks, filename):
    global batch

    if n_clicks>0:
        if batch==None:
            return ['''Failed to save SNAQS object - object is empty!''']

        if len(filename)==0:
            return ['''Failed to save SNAQS object - please choose a filename and/or path!''']
        else:
            with open(f"{filename}" + ".pkl", "wb") as f:
                dill.dump(batch, f)
            return [f'''Successfully saved SNAQS object: {filename} - it can now be loaded from file!''']
    else:
        return None

@callback(
    Output("load-object-output", "children"),
    Input("SNAQS-object-load", "n_clicks"),
    State("SNAQS-object-filename", "value")
    )
def load_SNAQS_object(n_clicks, filename_obj):
    global batch
    if n_clicks>0:
        if batch!=None:
            return ['''WARNING: An already loaded object has been overwritten by loading this file!''']

        if len(filename_obj)==0:
            return ['''Failed to load - Please specify a filepath and filename (without file extension!) ''']
        else:
            try:
                with open(f"{filename_obj}" + ".pkl", "rb") as f:
                    batch = dill.load(f)
                return ['''Object loaded successfully! You can now update the interface.''']
            except:
                return ['''Failed to load object - could not find object with the provided filepath and/or filename!''']
    else:
        return None
###### Main plot stuff
@callback(
    Output("Main-graph", "figure"),
    Input("main_plot_selector", "value"),
    Input("submit-states", "n_clicks"),
    State('input-zmin-state', "value"),
    State('input-zmax-state', "value"),
    State('input-min-err-y-state', "value"),
    State('input-max-err-y-state', "value"),
    State('input-min-err-x-state', "value"),
    State('input-max-err-x-state', "value"),
    State('input-min-LOF', "value"),
    State('input-max-LOF', "value"),
    State('export-data-path', "value"),
    )
def update_main_plot(plot_type, n_clicks, zmin, zmax, min_err_y, max_err_y, min_err_x, max_err_x, min_LOF, max_LOF, df_path):
    global df, batch
    try:
        df = pd.read_csv(df_path)
        batch=None
    except:
        try:
            df = batch.export_df
        except:
            raise PreventUpdate

    if zmin!="" and zmax!="":
        filtered_mask = (df["z"]>float(zmin)) & (df["z"]<float(zmax))
    if min_LOF!="" and max_LOF!="":
        try:
            filtered_mask += (df["LOF_val"]<float(max_LOF)) & (float(min_LOF)<df["LOF_val"])
        except:
            filtered_mask = (df["LOF_val"]<float(max_LOF)) & (float(min_LOF)<df["LOF_val"])
    try:
        filtered_df = df[filtered_mask]
    except:
        filtered_df = df

    JK = filtered_df["UKIDSS_J"]-filtered_df["UKIDSS_K"]
    gr = filtered_df["SDSS-g"]-filtered_df["SDSS-r"]
    ug = filtered_df["SDSS-u"]-filtered_df["SDSS-g"]
    W2W3 = filtered_df["WISE_W2"]-filtered_df["WISE_W3"]
    W1W2 = filtered_df["WISE_W1"]-filtered_df["WISE_W2"]
    gJ = filtered_df["SDSS-g"]-filtered_df["UKIDSS_J"]
    try:
        G = filtered_df["phot_g_mean_mag"]
    except:
        G = np.empty(len(filtered_df))
        G[:] = np.nan

    err_JK = (filtered_df["err_UKIDSS_J"]**2+filtered_df["err_UKIDSS_K"]**2)**0.5
    err_gr = (filtered_df["err_SDSS-g"]**2+filtered_df["err_SDSS-r"]**2)**0.5
    err_ug = (filtered_df["err_SDSS-u"]**2+filtered_df["err_SDSS-g"]**2)**0.5
    err_W2W3 = (filtered_df["err_WISE_W2"]**2+filtered_df["err_WISE_W3"]**2)**0.5
    err_W1W2 = (filtered_df["err_WISE_W1"]**2+filtered_df["err_WISE_W2"]**2)**0.5
    err_gJ = (filtered_df["err_SDSS-g"]**2+filtered_df["err_UKIDSS_J"]**2)**0.5

    if plot_type=="Right Ascension vs. Declination":
        fig = px.scatter(filtered_df, x="RA", y="Dec", color="Type", labels={"RA": "Right Ascension [A.U.]", "Dec": "Declination [A.U.]"}, hover_data=["LOF_val", "Object_name", "GAIA_ID", "Subtype", "z"], height=1200)
    elif plot_type=="g-r (SDSS) vs. J-K (UKIDSS)":
        mask = (float(min_err_x)<np.abs(err_JK)) & (np.abs(err_JK)<float(max_err_x)) & (float(min_err_y)<np.abs(err_gr)) & (np.abs(err_gr)<float(max_err_y))
        fig = px.scatter(filtered_df[mask], x=JK[mask], y=gr[mask], error_x=err_JK[mask], error_y=err_gr[mask], color="Type", labels={"x": "J-K (UKIDSS) [A.U.]", "y": "g-r (SDSS) [A.U.]"}, hover_data=["LOF_val", "Object_name", "GAIA_ID", "Subtype", "z"], height=1200)
    elif plot_type=="g-r (SDSS) vs. u-g (SDSS)":
        mask = (float(min_err_x)<np.abs(err_ug)) & (err_ug<float(max_err_x)) & (float(min_err_y)<np.abs(err_gr)) & (np.abs(err_gr)<float(max_err_y))
        fig = px.scatter(filtered_df[mask], x=ug[mask], y=gr[mask], error_x=err_ug[mask], error_y=err_gr[mask], color="Type", labels={"x": "u-g (SDSS) [A.U.]", "y": "g-r (SDSS) [A.U.]"}, hover_data=["LOF_val", "Object_name", "GAIA_ID", "Subtype", "z"], height=1200)
    elif plot_type=="W1-W2 (WISE) vs. W2-W3 (WISE)":
        mask = (float(min_err_x)<np.abs(err_W2W3)) & (np.abs(err_W2W3)<float(max_err_x)) & (float(min_err_y)<np.abs(err_W1W2)) & (np.abs(err_W1W2)<float(max_err_y))
        fig = px.scatter(filtered_df[mask], x=W2W3[mask], y=W1W2[mask], error_x=err_W2W3[mask], error_y=err_W1W2[mask], color="Type", labels={"x": "W2-W3 (WISE) [A.U.]", "y": "W1-W2 (WISE) [A.U.]"}, hover_data=["LOF_val", "Object_name", "GAIA_ID", "Subtype", "z"], height=1200)
    elif plot_type=="g-J (SDSS/UKIDSS) vs. J-K (UKIDSS)":
        mask = (float(min_err_x)<np.abs(err_JK)) & (np.abs(err_JK)<float(max_err_x)) & (float(min_err_y)<np.abs(err_gJ)) & (np.abs(err_gJ)<float(max_err_y))
        fig = px.scatter(filtered_df[mask], x=JK[mask], y=gJ[mask], error_x=err_JK[mask], error_y=err_gJ[mask], color="Type", labels={"x": "J-K (UKIDSS) [A.U.]", "y": "g-J (SDSS/UKIDSS) [A.U.]"}, hover_data=["LOF_val", "Object_name", "GAIA_ID", "Subtype", "z"], height=1200)
    elif plot_type=="W1-W2 (WISE) vs. J-K (UKIDSS)":
        mask = (float(min_err_x)<np.abs(err_JK)) & (np.abs(err_JK)<float(max_err_x)) & (float(min_err_y)<np.abs(err_W1W2)) & (np.abs(err_W1W2)<float(max_err_y))
        fig = px.scatter(filtered_df[mask], x=JK[mask], y=W1W2[mask], error_x=err_JK[mask], error_y=err_W1W2[mask], color="Type", labels={"x": "J-K (UKIDSS) [A.U.]", "y": "W1-W2 (WISE) [A.U.]"}, hover_data=["LOF_val", "Object_name", "GAIA_ID", "Subtype", "z"], height=1200)
    elif plot_type=="g-r (SDSS) vs. G (GAIA)":
        mask = (float(min_err_y)<np.abs(err_gr)) & (np.abs(err_gr)<float(max_err_y))
        fig = px.scatter(filtered_df[mask], x=G[mask], y=gr[mask], error_y=err_gr[mask], color="Type", labels={"x": "G (GAIA) [A.U.]", "y": "g-r (SDSS) [A.U.]"}, hover_data=["LOF_val", "Object_name", "GAIA_ID", "Subtype", "z"], height=1200)
    elif plot_type=="u-g (SDSS) vs. G (GAIA)":
        mask = (float(min_err_y)<np.abs(err_ug)) & (np.abs(err_ug)<float(max_err_y))
        fig = px.scatter(filtered_df[mask], x=G[mask], y=ug[mask], error_y=err_ug[mask], color="Type", labels={"x": "G (GAIA) [A.U.]", "y": "u-g (SDSS) [A.U.]"}, hover_data=["LOF_val", "Object_name", "GAIA_ID", "Subtype", "z"], height=1200)
    elif plot_type=="J-K (UKIDSS) vs. G (GAIA)":
        mask = (float(min_err_y)<np.abs(err_JK)) & (np.abs(err_JK)<float(max_err_y))
        fig = px.scatter(filtered_df[mask], x=G[mask], y=JK[mask], error_y=err_JK[mask], color="Type", labels={"x": "G (GAIA) [A.U.]", "y": "J-K (UKIDSS) [A.U.]"}, hover_data=["LOF_val", "Object_name", "GAIA_ID", "Subtype", "z"], height=1200)
    return fig


####### Stuff related to object spectrum plot
@callback(
    Output("Object-spectrum", "figure"),
    Input("Main-graph", "clickData"),
    Input("spectra-path", "value"),
    Input("template-selection", "value")
    )
def update_spectrum_plot(clickData, spectra_path, template_selected):
    global batch

    if clickData==None:
        raise PreventUpdate

    filename = clickData['points'][0]['customdata'][-4]
    path = spectra_path
    if batch==None:
        if filename[-4:]=="fits":
            try:
                hdu = fits.open(os.path.join(path, filename))
                wave = 10**hdu[1].data["loglam"]
                flux = hdu[1].data["flux"]
                error = (1./hdu[1].data["ivar"])**0.5
                fig2 = px.line(x=wave, y=flux, labels={"x": "Wavelength [Å]", "y": "Flux [10^(-17) erg/cm^(2)/s/Å]"}, subtitle=f"GAIA ID: {clickData['points'][0]['customdata'][-3]}", title=f"{filename}")
                fig2.add_trace(go.Scatter(x=wave,y=error, name="σ", line=dict(color='rgba(255, 0, 0, 1)')))
                hdu.close()
            except:
                fig2 = px.line(x=None, y=None)
        else:
            try:
                data = pd.read_csv(path + filename, sep="\s+")
                data = data[(data["calibrated_flux"].notna()) & (data["wavelength"]>4000) & (data["wavelength"]<8800)]
                flux = data["calibrated_flux"].values
                wave = data["wavelength"].values
                error = (data["flux_var"].values)**0.5
                fig2 = px.line(x=wave, y=flux, labels={"x": "Wavelength [Å]", "y": "Flux [erg/cm^(2)/s/Å]"}, subtitle=f"GAIA ID: {clickData['points'][0]['customdata'][-3]}", title=f"{filename}")
                fig2.add_trace(go.Scatter(x=wave,y=error, name="σ", line=dict(color='rgba(255, 0, 0, 1)')))
            except:
                fig2 = px.line(x=None, y=None)
        return fig2
    else:
        wave = batch.objects[filename].wave
        flux = batch.objects[filename].flux
        error = batch.objects[filename].error
        sub_export = batch.export_df[batch.export_df["Object_name"]==filename]
        if np.mean(flux)<10**(-10):
            fig2 = px.line(x=wave, y=flux, labels={"x": "Wavelength [Å]", "y": "Flux [erg/cm^(2)/s/Å]"})
            fig2.update_layout(title=go.layout.Title(text=f"{filename} <br><sup>GAIA ID: {clickData['points'][0]['customdata'][-3]}</sup>"))
        else:
            fig2 = px.line(x=wave, y=flux, labels={"x": "Wavelength [Å]", "y": "Flux [10^(-17) erg/cm^(2)/s/Å]"})
            fig2.update_layout(title=go.layout.Title(text=f"{filename} <br><sup>GAIA ID: {clickData['points'][0]['customdata'][-3]}</sup>"))
        fig2.add_trace(go.Scatter(x=wave,y=error, name="σ", line=dict(color='rgba(255, 0, 0, 1)')))
        if template_selected=="xPCA":
            model_flux = batch.objects[filename].xpca["BestModel_flux"]
            model_wave = batch.objects[filename].xpca["BestModel_wave"]
            z = batch.objects[filename].xpca["zBest"]
            z_std = batch.objects[filename].xpca["zBestErr"]
            AB = sub_export["compoM_AB"].values[0]
            AB_std = sub_export["compoM_AB_std"].values[0]

            fig2.update_layout(title=go.layout.Title(text=f"{filename} <br><sup>GAIA ID: {clickData['points'][0]['customdata'][-3]}  z: {np.round(z, f_n(z_std))} +/- {np.round(z_std, f_n(z_std))}  Chi2: {batch.objects[filename].xpca["zBestChi2"]}  <br>Type: {batch.objects[filename].xpca["zBestType"]}  Subtype: {batch.objects[filename].xpca["zBestSubType"]}  <br>### Reddening (source: best-fit composite model) ### <br> AB: {np.round(AB, f_n(AB_std))} +/- {np.round(AB_std, f_n(AB_std))}  <br> extinct params: {sub_export["best_compoM_extinct_params"].values[0]}  chi2_compoM: {sub_export["best_compoM_Chi2"].values[0]}</sup>", xref="paper", x=0))
            fig2.add_trace(go.Scatter(x=model_wave,y=model_flux, name="Best-fit xPCA model", line=dict(color='rgba(0, 255, 0, 1)')))
        elif template_selected=="CompoM LMC":
            model_flux = batch.objects[filename].compoM_LMC["model_flux"]
            model_wave = batch.objects[filename].wave
            z = batch.objects[filename].compoM_LMC["z"]
            z_std = batch.objects[filename].compoM_LMC["z_std"]
            AB = batch.objects[filename].compoM_LMC["AB"]
            AB_std = batch.objects[filename].compoM_LMC["AB_std"]
            norm = batch.objects[filename].compoM_LMC["norm"]
            norm_std = batch.objects[filename].compoM_LMC["norm_std"]

            fig2.update_layout(title=go.layout.Title(text=f"{filename} <br><sup>GAIA ID: {clickData['points'][0]['customdata'][-3]}  z: {np.round(z, f_n(z_std))} +/- {np.round(z_std, f_n(z_std))}  AB: {np.round(AB, f_n(AB_std))} +/-  {np.round(AB_std, f_n(AB_std))}  <br>Norm: {np.round(norm, f_n(norm_std))} +/-  {np.round(norm_std, f_n(norm_std))}  Chi2: {batch.objects[filename].compoM_LMC["chi2"]}  <br>Type: QSO</sup>", xref="paper", x=0))
            fig2.add_trace(go.Scatter(x=model_wave,y=model_flux, name="Best-fit Composite Model - LMC params", line=dict(color='rgba(0, 255, 0, 1)')))
        elif template_selected=="CompoM SMC":
            model_flux = batch.objects[filename].compoM_SMC["model_flux"]
            model_wave = batch.objects[filename].wave
            z = batch.objects[filename].compoM_SMC["z"]
            z_std = batch.objects[filename].compoM_SMC["z_std"]
            AB = batch.objects[filename].compoM_SMC["AB"]
            AB_std = batch.objects[filename].compoM_SMC["AB_std"]
            norm = batch.objects[filename].compoM_SMC["norm"]
            norm_std = batch.objects[filename].compoM_SMC["norm_std"]

            fig2.update_layout(title=go.layout.Title(text=f"{filename} <br><sup>GAIA ID: {clickData['points'][0]['customdata'][-3]}  z: {np.round(z, f_n(z_std))} +/- {np.round(z_std, f_n(z_std))}  AB: {np.round(AB, f_n(AB_std))} +/-  {np.round(AB_std, f_n(AB_std))}  <br>Norm: {np.round(norm, f_n(norm_std))} +/-  {np.round(norm_std, f_n(norm_std))}  Chi2: {batch.objects[filename].compoM_SMC["chi2"]}  <br>Type: QSO</sup>", xref="paper", x=0))
            fig2.add_trace(go.Scatter(x=model_wave,y=model_flux, name="Best-fit Composite Model - SMC params", line=dict(color='rgba(0, 255, 0, 1)')))
        elif template_selected=="CompoM MW":
            model_flux = batch.objects[filename].compoM_MW["model_flux"]
            model_wave = batch.objects[filename].wave
            z = batch.objects[filename].compoM_MW["z"]
            z_std = batch.objects[filename].compoM_MW["z_std"]
            AB = batch.objects[filename].compoM_MW["AB"]
            AB_std = batch.objects[filename].compoM_MW["AB_std"]
            norm = batch.objects[filename].compoM_MW["norm"]
            norm_std = batch.objects[filename].compoM_MW["norm_std"]

            fig2.update_layout(title=go.layout.Title(text=f"{filename} <br><sup>GAIA ID: {clickData['points'][0]['customdata'][-3]}  z: {np.round(z, f_n(z_std))} +/- {np.round(z_std, f_n(z_std))}  AB: {np.round(AB, f_n(AB_std))} +/-  {np.round(AB_std, f_n(AB_std))}  <br>Norm: {np.round(norm, f_n(norm_std))} +/-  {np.round(norm_std, f_n(norm_std))}  Chi2: {batch.objects[filename].compoM_MW["chi2"]}  <br>Type: QSO</sup>", xref="paper", x=0))
            fig2.add_trace(go.Scatter(x=model_wave,y=model_flux, name="Best-fit Composite Model - MW params", line=dict(color='rgba(0, 255, 0, 1)')))
        elif template_selected=="Stellar classification":
            model_flux = batch.objects[filename].stellar_classification["model_flux"]
            model_wave = batch.objects[filename].wave
            fig2.update_layout(title=go.layout.Title(text=f"{filename} <br><sup>GAIA ID: {clickData['points'][0]['customdata'][-3]}  Chi2: {batch.objects[filename].stellar_classification["Chi2"]}</sup>", xref="paper", x=0))
            fig2.add_trace(go.Scatter(x=model_wave,y=model_flux, name=f"Best-fit stellar template - {batch.objects[filename].stellar_classification["Template_file"]}", line=dict(color='rgba(0, 255, 0, 1)')))
        return fig2


###### Stuff related to analysis plot
@callback(
    Output("Analysis-graph", "figure"),
    Input("Main-graph", "selectedData"),
    Input("analysis_plot_selector", "value")
    )
def update_analysis_plot(selectedData, plot_type):
    global df
    if selectedData==None:
        raise PreventUpdate

    filenames = pd.Series([selectedData["points"][i]["customdata"][-4] for i in range(len(selectedData["points"]))])
    selected_df = df[df["Object_name"].isin(filenames)]
    if plot_type=="Redshift histogram":
        fig3 = px.histogram(selected_df, x="z", labels={"x": "z [A.U.]"})
    elif plot_type=="Subtype histogram":
        fig3 = px.histogram(selected_df, x="Subtype", labels={"x": "Subtypes [A.U.]"})
    elif plot_type=="Reddening histogram":
        fig3 = px.histogram(selected_df, x="compoM_AB", labels={"x": "Reddening [A.U.]"})

    return fig3
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
