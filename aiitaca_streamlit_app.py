import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from core_functions_3 import *
from cube_functions import *
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
import gdown
import shutil
import time
from astropy.io import fits
import warnings
from paths import GDRIVE_FOLDER_URL, TEMP_MODEL_DIR
from texts import (
    PROJECT_DESCRIPTION,
    PARAMS_EXPLANATION,
    TRAINING_DATASET,
    MAIN_TITLE,
    SUBTITLE,
    FLOW_OF_WORK,
    ACKNOWLEDGMENTS,
    CUBE_VISUALIZER_DESCRIPTION
)

warnings.filterwarnings('ignore')

# === PAGE CONFIGURATION ===
st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔭" 
)

# ===  CSS STYLES ===
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# === BLOCK BUTTONGS DURING PROCESSING ===
def disable_widgets():
    """Deshabilita todos los widgets cuando hay procesamiento en curso"""
    processing = st.session_state.get('processing', False)
    return processing

# === HEADER AND PROJECT DESCRIPTION ===
st.image("NGC6523_BVO_2.jpg", use_container_width=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.empty()
    
with col2:
    st.markdown(f'<p class="main-title">{MAIN_TITLE}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{SUBTITLE}</p>', unsafe_allow_html=True)

st.markdown(PROJECT_DESCRIPTION, unsafe_allow_html=True)

# === MODELS CONFIGURATION ===
if not os.path.exists(TEMP_MODEL_DIR):
    os.makedirs(TEMP_MODEL_DIR)

@st.cache_data(ttl=3600, show_spinner=True)
def download_models_from_drive(folder_url, output_dir):
    model_files = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
    data_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]

    if model_files and data_files:
        return model_files, data_files, True

    try:
        st.session_state['processing'] = True
        progress_text = st.sidebar.empty()
        progress_bar = st.sidebar.progress(0)
        progress_text.text("📥 Preparing to download models...")
        
        file_count = 0
        try:
            file_count = 10  # Valor estimado para la simulación
        except:
            file_count = 10  # Valor por defecto si no podemos obtener el conteo real
            
        with st.spinner("📥 Downloading models from Google Drive..."):
            gdown.download_folder(
                folder_url, 
                output=output_dir, 
                quiet=True,
                use_cookies=False
            )
            for i in range(file_count):
                time.sleep(0.5)
                progress = int((i + 1) / file_count * 100)
                progress_bar.progress(progress)
                progress_text.text(f"📥 Downloading models... {progress}%")
        
        model_files = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
        data_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
        
        progress_bar.progress(100)
        progress_text.text("Process Completed")
        
        if model_files and data_files:
            st.sidebar.success("✅ Models downloaded successfully!")
        else:
            st.sidebar.error("❌ No models found in the specified folder")
            
        return model_files, data_files, True
    except Exception as e:
        st.sidebar.error(f"❌ Error downloading models: {str(e)}")
        return [], [], False
    finally:
        st.session_state['processing'] = False

# === LATERAL BAR ===
st.sidebar.title("Configuration")

model_files, data_files, models_downloaded = download_models_from_drive(GDRIVE_FOLDER_URL, TEMP_MODEL_DIR)

# upload file
if 'prev_uploaded_file' not in st.session_state:
    st.session_state.prev_uploaded_file = None

current_uploaded_file = st.sidebar.file_uploader(
    "Input Spectrum File ( . | .txt | .dat | .fits | .spec )",
    type=None,
    help="Drag and drop file here ( . | .txt | .dat | .fits | .spec ). Limit 200MB per file",
    disabled=disable_widgets()
)

# Units Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("Units Configuration")

freq_unit = st.sidebar.selectbox(
    "Frequency Units",
    ["GHz", "MHz", "kHz", "Hz"],
    index=0,
    help="Select the frequency units for the input spectrum",
    disabled=disable_widgets()
)

intensity_unit = st.sidebar.selectbox(
    "Intensity Units",
    ["K", "Jy"],
    index=0,
    help="Select the intensity units for the input spectrum",
    disabled=disable_widgets()
)

# COnversion Factors
freq_conversion = {
    "GHz": 1e9,
    "MHz": 1e6,
    "kHz": 1e3,
    "Hz": 1.0
}

intensity_conversion = {
    "K": 1.0,
    "Jy": 1.0
}

if current_uploaded_file != st.session_state.prev_uploaded_file:
    if 'analysis_results' in st.session_state:
        del st.session_state['analysis_results']
    if 'analysis_done' in st.session_state:
        del st.session_state['analysis_done']
    if 'base_fig' in st.session_state:
        del st.session_state['base_fig']
    if 'input_spec' in st.session_state:
        del st.session_state['input_spec']
    
    st.session_state.prev_uploaded_file = current_uploaded_file
    st.rerun()

# Analisis Parameters
st.sidebar.subheader("Peak Matching Parameters")
sigma_emission = st.sidebar.slider("Sigma Emission", 0.1, 5.0, 1.5, step=0.1, key="sigma_emission_slider", disabled=disable_widgets())
window_size = st.sidebar.slider("Window Size", 1, 20, 3, step=1, disabled=disable_widgets())
sigma_threshold = st.sidebar.slider("Sigma Threshold", 0.1, 5.0, 2.0, step=0.1, key="sigma_threshold_slider", disabled=disable_widgets())
fwhm_ghz = st.sidebar.slider("FWHM (GHz)", 0.01, 0.5, 0.05, step=0.01, disabled=disable_widgets())
tolerance_ghz = st.sidebar.slider("Tolerance (GHz)", 0.01, 1.0, 0.1, step=0.01, disabled=disable_widgets())
min_peak_height_ratio = st.sidebar.slider("Min Peak Height Ratio", 0.1, 1.0, 0.3, step=0.05, disabled=disable_widgets())
top_n_lines = st.sidebar.slider("Top N Lines", 5, 100, 30, step=5, disabled=disable_widgets())
top_n_similar = st.sidebar.slider("Top N Similar", 50, 5000, 50, step=50, disabled=disable_widgets())

config = {
    'trained_models_dir': TEMP_MODEL_DIR,
    'peak_matching': {
        'sigma_emission': sigma_emission,
        'window_size': window_size,
        'sigma_threshold': sigma_threshold,
        'fwhm_ghz': fwhm_ghz,
        'tolerance_ghz': tolerance_ghz,
        'min_peak_height_ratio': min_peak_height_ratio,
        'top_n_lines': top_n_lines,
        'debug': True,
        'top_n_similar': top_n_similar
    },
    'units': {
        'frequency': freq_unit,
        'intensity': intensity_unit
    }
}

#@st.cache_data(ttl=3600, max_entries=3, show_spinner="Loading ALMA cube...")

# === PESTAÑAS PRINCIPALES ===
tab_molecular, tab_cube = st.tabs(["Molecular Analyzer", "Cube Visualizer"])

with tab_molecular:
    # === MOLECULAR ANALYZER ===
    st.title("Molecular Spectrum Analyzer | AI - ITACA")

    #Information Buttons
    st.markdown('<div class="buttons-container"></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([0.5, 0.5, 0.5, 0.5])
    with col1:
        params_tab = st.button("📝 Parameters Explanation", key="params_btn", 
                            help="Click to show parameters explanation",
                            disabled=disable_widgets())
    with col2:
        training_tab = st.button("📊 Training dataset", key="training_btn", 
                            help="Click to show training dataset information",
                            disabled=disable_widgets())
    with col3:
        flow_tab = st.button("📊 Flow of Work Diagram", key="flow_btn", 
                        help="Click to show the workflow diagram",
                        disabled=disable_widgets())
    with col4:
        Acknowledgments_tab = st.button("✅ Acknowledgments", key="Acknowledgments_tab", 
                        help="Click to show Acknowledgments",
                        disabled=disable_widgets())

    # Show contents
    if params_tab:
        with st.container():
            st.markdown(PARAMS_EXPLANATION, unsafe_allow_html=True)

    if training_tab:
        with st.container():
            st.markdown("""
                <div class="info-panel">
                    <h3 style="text-align: center; color: black; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">Training Dataset Parameters</h3>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(TRAINING_DATASET, unsafe_allow_html=True)
            st.image("Table_of_Mol_Params.jpg", 
                    use_container_width=True,
                    output_format="JPEG")
            st.markdown("""
            <div class="pro-tip">
                <p><strong>Note:</strong> The training dataset was generated using LTE radiative transfer models under typical ISM conditions.</p>
            </div>
            """, unsafe_allow_html=True)

    if flow_tab:
        with st.container():
            st.markdown("""
                <div class="info-panel">
                    <h3 style="text-align: center; 
                              color: white; 
                              border-bottom: 2px solid #1E88E5; 
                              padding-bottom: 10px;
                              margin-bottom: 20px;">
                        Flow of Work Diagram
                    </h3>
                </div>
            """, unsafe_allow_html=True)
    
            st.image("Flow_of_Work.jpg", 
                    use_container_width=True,
                    output_format="JPEG")
    
            st.markdown(FLOW_OF_WORK, unsafe_allow_html=True)
    
    if Acknowledgments_tab:
        with st.container():
            st.markdown("""
                <div class="info-panel">
                    <h3 style="text-align: center; 
                              color: white; 
                              border-bottom: 2px solid #1E88E5; 
                              padding-bottom: 10px;
                              margin-bottom: 20px;">
                        Project Acknowledgments
                    </h3>
                </div>
            """, unsafe_allow_html=True)
    
            st.image("Acknowledgments.png", 
                    use_container_width=True,
                    output_format="PNG")
    
            st.markdown(ACKNOWLEDGMENTS, unsafe_allow_html=True)

    if current_uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(current_uploaded_file.getvalue())
            tmp_path = tmp_file.name

        if not model_files:
            st.error("No trained models were found in Google Drive.")
        else:
            selected_model = st.selectbox(
                "Select Molecule Model", 
                model_files,
                disabled=disable_widgets()
            )
            
            analyze_btn = st.button(
                "Analyze Spectrum",
                disabled=disable_widgets()
            )

            if analyze_btn:
                try:
                    st.session_state['processing'] = True
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    def update_analysis_progress(step, total_steps=6):
                        progress = int((step / total_steps) * 100)
                        progress_bar.progress(progress)
                        steps = [
                            "Loading model...",
                            "Processing spectrum...",
                            "Detecting peaks...",
                            "Matching with database...",
                            "Calculating parameters...",
                            "Generating visualizations..."
                        ]
                        progress_text.text(f"🔍 Analyzing spectrum... {steps[step-1]} ({progress}%)")
                    
                    update_analysis_progress(1)
                    mol_name = selected_model.replace('_model.keras', '')

                    model_path = os.path.join(TEMP_MODEL_DIR, selected_model)
                    model = tf.keras.models.load_model(model_path)

                    update_analysis_progress(2)
                    data_file = os.path.join(TEMP_MODEL_DIR, f'{mol_name}_train_data.npz')
                    if not os.path.exists(data_file):
                        st.error(f"Training data not found for {mol_name}")
                    else:
                        with np.load(data_file) as data:
                            train_freq = data['train_freq']
                            train_data = data['train_data']
                            train_logn = data['train_logn']
                            train_tex = data['train_tex']
                            headers = data['headers']
                            filenames = data['filenames']

                        update_analysis_progress(3)
                        results = analyze_spectrum(
                            tmp_path, model, train_data, train_freq,
                            filenames, headers, train_logn, train_tex,
                            config, mol_name
                        )

                        if freq_unit != 'GHz':
                            results['input_freq'] = results['input_freq'] * 1e9 / freq_conversion[freq_unit]
                            results['best_match']['x_synth'] = results['best_match']['x_synth'] * 1e9 / freq_conversion[freq_unit]
                        
                        if intensity_unit != 'K':
                            results['input_spec'] = results['input_spec'] * intensity_conversion[intensity_unit]
                            results['best_match']['y_synth'] = results['best_match']['y_synth'] * intensity_conversion[intensity_unit]

                        update_analysis_progress(6)
                        st.success("Analysis completed successfully!")

                        st.session_state['analysis_results'] = results
                        st.session_state['analysis_done'] = True
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=results['input_freq'],
                            y=results['input_spec'],
                            mode='lines',
                            name='Input Spectrum',
                            line=dict(color='white', width=2)))
                        
                        fig.add_trace(go.Scatter(
                            x=results['best_match']['x_synth'],
                            y=results['best_match']['y_synth'],
                            mode='lines',
                            name='Best Match',
                            line=dict(color='red', width=2)))
                        
                        fig.update_layout(
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14',
                            margin=dict(l=50, r=50, t=60, b=50),
                            xaxis_title=f'Frequency ({freq_unit})',
                            yaxis_title=f'Intensity ({intensity_unit})',
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            height=600,
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='#3A3A3A'),
                            yaxis=dict(gridcolor='#3A3A3A')
                        )
                        
                        st.session_state['base_fig'] = fig
                        st.session_state['input_spec'] = results['input_spec']

                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                finally:
                    st.session_state['processing'] = False

            #Show results
            if 'analysis_done' in st.session_state and st.session_state['analysis_done']:
                tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Interactive Summary", 
                    "Molecule Best Match", 
                    "Peak Matching", 
                    "CNN Training", 
                    "Top Selection: Tex", 
                    "Top Selection: LogN"
                ])

                with tab0:
                    results = st.session_state['analysis_results']
                    st.markdown(f"""
                    <div class="summary-panel">
                        <h4 style="color: #1E88E5; margin-top: 0;">Detection of Physical Parameters</h4>
                        <p class="physical-params"><strong>LogN:</strong> {results['best_match']['logn']:.2f} cm⁻²</p>
                        <p class="physical-params"><strong>Tex:</strong> {results['best_match']['tex']:.2f} K</p>
                        <p class="physical-params"><strong>File (Top CNN Train):</strong> {results['best_match']['filename']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        show_sigma = st.checkbox("Visualize Sigma Emission", value=True, 
                                            key="show_sigma_checkbox",
                                            disabled=disable_widgets())
                    with col2:
                        show_threshold = st.checkbox("Visualize Sigma Threshold", value=True,
                                                key="show_threshold_checkbox",
                                                disabled=disable_widgets())
                    
                    fig = go.Figure(st.session_state['base_fig'])
                    
                    if show_sigma:
                        sigma_line_y = sigma_emission * np.std(st.session_state['input_spec'])
                        fig.add_hline(y=sigma_line_y, line_dash="dot",
                                    annotation_text=f"Sigma Emission: {sigma_emission}",
                                    annotation_position="bottom right",
                                    line_color="yellow")
                    
                    if show_threshold:
                        threshold_line_y = sigma_threshold * np.std(st.session_state['input_spec'])
                        fig.add_hline(y=threshold_line_y, line_dash="dot",
                                    annotation_text=f"Sigma Threshold: {sigma_threshold}",
                                    annotation_position="bottom left",
                                    line_color="cyan")
                    
                    st.plotly_chart(fig, use_container_width=True, key="main_plot")

                with tab1:
                    if 'analysis_results' in st.session_state:
                        results = st.session_state['analysis_results']
                        st.pyplot(plot_summary_comparison(
                            results['input_freq'], results['input_spec'],
                            results['best_match'], tmp_path
                        ))

                with tab2:
                    if 'analysis_results' in st.session_state:
                        results = st.session_state['analysis_results']
                        st.pyplot(plot_zoomed_peaks_comparison(
                            results['input_spec'], results['input_freq'],
                            results['best_match']
                        ))

                with tab3:
                    if 'analysis_results' in st.session_state:
                        results = st.session_state['analysis_results']
                        st.pyplot(plot_best_matches(
                            results['train_logn'], results['train_tex'],
                            results['similarities'], results['distances'],
                            results['closest_idx_sim'], results['closest_idx_dist'],
                            results['train_filenames'], results['input_logn']
                        ))

                with tab4:
                    if 'analysis_results' in st.session_state:
                        results = st.session_state['analysis_results']
                        st.pyplot(plot_tex_metrics(
                            results['train_tex'], results['train_logn'],
                            results['similarities'], results['distances'],
                            results['top_similar_indices'],
                            results['input_tex'], results['input_logn']
                        ))

                with tab5:
                    if 'analysis_results' in st.session_state:
                        results = st.session_state['analysis_results']
                        st.pyplot(plot_similarity_metrics(
                            results['train_logn'], results['train_tex'],
                            results['similarities'], results['distances'],
                            results['top_similar_indices'],
                            results['input_logn'], results['input_tex']
                        ))

    # Instructions
    st.sidebar.markdown("""
    **Instructions:**
    1. Select the directory containing the trained models
    2. Upload your input spectrum file ( . | .txt | .dat | .fits | .spec )
    3. Adjust the peak matching parameters as needed
    4. Select the model to use for analysis
    5. Click 'Analyze Spectrum' to run the analysis

    **Interactive Plot Controls:**
    - 🔍 Zoom: Click and drag to select area
    - 🖱️ Hover: View exact values
    - 🔄 Reset: Double-click
    - 🏎️ Pan: Shift+click+drag
    - 📊 Range Buttons: Quick zoom to percentage ranges
    """)

with tab_cube:
    # === CUBE VISUALIZER ===
    st.title("Cube Visualizer | AI-ITACA")
    st.markdown(CUBE_VISUALIZER_DESCRIPTION, unsafe_allow_html=True)
    
    cube_file = st.file_uploader(
        "Upload ALMA Cube (FITS format)",
        type=["fits", "FITS"],
        help="Drag and drop ALMA cube FITS file here (up to 2GB)",
        disabled=disable_widgets()
    )
    
    if cube_file is not None:
        with st.spinner("Processing ALMA cube..."):
            st.session_state['processing'] = True
            with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp_cube:
                tmp_cube.write(cube_file.getvalue())
                tmp_cube_path = tmp_cube.name
            
            try:
                cube_info = load_alma_cube(tmp_cube_path)
                display_cube_info(cube_info)
                
                st.success("ALMA cube loaded successfully!")
                
                st.markdown("""
                <div class="cube-controls">
                    <h4 style="color: #1E88E5; margin-top: 0;">Cube Visualization Controls</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    channel = st.slider(
                        "Select Channel",
                        0, cube_info['n_chan']-1, cube_info['n_chan']//2,
                        help="Navigate through spectral channels",
                        disabled=disable_widgets()
                    )
                    
                    st.markdown(f"""
                    <div class="cube-status">
                        <strong>Current Channel:</strong> {channel}<br>
                        {f"Frequency: {cube_info['freq_axis'][channel]/1e9:.4f} GHz" if cube_info['freq_axis'] is not None else ""}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style="margin-top: 20px;">
                        <strong>Visualization Options:</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    show_rms = st.checkbox("Show RMS noise level", value=True, disabled=disable_widgets())
                    scale = st.selectbox("Image Scale", ["Linear", "Log", "Sqrt"], index=0, disabled=disable_widgets())
                
                if len(cube_info['data'].shape) == 3:
                    img_data = cube_info['data'][channel, :, :]
                else:
                    img_data = cube_info['data']
                
                if scale == "Log":
                    img_data = np.log10(img_data - np.nanmin(img_data) + 1)
                elif scale == "Sqrt":
                    img_data = np.sqrt(img_data - np.nanmin(img_data))
                
                fig = px.imshow(
                    img_data,
                    origin='lower',
                    color_continuous_scale='viridis',
                    labels={'color': 'Intensity (K)'},
                    title=f"Channel {channel}" + (f" ({cube_info['freq_axis'][channel]/1e9:.4f} GHz)" if cube_info['freq_axis'] is not None else "")
                )
                
                fig.update_layout(
                    plot_bgcolor='#0D0F14',
                    paper_bgcolor='#0D0F14',
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis_title="RA (pixels)",
                    yaxis_title="Dec (pixels)",
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#3A3A3A'),
                    yaxis=dict(gridcolor='#3A3A3A')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="cube-controls">
                    <h4 style="color: #1E88E5; margin-top: 0;">Region Selection</h4>
                    <p>Click on the image to select a pixel or draw a rectangle to select a region.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if len(cube_info['data'].shape) == 3 and cube_info['freq_axis'] is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_range = st.slider("X Range (pixels)", 0, cube_info['ra_size']-1, (0, cube_info['ra_size']-1), disabled=disable_widgets())
                    with col2:
                        y_range = st.slider("Y Range (pixels)", 0, cube_info['dec_size']-1, (0, cube_info['dec_size']-1), disabled=disable_widgets())
                    
                    spectrum = extract_spectrum_from_region(
                        cube_info['data'],
                        x_range,
                        y_range
                    )
                    
                    if spectrum is not None:
                        fig_spec = go.Figure()
                        fig_spec.add_trace(go.Scatter(
                            x=cube_info['freq_axis']/1e9,
                            y=spectrum,
                            mode='lines',
                            line=dict(color='#1E88E5', width=2)
                        ))
                        
                        fig_spec.update_layout(
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14',
                            margin=dict(l=50, r=50, t=60, b=50),
                            xaxis_title='Frequency (GHz)',
                            yaxis_title='Intensity (K)',
                            hovermode='x unified',
                            height=400,
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='#3A3A3A'),
                            yaxis=dict(gridcolor='#3A3A3A')
                        )
                        
                        st.markdown("""
                        <div class="spectrum-display">
                            <h4 style="color: #1E88E5; margin-top: 0;">Extracted Spectrum</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.plotly_chart(fig_spec, use_container_width=True)
                        
                        spectrum_content = create_spectrum_download(
                            cube_info['freq_axis'], 
                            spectrum,
                            freq_unit='GHz',
                            intensity_unit='K'
                        )
                        if spectrum_content:
                            st.download_button(
                                label="Download Spectrum as TXT",
                                data=spectrum_content,
                                file_name="extracted_spectrum.txt",
                                mime="text/plain",
                                disabled=disable_widgets()
                            )
                
                os.unlink(tmp_cube_path)
                
            except Exception as e:
                st.error(f"Error processing ALMA cube: {str(e)}")
                if os.path.exists(tmp_cube_path):
                    os.unlink(tmp_cube_path)
            finally:
                st.session_state['processing'] = False

# === CACHÉ FUNCTIONS ===
@st.cache_data(ttl=3600)
def load_model(_model_path):
    return tf.keras.models.load_model(_model_path)

@st.cache_data(ttl=3600)
def load_training_data(data_file):
    with np.load(data_file) as data:
        return {
            'train_freq': data['train_freq'],
            'train_data': data['train_data'],
            'train_logn': data['train_logn'],
            'train_tex': data['train_tex'],
            'headers': data['headers'],
            'filenames': data['filenames']
        }

@st.cache_data(ttl=3600)
def cached_analyze_spectrum(tmp_path, model, train_data, train_freq, filenames, headers, train_logn, train_tex, config, mol_name):
    return analyze_spectrum(
        tmp_path, model, train_data, train_freq,
        filenames, headers, train_logn, train_tex,
        config, mol_name
    )
