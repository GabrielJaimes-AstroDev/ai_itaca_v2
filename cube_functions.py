
# === FUNCTIONS FOR CUBE VISUALIZING ===

def load_alma_cube(file_path, max_mb=2048):
    """Carga cubo ALMA desde archivo FITS con gestión de memoria"""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    if file_size_mb > max_mb:
        raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed ({max_mb} MB)")
    
    with fits.open(file_path) as hdul:
        cube_data = hdul[0].data
        header = hdul[0].header
        
        n_chan = cube_data.shape[0] if len(cube_data.shape) == 3 else 1
        ra_size = cube_data.shape[-2] if len(cube_data.shape) >= 2 else 1
        dec_size = cube_data.shape[-1] if len(cube_data.shape) >= 2 else 1
        
        try:
            freq0 = header['CRVAL3']
            dfreq = header['CDELT3']
            freq_axis = freq0 + dfreq * np.arange(n_chan)
        except:
            freq_axis = None
        
        cube_info = {
            'data': cube_data,
            'header': header,
            'n_chan': n_chan,
            'ra_size': ra_size,
            'dec_size': dec_size,
            'freq_axis': freq_axis,
            'file_size_mb': file_size_mb
        }
    
    return cube_info

def display_cube_info(cube_info):
    """Muestra información básica del cubo cargado"""
    st.markdown(f"""
    <div class="cube-status">
        <strong>Cube Information:</strong><br>
        Dimensions: {cube_info['data'].shape}<br>
        Channels: {cube_info['n_chan']}<br>
        RA size: {cube_info['ra_size']} pixels<br>
        Dec size: {cube_info['dec_size']} pixels<br>
        File size: {cube_info['file_size_mb']:.2f} MB<br>
    </div>
    """, unsafe_allow_html=True)

def extract_spectrum_from_region(cube_data, x_range, y_range):
    """Extrae espectro promedio de la región seleccionada"""
    if len(cube_data.shape) == 3:
        spectrum = np.mean(cube_data[:, y_range[0]:y_range[1], x_range[0]:x_range[1]], axis=(1, 2))
    else:
        spectrum = None
    return spectrum

def create_spectrum_download(freq_axis, spectrum, freq_unit='GHz', intensity_unit='K'):
    """Crea archivo descargable con datos del espectro"""
    if freq_axis is None or spectrum is None:
        return None
    
    if freq_unit == 'GHz':
        freq_values = freq_axis / 1e9
    elif freq_unit == 'MHz':
        freq_values = freq_axis / 1e6
    elif freq_unit == 'kHz':
        freq_values = freq_axis / 1e3
    else:
        freq_values = freq_axis
    
    content = f"!xValues({freq_unit})\tyValues({intensity_unit})\n"
    for freq, val in zip(freq_values, spectrum):
        content += f"{freq:.8f}\t{val:.6f}\n"
    
    return content
