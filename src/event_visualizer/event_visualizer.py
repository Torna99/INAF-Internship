import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")

def pixel_conversion(pixel_ijk):
    """
    Convert pixel_ijk string to a list of tuples (i, j, k).
    Pixels are in format "[i, j , k];[i, j, k];...;[i, j, k]"
    """
    pixels_no_brackets = pixel_ijk.replace("[", "").replace("]", "")
    pixel_coord = pixels_no_brackets.split(",")

    i, j, k = map(int, pixel_coord)
    return i, j, k

def create_detection_cube(hits_matrix, gap=0.02, colorscale="Plasma", opacity=0.9, min_hitted=0, show_arrows=True):
    """
    Create a 3D visualization of the detector activation with proportional 3D Cartesian arrow axes.

    Parameters:
        hits_matrix: A 3D numpy array representing the total hits in each sub-cube of the detector.
        gap: Spacing between sub-cubes.
        colorscale: Plotly colorscale for the voxels.
        opacity: Opacity of the voxel meshes.
        min_hitted: Minimum threshold for hits to be displayed.
        show_arrows: If True, draws 3D arrow axes (I, J, K) starting from the origin.
    """
    values = np.asarray(hits_matrix)
    nx, ny, nz = values.shape
    size = 1.0 - gap

    # Vertex and face definitions for a unit cube
    base_v = np.array([
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
        [0, 0, size], [size, 0, size], [size, size, size], [0, size, size]
    ])
    base_triangles = np.array([
        [7, 3, 0], [0, 4, 7], [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7], [6, 5, 1], [6, 1, 2],
        [4, 0, 1], [4, 1, 5], [3, 2, 6], [3, 6, 7]
    ])
    edge_seq = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, None, 1, 5, None, 2, 6, None, 3, 7]

    all_vx, all_vy, all_vz = [], [], []
    all_i, all_j, all_k = [], [], []
    intensity_list = []
    edge_x, edge_y, edge_z = [], [], []

    # Takes only coordinates with hits strictly greater than min_hitted
    active_coords = np.argwhere(values > min_hitted)

    if len(active_coords) == 0:
        print("Attenzione: nessun voxel con hit >", min_hitted)
        return go.Figure()

    vert_offset = 0
    for x, y, z in active_coords:
        val = values[x, y, z]
        cube_v = base_v + np.array([x, y, z])

        all_vx.extend(cube_v[:, 0])
        all_vy.extend(cube_v[:, 1])
        all_vz.extend(cube_v[:, 2])

        intensity_list.extend([val] * 8)

        shifted_tri = base_triangles + vert_offset
        all_i.extend(shifted_tri[:, 0])
        all_j.extend(shifted_tri[:, 1])
        all_k.extend(shifted_tri[:, 2])
        vert_offset += 8

        # Black edges for each cube
        for idx in edge_seq:
            if idx is None:
                edge_x.append(None); edge_y.append(None); edge_z.append(None)
            else:
                edge_x.append(cube_v[idx, 0])
                edge_y.append(cube_v[idx, 1])
                edge_z.append(cube_v[idx, 2])
        edge_x.append(None); edge_y.append(None); edge_z.append(None)

    # 1. Mesh of active voxels
    mesh = go.Mesh3d(
        x=all_vx, y=all_vy, z=all_vz,
        i=all_i, j=all_j, k=all_k,
        intensity=intensity_list,
        intensitymode='vertex',
        colorscale=colorscale,
        cmin=float(np.min(values[values > min_hitted])),
        cmax=float(np.max(values)),
        opacity=opacity,
        flatshading=True,
        showscale=False, # to visualize the colorbar use instead colorbar=dict(title="Hits"),
        name="Hits"
    )

    # 2. Voxel outlines
    edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=1.5),
        hoverinfo='none',
        showlegend=False
    )

    # 3. Detector bounding box
    bbox_x = [0, nx, nx, 0, 0, 0, nx, nx, 0, 0, None, nx, nx, None, nx, nx, None, 0, 0]
    bbox_y = [0, 0, ny, ny, 0, 0, 0, ny, ny, 0, None, 0, 0, None, ny, ny, None, ny, ny]
    bbox_z = [0, 0, 0, 0, 0, nz, nz, nz, nz, nz, None, 0, nz, None, 0, nz, None, 0, nz]
    
    bounding_box = go.Scatter3d(
        x=bbox_x, y=bbox_y, z=bbox_z,
        mode='lines',
        line=dict(color='gray', width=2, dash='dot'),
        hoverinfo='none',
        showlegend=False
    )

    data = [mesh, edges, bounding_box]

    # 4. Proportional Cartesian Arrow Axes
    if show_arrows:
        max_d = max(nx, ny, nz)
        axis_len = max_d * 1.15
        head_len = max_d * 0.12

        axes_cfg = [
            ('I', 'red',   [0, axis_len, 0, 0, 0, 0], [head_len, 0, 0], [axis_len * 1.12, 0, 0]),
            ('J', 'green', [0, 0, 0, axis_len, 0, 0], [0, head_len, 0], [0, axis_len * 1.12, 0]),
            ('K', 'blue',  [0, 0, 0, 0, 0, axis_len], [0, 0, head_len], [0, 0, axis_len * 1.12])
        ]

        for name, color, coords, vec, label_p in axes_cfg:
            # Shaft line
            data.append(go.Scatter3d(
                x=[coords[0], coords[1]],
                y=[coords[2], coords[3]],
                z=[coords[4], coords[5]],
                mode='lines',
                line=dict(color=color, width=5),
                hoverinfo='none',
                showlegend=False
            ))

            # Arrowhead cone
            data.append(go.Cone(
                x=[coords[1]], y=[coords[3]], z=[coords[5]],
                u=[vec[0]], v=[vec[1]], w=[vec[2]],
                anchor='tail',
                sizemode='scaled',
                sizeref=1.0,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                hoverinfo='none'
            ))

            # Axis label
            data.append(go.Scatter3d(
                x=[label_p[0]], y=[label_p[1]], z=[label_p[2]],
                mode='text',
                text=[f'<b>{name}</b>'],
                textfont=dict(color=color, size=15),
                hoverinfo='none',
                showlegend=False
            ))

    fig = go.Figure(data=data)
    
    max_range = max(nx, ny, nz) * 1.25
    max_range = max(nx, ny, nz) * 1.25
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(
                title='I (pixel)', 
                range=[-0.5, max_range],
                showbackground=False,  
                showgrid=False,        
                showline=False,       
                zeroline=False
            ),
            yaxis=dict(
                title='J (pixel)', 
                range=[-0.5, max_range],
                showbackground=False,
                showgrid=False,
                showline=False,
                zeroline=False
            ),
            zaxis=dict(
                title='K (pixel)', 
                range=[-0.5, max_range],
                showbackground=False,
                showgrid=False,
                showline=False,
                zeroline=False
            )
        ),
        paper_bgcolor='white',  
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

@st.cache_data
def load_data(path, columns):
    df = pd.read_csv(path, usecols=lambda c: c in columns or c == "hits_sum")
    df = df[df["hits_sum"] >= 3]
    df = df[df["primary_e0_kev"] > 0]
    return df[columns].reset_index(drop=True)

### ======================================================================
### Definition of constants and data loading
### ======================================================================
DIM_CUBE = 10
PATH = "./visualizer_data.csv"

ordering_columns = [
    'event_index', 
    # 'pixel_ijk', 
    'primary_e0_kev',
    # 'n_voxels', 
    'E_sum', 
    # 'E_mean', 
    # 'E_std', 
    # 'E_min', 
    # 'E_max', 
    # 'E_median', 
    # 'E_p10', 
    # 'E_p25', 
    # 'E_p75', 
    # 'E_p90', 
    # 'E_p95', 
    # 'E_p99', 
    # 'E_iqr', 
    # 'E_mad', 
    # 'E_skew', 
    # 'E_kurt', 
    # 'entropy', 
    # 'gini', 
    # 'n_active', 
    # 'frac_active', 
    # 'frac_top1', 
    # 'frac_top3', 
    # 'eff_E_ge_T', 
    'com_x', 
    'com_y', 
    'com_z', 
    'rg', 
    # 'lam1', 
    # 'lam2', 
    # 'lam3', 
    'linearity', 
    'planarity', 
    'sphericity', 
    # 'asym_x', 
    # 'asym_y', 
    # 'asym_z', 
    'span_x', 
    'span_y', 
    'span_z', 
    'hits_sum', 
    # 'hits_mean_active', 
    # 'corr_E_hits', 
    'D_f', 
    'D_f_r2', 
    # 'f_occ', 
    'L_max_hops', 
    'L_max_metric', 
    'N_b', 
    'L_skel'
]
showing_columns = ordering_columns + ["particle_type", "pixel_ijk"]

data = load_data(PATH, showing_columns)
particle_types = sorted(data["particle_type"].unique())
data = data[data["hits_sum"] >= 3]
data = data[showing_columns].reset_index(drop=True)


st.title("Event Visualizer")
st.subheader("Shows events with hits number >= 3 in a 3D cube ", divider="gray")

### ======================================================================
### Selection part
### ======================================================================
col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    particle_type = st.selectbox(
        "Select a particle type",
        particle_types
    )
    selected_data = data[data["particle_type"] == particle_type]
    st.markdown(f"N. events for: {len(selected_data)}")

with col2:
    order = st.selectbox(
        "Select an ordering feature",
        ordering_columns
    )
    asc_desc_str = st.radio(
        "Select ordering direction",
        options=["Descending", "Ascending"],
    )
    is_ascending = (asc_desc_str == "Ascending")

ordered_data = selected_data.sort_values(
    by=order, 
    ascending=is_ascending
).reset_index(drop=True)

with col3:
    available_events = ordered_data["event_index"].tolist()

    if "current_event_id" not in st.session_state or st.session_state.current_event_id not in available_events:
        st.session_state.current_event_id = available_events[0]

    # Find the index of the current event in the available events list
    current_idx_in_options = available_events.index(st.session_state.current_event_id)

    event_label_map = {
        int(idx): f"Event {int(idx)} | {order} = {val:.3f}"
        for idx, val in zip(ordered_data["event_index"], ordered_data[order])
    }

    selected_event = st.selectbox(
        "Select event",
        options=available_events,
        index=current_idx_in_options,
        format_func=lambda eid: event_label_map.get(int(eid), str(eid)),
        key=f"event_selector_{particle_type}_{order}_{asc_desc_str}",
    )

    # update the current event ID in the session state ONLY IF the selected event has changed by the user
    st.session_state.current_event_id = selected_event

    current_event_row = ordered_data[
        ordered_data["event_index"] == st.session_state.current_event_id
    ].iloc[0]

### ======================================================================
### Hits matrix creation and 3D visualization
### ======================================================================
st.divider()

hits_matrix = np.zeros((DIM_CUBE, DIM_CUBE, DIM_CUBE))
# for pixel_str in st.session_state.ordered_data.iloc[row_idx]["pixel_ijk"].split(";"):
for pixel_str in current_event_row["pixel_ijk"].split(";"):
    i, j, k = pixel_conversion(pixel_str)
    hits_matrix[i, j, k] += 1

fig = create_detection_cube(
    hits_matrix,
    gap=0.08,
    colorscale="Plasma",
    opacity=0.9,
    min_hitted=0.0,
)
fig.update_layout(height=700)

plot_col, info_col = st.columns([2, 1])  

with plot_col:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with info_col:
    row = current_event_row

    st.markdown(f"### Event {int(row['event_index'])}  ·  {row['particle_type']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Primary $E_0$ [keV]", f"{row['primary_e0_kev']:.1f}")
    c2.metric("E sum", f"{row['E_sum']:.1f}")
    c3.metric("Hits sum", f"{int(row['hits_sum'])}")

    # c1, c2, c3 = st.columns(3)
    # c1.metric("COM (x, y, z)", f"({row['com_x']:.2f}, {row['com_y']:.2f}, {row['com_z']:.2f})")
    # c2.metric("Span (x, y, z)", f"({row['span_x']:.2f}, {row['span_y']:.2f}, {row['span_z']:.2f})")
    # c3.metric("Rg", f"{row['rg']:.2f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Linearity", f"{row['linearity']:.3f}")
    c2.metric("Planarity", f"{row['planarity']:.3f}")
    c3.metric("Sphericity", f"{row['sphericity']:.3f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("L_skel", f"{row['L_skel']:.3f}")
    c2.metric("N_b", f"{row['N_b']:.3f}")
    c3.metric("L_max_hops", f"{row['L_max_hops']:.3f}") 

    st.markdown("⚠️Full information of the event on the following section!")

### ======================================================================
### Full information of the event
### ======================================================================
st.divider()
st.markdown("### Every feature of the event")
st.dataframe(current_event_row.drop(columns=["pixel_ijk"]).T, width="stretch", height=800)