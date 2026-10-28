import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyproj
from scipy.interpolate import griddata
from scipy.spatial import Delaunay, KDTree
import streamlit as st

try:
  import triangle as tr

  HAS_TRIANGLE = True
except ImportError:
  HAS_TRIANGLE = False

st.set_page_config(
    page_title="Advanced Topographic & Contour Generator", layout="wide"
)

st.title("Advanced Topographic Contour Generator & Terrain Modeler")
st.markdown(
    "Upload a single survey CSV file, choose your input coordinate system,"
    " map columns dynamically, and render interactive 3D terrain meshes."
)

st.sidebar.header("1. Data Input & Coordinate System")
uploaded_file = st.sidebar.file_uploader(
    "Upload Combined Survey File (CSV or TXT)", type=["csv", "txt"]
)

# --- Coordinate Reference System (CRS) Selection ---
crs_choice = st.sidebar.selectbox(
    "Select Input Coordinate System:",
    [
        "UTM Zone 36N (Meters - East Africa / Uganda)",
        "UTM Zone 35N (Meters - West / Central Uganda)",
        "WGS 84 (Latitude / Longitude - Degrees)",
    ],
)

if uploaded_file is not None:
  try:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
  except Exception as e:
    st.error(f"Error reading survey file: {e}")
    st.stop()

  st.subheader("Raw Survey Data Preview")
  st.dataframe(df.head(10), use_container_width=True)

  st.sidebar.header("2. Column Mapping")
  columns = list(df.columns)

  default_east = columns[0] if len(columns) > 0 else ""
  default_north = columns[1] if len(columns) > 1 else ""
  default_elev = columns[2] if len(columns) > 2 else ""
  default_code = columns[3] if len(columns) > 3 else columns[0]

  easting_col = st.sidebar.selectbox(
      "Easting / X Coordinate / Longitude",
      columns,
      index=columns.index(default_east) if default_east in columns else 0,
  )
  northing_col = st.sidebar.selectbox(
      "Northing / Y Coordinate / Latitude",
      columns,
      index=columns.index(default_north) if default_north in columns else 1,
  )
  elevation_col = st.sidebar.selectbox(
      "Elevation / Z Coordinate",
      columns,
      index=columns.index(default_elev) if default_elev in columns else 2,
  )
  code_col = st.sidebar.selectbox(
      "Code / Feature Column",
      columns,
      index=columns.index(default_code) if default_code in columns else 3,
  )

  spot_code_value = st.sidebar.text_input(
      "Spot Height Code Identifier", value="sh"
  )

  st.sidebar.header("3. Surface & Styling Parameters")
  grid_resolution = st.sidebar.slider("Grid Resolution (Mesh Density)", 20, 150, 50)
  interpolation_method = st.sidebar.selectbox(
      "Interpolation Method", ["linear", "nearest", "cubic"]
  )
  color_palette = st.sidebar.selectbox(
      "Color Palette",
      ["Viridis", "Plasma", "Jet", "Cividis", "Blues", "Greens", "YlOrRd", "Turbo"],
  )
  use_spatial_index = st.sidebar.checkbox(
      "Enable Spatial Indexing (KDTree Optimization)", value=True
  )
  max_edge_length = st.sidebar.slider(
      "Max Triangle Edge Length (Gap Filter)", 5.0, 500.0, 50.0, 5.0
  )

  if not HAS_TRIANGLE:
    st.sidebar.warning(
        "The 'triangle' library is missing. Install via terminal: `pip install"
        " triangle`"
    )

  if st.sidebar.button("Generate Surface & Contours", type="primary"):
    try:
      all_x = pd.to_numeric(df[easting_col], errors="coerce").values
      all_y = pd.to_numeric(df[northing_col], errors="coerce").values
      all_z = pd.to_numeric(df[elevation_col], errors="coerce").values
      codes = df[code_col].astype(str).str.strip().values

      valid_mask = ~np.isnan(all_x) & ~np.isnan(all_y) & ~np.isnan(all_z)

      if valid_mask.sum() < 4:
        st.error(
            "The dataset must contain at least 4 valid numeric points for"
            " surface interpolation."
        )
        st.stop()

      raw_x = all_x[valid_mask]
      raw_y = all_y[valid_mask]
      z_values = all_z[valid_mask]

      # --- Coordinate System Transformation ---
      if "WGS 84" in crs_choice:
        st.info(
            "Transforming WGS 84 (Lat/Long degrees) to UTM Zone 36N projected"
            " meters for modeling..."
        )
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:32636", always_xy=True
        )
        x_coords, y_coords = transformer.transform(raw_x, raw_y)
      elif "Zone 35N" in crs_choice:
        st.info(
            "Mapping input coordinates under UTM Zone 35N (EPSG:32635)..."
        )
        transformer = pyproj.Transformer.from_crs(
            "EPSG:32635", "EPSG:32635", always_xy=True
        )
        x_coords, y_coords = transformer.transform(raw_x, raw_y)
      else:
        # Default UTM Zone 36N projected meters
        x_coords, y_coords = raw_x, raw_y

      vertices = np.column_stack((x_coords, y_coords))

      if use_spatial_index:
        spatial_tree = KDTree(vertices)

      # Generate robust Delaunay triangulation directly from processed vertices
      triangles = None
      if len(vertices) >= 3:
        try:
          tri = Delaunay(vertices)
          raw_triangles = tri.simplices

          filtered_triangles = []
          max_v_idx = len(vertices)
          for simplex in raw_triangles:
            if (
                simplex[0] < max_v_idx
                and simplex[1] < max_v_idx
                and simplex[2] < max_v_idx
            ):
              p0, p1, p2 = (
                  vertices[simplex[0]],
                  vertices[simplex[1]],
                  vertices[simplex[2]],
              )
              e0 = np.linalg.norm(p0 - p1)
              e1 = np.linalg.norm(p1 - p2)
              e2 = np.linalg.norm(p2 - p0)
              if (
                  e0 <= max_edge_length
                  and e1 <= max_edge_length
                  and e2 <= max_edge_length
              ):
                filtered_triangles.append(simplex)

          triangles = (
              np.array(filtered_triangles, dtype=int)
              if filtered_triangles
              else None
          )
          st.info(
              f"Valid Triangles (after edge filtering):"
              f" {len(triangles) if triangles is not None else 0}"
          )
        except Exception as dt_err:
          st.warning(
              f"Delaunay triangulation skipped: {dt_err}. Falling back to grid."
          )
          triangles = None

      xi = np.linspace(vertices[:, 0].min(), vertices[:, 0].max(), grid_resolution)
      yi = np.linspace(vertices[:, 1].min(), vertices[:, 1].max(), grid_resolution)
      XI, YI = np.meshgrid(xi, yi)

      ZI = griddata(vertices, z_values, (XI, YI), method=interpolation_method)
      ZI = np.nan_to_num(ZI, nan=np.nanmean(z_values))

      if use_spatial_index:
        distances, _ = spatial_tree.query(
            np.column_stack((XI.ravel(), YI.ravel()))
        )
        grid_mask = distances.reshape(XI.shape) > (max_edge_length * 1.0)
        ZI[grid_mask] = np.nan

      st.success("Advanced surface modeling and interpolation computed successfully!")

      col1, col2 = st.columns(2)

      with col1:
        st.subheader("Interactive 3D Terrain Mesh")
        if triangles is not None and len(triangles) > 0:
          fig_3d = go.Figure(
              data=[
                  go.Mesh3d(
                      x=vertices[:, 0],
                      y=vertices[:, 1],
                      z=z_values,
                      i=triangles[:, 0],
                      j=triangles[:, 1],
                      k=triangles[:, 2],
                      intensity=z_values,
                      colorscale=color_palette,
                      showscale=True,
                      lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
                  )
              ]
          )
        else:
          fig_3d = go.Figure(
              data=[go.Surface(z=ZI, x=xi, y=yi, colorscale=color_palette)]
          )

        fig_3d.update_layout(
            title="3D Digital Elevation Model (DEM)",
            scene=dict(
                xaxis_title="Easting / Longitude (m or deg)",
                yaxis_title="Northing / Latitude (m or deg)",
                zaxis_title="Elevation (m)",
            ),
            autosize=True,
            margin=dict(l=0, r=0, b=0, t=40),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

      with col2:
        st.subheader("Topographic Contour Map")
        fig_contour = go.Figure(
            data=[
                go.Contour(
                    z=ZI,
                    x=xi,
                    y=yi,
                    colorscale=color_palette,
                    contours=dict(
                        showlabels=True,
                        labelfont=dict(size=12, color="white"),
                    ),
                )
            ]
        )
        fig_contour.update_layout(
            title="2D Contour Plot",
            xaxis_title="Easting / Longitude",
            yaxis_title="Northing / Latitude",
            autosize=True,
            margin=dict(l=0, r=0, b=0, t=40),
        )
        st.plotly_chart(fig_contour, use_container_width=True)

      st.subheader("Export Processed Grid Data")
      grid_df = pd.DataFrame({
          "Easting": XI.flatten(),
          "Northing": YI.flatten(),
          "Elevation": ZI.flatten(),
      })
      csv_export = grid_df.dropna().to_csv(index=False).encode("utf-8")

      st.download_button(
          label="Download Interpolated Grid (CSV)",
          data=csv_export,
          file_name="advanced_terrain_interpolated_grid.csv",
          mime="text/csv",
      )

    except Exception as ex:
      st.error(f"An error occurred during processing: {ex}")
else:
  st.info(
      "Please upload a single survey data file using the sidebar panel to get"
      " started."
  )