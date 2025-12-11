import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import re
import io
import cv2
import numpy as np
from PIL import Image

# Import backend logic
from graph import Graph
from src.graph_title import TITLE_DB
from automorphisms import (
    count_automorphisms, 
    are_graphs_isomorphic, 
    compare_graphs,
    find_isomorphism
)
from image_to_description.improved_detector import ImprovedGraphDetector

# ==========================================
# 1. HELPER FUNCTIONS (CONVERSION LOGIC)
# ==========================================

def parse_description_to_graph(text):
    """
    Parses text input in the format: V = {1, 2, 3}, E = {(1, 2), (2, 3)}
    Returns a Graph object.
    """
    try:
        # Extract Edges (E)
        e_match = re.search(r"E\s*=\s*\{(.*?)\}", text)
        if e_match:
            edges_str = e_match.group(1)
            # Regex to find tuples like (1, 2) or (1,2)
            edges = re.findall(r"\((\d+),\s*(\d+)\)", edges_str)
            edge_list = [(int(u), int(v)) for u, v in edges]
            
            if not edge_list:
                return None, "No edges found. Ensure format matches: E = {(1, 2), (2, 3)}"
            
            # Create Graph object with description (edge list)
            g = Graph(description=edge_list)
            return g, None
        else:
            return None, "Could not parse edge list. Ensure format matches: E = {(1, 2), (2, 3)}"
            
    except Exception as e:
        return None, f"Parsing Error: {str(e)}"

def model_title_to_graph(title_query):
    """
    Converts a text title (e.g., "Cycle graph C5") into a Graph object.
    Uses the Graph class to handle title-to-description conversion.
    """
    try:
        # Try to create a Graph object from the title
        g = Graph(title=title_query)
        
        if g.description:
            return g, f"Successfully created graph from title: {title_query}"
        else:
            return None, "Could not interpret title. Title may not be in database."
            
    except Exception as e:
        # If Graph class can't handle it, try fallback regex approach
        clean_query = title_query.strip().lower()
        
        # Cycle Graph
        match_c = re.search(r"(?:cycle|c)[^0-9]*(\d+)", clean_query)
        if match_c:
            n = int(match_c.group(1))
            nx_graph = nx.cycle_graph(n)
            edge_list = list(nx_graph.edges())
            g = Graph(description=edge_list)
            return g, f"Generated Cycle Graph C{n}"

        # Complete Graph
        match_k = re.search(r"(?:complete|k)[^0-9]*(\d+)", clean_query)
        if match_k:
            n = int(match_k.group(1))
            nx_graph = nx.complete_graph(n)
            edge_list = list(nx_graph.edges())
            g = Graph(description=edge_list)
            return g, f"Generated Complete Graph K{n}"

        # Path Graph
        match_p = re.search(r"(?:path|p)[^0-9]*(\d+)", clean_query)
        if match_p:
            n = int(match_p.group(1))
            nx_graph = nx.path_graph(n)
            edge_list = list(nx_graph.edges())
            g = Graph(description=edge_list)
            return g, f"Generated Path Graph P{n}"

        # Star Graph
        match_s = re.search(r"(?:star|s)[^0-9]*(\d+)", clean_query)
        if match_s:
            n = int(match_s.group(1))
            nx_graph = nx.star_graph(n)
            edge_list = list(nx_graph.edges())
            g = Graph(description=edge_list)
            return g, f"Generated Star Graph with {n} leaves"

        return None, f"Could not interpret title: {str(e)}"

def model_image_to_data(uploaded_image, min_radius=20, max_radius=100, detect_arrows=True, use_ocr=True, edge_min_pixels=2, node_proximity=40):
    """
    Converts an uploaded image to a Graph object using image detection.
    OCR is enabled by default to detect node labels and edge weights.
    """
    try:
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_image)
        
        # Use detector directly for better control
        from image_to_description.improved_detector import ImprovedGraphDetector
        detector = ImprovedGraphDetector(image, use_ocr=use_ocr)
        
        # Detect with parameters optimized for various graphs
        nodes = detector.detect_nodes(min_radius=min_radius, max_radius=max_radius)
        edges = detector.detect_edges(detect_arrows=detect_arrows, edge_min_pixels=edge_min_pixels, node_proximity=node_proximity)
        
        # Store detection results in session state for editing
        import streamlit as st
        st.session_state['last_detection'] = {
            'nodes': nodes,
            'edges': edges,
            'detector': detector
        }
        
        if nodes and edges:
            # Create mapping from node IDs to labels
            node_id_to_label = {}
            for node in nodes:
                if node['text']:
                    node_id_to_label[node['id']] = node['text']
                else:
                    node_id_to_label[node['id']] = str(node['id'])
            
            # Build edge list using labels and weights
            edge_list = []
            for e in edges:
                src_label = node_id_to_label[e['source']]
                tgt_label = node_id_to_label[e['target']]
                
                if e.get('weight') is not None:
                    # Weighted edge: (source, target, weight)
                    edge_list.append((src_label, tgt_label, e['weight']))
                else:
                    # Unweighted edge: (source, target)
                    edge_list.append((src_label, tgt_label))
            
            # Count detected labels and weights
            node_labels = {node['id']: node['text'] for node in nodes if node['text']}
            edge_weights = [e for e in edges if e.get('weight')]
            
            g = Graph(description=edge_list, image=image)
            
            labels_info = f", {len(node_labels)} with labels" if node_labels else ""
            weights_info = f", {len(edge_weights)} with weights" if edge_weights else ""
            
            return g, f"Detected {len(nodes)} nodes{labels_info} and {len(edges)} edges{weights_info}"
        elif nodes:
            return None, f"Detected {len(nodes)} nodes but no edges found. Try adjusting Edge Sensitivity."
        else:
            return None, "Could not detect graph structure. Try adjusting Min/Max Node Radius."
            
    except Exception as e:
        import traceback
        return None, f"Image processing error: {str(e)}\n{traceback.format_exc()}"

def render_graph(graph_obj):
    """Standard visualization using Matplotlib from Graph object"""
    # If the Graph object has an image, we could display it directly
    # But for consistency, we'll render from the description
    if graph_obj.description:
        G = nx.Graph()
        
        # Check if edges are weighted (3-tuples) or unweighted (2-tuples)
        edge_weights = {}
        for edge in graph_obj.description:
            if len(edge) == 3:
                # Weighted edge: (source, target, weight)
                G.add_edge(edge[0], edge[1])
                edge_weights[(edge[0], edge[1])] = edge[2]
            else:
                # Unweighted edge: (source, target)
                G.add_edge(edge[0], edge[1])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        pos = nx.kamada_kawai_layout(G) 
        
        # Draw nodes and edges
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='#d1c4e9', edge_color='#5e35b1', 
                node_size=600, font_weight='bold', font_size=12)
        
        # Draw edge weights if present
        if edge_weights:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, 
                                        font_size=10, ax=ax)
        
        return fig
    elif graph_obj.image:
        # If we only have image, display it
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(graph_obj.image)
        ax.axis('off')
        return fig
    return None

def process_camera_frame(image_pil):
    """Process a camera frame using the improved detector"""
    try:
        detector = ImprovedGraphDetector(image_pil)
        
        # Detect nodes and edges
        nodes = detector.detect_nodes(min_radius=20, max_radius=80)
        edges = detector.detect_edges(detect_arrows=True)
        
        if nodes and edges:
            # Get edge list
            edge_list = [(e['source'], e['target']) for e in edges]
            
            # Visualize detection
            result_cv = detector.visualize(show_representation=False)
            # Convert BGR to RGB for display
            result_rgb = cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            
            return edge_list, result_pil, f"Detected {len(nodes)} nodes, {len(edges)} edges"
        else:
            return None, image_pil, "No graph detected in frame"
    except Exception as e:
        return None, image_pil, f"Detection error: {str(e)}"

# ==========================================
# 2. MAIN APPLICATION (UI)
# ==========================================

def main():
    st.set_page_config(layout="wide", page_title="Graph Rep Generator")

    # CSS for styling
    st.markdown("""
    <style>
        .block-container {padding-top: 2rem;}
        div.stButton > button:first-child {width: 100%;}
        .stTextArea textarea {font-family: monospace;}
    </style>
    """, unsafe_allow_html=True)

    st.title("Multiple Graph Representations Generator")
    st.markdown("Generate and interpret diverse graph representations: text, images, and structural analysis.")
    
    # Add tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Generator", "Real-Time Detection", "Isomorphism Checker"])
    
    # ===== TAB 1: MAIN GENERATOR =====
    with tab1:
        main_generator_ui()
    
    # ===== TAB 2: REAL-TIME DETECTION =====
    with tab2:
        realtime_detection_ui()
    
    # ===== TAB 3: ISOMORPHISM CHECKER =====
    with tab3:
        isomorphism_checker_ui()

def main_generator_ui():
    """Main generator UI (original functionality)"""
    
    # Initialize session state for persistence
    if 'gen_results' not in st.session_state:
        st.session_state['gen_results'] = None
    if 'gen_status' not in st.session_state:
        st.session_state['gen_status'] = ""
    if 'gen_mode' not in st.session_state:
        st.session_state['gen_mode'] = ""
    
    col_input, col_output = st.columns([1, 1], gap="medium")

    # --- LEFT COLUMN: INPUT ---
    with col_input:
        with st.container(border=True):
            st.subheader("Input")
            
            # 1. Title Input
            use_title = st.toggle("Title")
            val_title = ""
            if use_title:
                val_title = st.text_input("Enter Graph Title", placeholder="e.g. Cycle graph C5")

            # 2. Description Input
            st.markdown("---")
            use_desc = st.toggle("Description")
            val_desc = ""
            if use_desc:
                default_notation = "V = {1, 2, 3, 4, 5}\nE = {(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)}"
                val_desc = st.text_area("Set Notation", value=default_notation, height=120)

            # 3. Image Input
            st.markdown("---")
            use_img = st.toggle("Image")
            val_img = None
            
            # Graph detection options (shown when image is selected)
            use_ocr_gen = True
            is_weighted_gen = True
            is_directed_gen = False
            preset = "Hand-drawn"
            min_radius_gen = 20
            max_radius_gen = 40
            edge_min_pixels_gen = 3
            node_proximity_gen = 40
            
            if use_img:
                # Option to use camera or upload
                img_source = st.radio("Image Source", ["Camera", "Upload"], horizontal=True)
                
                if img_source == "Camera":
                    val_img = st.camera_input("Take a picture of your graph")
                else:
                    val_img = st.file_uploader("Upload Graph Image", type=["png", "jpg", "jpeg"])
                
                if val_img:
                    st.image(val_img, width=150)
                    st.success("Image Ready")
                    
                    # Graph type options
                    st.markdown("**Graph Options:**")
                    col_opt1, col_opt2, col_opt3 = st.columns(3)
                    with col_opt1:
                        use_ocr_gen = st.checkbox("Use OCR", value=True, key="gen_ocr", help="Detect node labels and edge weights")
                    with col_opt2:
                        is_weighted_gen = st.checkbox("Weighted", value=True, key="gen_weighted", help="Graph has edge weights")
                    with col_opt3:
                        is_directed_gen = st.checkbox("Directed", value=False, key="gen_directed", help="Graph has directed edges")
                    
                    st.markdown("**Detection Preset:**")
                    preset = st.radio(
                        "Image Type",
                        ["Hand-drawn", "Computer-rendered", "Custom"],
                        horizontal=True,
                        key="gen_preset",
                        help="Hand-drawn: photos of hand-drawn graphs. Computer-rendered: matplotlib screenshots."
                    )
                    
                    # Set parameters based on preset
                    if preset == "Hand-drawn":
                        min_radius_gen, max_radius_gen = 20, 40
                        edge_sensitivity_gen = 8
                        node_proximity_gen = 40
                    elif preset == "Computer-rendered":
                        # Ultra-aggressive for matplotlib screenshots  
                        min_radius_gen, max_radius_gen = 8, 35
                        edge_sensitivity_gen = 11  # Maps to edge_min_pixels=0 (most sensitive)
                        node_proximity_gen = 225  # Fine-tuned for matplotlib graphs
                    else:  # Custom
                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            min_radius_gen = st.slider("Min Node Radius", 5, 50, 20, key="gen_min_r")
                            max_radius_gen = st.slider("Max Node Radius", 20, 100, 40, key="gen_max_r")
                        with col_p2:
                            edge_sensitivity_gen = st.slider("Edge Sensitivity", 1, 10, 8, key="gen_edge")
                            node_proximity_gen = st.slider("Node Proximity", 20, 150, 40, key="gen_prox")
                    
                    if preset != "Custom":
                        st.caption(f"Preset params: min_r={min_radius_gen}, max_r={max_radius_gen}, proximity={node_proximity_gen}")
                    
                    edge_min_pixels_gen = 11 - edge_sensitivity_gen  # Invert

            # Action Buttons
            st.markdown("###")
            b_col1, b_col2 = st.columns([1, 2])
            with b_col1:
                if st.button("Clear"):
                    # Clear session state
                    st.session_state['gen_results'] = None
                    st.session_state['gen_status'] = ""
                    st.session_state['gen_mode'] = ""
                    if 'last_detection' in st.session_state:
                        del st.session_state['last_detection']
                    if 'generated_graph' in st.session_state:
                        del st.session_state['generated_graph']
                    st.rerun()
            with b_col2:
                run_btn = st.button("Generate", type="primary")
            
            # Process generation when button is clicked
            if run_btn:
                results_graph = None
                status_msg = ""
                mode_msg = ""

                # PRIORITY 1: Description -> Image & Title
                if use_desc and not use_img and val_desc:
                    mode_msg = "Mode: Description → Image & Title"
                    results_graph, err = parse_description_to_graph(val_desc)
                    if err:
                        status_msg = f"Error: {err}"
                    else:
                        status_msg = "Graph parsed successfully from set notation."

                # PRIORITY 2: Image -> Description & Title
                elif use_img and val_img:
                    mode_msg = f"Mode: Image → Description & Title ({'Weighted' if is_weighted_gen else 'Unweighted'}, {'Directed' if is_directed_gen else 'Undirected'}, OCR: {'On' if use_ocr_gen else 'Off'}, Preset: {preset})"
                    
                    # Use detection parameters set in left column
                    results_graph, note = model_image_to_data(
                        val_img, 
                        min_radius=min_radius_gen,
                        max_radius=max_radius_gen,
                        detect_arrows=is_directed_gen, 
                        use_ocr=use_ocr_gen,
                        edge_min_pixels=edge_min_pixels_gen,
                        node_proximity=node_proximity_gen
                    )
                    status_msg = note

                # PRIORITY 3: Title -> Image & Description
                elif use_title and val_title:
                    mode_msg = f"Mode: Title '{val_title}' → Graph"
                    results_graph, note = model_title_to_graph(val_title)
                    status_msg = note
                
                # Store results in session state for persistence
                st.session_state['gen_results'] = results_graph
                st.session_state['gen_status'] = status_msg
                st.session_state['gen_mode'] = mode_msg
                
                # Save for isomorphism checking
                if results_graph and results_graph.description:
                    st.session_state['generated_graph'] = results_graph.description

    # --- RIGHT COLUMN: OUTPUT ---
    with col_output:
        with st.container(border=True):
            st.subheader("Output")
            
            # Display results from session state (persists across reruns)
            if st.session_state['gen_results']:
                results_graph = st.session_state['gen_results']
                status_msg = st.session_state['gen_status']
                mode_msg = st.session_state['gen_mode']
                
                if mode_msg:
                    if "Error" in status_msg:
                        st.error(status_msg)
                    else:
                        st.info(mode_msg)
                        if "Image" in mode_msg:
                            st.warning("⚠️ Handwritten graphs may have detection errors. You can manually edit below.")
                
                # Show manual editing if detection data is available
                if 'last_detection' in st.session_state:
                    with st.expander("📝 Manual Edit Detected Data", expanded=False):
                        st.info("Edit the edges below, then click 'Apply Edits' to regenerate the graph.")
                        
                        detected_nodes = st.session_state['last_detection']['nodes']
                        detected_edges = st.session_state['last_detection']['edges']
                        
                        # Editable edges (simplified format)
                        st.markdown("**Edit Edges** (format: `a → b` or `a → b (weight: 5)`)")
                        edge_data = []
                        for edge in detected_edges:
                            src_label = next((n['text'] for n in detected_nodes if n['id'] == edge['source'] and n['text']), f"node{edge['source']}")
                            tgt_label = next((n['text'] for n in detected_nodes if n['id'] == edge['target'] and n['text']), f"node{edge['target']}")
                            directed = "→" if edge.get('directed') else "—"
                            
                            if edge.get('weight'):
                                edge_data.append(f"{src_label} {directed} {tgt_label} (weight: {edge['weight']})")
                            else:
                                edge_data.append(f"{src_label} {directed} {tgt_label}")
                        
                        edges_text = "\n".join(edge_data)
                        edited_edges = st.text_area("Edges:", value=edges_text, height=150, key="manual_edges")
                        
                        if st.button("✓ Apply Edits", key="apply_manual_edits", type="primary"):
                            # Parse manual edits
                            new_edges = []
                            try:
                                for line in edited_edges.strip().split('\n'):
                                    if not line.strip():
                                        continue
                                    
                                    # Parse edge: "a → b (weight: 5)" or "a → b"
                                    if '(weight:' in line:
                                        edge_part, weight_part = line.split('(weight:')
                                        weight = float(weight_part.replace(')', '').strip())
                                        edge_part = edge_part.strip()
                                    else:
                                        weight = None
                                        edge_part = line.strip()
                                    
                                    # Split by arrow
                                    if '→' in edge_part:
                                        src, tgt = edge_part.split('→')
                                    elif '—' in edge_part:
                                        src, tgt = edge_part.split('—')
                                    else:
                                        continue
                                    
                                    src = src.strip()
                                    tgt = tgt.strip()
                                    
                                    if weight is not None:
                                        new_edges.append((src, tgt, weight))
                                    else:
                                        new_edges.append((src, tgt))
                                
                                # Create new graph with edited edges
                                updated_graph = Graph(description=new_edges)
                                
                                # Update session state
                                st.session_state['gen_results'] = updated_graph
                                st.session_state['generated_graph'] = new_edges
                                st.success(f"✓ Applied edits! Graph updated with {len(new_edges)} edges.")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error parsing edits: {e}")
                
                st.markdown("---")
                st.markdown("### Generated Representations")
                
                # 1. Display Title (from Graph object)
                if results_graph.title:
                    st.success(f"**Title:** {results_graph.title}")
                else:
                    st.warning("Could not generate canonical title")

                # 2. Generate Description (V/E Sets)
                if results_graph.description:
                    # Get all unique nodes from edges
                    nodes = set()
                    edges_for_display = []
                    
                    for edge_tuple in results_graph.description:
                        if len(edge_tuple) == 3:
                            # Weighted edge
                            u, v, w = edge_tuple
                            nodes.add(u)
                            nodes.add(v)
                            edges_for_display.append(f"({u},{v},{w})")
                        else:
                            # Unweighted edge
                            u, v = edge_tuple
                            nodes.add(u)
                            nodes.add(v)
                            edges_for_display.append(f"({u},{v})")
                    
                    # Sort nodes (handle both strings and numbers)
                    try:
                        nodes_sorted = sorted(list(nodes))
                    except:
                        nodes_sorted = sorted(list(nodes), key=str)
                    
                    # Format as set notation
                    v_str = "V = {" + ", ".join(str(n) for n in nodes_sorted) + "}"
                    e_str = "E = {" + ", ".join(edges_for_display) + "}"
                    desc_text = f"{v_str}\n{e_str}"
                    st.text_area("Generated Description", value=desc_text, height=100, key="final_desc")

                # 3. Generate Image
                fig = render_graph(results_graph)
                if fig:
                    st.pyplot(fig)
                
                if status_msg and "Error" not in status_msg:
                    st.caption(f"Status: {status_msg}")
                
                # Add helper message for isomorphism checking
                st.info("💡 Tip: Use 'From Generator' in Isomorphism Checker tab to compare this graph without re-detecting from the rendered image.")
            
            else:
                # Empty state - no results yet
                st.info("Waiting for input...")
                st.caption("Select an input type (Title, Description, or Image) on the left and click 'Generate'.")

def realtime_detection_ui():
    """Real-time detection interface with webcam"""
    st.subheader("Real-Time Graph Detection")
    st.markdown("Take photos with your webcam and detect graph structures with OCR support.")
    st.warning("⚠️ **Note:** Handwritten graphs may have detection errors. Review detected nodes and edges carefully.")
    
    # Detection settings
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Detection Settings")
        
        st.markdown("**Graph Type:**")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            use_ocr = st.checkbox("Use OCR", value=True, key="rt_ocr",
                                 help="Detect node labels and edge weights")
        with col_opt2:
            is_weighted_rt = st.checkbox("Weighted", value=True, key="rt_weighted",
                                        help="Graph has edge weights")
        
        is_directed_rt = st.checkbox("Directed Graph", value=False, key="rt_directed",
                                     help="Graph has directed edges")
        
        st.markdown("---")
        st.markdown("**Detection Preset:**")
        preset_rt = st.radio(
            "Image Type",
            ["Hand-drawn", "Computer-rendered", "Custom"],
            horizontal=True,
            key="rt_preset",
            help="Hand-drawn: photos. Computer-rendered: matplotlib screenshots."
        )
        
        # Set parameters based on preset
        if preset_rt == "Hand-drawn":
            min_radius = 20
            max_radius = 40
            edge_sensitivity = 8
            node_proximity = 40
        elif preset_rt == "Computer-rendered":
            min_radius = 8
            max_radius = 35
            edge_sensitivity = 11
            node_proximity = 225
        else:  # Custom
            min_radius = st.slider("Min Node Radius", 10, 80, 20, key="rt_min", 
                                  help="Lower for smaller nodes")
            max_radius = st.slider("Max Node Radius", 40, 200, 40, key="rt_max",
                                  help="Increase for hand-drawn graphs")
            edge_sensitivity = st.slider("Edge Sensitivity (higher = more edges)", 1, 10, 8, key="rt_edge",
                                         help="Start at 8, increase to 9-10 for hand-drawn")
            node_proximity = st.slider("Node Proximity", 15, 150, 40, key="rt_prox",
                                       help="Distance to connect edges to nodes")
        
        if preset_rt != "Custom":
            st.caption(f"Preset: min_r={min_radius}, max_r={max_radius}, proximity={node_proximity}")
        
        edge_min_pixels = 11 - edge_sensitivity  # Invert: 10->1, 1->10, 8->3
        detect_arrows = is_directed_rt  # Use the directed graph setting
        
        st.markdown("---")
        show_overlay = st.checkbox("Show detection overlay", value=True)
        
        # Manual detection button
        detect_button = st.button("Detect Graph Now", type="primary", key="rt_detect")
    
    with col2:
        st.markdown("#### Live Feed")
        
        # Streamlit camera for photos
        camera_input_rt = st.camera_input("Take a photo to detect", key="realtime_camera")
        
        if camera_input_rt is not None:
            try:
                image = Image.open(camera_input_rt)
                
                # Show detection parameters being used
                st.caption(f"Using: min_radius={min_radius}, max_radius={max_radius}, edge_min_pixels={edge_min_pixels}, proximity={node_proximity}")
                
                # Real-time detection (runs automatically when new photo taken or button pressed)
                with st.spinner("Detecting graph..."):
                    detector = ImprovedGraphDetector(image, use_ocr=use_ocr)
                    nodes = detector.detect_nodes(min_radius=min_radius, max_radius=max_radius)
                    
                    if nodes:
                        edges = detector.detect_edges(
                            detect_arrows=detect_arrows,
                            edge_min_pixels=edge_min_pixels,
                            node_proximity=node_proximity,
                            debug=False
                        )
                    else:
                        edges = []
                    
                    # Create unique key based on detection parameters to force text area updates
                    detection_key = f"rt_{min_radius}_{max_radius}_{edge_min_pixels}_{node_proximity}_{len(nodes)}_{len(edges)}"
                
                # Display results
                if nodes:
                    if show_overlay:
                        # Show detected overlay
                        result_cv = detector.visualize(show_representation=False)
                        result_rgb = cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, use_container_width=True)
                    else:
                        st.image(image, use_container_width=True)
                    
                    # Show stats
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Nodes", len(nodes))
                    with col_b:
                        st.metric("Edges", len(edges))
                    with col_c:
                        if edges:
                            st.success("Graph detected!")
                        else:
                            st.warning("No edges found")
                    
                    # Show graph representation with editing capability
                    if edges:
                        st.markdown("---")
                        st.markdown("### Detected Graph Data")
                        
                        # Create mapping from node IDs to labels for consistent display
                        node_id_to_label = {}
                        for node in nodes:
                            if node['text']:
                                node_id_to_label[node['id']] = node['text']
                            else:
                                node_id_to_label[node['id']] = f"Node_{node['id']}"
                        
                        # Editable nodes section - show ALL detected nodes
                        with st.expander("📝 Edit Detected Nodes", expanded=False):
                            st.info(f"Review and edit node labels. Format: ID: Label (Detected {len(nodes)} nodes)")
                            # Debug: Show actual node IDs detected
                            actual_node_ids = [node['id'] for node in nodes]
                            st.caption(f"Debug: Node IDs detected: {sorted(actual_node_ids)}")
                            
                            node_lines = []
                            # Sort nodes by ID for consistent display
                            sorted_nodes = sorted(nodes, key=lambda n: n['id'])
                            for node in sorted_nodes:
                                label = node_id_to_label[node['id']]
                                node_lines.append(f"{node['id']}: {label}")
                            
                            nodes_text = "\n".join(node_lines)
                            # Use unique key based on detection to force update when parameters change
                            edited_nodes = st.text_area("Node Labels", value=nodes_text, height=150, key=f"rt_edit_nodes_{detection_key}")
                        
                        # Editable edges section - show ALL detected edges
                        with st.expander("📝 Edit Detected Edges", expanded=False):
                            st.info(f"Review and edit edges. Format: source → target or source → target (weight: N) (Detected {len(edges)} edges)")
                            # Debug: Show actual edge count
                            st.caption(f"Debug: Processing {len(edges)} edges from detector")
                            
                            edge_lines = []
                            for edge in edges:
                                src_id = edge['source']
                                tgt_id = edge['target']
                                
                                # Get labels, fallback to node ID if no label
                                src_label = node_id_to_label.get(src_id, str(src_id))
                                tgt_label = node_id_to_label.get(tgt_id, str(tgt_id))
                                
                                arrow = "→" if edge.get('directed') else "—"
                                weight = edge.get('weight')
                                
                                if weight:
                                    edge_lines.append(f"{src_label} {arrow} {tgt_label} (weight: {weight})")
                                else:
                                    edge_lines.append(f"{src_label} {arrow} {tgt_label}")
                            
                            edges_text = "\n".join(edge_lines)
                            # Use unique key based on detection to force update when parameters change
                            edited_edges = st.text_area("Edges", value=edges_text, height=200, key=f"rt_edit_edges_{detection_key}")
                        
                        # Graph representation - build from actual detected data
                        # Get all unique nodes (both from edges and standalone)
                        nodes_in_edges = set()
                        all_node_ids = set([node['id'] for node in nodes])
                        edge_list = []
                        for edge in edges:
                            src_id = edge['source']
                            tgt_id = edge['target']
                            nodes_in_edges.add(src_id)
                            nodes_in_edges.add(tgt_id)
                            
                            src_label = node_id_to_label.get(src_id, str(src_id))
                            tgt_label = node_id_to_label.get(tgt_id, str(tgt_id))
                            
                            if edge.get('weight') is not None:
                                edge_list.append((src_label, tgt_label, edge['weight']))
                            else:
                                edge_list.append((src_label, tgt_label))
                        
                        # Build representation string - include ALL detected nodes
                        all_node_labels = sorted([node_id_to_label[nid] for nid in all_node_ids])
                        v_str = "V = {" + ", ".join(all_node_labels) + "}"
                        edge_strs = []
                        for e in edge_list:
                            if len(e) == 3:
                                edge_strs.append(f"({e[0]},{e[1]},{e[2]})")
                            else:
                                edge_strs.append(f"({e[0]},{e[1]})")
                        e_str = "E = {" + ", ".join(edge_strs) + "}"
                        rep = f"{v_str}\n{e_str}"
                        # Use unique key based on detection to force update when parameters change
                        st.text_area("Graph Representation", value=rep, height=100, key=f"rt_rep_{detection_key}")
                        
                        # Show statistics
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        with col_stats1:
                            labels_count = len([n for n in nodes if n['text']])
                            st.metric("Nodes with Labels", f"{labels_count}/{len(nodes)}")
                        with col_stats2:
                            weights_count = len([e for e in edges if e.get('weight')])
                            st.metric("Edges with Weights", f"{weights_count}/{len(edges)}")
                        with col_stats3:
                            directed_count = len([e for e in edges if e.get('directed')])
                            st.metric("Directed Edges", f"{directed_count}/{len(edges)}")
                        
                        # Save option
                        if st.button("Save for Isomorphism Check", key="rt_save"):
                            # Create mapping from node IDs to labels
                            node_id_to_label = {}
                            for node in nodes:
                                if node['text']:
                                    node_id_to_label[node['id']] = node['text']
                                else:
                                    node_id_to_label[node['id']] = str(node['id'])
                            
                            # Build edge list using labels and weights
                            edge_list = []
                            for e in edges:
                                src_label = node_id_to_label[e['source']]
                                tgt_label = node_id_to_label[e['target']]
                                
                                if e.get('weight') is not None:
                                    # Weighted edge: (source, target, weight)
                                    edge_list.append((src_label, tgt_label, e['weight']))
                                else:
                                    # Unweighted edge: (source, target)
                                    edge_list.append((src_label, tgt_label))
                            
                            st.session_state['detected_graph'] = edge_list
                            st.success(f"Saved {len(edges)} edges! Go to Isomorphism Checker tab.")
                    else:
                        st.info("Detected nodes but no edges. Try adjusting Edge Sensitivity or Node Proximity.")
                        # Show node info for debugging
                        with st.expander("Debug: Node Information"):
                            for node in nodes:
                                label = f" ('{node['text']}')" if node['text'] else ""
                                st.write(f"Node {node['id']}{label}: center={node['center']}, radius={node.get('radius', 'N/A')}")
                else:
                    st.image(image, use_container_width=True)
                    st.warning("No nodes detected. Try adjusting Min/Max Node Radius.")
                    
            except Exception as e:
                st.error(f"Detection error: {str(e)}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        else:
            st.info("Click the camera button above to take a photo")
            st.markdown("""
            **Tips for hand-drawn graphs:**
            
            **Drawing:**
            - Draw nodes as circles (doesn't have to be perfect!)
            - Make nodes large (at least 2-3 cm diameter)
            - Use thick pen/marker for both nodes and edges
            - Leave space between nodes
            
            **Photography:**
            - Use good lighting (no shadows)
            - Keep graph flat and centered
            - Take photo straight-on (not angled)
            - Ensure high contrast (dark pen on light paper)
            
            **If detection fails:**
            1. Increase Max Node Radius to 120
            2. Set Edge Sensitivity to 9 or 10
            3. Increase Node Proximity to 40-50
            4. Make sure edges connect to node boundaries
            """)

def isomorphism_checker_ui():
    """Graph isomorphism comparison interface"""
    st.subheader("Graph Isomorphism Checker")
    st.markdown("Compare two graphs to determine if they are isomorphic (structurally identical).")
    
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    
    # ===== GRAPH 1 INPUT =====
    with col1:
        with st.container(border=True):
            st.markdown("#### Graph 1")
            
            g1_input_type = st.radio(
                "Input Type",
                ["Description", "Image", "Saved", "From Generator"],
                key="g1_type"
            )
            
            graph1_edges = None
            
            if g1_input_type == "Description":
                g1_desc = st.text_area(
                    "Enter edges (format: E = {(1,2), (2,3), ...})",
                    value="E = {(1, 2), (2, 3), (3, 1)}",
                    key="g1_desc",
                    height=100
                )
                if st.button("Parse Graph 1"):
                    g1_obj, err = parse_description_to_graph(g1_desc)
                    if g1_obj and g1_obj.description:
                        graph1_edges = g1_obj.description
                        st.session_state['graph1_edges'] = graph1_edges
                        st.success(f"Graph 1 loaded: {len(graph1_edges)} edges")
                    else:
                        st.error(err or "Failed to parse")
            
            elif g1_input_type == "Image":
                g1_img = st.file_uploader("Upload Graph 1 Image", type=["png", "jpg"], key="g1_img")
                
                if g1_img:
                    st.markdown("**Graph Options:**")
                    g1_ocr = st.checkbox("Use OCR", value=True, key="g1_ocr",
                                        help="Detect node labels and edge weights")
                    g1_weighted = st.checkbox("Weighted", value=True, key="g1_weighted",
                                            help="Graph has edge weights")
                    g1_directed = st.checkbox("Directed", value=False, key="g1_directed",
                                            help="Graph has directed edges")
                    
                    g1_preset = st.radio("Image Type", ["Hand-drawn", "Computer-rendered", "Custom"], key="g1_preset", horizontal=True)
                    
                    # Set parameters based on preset
                    if g1_preset == "Hand-drawn":
                        g1_min_r, g1_max_r, g1_edge_pix, g1_prox = 20, 40, 2, 40
                    elif g1_preset == "Computer-rendered":
                        g1_min_r, g1_max_r, g1_edge_pix, g1_prox = 8, 35, 0, 225
                    else:  # Custom
                        col_g1_p1, col_g1_p2 = st.columns(2)
                        with col_g1_p1:
                            g1_min_r = st.slider("Min Node Radius", 5, 50, 20, key="g1_min_r")
                            g1_max_r = st.slider("Max Node Radius", 20, 100, 40, key="g1_max_r")
                        with col_g1_p2:
                            g1_edge_sens = st.slider("Edge Sensitivity", 1, 10, 8, key="g1_edge_sens")
                            g1_prox = st.slider("Node Proximity", 20, 150, 40, key="g1_prox")
                        g1_edge_pix = 11 - g1_edge_sens
                    
                    if st.button("Detect Graph 1"):
                        g1_obj, msg = model_image_to_data(
                            g1_img, 
                            min_radius=g1_min_r, 
                            max_radius=g1_max_r, 
                            detect_arrows=g1_directed, 
                            use_ocr=g1_ocr,
                            edge_min_pixels=g1_edge_pix,
                            node_proximity=g1_prox
                        )
                        if g1_obj and g1_obj.description:
                            graph1_edges = g1_obj.description
                            st.session_state['graph1_edges'] = graph1_edges
                            st.success(msg)
                        else:
                            st.error(msg)
            
            elif g1_input_type == "Saved":
                if 'detected_graph' in st.session_state:
                    graph1_edges = st.session_state['detected_graph']
                    st.session_state['graph1_edges'] = graph1_edges
                    st.success(f"Loaded from detection: {len(graph1_edges)} edges")
                else:
                    st.info("No saved graph available. Use Camera Detection tab first.")
            
            elif g1_input_type == "From Generator":
                if 'generated_graph' in st.session_state:
                    graph1_edges = st.session_state['generated_graph']
                    # Ensure edges are properly formatted
                    if graph1_edges and isinstance(graph1_edges, list):
                        # Validate edges are tuples
                        valid_edges = []
                        for edge in graph1_edges:
                            if isinstance(edge, (tuple, list)) and len(edge) >= 2:
                                valid_edges.append(tuple(edge))
                        if valid_edges:
                            st.session_state['graph1_edges'] = valid_edges
                            st.success(f"Loaded from Generator: {len(valid_edges)} edges")
                        else:
                            st.error("Invalid edge format in generated graph")
                    else:
                        st.error("Generated graph has invalid format")
                else:
                    st.info("No generated graph available. Use Generator tab first.")
            
            # Display Graph 1
            if 'graph1_edges' in st.session_state:
                try:
                    G1 = nx.Graph()
                    edges = st.session_state['graph1_edges']
                    # Handle both 2-tuples and 3-tuples (with weights)
                    for edge in edges:
                        if isinstance(edge, (tuple, list)) and len(edge) >= 2:
                            if len(edge) == 3:
                                # Weighted edge: (u, v, weight)
                                G1.add_edge(edge[0], edge[1], weight=edge[2])
                            else:
                                # Unweighted edge: (u, v)
                                G1.add_edge(edge[0], edge[1])
                    fig1, ax1 = plt.subplots(figsize=(4, 3))
                    pos1 = nx.spring_layout(G1)
                    nx.draw(G1, pos1, ax=ax1, with_labels=True, node_color='#ffcdd2', 
                           edge_color='#e57373', node_size=400, font_weight='bold')
                    st.pyplot(fig1)
                except Exception as e:
                    st.error(f"Error displaying Graph 1: {str(e)}")
                    st.code(f"Edges: {st.session_state.get('graph1_edges', [])}")
    
    # ===== GRAPH 2 INPUT =====
    with col2:
        with st.container(border=True):
            st.markdown("#### Graph 2")
            
            g2_input_type = st.radio(
                "Input Type",
                ["Description", "Image", "Saved", "From Generator"],
                key="g2_type"
            )
            
            graph2_edges = None
            
            if g2_input_type == "Description":
                g2_desc = st.text_area(
                    "Enter edges (format: E = {(1,2), (2,3), ...})",
                    value="E = {(10, 11), (11, 12), (12, 10)}",
                    key="g2_desc",
                    height=100
                )
                if st.button("Parse Graph 2"):
                    g2_obj, err = parse_description_to_graph(g2_desc)
                    if g2_obj and g2_obj.description:
                        graph2_edges = g2_obj.description
                        st.session_state['graph2_edges'] = graph2_edges
                        st.success(f"Graph 2 loaded: {len(graph2_edges)} edges")
                    else:
                        st.error(err or "Failed to parse")
            
            elif g2_input_type == "Image":
                g2_img = st.file_uploader("Upload Graph 2 Image", type=["png", "jpg"], key="g2_img")
                
                if g2_img:
                    st.markdown("**Graph Options:**")
                    g2_ocr = st.checkbox("Use OCR", value=True, key="g2_ocr",
                                        help="Detect node labels and edge weights")
                    g2_weighted = st.checkbox("Weighted", value=True, key="g2_weighted",
                                            help="Graph has edge weights")
                    g2_directed = st.checkbox("Directed", value=False, key="g2_directed",
                                            help="Graph has directed edges")
                    
                    g2_preset = st.radio("Image Type", ["Hand-drawn", "Computer-rendered", "Custom"], key="g2_preset", horizontal=True)
                    
                    # Set parameters based on preset
                    if g2_preset == "Hand-drawn":
                        g2_min_r, g2_max_r, g2_edge_pix, g2_prox = 20, 40, 2, 40
                    elif g2_preset == "Computer-rendered":
                        g2_min_r, g2_max_r, g2_edge_pix, g2_prox = 8, 35, 0, 225
                    else:  # Custom
                        col_g2_p1, col_g2_p2 = st.columns(2)
                        with col_g2_p1:
                            g2_min_r = st.slider("Min Node Radius", 5, 50, 20, key="g2_min_r")
                            g2_max_r = st.slider("Max Node Radius", 20, 100, 40, key="g2_max_r")
                        with col_g2_p2:
                            g2_edge_sens = st.slider("Edge Sensitivity", 1, 10, 8, key="g2_edge_sens")
                            g2_prox = st.slider("Node Proximity", 20, 150, 40, key="g2_prox")
                        g2_edge_pix = 11 - g2_edge_sens
                    
                    if st.button("Detect Graph 2"):
                        g2_obj, msg = model_image_to_data(
                            g2_img, 
                            min_radius=g2_min_r, 
                            max_radius=g2_max_r, 
                            detect_arrows=g2_directed, 
                            use_ocr=g2_ocr,
                            edge_min_pixels=g2_edge_pix,
                            node_proximity=g2_prox
                        )
                        if g2_obj and g2_obj.description:
                            graph2_edges = g2_obj.description
                            st.session_state['graph2_edges'] = graph2_edges
                            st.success(msg)
                        else:
                            st.error(msg)
            
            elif g2_input_type == "Saved":
                if 'detected_graph' in st.session_state:
                    graph2_edges = st.session_state['detected_graph']
                    st.session_state['graph2_edges'] = graph2_edges
                    st.success(f"Loaded from detection: {len(graph2_edges)} edges")
                else:
                    st.info("No saved graph available. Use Camera Detection tab first.")
            
            elif g2_input_type == "From Generator":
                if 'generated_graph' in st.session_state:
                    graph2_edges = st.session_state['generated_graph']
                    # Ensure edges are properly formatted
                    if graph2_edges and isinstance(graph2_edges, list):
                        # Validate edges are tuples
                        valid_edges = []
                        for edge in graph2_edges:
                            if isinstance(edge, (tuple, list)) and len(edge) >= 2:
                                valid_edges.append(tuple(edge))
                        if valid_edges:
                            st.session_state['graph2_edges'] = valid_edges
                            st.success(f"Loaded from Generator: {len(valid_edges)} edges")
                        else:
                            st.error("Invalid edge format in generated graph")
                    else:
                        st.error("Generated graph has invalid format")
                else:
                    st.info("No generated graph available. Use Generator tab first.")
            
            # Display Graph 2
            if 'graph2_edges' in st.session_state:
                try:
                    G2 = nx.Graph()
                    edges = st.session_state['graph2_edges']
                    # Handle both 2-tuples and 3-tuples (with weights)
                    for edge in edges:
                        if isinstance(edge, (tuple, list)) and len(edge) >= 2:
                            if len(edge) == 3:
                                # Weighted edge: (u, v, weight)
                                G2.add_edge(edge[0], edge[1], weight=edge[2])
                            else:
                                # Unweighted edge: (u, v)
                                G2.add_edge(edge[0], edge[1])
                    fig2, ax2 = plt.subplots(figsize=(4, 3))
                    pos2 = nx.spring_layout(G2)
                    nx.draw(G2, pos2, ax=ax2, with_labels=True, node_color='#c5e1a5', 
                           edge_color='#9ccc65', node_size=400, font_weight='bold')
                    st.pyplot(fig2)
                except Exception as e:
                    st.error(f"Error displaying Graph 2: {str(e)}")
                    st.code(f"Edges: {st.session_state.get('graph2_edges', [])}")
    
    # ===== COMPARISON RESULTS =====
    with col3:
        with st.container(border=True):
            st.markdown("#### Comparison Results")
            
            if st.button("Check Isomorphism", type="primary", use_container_width=True):
                if 'graph1_edges' in st.session_state and 'graph2_edges' in st.session_state:
                    try:
                        g1_edges = st.session_state['graph1_edges']
                        g2_edges = st.session_state['graph2_edges']
                        
                        # Perform detailed comparison
                        comparison = compare_graphs(g1_edges, g2_edges)
                        
                        st.markdown("---")
                        
                        # Show result
                        if comparison['are_isomorphic']:
                            st.success("**ISOMORPHIC**")
                            st.markdown("These graphs are structurally identical.")
                        else:
                            st.error("**NOT ISOMORPHIC**")
                            st.markdown("These graphs have different structures.")
                        
                        st.markdown("---")
                        
                        # Show statistics
                        st.markdown("**Statistics:**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Graph 1 Nodes", comparison['graph1_nodes'])
                            st.metric("Graph 1 Edges", comparison['graph1_edges'])
                            st.metric("G1 Symmetries", comparison['graph1_symmetries'])
                        with col_b:
                            st.metric("Graph 2 Nodes", comparison['graph2_nodes'])
                            st.metric("Graph 2 Edges", comparison['graph2_edges'])
                            st.metric("G2 Symmetries", comparison['graph2_symmetries'])
                        
                        # Show mapping if isomorphic
                        if comparison['are_isomorphic'] and comparison['isomorphism_mapping']:
                            st.markdown("**Node Mapping:**")
                            mapping_str = "\n".join([f"{k} → {v}" for k, v in comparison['isomorphism_mapping'].items()])
                            st.code(mapping_str)
                        
                    except Exception as e:
                        st.error(f"Error comparing graphs: {str(e)}")
                else:
                    st.warning("Please load both graphs first!")
            else:
                st.info("Load both graphs and click 'Check Isomorphism'")
                
                # Show quick info
                if 'graph1_edges' in st.session_state:
                    st.markdown(f"**Graph 1:** {len(st.session_state['graph1_edges'])} edges loaded")
                else:
                    st.markdown("**Graph 1:** Not loaded")
                
                if 'graph2_edges' in st.session_state:
                    st.markdown(f"**Graph 2:** {len(st.session_state['graph2_edges'])} edges loaded")
                else:
                    st.markdown("**Graph 2:** Not loaded")

if __name__ == "__main__":
    main()