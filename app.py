import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import re
import random

# Import backend logic from your uploaded file
#
from src.graph_title import generate_title, TITLE_DB

# ==========================================
# 1. HELPER FUNCTIONS (CONVERSION LOGIC)
# ==========================================

def parse_description_to_graph(text):
    """
    Parses text input in the format: V = {1, 2, 3}, E = {(1, 2), (2, 3)}
    Returns a NetworkX graph.
    """
    G = nx.Graph()
    try:
        # Extract Nodes (V)
        v_match = re.search(r"V\s*=\s*\{([0-9,\s]+)\}", text)
        if v_match:
            # Split by comma, filter empty strings, convert to int
            nodes = [int(n.strip()) for n in v_match.group(1).split(',') if n.strip()]
            G.add_nodes_from(nodes)

        # Extract Edges (E)
        e_match = re.search(r"E\s*=\s*\{(.*?)\}", text)
        if e_match:
            edges_str = e_match.group(1)
            # Regex to find tuples like (1, 2) or (1,2)
            edges = re.findall(r"\((\d+),\s*(\d+)\)", edges_str)
            for u, v in edges:
                G.add_edge(int(u), int(v))
        
        if G.number_of_nodes() == 0:
            return None, "No nodes found. Ensure format matches: V = {1, 2}"

        return G, None
    except Exception as e:
        return None, f"Parsing Error: {str(e)}"

def model_title_to_graph(title_query):
    """
    Converts a text title (e.g., "Cycle graph C5") into a NetworkX graph.
    1. Checks TITLE_DB for exact matches.
    2. Uses Regex for dynamic generation of common graphs.
    """
    clean_query = title_query.strip().lower()

    # --- STRATEGY 1: Reverse Lookup in TITLE_DB ---
    for g6_str, db_title in TITLE_DB.items():
        if clean_query == db_title.lower():
            G = nx.from_graph6_bytes(g6_str.encode())
            return G, f"Found in database: {db_title}"

    # --- STRATEGY 2: Dynamic Generation (Regex) ---
    # Cycle Graph
    match_c = re.search(r"(?:cycle|c)[^0-9]*(\d+)", clean_query)
    if match_c:
        n = int(match_c.group(1))
        return nx.cycle_graph(n), f"Generated Cycle Graph C{n}"

    # Complete Graph
    match_k = re.search(r"(?:complete|k)[^0-9]*(\d+)", clean_query)
    if match_k:
        n = int(match_k.group(1))
        return nx.complete_graph(n), f"Generated Complete Graph K{n}"

    # Path Graph
    match_p = re.search(r"(?:path|p)[^0-9]*(\d+)", clean_query)
    if match_p:
        n = int(match_p.group(1))
        return nx.path_graph(n), f"Generated Path Graph P{n}"

    # Star Graph
    match_s = re.search(r"(?:star|s)[^0-9]*(\d+)", clean_query)
    if match_s:
        n = int(match_s.group(1))
        return nx.star_graph(n), f"Generated Star Graph with {n} leaves"

    return None, "Could not interpret title. Try 'Cycle graph C5', 'K4', or 'Star 6'."

def model_image_to_data(uploaded_image):
    """
    PLACEHOLDER: Connect your AI Model (CaMeRa-style) here.
    Currently returns a mock result.
    """
    # Mock simulation
    n = random.randint(4, 8)
    G = nx.cycle_graph(n)
    return G, f"Simulated AI prediction: Detected Cycle Graph C{n}"

def render_graph(G):
    """Standard visualization using Matplotlib"""
    fig, ax = plt.subplots(figsize=(5, 4))
    pos = nx.kamada_kawai_layout(G) 
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='#d1c4e9', edge_color='#5e35b1', 
            node_size=500, font_weight='bold')
    return fig

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
    st.markdown("Generate and interpret diverse representations (Text ↔ Image ↔ Graph).")

    col_input, col_output = st.columns([1, 1], gap="medium")

    # --- LEFT COLUMN: INPUT ---
    with col_input:
        with st.container(border=True):
            st.subheader("📝 Input")
            
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
            if use_img:
                val_img = st.file_uploader("Upload Graph Image", type=["png", "jpg", "jpeg"])
                if val_img:
                    st.image(val_img, width=150)
                    st.success("Image Uploaded")

            # Action Buttons
            st.markdown("###")
            b_col1, b_col2 = st.columns([1, 2])
            with b_col1:
                if st.button("Clear"):
                    st.rerun()
            with b_col2:
                run_btn = st.button("Generate ✨", type="primary")

    # --- RIGHT COLUMN: OUTPUT ---
    with col_output:
        with st.container(border=True):
            st.subheader("✨ Output")
            
            if run_btn:
                results_G = None
                status_msg = ""

                # PRIORITY 1: Description -> Image & Title
                if use_desc and not use_img and val_desc:
                    st.info("Mode: Description ➔ Image & Title")
                    results_G, err = parse_description_to_graph(val_desc)
                    if err:
                        st.error(err)
                    else:
                        status_msg = "Graph parsed successfully from set notation."

                # PRIORITY 2: Image -> Description & Title
                elif use_img and val_img:
                    st.info("Mode: Image ➔ Description & Title")
                    results_G, note = model_image_to_data(val_img)
                    status_msg = note

                # PRIORITY 3: Title -> Image & Description
                elif use_title and val_title:
                    st.info(f"Mode: Title ('{val_title}') ➔ Graph")
                    results_G, note = model_title_to_graph(val_title)
                    status_msg = note
                
                # --- RENDER RESULTS ---
                if results_G:
                    # 1. Generate Title (using your backend)
                    try:
                        # If user provided a title, we might show it or regenerate a canonical one
                        generated_title = generate_title(results_G)
                        st.success(f"**Title:** {generated_title}")
                    except Exception as e:
                        st.warning(f"Could not generate canonical title: {e}")

                    # 2. Generate Description (V/E Sets)
                    nodes = sorted(list(results_G.nodes()))
                    edges = sorted(list(results_G.edges()))
                    # Format as set notation
                    desc_text = f"V = {set(nodes)}\nE = {set(edges)}"
                    st.text_area("Generated Description", value=desc_text, height=100)

                    # 3. Generate Image
                    fig = render_graph(results_G)
                    st.pyplot(fig)
                    
                    if status_msg:
                        st.caption(f"Status: {status_msg}")
                
                elif not results_G and not status_msg:
                     st.warning("Please select an input mode and provide valid data.")
            else:
                # Empty state
                st.markdown("*Waiting for input...*")

if __name__ == "__main__":
    main()