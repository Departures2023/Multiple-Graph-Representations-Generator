import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import re
import io
from PIL import Image

# Import backend logic
from graph import Graph
from src.graph_title import TITLE_DB

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

def model_image_to_data(uploaded_image):
    """
    Converts an uploaded image to a Graph object using image detection.
    """
    try:
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_image)
        
        # Create Graph object from image
        g = Graph(image=image)
        
        if g.description:
            return g, f"Successfully detected graph from image"
        else:
            return None, "Could not detect graph structure from image"
            
    except Exception as e:
        return None, f"Image processing error: {str(e)}"

def render_graph(graph_obj):
    """Standard visualization using Matplotlib from Graph object"""
    # If the Graph object has an image, we could display it directly
    # But for consistency, we'll render from the description
    if graph_obj.description:
        G = nx.Graph()
        G.add_edges_from(graph_obj.description)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        pos = nx.kamada_kawai_layout(G) 
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='#d1c4e9', edge_color='#5e35b1', 
                node_size=500, font_weight='bold')
        return fig
    elif graph_obj.image:
        # If we only have image, display it
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(graph_obj.image)
        ax.axis('off')
        return fig
    return None

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
                results_graph = None
                status_msg = ""

                # PRIORITY 1: Description -> Image & Title
                if use_desc and not use_img and val_desc:
                    st.info("Mode: Description ➔ Image & Title")
                    results_graph, err = parse_description_to_graph(val_desc)
                    if err:
                        st.error(err)
                    else:
                        status_msg = "Graph parsed successfully from set notation."

                # PRIORITY 2: Image -> Description & Title
                elif use_img and val_img:
                    st.info("Mode: Image ➔ Description & Title")
                    results_graph, note = model_image_to_data(val_img)
                    status_msg = note

                # PRIORITY 3: Title -> Image & Description
                elif use_title and val_title:
                    st.info(f"Mode: Title ('{val_title}') ➔ Graph")
                    results_graph, note = model_title_to_graph(val_title)
                    status_msg = note
                
                # --- RENDER RESULTS ---
                if results_graph:
                    # 1. Display Title (from Graph object)
                    if results_graph.title:
                        st.success(f"**Title:** {results_graph.title}")
                    else:
                        st.warning("Could not generate canonical title")

                    # 2. Generate Description (V/E Sets)
                    if results_graph.description:
                        # Get all unique nodes from edges
                        nodes = set()
                        for u, v in results_graph.description:
                            nodes.add(u)
                            nodes.add(v)
                        nodes = sorted(list(nodes))
                        edges = sorted(results_graph.description)
                        # Format as set notation
                        desc_text = f"V = {set(nodes)}\nE = {set(edges)}"
                        st.text_area("Generated Description", value=desc_text, height=100)

                    # 3. Generate Image
                    fig = render_graph(results_graph)
                    if fig:
                        st.pyplot(fig)
                    
                    if status_msg:
                        st.caption(f"Status: {status_msg}")
                
                elif not results_graph and not status_msg:
                     st.warning("Please select an input mode and provide valid data.")
            else:
                # Empty state
                st.markdown("*Waiting for input...*")

if __name__ == "__main__":
    main()