"""
Enhanced app.py with real-time WebRTC streaming support
Install: pip install streamlit-webrtc
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Try to import streamlit-webrtc
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

from image_to_description.improved_detector import ImprovedGraphDetector

class GraphDetectorTransformer(VideoTransformerBase):
    """Transform video frames with graph detection"""
    
    def __init__(self):
        self.detector = ImprovedGraphDetector()
        self.detect_arrows = True
        self.min_radius = 30
        self.max_radius = 100
        self.edge_min_pixels = 3
        self.node_proximity = 35
        self.show_overlay = True
        self.last_nodes = []
        self.last_edges = []
    
    def transform(self, frame):
        """Process each frame from webcam"""
        img = frame.to_ndarray(format="bgr24")
        
        if not self.show_overlay:
            return img
        
        try:
            # Set image and detect
            self.detector.set_image(img)
            self.last_nodes = self.detector.detect_nodes(
                min_radius=self.min_radius, 
                max_radius=self.max_radius
            )
            
            if self.last_nodes:
                self.last_edges = self.detector.detect_edges(
                    detect_arrows=self.detect_arrows,
                    edge_min_pixels=self.edge_min_pixels,
                    node_proximity=self.node_proximity
                )
                
                # Visualize
                result = self.detector.visualize(show_representation=False)
                return result
            else:
                return img
                
        except Exception as e:
            # Return original frame on error
            return img

def realtime_webrtc_ui():
    """Real-time detection using WebRTC"""
    st.subheader("Real-Time Graph Detection (WebRTC)")
    
    if not WEBRTC_AVAILABLE:
        st.error("streamlit-webrtc not installed!")
        st.code("pip install streamlit-webrtc")
        st.info("After installing, restart the app to use real-time detection.")
        return
    
    st.markdown("True real-time detection with continuous video streaming")
    
    # Settings in sidebar
    with st.sidebar:
        st.markdown("### Detection Settings")
        detect_arrows = st.checkbox("Detect Arrows", value=True, key="webrtc_arrows")
        min_radius = st.slider("Min Radius", 10, 80, 30, key="webrtc_min")
        max_radius = st.slider("Max Radius", 40, 200, 100, key="webrtc_max")
        edge_sensitivity = st.slider("Edge Sensitivity", 1, 10, 8, key="webrtc_edge")
        edge_min_pixels = 11 - edge_sensitivity
        node_proximity = st.slider("Node Proximity", 15, 60, 35, key="webrtc_prox")
        show_overlay = st.checkbox("Show Detection Overlay", value=True)
    
    # WebRTC streamer
    ctx = webrtc_streamer(
        key="graph-detection",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=GraphDetectorTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Update transformer settings
    if ctx.video_transformer:
        ctx.video_transformer.detect_arrows = detect_arrows
        ctx.video_transformer.min_radius = min_radius
        ctx.video_transformer.max_radius = max_radius
        ctx.video_transformer.edge_min_pixels = edge_min_pixels
        ctx.video_transformer.node_proximity = node_proximity
        ctx.video_transformer.show_overlay = show_overlay
        
        # Show current detection stats
        if ctx.video_transformer.last_nodes:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nodes", len(ctx.video_transformer.last_nodes))
            with col2:
                st.metric("Edges", len(ctx.video_transformer.last_edges))

# Usage example
if __name__ == "__main__":
    st.set_page_config(page_title="Real-Time Graph Detection", layout="wide")
    st.title("Real-Time Graph Detection with WebRTC")
    
    tab1, tab2 = st.tabs(["WebRTC (Real-Time)", "Photo-Based"])
    
    with tab1:
        realtime_webrtc_ui()
    
    with tab2:
        st.info("Use the main app.py for photo-based detection")


