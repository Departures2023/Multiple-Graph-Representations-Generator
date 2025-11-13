"""
Real-time Graph Detector with Live Representation Display
Shows V = {...} and E = {...} format in real-time
"""

from improved_detector import ImprovedGraphDetector
import cv2
import numpy as np


class RealtimeGraphDetector:
    def __init__(self):
        self.detector = ImprovedGraphDetector()
        self.last_representation = "No graph detected yet"
        
    def process_frame(self, frame, detect_arrows=True, min_radius=30, max_radius=60, 
                      edge_min_pixels=3, node_proximity=25):
        """Process a frame and detect graph"""
        self.detector.set_image(frame)
        
        # Detect nodes and edges
        nodes = self.detector.detect_nodes(min_radius=min_radius, max_radius=max_radius)
        edges = self.detector.detect_edges(detect_arrows=detect_arrows, 
                                          edge_min_pixels=edge_min_pixels,
                                          node_proximity=node_proximity)
        
        # Get representation
        if nodes:
            _, _, representation = self.detector.get_graph_representation()
            self.last_representation = representation
        else:
            self.last_representation = "No graph detected"
        
        # Visualize with representation overlay
        result = self.detector.visualize(show_representation=True)
        
        return result, nodes, edges
    
    def run_webcam(self):
        """Run real-time detection from webcam"""
        cap = cv2.VideoCapture(0)
        detect_arrows = True
        auto_detect = False
        min_radius = 30
        max_radius = 60
        edge_min_pixels = 3
        node_proximity = 25
        
        print("\n" + "="*60)
        print("Real-Time Graph Detector")
        print("="*60)
        print("\nControls:")
        print("  SPACE   - Capture and detect graph")
        print("  'a'     - Toggle auto-detect mode")
        print("  'd'     - Toggle arrow detection (directed/undirected)")
        print("  'r'     - Show last representation in console")
        print("  '+'     - Increase node detection size")
        print("  '-'     - Decrease node detection size")
        print("  ']'     - Increase edge sensitivity (more edges)")
        print("  '['     - Decrease edge sensitivity (fewer edges)")
        print("  'q'     - Quit")
        print("\n" + "="*60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            display = frame.copy()
            
            # Add status text
            status = f"Arrow: {'ON' if detect_arrows else 'OFF'} | Auto: {'ON' if auto_detect else 'OFF'}"
            cv2.putText(display, status, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            status2 = f"Node: {min_radius}-{max_radius}px | Edge Prox: {node_proximity}px | Min: {edge_min_pixels}px"
            cv2.putText(display, status2, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Webcam Feed", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('a'):
                auto_detect = not auto_detect
                print(f"Auto-detect: {'ON' if auto_detect else 'OFF'}")
            
            elif key == ord('d'):
                detect_arrows = not detect_arrows
                print(f"Arrow detection: {'ON' if detect_arrows else 'OFF'}")
            
            elif key == ord('+') or key == ord('='):
                min_radius += 5
                max_radius += 5
                print(f"Node size increased: {min_radius}-{max_radius}px")
            
            elif key == ord('-') or key == ord('_'):
                min_radius = max(10, min_radius - 5)
                max_radius = max(20, max_radius - 5)
                print(f"Node size decreased: {min_radius}-{max_radius}px")
            
            elif key == ord(']') or key == ord('}'):
                node_proximity += 5
                edge_min_pixels = max(1, edge_min_pixels - 1)
                print(f"Edge sensitivity increased: proximity={node_proximity}px, min_pixels={edge_min_pixels}")
            
            elif key == ord('[') or key == ord('{'):
                node_proximity = max(10, node_proximity - 5)
                edge_min_pixels += 1
                print(f"Edge sensitivity decreased: proximity={node_proximity}px, min_pixels={edge_min_pixels}")
            
            elif key == ord('r'):
                print("\nCurrent Graph Representation:")
                print("-" * 60)
                print(self.last_representation)
                print("-" * 60 + "\n")
            
            elif key == ord(' ') or auto_detect:
                result, nodes, edges = self.process_frame(frame, detect_arrows, min_radius, max_radius,
                                                         edge_min_pixels, node_proximity)
                
                cv2.imshow("Detected Graph", result)
                
                # Print to console
                print("\n" + "="*60)
                print(f"Detected: {len(nodes)} nodes, {len(edges)} edges")
                print(f"Settings: node_proximity={node_proximity}px, edge_min_pixels={edge_min_pixels}")
                print("="*60)
                print(self.last_representation)
                print("="*60 + "\n")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_on_image(self, image_path, detect_arrows=True, min_radius=30, max_radius=60):
        """Run detection on a static image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        print(f"\nProcessing: {image_path}")
        print(f"Arrow detection: {'ON' if detect_arrows else 'OFF'}")
        print(f"Node size range: {min_radius}-{max_radius}px\n")
        
        result, nodes, edges = self.process_frame(image, detect_arrows, min_radius, max_radius)
        
        print("="*60)
        print(f"Detection Results")
        print("="*60)
        print(f"\nNodes: {len(nodes)}")
        for node in nodes:
            text_info = f" '{node['text']}'" if node.get('text') else ""
            print(f"  Node {node['id']}: {node['shape']} at {node['center']}{text_info}")
        
        print(f"\nEdges: {len(edges)}")
        for edge in edges:
            arrow = " →" if edge['directed'] else " ─"
            print(f"  Edge {edge['id']}: {edge['source']}{arrow}{edge['target']}")
        
        print("\n" + "="*60)
        print("Graph Representation:")
        print("="*60)
        print(self.last_representation)
        print("="*60 + "\n")
        
        # Show results
        cv2.imshow("Original", image)
        cv2.imshow("Detected Graph", result)
        
        # Save output
        output_path = image_path.rsplit('.', 1)[0] + '_realtime_detected.png'
        cv2.imwrite(output_path, result)
        print(f"Saved to: {output_path}")
        
        # Save representation to text file
        txt_path = image_path.rsplit('.', 1)[0] + '_graph.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Graph Representation\n")
            f.write("="*60 + "\n\n")
            f.write(self.last_representation + "\n\n")
            f.write("="*60 + "\n\n")
            f.write(f"Nodes: {len(nodes)}\n")
            for node in nodes:
                text_info = f" '{node['text']}'" if node.get('text') else ""
                f.write(f"  Node {node['id']}: {node['shape']} at {node['center']}{text_info}\n")
            f.write(f"\nEdges: {len(edges)}\n")
            for edge in edges:
                arrow = " →" if edge['directed'] else " ─"
                f.write(f"  Edge {edge['id']}: {edge['source']}{arrow}{edge['target']}\n")
        print(f"Representation saved to: {txt_path}")
        
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def create_sample_graphs():
    """Create sample graphs for testing"""
    
    # Sample 1: Simple directed graph
    print("Creating sample graphs...")
    
    # Directed graph: A -> B -> C -> D
    img1 = np.ones((400, 800, 3), dtype=np.uint8) * 255
    
    nodes = [(150, 200, 'A'), (300, 200, 'B'), (450, 200, 'C'), (600, 200, 'D')]
    
    for x, y, label in nodes:
        cv2.circle(img1, (x, y), 40, (150, 200, 255), -1)
        cv2.circle(img1, (x, y), 40, (0, 0, 0), 2)
        cv2.putText(img1, label, (x-12, y+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Draw directed edges
    cv2.arrowedLine(img1, (190, 200), (260, 200), (0, 0, 0), 3, tipLength=0.05)
    cv2.arrowedLine(img1, (340, 200), (410, 200), (0, 0, 0), 3, tipLength=0.05)
    cv2.arrowedLine(img1, (490, 200), (560, 200), (0, 0, 0), 3, tipLength=0.05)
    
    cv2.imwrite('sample_directed.png', img1)
    print("✓ Created: sample_directed.png")
    
    # Sample 2: Undirected graph with cycle
    img2 = np.ones((500, 500, 3), dtype=np.uint8) * 255
    
    nodes = [(250, 100, 'A'), (400, 250, 'B'), (250, 400, 'C'), (100, 250, 'D')]
    
    for x, y, label in nodes:
        cv2.circle(img2, (x, y), 35, (200, 150, 255), -1)
        cv2.circle(img2, (x, y), 35, (0, 0, 0), 2)
        cv2.putText(img2, label, (x-10, y+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Draw undirected edges (square)
    cv2.line(img2, (285, 120), (370, 230), (0, 0, 0), 3)  # A-B
    cv2.line(img2, (380, 285), (270, 370), (0, 0, 0), 3)  # B-C
    cv2.line(img2, (220, 380), (130, 280), (0, 0, 0), 3)  # C-D
    cv2.line(img2, (120, 220), (220, 130), (0, 0, 0), 3)  # D-A
    cv2.line(img2, (250, 135), (250, 365), (0, 0, 0), 3)  # A-C diagonal
    
    cv2.imwrite('sample_undirected.png', img2)
    print("✓ Created: sample_undirected.png")
    
    # Sample 3: Complex directed graph
    img3 = np.ones((500, 600, 3), dtype=np.uint8) * 255
    
    nodes = [(150, 150, 'a'), (450, 150, 'b'), (150, 350, 'c'), (450, 350, 'd'), (300, 250, 'e')]
    
    for x, y, label in nodes:
        cv2.circle(img3, (x, y), 40, (255, 200, 150), -1)
        cv2.circle(img3, (x, y), 40, (0, 0, 0), 2)
        cv2.putText(img3, label, (x-12, y+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Complex edges
    cv2.arrowedLine(img3, (190, 150), (410, 150), (0, 0, 0), 3, tipLength=0.03)  # a->b
    cv2.arrowedLine(img3, (150, 190), (150, 310), (0, 0, 0), 3, tipLength=0.03)  # a->c
    cv2.arrowedLine(img3, (450, 190), (450, 310), (0, 0, 0), 3, tipLength=0.03)  # b->d
    cv2.arrowedLine(img3, (190, 350), (410, 350), (0, 0, 0), 3, tipLength=0.03)  # c->d
    cv2.arrowedLine(img3, (175, 180), (275, 230), (0, 0, 0), 3, tipLength=0.05)  # a->e
    cv2.arrowedLine(img3, (325, 230), (425, 180), (0, 0, 0), 3, tipLength=0.05)  # e->b
    
    cv2.imwrite('sample_complex.png', img3)
    print("✓ Created: sample_complex.png")
    
    print("\nSample graphs created successfully!\n")


def main():
    import sys
    
    rt_detector = RealtimeGraphDetector()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "webcam":
            # Webcam mode
            rt_detector.run_webcam()
        
        elif sys.argv[1] == "create-samples":
            # Create sample graphs
            create_sample_graphs()
        
        else:
            # Process image file
            image_path = sys.argv[1]
            detect_arrows = True
            if len(sys.argv) > 2:
                detect_arrows = sys.argv[2].lower() in ['true', '1', 'yes', 'directed']
            
            rt_detector.run_on_image(image_path, detect_arrows)
    
    else:
        print("\nReal-time Graph Detector")
        print("="*60)
        print("\nUsage:")
        print("  python realtime_graph_detector.py webcam")
        print("  python realtime_graph_detector.py <image_path> [directed]")
        print("  python realtime_graph_detector.py create-samples")
        print("\nExamples:")
        print("  python realtime_graph_detector.py webcam")
        print("  python realtime_graph_detector.py graph.png")
        print("  python realtime_graph_detector.py graph.png directed")
        print("  python realtime_graph_detector.py create-samples")
        print("\n" + "="*60)
        
        # Interactive menu
        print("\nWhat would you like to do?")
        print("1. Run webcam detection")
        print("2. Create sample graphs")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            rt_detector.run_webcam()
        elif choice == "2":
            create_sample_graphs()
            print("\nProcess a sample? (y/n): ", end='')
            if input().lower() == 'y':
                print("\n1. sample_directed.png")
                print("2. sample_undirected.png")
                print("3. sample_complex.png")
                sample_choice = input("\nChoose sample (1-3): ").strip()
                
                samples = {
                    '1': ('sample_directed.png', True),
                    '2': ('sample_undirected.png', False),
                    '3': ('sample_complex.png', True)
                }
                
                if sample_choice in samples:
                    path, arrows = samples[sample_choice]
                    rt_detector.run_on_image(path, arrows)
        else:
            print("Goodbye!")


if __name__ == "__main__":
    main()

