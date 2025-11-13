import cv2
import numpy as np
import json


class ImprovedGraphDetector:
    def __init__(self, image_path=None):
        self.image = cv2.imread(image_path) if image_path else None
        self.nodes = []
        self.edges = []
    
    def set_image(self, image):
        self.image = image.copy()
    
    def detect_nodes(self, min_radius=20, max_radius=100):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        self.nodes = []
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=100, param2=30, 
                                   minRadius=min_radius, maxRadius=max_radius)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i, (x, y, r) in enumerate(circles[0, :]):
                # Extract node region
                x1, y1 = max(0, x-r-10), max(0, y-r-10)
                x2, y2 = min(self.image.shape[1], x+r+10), min(self.image.shape[0], y+r+10)
                node_region = self.image[y1:y2, x1:x2]
                
                # Get dominant color
                mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
                cv2.circle(mask, (x-x1, y-y1), r, 255, -1)
                color = cv2.mean(node_region, mask=mask)[:3]
                color = tuple(map(int, color))
                
                # Try to extract text using simple method
                text = self._extract_text_simple(node_region)
                
                node = {
                    'id': i,
                    'center': (int(x), int(y)),
                    'radius': int(r),
                    'shape': 'circle',
                    'bbox': (x-r, y-r, 2*r, 2*r),
                    'color': color,
                    'text': text.strip()
                }
                self.nodes.append(node)
        
        # Also detect rectangles/squares
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_radius**2 * 3:
                continue
            
            # Check if it's a rectangle
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if this overlaps with existing circles
                center = (x + w//2, y + h//2)
                overlaps = False
                for node in self.nodes:
                    if node['shape'] == 'circle':
                        dist = np.sqrt((center[0] - node['center'][0])**2 + 
                                     (center[1] - node['center'][1])**2)
                        if dist < node['radius'] + max(w, h)//2:
                            overlaps = True
                            break
                
                if not overlaps:
                    node_region = self.image[y:y+h, x:x+w]
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    color = cv2.mean(self.image, mask=mask[y:y+h, x:x+w])[:3]
                    color = tuple(map(int, color))
                    
                    text = self._extract_text_simple(node_region)
                    
                    node = {
                        'id': len(self.nodes),
                        'center': center,
                        'shape': 'rectangle',
                        'bbox': (x, y, w, h),
                        'color': color,
                        'text': text.strip()
                    }
                    self.nodes.append(node)
        
        return self.nodes
    
    def _extract_text_simple(self, region):
        """Simple text extraction - looks for dark/light patterns"""
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find text contours (small regions)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_chars = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 20 < area < 1000:  # Likely text size
                    x, y, w, h = cv2.boundingRect(cnt)
                    if 0.3 < w/h < 3:  # Aspect ratio of text
                        text_chars.append(x)  # Just track that text exists
            
            # For now, return empty - OCR would go here
            return ""
        except:
            return ""
    
    def detect_edges(self, detect_arrows=True, edge_min_pixels=5, node_proximity=20):
        """Detect edges by removing nodes and finding remaining lines
        
        Args:
            detect_arrows: Whether to detect arrow directions
            edge_min_pixels: Minimum pixels for valid edge component
            node_proximity: Distance threshold for edge-node connection (pixels)
        """
        if not self.nodes:
            return []
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive threshold for better edge detection in varying lighting
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Also try regular threshold and combine
        _, binary2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_or(binary, binary2)
        
        # Create mask of nodes to remove them (with larger margin)
        node_mask = np.zeros_like(binary)
        for node in self.nodes:
            if node['shape'] == 'circle':
                cv2.circle(node_mask, node['center'], node['radius'] + 10, 255, -1)
            else:
                x, y, w, h = node['bbox']
                cv2.rectangle(node_mask, (x-5, y-5), (x+w+5, y+h+5), 255, -1)
        
        # Get edges (everything minus nodes)
        edges_only = cv2.bitwise_and(binary, cv2.bitwise_not(node_mask))
        
        # Thin the edges
        try:
            skeleton = cv2.ximgproc.thinning(edges_only)
        except:
            skeleton = edges_only
        
        # Find connected components in edge skeleton
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
        
        self.edges = []
        edge_id = 0
        
        # For each edge component, find which nodes it connects
        for label in range(1, num_labels):  # Skip background (0)
            # Get pixels of this edge component
            edge_mask = (labels == label).astype(np.uint8) * 255
            edge_pixels = cv2.findNonZero(edge_mask)
            
            if edge_pixels is None or len(edge_pixels) < edge_min_pixels:
                continue
            
            # Find which nodes this edge touches
            connected_nodes = []
            for node in self.nodes:
                center = node['center']
                radius = node.get('radius', max(node['bbox'][2], node['bbox'][3]) // 2)
                
                # Check if any edge pixel is near this node
                min_dist = float('inf')
                for pixel in edge_pixels:
                    px, py = pixel[0]
                    dist = np.sqrt((px - center[0])**2 + (py - center[1])**2)
                    min_dist = min(min_dist, dist)
                    
                    if dist < radius + node_proximity:  # Within reach of node
                        if node['id'] not in connected_nodes:
                            connected_nodes.append(node['id'])
                        break
            
            # If this edge connects exactly 2 nodes, it's valid
            if len(connected_nodes) == 2:
                source_id, target_id = connected_nodes[0], connected_nodes[1]
                
                # Determine direction based on arrow detection
                arrow_direction = None
                if detect_arrows:
                    arrow_direction = self._detect_arrow_direction(
                        edge_mask, self.nodes[source_id], self.nodes[target_id])
                
                # If arrow points to source, swap source and target
                if arrow_direction == 'source':
                    source_id, target_id = target_id, source_id
                    arrow_direction = 'target'  # Now it points to (new) target
                
                edge = {
                    'id': edge_id,
                    'source': source_id,
                    'target': target_id,
                    'directed': arrow_direction is not None,
                    'arrow_at_target': arrow_direction == 'target' if arrow_direction else False,
                    'color': (0, 0, 0),
                    'thickness': 2
                }
                self.edges.append(edge)
                edge_id += 1
        
        return self.edges
    
    def _detect_arrow_direction(self, edge_mask, source_node, target_node):
        """Detect if edge has arrow and which direction by looking for arrowhead"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        source_center = source_node['center']
        target_center = target_node['center']
        
        # Get node radii
        source_radius = source_node.get('radius', 40)
        target_radius = target_node.get('radius', 40)
        
        # Check for arrowhead near each node
        # Arrowhead detection: look for triangular filled region near node
        target_has_arrow = self._check_arrowhead_near_node(binary, target_center, 
                                                           source_center, target_radius)
        source_has_arrow = self._check_arrowhead_near_node(binary, source_center, 
                                                           target_center, source_radius)
        
        # Determine direction
        if target_has_arrow and not source_has_arrow:
            return 'target'  # Arrow points to target (source -> target)
        elif source_has_arrow and not target_has_arrow:
            return 'source'  # Arrow points to source (target -> source)
        elif target_has_arrow and source_has_arrow:
            # Both ends have arrows (bidirectional) - treat as directed to target
            return 'target'
        
        return None  # Undirected (no arrows detected)
    
    def _check_arrowhead_near_node(self, binary, node_center, other_center, radius):
        """Check if there's an arrowhead pattern near a node"""
        # Define search region near the node (outside the node circle)
        search_radius = int(radius * 1.5)
        
        # Calculate the direction from other node to this node
        dx = node_center[0] - other_center[0]
        dy = node_center[1] - other_center[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 1:
            return False
        
        # Normalize direction
        dx /= dist
        dy /= dist
        
        # Sample region just outside the node where arrowhead would be
        # Start from edge of node and look outward
        sample_start = radius + 2
        sample_end = radius + 20
        
        # Sample multiple points along the approach direction
        arrow_pixels = 0
        total_samples = 0
        
        for d in range(int(sample_start), int(sample_end), 2):
            # Point along the line from other node to this node
            px = int(node_center[0] - dx * d)
            py = int(node_center[1] - dy * d)
            
            if 0 <= px < binary.shape[1] and 0 <= py < binary.shape[0]:
                # Check a small perpendicular area (for arrowhead width)
                perpendicular_samples = 0
                perpendicular_hits = 0
                
                for offset in range(-8, 9, 2):
                    # Perpendicular direction
                    perp_x = int(px - dy * offset)
                    perp_y = int(py + dx * offset)
                    
                    if 0 <= perp_x < binary.shape[1] and 0 <= perp_y < binary.shape[0]:
                        perpendicular_samples += 1
                        if binary[perp_y, perp_x] > 0:
                            perpendicular_hits += 1
                
                # If we find a wide region of pixels (arrowhead is wider than line)
                if perpendicular_samples > 0:
                    width_ratio = perpendicular_hits / perpendicular_samples
                    if width_ratio > 0.3:  # At least 30% filled in perpendicular
                        arrow_pixels += 1
                    total_samples += 1
        
        # If we found a thickening pattern (arrowhead), return True
        if total_samples > 0:
            arrow_ratio = arrow_pixels / total_samples
            return arrow_ratio > 0.3  # At least 30% of samples show thickening
        
        return False
    
    def get_graph_representation(self):
        """Get V and E representation"""
        vertices = set()
        node_names = {}
        
        for node in self.nodes:
            name = node['text'] if node['text'] else str(node['id'])
            vertices.add(name)
            node_names[node['id']] = name
        
        edges = []
        for edge in self.edges:
            source_name = node_names.get(edge['source'], str(edge['source']))
            target_name = node_names.get(edge['target'], str(edge['target']))
            
            if edge['directed']:
                edges.append((source_name, target_name))
            else:
                edge_pair = tuple(sorted([source_name, target_name]))
                if edge_pair not in edges:
                    edges.append(edge_pair)
        
        v_str = "V = {" + ", ".join(sorted(vertices)) + "}"
        e_str = "E = {"
        if edges:
            e_str += ", ".join([f"({s},{t})" for s, t in edges])
        e_str += "}"
        
        return vertices, edges, f"{v_str}\n{e_str}"
    
    def visualize(self, show_representation=True):
        """Visualize detected graph"""
        result = self.image.copy()
        
        # Draw edges
        for edge in self.edges:
            src_node = self.nodes[edge['source']]
            tgt_node = self.nodes[edge['target']]
            
            cv2.line(result, src_node['center'], tgt_node['center'], (0, 255, 0), 2)
            
            if edge['directed']:
                # Draw arrow indicator
                cv2.circle(result, tgt_node['center'], 10, (0, 0, 255), 2)
        
        # Draw nodes
        for node in self.nodes:
            if node['shape'] == 'circle':
                cv2.circle(result, node['center'], node['radius'], (0, 255, 0), 3)
            else:
                x, y, w, h = node['bbox']
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Label
            label = node['text'] if node['text'] else str(node['id'])
            cv2.putText(result, label, (node['center'][0] + 20, node['center'][1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Add representation
        if show_representation:
            _, _, rep = self.get_graph_representation()
            self._add_overlay(result, rep)
        
        return result
    
    def _add_overlay(self, image, text):
        """Add text overlay"""
        lines = text.split('\n')
        y_offset = 20
        
        for line in lines:
            cv2.putText(image, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 5)
            cv2.putText(image, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            y_offset += 30


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'sample_complex.png'
    
    print(f"Processing: {image_path}\n")
    
    detector = ImprovedGraphDetector(image_path)
    
    # Detect nodes
    nodes = detector.detect_nodes(min_radius=30, max_radius=60)
    print(f"✓ Found {len(nodes)} nodes")
    for node in nodes:
        print(f"  Node {node['id']}: {node['shape']} at {node['center']}, text='{node['text']}'")
    
    # Detect edges
    edges = detector.detect_edges(detect_arrows=True)
    print(f"\n✓ Found {len(edges)} edges")
    for edge in edges:
        arrow = "→" if edge['directed'] else "─"
        print(f"  Edge {edge['id']}: {edge['source']} {arrow} {edge['target']}")
    
    # Get representation
    print("\n" + "="*60)
    v, e, rep = detector.get_graph_representation()
    print(rep)
    print("="*60 + "\n")
    
    # Visualize
    result = detector.visualize()
    
    cv2.imshow("Original", detector.image)
    cv2.imshow("Detected", result)
    
    output_path = image_path.rsplit('.', 1)[0] + '_improved_detected.png'
    cv2.imwrite(output_path, result)
    print(f"Saved to: {output_path}")
    
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

