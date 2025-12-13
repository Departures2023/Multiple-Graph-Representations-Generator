"""
Test edge detection on all images in image_to_description directory
"""
import os
from PIL import Image
from image_to_description.improved_detector import ImprovedGraphDetector

# Get all PNG images
image_dir = "image_to_description"
images = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.endswith('_detected.png')]

print(f"Found {len(images)} images to test\n")
print("="*80)

results = []

for img_file in sorted(images):
    img_path = os.path.join(image_dir, img_file)
    print(f"\nTesting: {img_file}")
    print("-" * 80)
    
    try:
        image = Image.open(img_path)
        detector = ImprovedGraphDetector(image, use_ocr=False)  # Disable OCR for speed
        
        # Try different parameter sets
        param_sets = [
            {"name": "Default", "min_r": 20, "max_r": 40, "edge_pix": 2, "prox": 40},
            {"name": "Aggressive", "min_r": 10, "max_r": 60, "edge_pix": 0, "prox": 60},
            {"name": "Computer-rendered", "min_r": 8, "max_r": 35, "edge_pix": 0, "prox": 225},
        ]
        
        best_result = None
        best_nodes = 0
        best_edges = 0
        
        for params in param_sets:
            try:
                nodes = detector.detect_nodes(min_radius=params["min_r"], max_radius=params["max_r"])
                edges = detector.detect_edges(edge_min_pixels=params["edge_pix"], node_proximity=params["prox"])
                
                if len(nodes) > best_nodes or (len(nodes) == best_nodes and len(edges) > best_edges):
                    best_result = params
                    best_nodes = len(nodes)
                    best_edges = len(edges)
                    
                print(f"  {params['name']:20s}: {len(nodes):2d} nodes, {len(edges):2d} edges")
            except Exception as e:
                print(f"  {params['name']:20s}: ERROR - {str(e)}")
        
        if best_result:
            results.append({
                "file": img_file,
                "best_params": best_result,
                "nodes": best_nodes,
                "edges": best_edges
            })
            print(f"\n  ✓ Best: {best_result['name']} - {best_nodes} nodes, {best_edges} edges")
        else:
            results.append({
                "file": img_file,
                "best_params": None,
                "nodes": 0,
                "edges": 0
            })
            print(f"\n  ✗ No successful detection")
            
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        results.append({
            "file": img_file,
            "best_params": None,
            "nodes": 0,
            "edges": 0,
            "error": str(e)
        })

print("\n" + "="*80)
print("\nSUMMARY:")
print("="*80)
for r in results:
    if r['best_params']:
        print(f"{r['file']:40s}: {r['nodes']:2d} nodes, {r['edges']:2d} edges ({r['best_params']['name']})")
    else:
        print(f"{r['file']:40s}: FAILED")

