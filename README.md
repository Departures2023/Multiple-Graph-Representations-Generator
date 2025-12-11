# Multiple Graph Representations Generator

A comprehensive tool for working with graphs in multiple representations: text descriptions, images, titles, and real-time camera detection. Includes powerful graph isomorphism checking capabilities for structural analysis.

## Features

### Multi-Representation Conversion
- **Description ↔ Image ↔ Title**: Seamlessly convert between different graph representations
- **Edge List Format**: Parse and generate edge lists in set notation (V = {...}, E = {...})
- **Visual Rendering**: Generate beautiful graph visualizations using NetworkX and Matplotlib

### Real-Time Detection
- **Camera Integration**: Capture graphs directly from your webcam or camera
- **Image Upload**: Upload images of hand-drawn or printed graphs
- **Smart Detection**: Automatically detect nodes (circles, rectangles) and edges (directed/undirected)
- **Adjustable Parameters**: Fine-tune detection sensitivity for different graph types

### Graph Isomorphism Checker
- **Isomorphism Testing**: Check if two graphs are structurally identical
- **Multiple Input Methods**: Compare graphs from descriptions, images, or camera captures
- **Detailed Analysis**: View node mappings, symmetry counts, and structural properties
- **Automorphism Counting**: Calculate graph symmetries automatically

### Graph Analysis
- **Automorphism Detection**: Count and enumerate graph symmetries
- **Vertex Transitivity**: Check if graphs are vertex-transitive
- **Canonical Titles**: Generate standard mathematical names for common graphs
- **Graph Properties**: Analyze nodes, edges, and structural characteristics

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Multiple-Graph-Representations-Generator.git
cd Multiple-Graph-Representations-Generator
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Dependencies
- `streamlit` - Web application framework
- `networkx` - Graph algorithms and analysis
- `matplotlib` - Graph visualization
- `opencv-python` - Image processing and detection
- `opencv-contrib-python` - Extended OpenCV features
- `numpy` - Numerical computations
- `Pillow` - Image handling

## Usage

### Running the Web Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your browser with three main tabs:

#### 1. Generator Tab
Convert between different graph representations:
- **Input**: Choose from Title, Description, or Image
- **Output**: Get all other representations automatically
- **Example**: Input "Cycle graph C5" → Get edge list + visualization

#### 2. Camera Detection Tab
Detect graphs from images or camera:
- **Camera Capture**: Use built-in camera to photograph graphs
- **Image Upload**: Upload existing graph images
- **Detection Settings**: Adjust node radius and arrow detection
- **Save Results**: Store detected graphs for isomorphism checking

#### 3. Isomorphism Checker Tab
Compare two graphs for structural equivalence:
- **Input Methods**: Description, Image, or Saved graphs
- **Visual Comparison**: Side-by-side graph visualization
- **Detailed Results**: Node mappings, symmetries, and statistics
- **Multiple Tests**: Compare any combination of input types

### Command-Line Tools

#### Real-Time Graph Detector
```bash
# Run with webcam
python image_to_description/realtime_graph_detector.py webcam

# Process a single image
python image_to_description/realtime_graph_detector.py path/to/image.png

# Create sample graphs for testing
python image_to_description/realtime_graph_detector.py create-samples
```

#### Improved Detector
```bash
# Detect graph from image
python image_to_description/improved_detector.py path/to/image.png
```

#### Integration Tests
```bash
# Run all tests
python test_integration.py
```

## Examples

### Example 1: Description to Image
**Input**:
```
V = {1, 2, 3, 4}
E = {(1, 2), (2, 3), (3, 4), (4, 1)}
```
**Output**: 
- Title: "Cycle graph C4"
- Visual graph rendering
- Edge list format

### Example 2: Image Detection
**Input**: Photo of a hand-drawn graph
**Output**:
- Detected nodes and edges
- V/E set notation
- Canonical title (if recognized)

### Example 3: Isomorphism Check
**Graph 1**: `E = {(0,1), (1,2), (2,0)}`  (Triangle)
**Graph 2**: `E = {(10,11), (11,12), (12,10)}`  (Triangle with different labels)
**Result**: ✅ ISOMORPHIC with mapping: {0→10, 1→11, 2→12}

## Testing

The project includes comprehensive integration tests:

```bash
python test_integration.py
```

Tests cover:
- Isomorphism detection
- Automorphism counting
- Image detection
- Graph class conversions
- Multi-representation integration

## Project Structure

```
Multiple-Graph-Representations-Generator/
├── app.py                          # Main Streamlit application
├── graph.py                        # Core Graph class
├── automorphisms.py                # Isomorphism & automorphism functions
├── main.py                         # Example usage
├── test_integration.py             # Integration tests
├── requirements.txt                # Python dependencies
├── description_to_image/           # Image generation module
│   ├── description_to_image.py
│   └── generate_graph_image.py
├── image_to_description/           # Image detection module
│   ├── improved_detector.py        # Advanced graph detector
│   ├── realtime_graph_detector.py  # Real-time detection
│   └── sample_*.png                # Sample images
└── src/
    ├── graph_title.py              # Title generation
    └── test.py                     # Unit tests
```

## API Reference

### Graph Class
```python
from graph import Graph

# Create from description
g = Graph(description=[(0,1), (1,2), (2,0)])

# Create from image
from PIL import Image
img = Image.open("graph.png")
g = Graph(image=img)

# Create from title
g = Graph(title="Cycle graph C5")

# Access representations
print(g.description)  # Edge list
print(g.title)        # Canonical name
g.image.show()        # Display image
```

### Isomorphism Functions
```python
from automorphisms import (
    are_graphs_isomorphic,
    compare_graphs,
    find_isomorphism,
    count_automorphisms
)

# Check if isomorphic
graph1 = [(0,1), (1,2), (2,0)]
graph2 = [(10,11), (11,12), (12,10)]
is_iso = are_graphs_isomorphic(graph1, graph2)

# Get detailed comparison
comparison = compare_graphs(graph1, graph2)
print(comparison['are_isomorphic'])
print(comparison['isomorphism_mapping'])

# Count symmetries
symmetries = count_automorphisms(graph1)
```

### Image Detection
```python
from image_to_description.improved_detector import ImprovedGraphDetector
from PIL import Image

# Detect from image
img = Image.open("graph.png")
detector = ImprovedGraphDetector(img)

# Get edge list directly
edges = detector.detect_and_get_edges(
    min_radius=20,
    max_radius=80,
    detect_arrows=True
)
```

## Use Cases

1. **Education**: Teach graph theory with visual representations
2. **Research**: Analyze graph structures and isomorphisms
3. **Documentation**: Convert hand-drawn graphs to digital format
4. **Verification**: Check if two graphs are structurally identical
5. **Analysis**: Study graph symmetries and properties

## Contributing

Contributions are welcome! Areas for improvement:
- Enhanced OCR for node labels
- Support for weighted edges
- More graph types in title database
- Improved arrow detection algorithms
- Export to various graph formats

## License

This project is open source and available under the MIT License.

## Acknowledgments

- NetworkX for graph algorithms
- OpenCV for image processing
- Streamlit for the web framework
- The graph theory community for canonical graph definitions

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

Made for the graph theory community
