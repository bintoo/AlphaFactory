
import os
import sys

# Ensure parent directory is in path so we can import alpha_factory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_factory.graph import AlphaGraphFactory

def generate_graph_image():
    print("Initializing AlphaGraphFactory...")
    factory = AlphaGraphFactory()
    
    print("Creating graph...")
    app = factory.create_graph()
    
    # Define output path
    output_dir = os.path.join(os.path.dirname(__file__), "graphStructure")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "graph.png")
    
    print(f"Generating Mermaid PNG to {output_path}...")
    try:
        # 1. Print Syntax
        mermaid_syntax = app.get_graph().draw_mermaid()
        print("\n--- MERMAID SYNTAX ---\n")
        print(mermaid_syntax)
        print("\n----------------------\n")

        # 2. Try PNG
        png_data = app.get_graph().draw_mermaid_png()
        
        with open(output_path, "wb") as f:
            f.write(png_data)
        
        print(f"Graph successfully saved to: {output_path}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error generating graph image: {str(e)}")
        print("Ensure you have 'langgraph' installed.")

if __name__ == "__main__":
    generate_graph_image()
