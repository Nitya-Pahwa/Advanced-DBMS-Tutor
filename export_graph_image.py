
"""
Purpose:
Exports the LangGraph pipeline as architecture.png
for the GitHub README.

Run ONCE:
    python export_graph_image.py

Output:
    architecture.png  (in project root)

Requirements:
    pip install langgraph grandalf   (grandalf enables PNG export)
"""

from graph_flow import compile_graph

def export_architecture_png(output_path: str = "architecture.png"):
    # Compile the graph (no display call inside now — see graph_flow fix)
    app = compile_graph()

    # Use LangGraph's built-in Mermaid → PNG renderer
    png_bytes = app.get_graph().draw_mermaid_png()

    with open(output_path, "wb") as f:
        f.write(png_bytes)

    print(f"  Architecture diagram saved → {output_path}")


if __name__ == "__main__":
    export_architecture_png()