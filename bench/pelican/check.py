# Run by scripts/agent_bench.py in the agent's copy of this folder, which does
# not contain this file. Exits non-zero unless pelican.svg is an SVG document
# that draws something. Whether it is a pelican on a bicycle takes a look.
import sys
import xml.etree.ElementTree as ET

SHAPES = {"path", "circle", "ellipse", "rect", "line", "polyline", "polygon"}

try:
    root = ET.parse("pelican.svg").getroot()
except (OSError, ET.ParseError) as e:
    sys.exit(f"pelican.svg: {e}")
if root.tag.rsplit("}", 1)[-1] != "svg":
    sys.exit(f"root element is {root.tag}, want svg")
shapes = sum(el.tag.rsplit("}", 1)[-1] in SHAPES for el in root.iter())
if shapes < 3:
    sys.exit(f"{shapes} shapes, want at least 3")
