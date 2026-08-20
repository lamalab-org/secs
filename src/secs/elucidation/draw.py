from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# ChemDraw-inspired palette from the smiles-hover renderer: muted, print-friendly.
CHEMDRAW_PALETTE = {
    1: (0.4, 0.4, 0.4),  # H
    5: (0.9, 0.55, 0.3),  # B
    6: (0.1, 0.1, 0.1),  # C
    7: (0.15, 0.3, 0.85),  # N -- deep blue
    8: (0.85, 0.15, 0.15),  # O -- crimson
    9: (0.2, 0.65, 0.3),  # F -- green
    15: (0.95, 0.55, 0.1),  # P -- orange
    16: (0.85, 0.7, 0.1),  # S -- gold
    17: (0.2, 0.65, 0.3),  # Cl -- green
    35: (0.55, 0.2, 0.1),  # Br -- dark brown-red
    53: (0.55, 0.1, 0.65),  # I -- purple
}

BASE_WIDTH, BASE_HEIGHT = 500, 350
PX_PER_BOND = 96  # a bond is 1 unit long
MIN_SIDE = 0.9  # so a flat or single-atom layout is not a sliver
MIN_W, MIN_H = 120, 90
MAX_W, MAX_H = 820, 620


def _canvas_size(mol, base_width: int, base_height: int) -> tuple[int, int]:
    """Size the canvas from the molecule's own 2D layout.

    Every structure is then drawn at one scale and line weight, so a small
    fragment and a fused polycyclic look like they came from the same page
    instead of being stretched to a common box.
    """
    conformer = mol.GetConformer()
    positions = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    xs = [p.x for p in positions]
    ys = [p.y for p in positions]
    width, height = max(xs) - min(xs), max(ys) - min(ys)

    unit = PX_PER_BOND * ((base_width * base_height) / (BASE_WIDTH * BASE_HEIGHT)) ** 0.5
    side = lambda extent, low, high: round(min(high, max(low, max(extent, MIN_SIDE) * unit)))  # noqa: E731
    return side(width, MIN_W, MAX_W), side(height, MIN_H, MAX_H)


def draw_molecule(
    smiles: str,
    base_width: int = BASE_WIDTH,
    base_height: int = BASE_HEIGHT,
    legend: str = "",
    auto_size: bool = True,
) -> bytes | None:
    """Render a SMILES to PNG bytes in the smiles-hover ChemDraw style.

    White background regardless of theme -- it reads as a paper document in
    both light and dark viewers. Returns None for SMILES RDKit cannot parse,
    so a bad candidate never breaks a logging loop.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)
    width, height = _canvas_size(mol, base_width, base_height) if auto_size else (base_width, base_height)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    # Geometry and labelling in the ACS house style.
    options.explicitMethyl = True
    options.multipleBondOffset = 0.18
    options.addStereoAnnotation = True
    # Bolder than ACS print defaults so the molecule reads on screen.
    options.bondLineWidth = 2.4
    options.scaleBondWidth = True
    # No fixed font size: it overlaps labels on anything dense. Scale, with a
    # floor that stays readable.
    options.minFontSize = 12
    options.maxFontSize = 22
    options.annotationFontScale = 0.75
    options.additionalAtomLabelPadding = 0.1
    options.padding = 0.1
    options.clearBackground = True
    options.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    options.updateAtomPalette(CHEMDRAW_PALETTE)

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend=legend)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
