"""Modality encoders, one subpackage per modality, one module per backbone.

Modules are named after the architecture ("cnn", "transformer") or, for
pretrained models, after the model itself ("molformer"). Importing this package
populates the encoder registry.
"""

from secs.models.encoders.c_nmr.transformer import CNmrTransformerEncoder
from secs.models.encoders.graph.molclr import GraphGINEncoder
from secs.models.encoders.h_nmr.cnn import HNmrCNNEncoder
from secs.models.encoders.hsqc.cnn import HsqcCNNEncoder
from secs.models.encoders.ir.cnn import IrCNNEncoder
from secs.models.encoders.smiles.molformer import MolformerEncoder

__all__ = [
    "CNmrTransformerEncoder",
    "GraphGINEncoder",
    "HNmrCNNEncoder",
    "HsqcCNNEncoder",
    "IrCNNEncoder",
    "MolformerEncoder",
]
