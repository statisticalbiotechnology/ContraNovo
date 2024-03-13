from .model import Spec2Pep
import heapq
import logging
import operator
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .. import masses
import einops
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter

# from depthcharge.components import ModelMixin, PeptideDecoder, SpectrumEncoder

from ..components import ModelMixin, PeptideDecoder, SpectrumEncoder
import torch.nn.functional as F
from . import evaluate
from ..denovo.clipmodel import PeptideEncoder


class Spec2Vector(Spec2Pep):
    def __init__(
        self,
        *args,
        **kwargs: Dict,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        # del self.decoder

    def encode_spectrum(self, encoder_features: torch.Tensor):

        pkt = encoder_features

        """-----------------------------------------------------"""
        # Extract the global features for Spectrum and Peptide.

        """Sprectrum Global Features:"""
        pkt = torch.transpose(pkt, 1, 2)
        ratiospkt = torch.matmul(self.global_spectrum, pkt)
        ratiospkt = torch.softmax(ratiospkt, dim=2)
        pkt = torch.transpose(pkt, 1, 2)
        pkt = torch.matmul(ratiospkt, pkt)
        pkt_features = pkt.squeeze(1)
        return pkt_features

    def forward(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> Tuple[List[List[str]], torch.Tensor]:
        clipSpectEncoderOutput, masks = self.encoder(spectra, precursors)
        vector = self.encode_spectrum(clipSpectEncoderOutput)
        return vector

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            spectrum identifiers as torch Tensors.

        Returns
        -------
        spectrum_idx : torch.Tensor
            The spectrum identifiers.
        precursors : torch.Tensor
            Precursor information for each spectrum.
        vectors : torch.Tensor of shape (n_spectra, 1, d_model)
            The spectrum embedding vectors
        """
        vectors = self.forward(batch[0], batch[1])
        return batch[2], batch[1], vectors

    def on_predict_epoch_end(
        self, results: List[List[Tuple[np.ndarray, List[str], torch.Tensor]]]
    ) -> None:
        """
        Write the predicted peptide sequences and amino acid scores to the
        output file.
        """
        if self.out_writer is None:
            return
        for batch in results:
            for step in batch:
                for spectrum_idx, precursor, vector in zip(*step):
                    # Get peptide sequence, amino acid and peptide-level
                    # confidence scores to write to output file.

                    # Compare the experimental vs calculated precursor m/z.
                    _, precursor_charge, precursor_mz = precursor
                    precursor_charge = int(precursor_charge.item())
                    precursor_mz = precursor_mz.item()

                    self.out_writer.append(
                        {
                            "spectrum_index": spectrum_idx,
                            "precursor_charge": precursor_charge,
                            "precursor_mz": precursor_mz,
                            "spectrum_embedding": vector.cpu().numpy().tolist(),
                        }
                    )
        self.out_writer.save()
