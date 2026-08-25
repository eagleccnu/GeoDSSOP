# Data and third-party license record

## Repository code

Original GeoDSSOP release code is under the MIT License in the repository root.

## PDB example

`examples/minimal_example/1ubq.cif.gz` is the unmodified PDB archive entry
1UBQ, downloaded from `https://files.rcsb.org/download/1UBQ.cif.gz` on
2026-08-25. Its SHA-256 is
`b1a85fb2761c9d2e36734e48706645878036d6c848ebc386611a1957c697d6b3`.

The wwPDB archive makes PDB data files available under the CC0 1.0 Universal
Public Domain Dedication:

- https://www.wwpdb.org/about/usage-policies
- https://www.rcsb.org/pages/usage-policy

Users should still cite PDB ID 1UBQ and its structure authors where possible.

## ESM-2

The feature extractor pins `facebook/esm2_t33_650M_UR50D` at revision
`08e4846e537177426273712802403f7ba8261b6c`. The upstream ESM project and model
card identify the code/model license as MIT:

- https://github.com/facebookresearch/esm/blob/main/LICENSE
- https://huggingface.co/facebook/esm2_t33_650M_UR50D

The repository does not redistribute the 2.6 GB upstream model. The bundled
1UBQ feature tensor is a deterministic derived representation used solely for
the software regression example and retains model/sequence provenance in its
metadata.

## Split metadata

`data/manifests/split-manifest.csv` contains accessions, source names, frozen
roles, homology-group identifiers, sequence hashes, and lengths. It contains no
sequences and no target values. It is released to document experimental design,
not to supersede original source terms.

## Labels and trajectories not redistributed

- MD-iRED residue labels and raw MD trajectories;
- S-OPPE pseudo-label values;
- experimental NMR order-parameter values;
- source PDB collections beyond the CC0 1UBQ demo;
- AlphaFold structures;
- article PDFs or Zotero attachments.

Access the original resources under their own terms and locally convert them to
the documented cached-data contract. Absence of a file here is deliberate and
must not be worked around by substituting a different dataset under the same
identifier.

## AlphaFold Database

No AlphaFold structure is bundled. If a future GeoDSSOP-AF release uses the
AlphaFold Protein Structure Database, its predicted structures are CC BY 4.0
and require attribution under the database license:

- https://alphafold.ebi.ac.uk/

That future deployment route is not part of the current validated PDB version.
