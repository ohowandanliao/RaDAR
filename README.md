# RaDAR

RaDAR: relation-aware diffusion and asymmetric contrastive learning for recommendation on user-item graphs.

## Origin
This repository is based on AdaGCL (HKUDS): https://github.com/HKUDS/AdaGCL

## Datasets
- **Binary-edge**: Last.FM, Yelp, BeerAdvocate (implicit feedback; edges indicate interaction presence).
- **Weighted-edge (multi-behavior)**: Tmall, RetailRocket, IJCAI15.
  - We follow the public DiffGraph/HEC-GCN protocol: multi-behavior interactions are merged into a **single binary graph** (edge presence).
  - In those reference implementations, raw multi-behavior values (e.g., timestamps) are binarized before graph construction; only edge presence is used.
  - Graph propagation uses standard **symmetric normalized adjacency** with self-loops.
  - Evaluation is the same Top-K all-ranking protocol as binary (mask train interactions; report Recall@20/NDCG@20).
  - Note: here “weighted-edge” refers to the **multi-behavior setting**, not real-valued edge weights.

## Reproduction
- Binary-edge: `scripts/reproduce_sota.sh`
- Weighted-edge (protocol-aligned): `scripts/reproduce_weighted.sh`

## Experimental (not used in paper)
- `scripts/reproduce_weighted_realw.sh` runs with real-valued weights in `Datasets/*/weighted/trnMat.pkl`.
  Results are **not** comparable to the paper tables.

## Acknowledgements
We thank AdaGCL and DiffGraph/HEC-GCN for their open-source implementations and public protocols.
