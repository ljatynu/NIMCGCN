# NIMCGCN
Copyright (C) 2019 Jin Li(lijin@ynu.edu.cn)

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see http://www.gnu.org/licenses/.

Jin Li(lijin@ynu.edu.cn) School of Software, Yunnan University, Kunming CHINA, 650000


NIMCGCN
NIMCGCN: Neural inductive matrix completion with graph convolutional networks for miRNA-disease association prediction (Bioinformatics).

Requirements
Pytorch (tested on version 1.1.1)
numpy (tested on version 1.16.2)
sklearn (tested on version 0.20.3)

Quick start
To reproduce our results:
Unzip data.zip in ./data.
Run main.py to RUN NIMCGCN.

Data description
d-d.csv:disease-disease similarity matrix.
m-m.csv: miRNA-miRNA similarity matrix.
disease name.csv: list of disease names.
miRNA name.csv: list of miRNA names
m-d.csv: miRNA-disease association matrix
