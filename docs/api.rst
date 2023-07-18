Preprocessing: `pp`
===================
.. module:: dandelion.preprocessing

.. autosummary::
   :toctree: modules

   assign_isotype
   assign_isotypes
   calculate_threshold
   create_germlines
   check_contigs
   filter_contigs
   format_fasta
   format_fastas
   quantify_mutations
   reannotate_genes
   reassign_alleles

Preprocessing (external): `pp.external`
=======================================
.. module:: dandelion.preprocessing.external

.. autosummary::
   :toctree: modules

   assigngenes_igblast
   creategermlines
   makedb_igblast
   parsedb_heavy
   parsedb_light
   recipe_scanpy_qc
   tigger_genotype

Tools: `tl`
===========
.. module:: dandelion.tools

.. autosummary::
   :toctree: modules

   clone_centrality
   clone_degree
   clone_diversity
   clone_overlap
   clone_rarefaction
   clone_size   
   define_clones
   extract_edge_weights
   find_clones
   generate_network
   transfer
   setup_vdj_pseudobulk
   vdj_pseudobulk
   pseudotime_transfer
   project_pseudotime_to_cell
   pseudobulk_gex
   bin_expression
   chatterjee_corr

Plotting: `pl`
==============
.. module:: dandelion.plotting

.. autosummary::
   :toctree: modules

   barplot
   clone_network
   clone_overlap
   clone_rarefaction   
   spectratype
   stackedbarplot   

Utilities: `utl`
================
.. module:: dandelion.utilities

.. autosummary::
   :toctree: modules
   
   load_data
   makeblastdb   
   read_h5ddl
   read_pkl
   read_10x_airr
   read_10x_vdj
   update_metadata
   concat
   to_scirpy
   from_scirpy
   write_fasta
   

Dandelion
=========
.. module:: dandelion.Dandelion

.. autosummary::
   :toctree: modules

   copy
   update_germline
   store_germline_reference
   update_metadata
   write
   write_h5ddl
   write_pkl
   write_airr

Logging
=========
.. module:: dandelion.logging

.. autosummary::
   :toctree: modules

   print_header
   print_versions
   