# release
sudo singularity build --notest sc-dandelion.sif sc-dandelion.def
sudo singularity test --writable-tmpfs sc-dandelion.sif
singularity sign sc-dandelion.sif
singularity verify sc-dandelion.sif
singularity push sc-dandelion.sif library://kt16/default/sc-dandelion

# devel and testing
sudo singularity build --notest sc-dandelion_dev.sif sc-dandelion_dev.def
sudo singularity test --writable-tmpfs sc-dandelion_dev.sif
singularity sign sc-dandelion_dev.sif
singularity verify sc-dandelion_dev.sif