# set up miniforge
apt-get --allow-releaseinfo-change update && apt-get install -y curl git
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p "/opt/conda"
rm Miniforge3-$(uname)-$(uname -m).sh
. /opt/conda/etc/profile.d/conda.sh
. /opt/conda/etc/profile.d/mamba.sh
# fix pathing
echo ". /opt/conda/etc/profile.d/conda.sh" | tee -a $SINGULARITY_ENVIRONMENT
echo ". /opt/conda/etc/profile.d/mamba.sh" | tee -a $SINGULARITY_ENVIRONMENT
echo "conda activate sc-dandelion-container" | tee -a $SINGULARITY_ENVIRONMENT
echo "export GERMLINE=/share/database/germlines/" | tee -a $SINGULARITY_ENVIRONMENT
echo "export IGDATA=/share/database/igblast/" | tee -a $SINGULARITY_ENVIRONMENT
echo "export BLASTDB=/share/database/blast/" | tee -a $SINGULARITY_ENVIRONMENT
chmod +x /share/dandelion_preprocess.py
chmod +x /share/changeo_clonotypes.py
# install dependencies
mamba env update --name sc-dandelion-container -f environment.yml
mamba activate sc-dandelion-container
