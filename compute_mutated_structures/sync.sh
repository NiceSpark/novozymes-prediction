rsync -au --progress relaxed_pdb/ ml:/home/ml/novozymes-prediction/compute_mutated_structures/relaxed_pdb/
rsync -au --progress mutations_logs/ ml:/home/ml/novozymes-prediction/compute_mutated_structures/mutations_logs/
scp 'ml:/home/ml/novozymes-prediction/compute_mutated_structures/mutations_GCP*.sh' .
scp 'ml:/home/ml/novozymes-prediction/compute_mutated_structures/main_mutations_GCP.sh' .
scp -r ml:/home/ml/novozymes-prediction/compute_mutated_structures/mutations/ . 
