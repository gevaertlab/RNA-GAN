# Synthetic whole-slide imaging tile generation with gene expression profiles infused deep generative models

---

## Abstract

The acquisition of multi-modal biological data has increased in recent years, enabling a multi-scale study of human biology. However, despite tremendous efforts, the majority of the datasets available only contain one modality, given a lack of the necessary personnel or means to obtain all of them at hospitals and research centers. Given these difficulties, the generation of synthetic biological data appears as a solution for the data scarcity problem. Currently, most studies focus on generating a single modality (e.g. histology images), without leveraging the information provided by other related ones (e.g. gene expression profiles). In this work, we propose a new approach to generate whole-slide-imaging healthy tissue tiles by using deep generative models infused with the gene expression profile of the patient. We firstly train a variational autoencoder that learns a latent representation of multi-tissue gene expression profiles, and we show that it is able to generate realistic synthetic ones. Then, we use this representation to infuse generative adversarial networks, generating healthy lung and brain cortex tiles with a new model that we called RNA-GAN. Tiles generated by RNA-GAN were preferred by expert pathologists in comparison to tiles generated using a traditional generative adversarial network and, in comparison to them, it needed fewer training epochs to generate high-quality tiles. In addition, RNA-GAN was able to generalize to gene expression profiles outside of the training set and we show that the synthetic tiles can be used to train machine learning models. A web-based quiz is available where users can try to distinguish between real and synthetic tiles here: [rna-gan.stanford.edu](rna-gan.stanford.edu)

<img src="imgs/generation.png" alt="generation" width="1500"/>

## Web quiz

A quiz is available to get a score on how well fake and real images are detected.

## Example usage

### betaVAE

**Training the model**

```bash
python3 betaVAE_training.py --seed 99 --config configs/betavae_tissues.json --log 1 --parallel 0
```
**Compute interpolation vectors**

```bash
python3 betaVAE_interpolation.py --seed 99 config --configs/betavae_tissues.json --log 0 --parallel 0
```
**Interpolating**

```bash
pythion3 betaVAE_sample.py --seed 99 --config configs/betavae_tissues.json --log 0 --parallel 0
```

### Normal GAN and RNA-GAN

**Normal GAN training**

```bash
python3 histopathology_gan.py --seed 99 --config configs/gan_run_brain.json --image_dir gan_generated_images/images_gan_brain --model_dir ./checkpoints/gan_brain/gan_brain --num_epochs 39 --gan_type dcgan --loss_type wgan --num_patches 600

python3 histopathology_gan.py --seed 99 --config configs/gan_run_lung.json --image_dir gan_generated_images/images_gan_lung --model_dir ./checkpoints/gan_lung/gan_lung --num_epochs 91 --gan_type dcgan --loss_type wgan --num_patches 600
```

**RNA-GAN training**

```bash
python3 histopathology_gan.py --seed 99 --config configs/gan_run_brain.json --image_dir gan_generated_images/images_rna-gan_brain --model_dir ./checkpoints/rna-gan_brain/rna-gan_brain --num_epochs 24 --gan_type dcgan --loss_type wganvae --num_patches 600
python3 histopathology_gan.py --seed 99 --config configs/gan_run_lung.json --image_dir gan_generated_images/images_rna-gan_lung --model_dir ./checkpoints/rna-gan_lung/rna-gan_lung --num_epochs 11 --gan_type dcgan --loss_type wganvae --num_patches 600
```

**Compute FID metrics**

```bash
# Real lung vs gan lung
python3 fid.py --checkpoint ./checkpoints/gan_lung/gan_lung.model --config configs/gan_run_lung.json \
        --sample_size 600 --vae --vae_checkpoint checkpoints/betavae_training_tissues/model_dict_best.pt \
        --patient1 GTEX-15RJ7-0625.svs

# Real lung vs rna-gan lung
python3 fid.py --checkpoint ./checkpoints/rna-gan_lung/rna-gan_lung.model --config configs/gan_run_lung.json \
        --sample_size 600 --vae --vae_checkpoint checkpoints/betavae_training_tissues/model_dict_best.pt \
        --patient1 GTEX-15RJ7-0625.svs

# Gan lung vs rna-gan lung
python3 fid.py --checkpoint ./checkpoints/rna-gan_lung/rna-gan_lung.model --checkpoint2 ./checkpoints/gan_lung/gan_lung.model --config configs/gan_run_lung.json \
        --sample_size 600 --vae --vae_checkpoint checkpoints/betavae_training_tissues/model_dict_best.pt \
        --patient1 GTEX-15RJ7-0625.svs

# Gan vs Real brain
python3 fid.py --checkpoint ./checkpoints/gan_lung/gan_lung.model --config configs/gan_run_brain.json \
        --sample_size 600

# Real brain vs rna-gan
python3 fid.py --checkpoint ./checkpoints/rna-gan_lung/rna-gan_lung.model --config configs/gan_run_brain.json \
        --sample_size 600 --vae --vae_checkpoint checkpoints/betavae_training_tissues/model_dict_best.pt \
        --patient1 GTEX-1C6WA-3025.svs

# Gan brain vs gan lung
python3 fid.py --checkpoint ./checkpoints/gan_lung/gan_lung.model --config configs/gan_run_brain.json \
        --sample_size 600 --checkpoint2 ./checkpoints/gan_brain/gan_brain.model

# brain rna-gan vs gan lung
python3 fid.py --checkpoint2 ./checkpoints/gan_lung/gan_lung.model --checkpoint ./checkpoints/rna-gan_brain/rna-gan_brain.model --config configs/gan_run_brain.json \
        --sample_size 600 --vae --vae_checkpoint checkpoints/betavae_training_tissues/model_dict_best.pt \
        --patient1 GTEX-1C6WA-3025.svs

# lung rna-gan vs gan brain
python3 fid.py --checkpoint2 ./checkpoints/gan_brain/gan_brain.model --checkpoint ./checkpoints/rna-gan_lung/rna-gan_lung.model --config configs/gan_run_lung.json \
        --sample_size 600 --vae --vae_checkpoint checkpoints/betavae_training_tissues/model_dict_best.pt \
        --patient1 GTEX-15RJ7-0625.svs

# lung rna-gan vs  brain rna-gan 
python3 fid.py --checkpoint2 ./checkpoints/rna-gan_brain/rna-gan_brain.model --checkpoint ./checkpoints/rna-gan_lung/rna-gan_lung.model --config configs/gan_run_lung.json \
        --config2 configs/gan_run_brain.json --sample_size 600 --vae --vae_checkpoint checkpoints/betavae_training_tissues/model_dict_best.pt \
        --patient1 GTEX-15RJ7-0625.svs --patient2 GTEX-1C6WA-3025.svs

#############

# Real brain vs gan brain
python3 fid.py --checkpoint ./checkpoints/gan_brain/gan_brain.model --config configs/gan_run_brain.json \
        --sample_size 600 \
        --patient1 GTEX-1C6WA-3025.svs

# Real brain vs rna-gan brain
python3 fid.py --checkpoint ./checkpoints/rna-gan_brain/rna-gan_brain.model --config configs/gan_run_brain.json \
        --sample_size 600 --vae --vae_checkpoint checkpoints/betavae_training_tissues/model_dict_best.pt \
        --patient1 GTEX-1C6WA-3025.svs

# Gan brain vs rna-gan brain
python3 fid.py --checkpoint ./checkpoints/rna-gan_brain/rna-gan_brain.model --checkpoint2 ./checkpoints/gan_brain/gan_brain.model --config configs/gan_run_brain.json \
        --sample_size 600 --vae --vae_checkpoint checkpoints/betavae_training_tissues/model_dict_best.pt \
        --patient1 GTEX-1C6WA-3025.svs

# brain Gan vs Real lung
python3 fid.py --checkpoint ./checkpoints/gan_brain/gan_brain.model --config configs/gan_run_lung.json \
        --sample_size 600

# Real lung vs rna-gan brain
python3 fid.py --checkpoint ./checkpoints/rna-gan_brain/rna-gan_brain.model --config configs/gan_run_lung.json \
        --sample_size 600 --vae --vae_checkpoint checkpoints/betavae_training_tissues/model_dict_best.pt \
        --patient1 GTEX-15RJ7-0625.svs

# lung rna-gan vs  brain rna-gan
python3 fid.py --checkpoint2 ./checkpoints/rna-gan_brain/rna-gan_brain.model --checkpoint ./checkpoints/rna-gan_lung/rna-gan_lung.model --config configs/gan_run_lung.json \
        --config2 configs/gan_run_brain.json --sample_size 600 --vae --vae_checkpoint checkpoints/betavae_training_tissues/model_dict_best.pt \
        --patient1 GTEX-15RJ7-0625.svs --patient2 GTEX-1C6WA-3025.svs

```

**Image generation**

```bash
python3 generate_tissue_images.py --checkpoint ./checkpoints/rna-gan_lung.model --checkpoint2 ./checkpoints/gan_lung.model --config configs/gan_run_lung.json --sample_size 600 --vae --vae_checkpoint checkpoints/betavae.pt --patient1 GTEX-15RJ7-0625.svs

python3 generate_tissue_images.py --checkpoint ./checkpoints/rna-gan_brain.model --checkpoint2 ./checkpoints/gan_brain.model --config configs/gan_run_brain.json --sample_size 600 --vae --vae_checkpoint checkpoints/betavae.pt --patient1 GTEX-1C6WA-3025.svs

# From GEO series

python3 generate_tissue_image.py --checkpoint ./checkpoints/rna-gan_lung.model --config configs/gan_run_brain.json --sample_size 600 --vae --vae_checkpoint checkpoints/betavae.pt --rna_data GSE120795_lung_proteincoding.csv --random_patient

python3 generate_tissue_image.py --checkpoint ./checkpoints/rna-gan_brain.model --config configs/gan_run_brain.json --sample_size 600 --vae --vae_checkpoint checkpoints/betavae.pt --rna_data GSE120795_brain_proteincoding.csv --random_patient
```

**Training toy classification model**

```bash

# Using real samples
python3 wsi_model.py --patch_data_path images_form/ --csv_path real_toy_example.csv

# Using generated samples
python3 wsi_model.py --patch_data_path images_form/ --csv_path rna-gan_toy_example.csv
```

