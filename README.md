# DAS-SPP: Transformer-Based Seismic Phase Picking on Distributed Acoustic Sensing data

PyTorch implementation of DAS-SPP for P/S phase picking on DAS (Distributed Acoustic Sensing) data. DAS-SPP captures both local spatial patterns and long-range dependencies within the data, demonstrating reliable performance on seismic phase picking and event detection with minimal labeled data. 

<img width="1254" height="929" alt="image" src="https://github.com/user-attachments/assets/61bf582a-bf52-442b-8d5d-bf66dcf76f64" />


## Getting started

Download the code:

```
git clone https://github.com/mirianacorsaro/das-spp
cd das-spp
```

Download the **pretrained DAS-SPP** & a subset of our datasets from [here](https://drive.google.com/drive/folders/1UkeosTOsF130dHzRfRU3FLvXnnzQXorB?usp=drive_link).

After download, place files as shown in the tree.

```
├─ data/
│  └─ das-picking/
│     └─ labeled-data/
│        ├─ train/                   
│        ├─ val/
│        ├─ test/
│        └─ targets_masks/           
├─ dataset/
│  └─ dataset.py                     
├─ models/
│  ├─ tunet.py                     
│  ├─ unet.py                      
│  ├─ swin_transformer_v2.py        
│  └─ trained_models/
│     └─ best_picking_model_v2.pth   
├─ training/
│  └─ train.py                     
├─ utils/
│  ├─ train_eval.py                 
│  ├─ save_pick.py                   
│  └─ utils.py                       
├─ main.py                          
└─ das-spp-example.ipynb            
```

With:
- Event filename pattern (train/val/test): ```<id>_<magnitude>_<YYYY-mm-dd>_<HH-MM-SS.micro>.npy```.
- Target masks: ```targets_masks/<id>.npy```   

Please follow the instructions here to setup all the required dependencies for training and evaluation:

```
conda env create -f environment.yml
conda activate das-ssp-env
```

## Training

Training from scratch with your own dataset (CLI):

```
python main.py train \
  --model-name das-spp --num-heads 2 --forward-expansion 2 \
  --batch-size 4 --lr 1e-4 --epochs 100 \
  --optimizer adam --criterion weighted_classes \
  --train-data data/campi_flegrei/labeled-data/data/train \
  --val-data   data/campi_flegrei/labeled-data/data/val \
  --test-data  data/campi_flegrei/labeled-data/data/test \
  --targets    data/campi_flegrei/labeled-data/masks \
  --das-val-dir data/campi_flegrei/labeled-data/masks \
  --exp-id 0 --gpu 0 \
  --eval-every 10 --save-ckpt-every 10 --gate-val-acc 0.80 \
  --num-plot-images 8 --plot-original-das --dpi 150
```

Fine-tuning from a checkpoint:

```
python main.py train \
  --finetuning models/trained_models/best_picking_model_v2.pth \
  --model-name das-spp --num-heads 2 --forward-expansion 2 \
  --batch-size 4 --lr 5e-5 --epochs 30 \
  --optimizer adam --criterion weighted_classes \
  --train-data data/campi_flegrei/labeled-data/data/train \
  --val-data   data/campi_flegrei/labeled-data/data/val \
  --test-data  data/campi_flegrei/labeled-data/data/test \
  --targets    data/campi_flegrei/labeled-data/masks \
  --das-val-dir data/campi_flegrei/labeled-data/masks \
  --exp-id 10 --gpu 0
```

## Inference:

Run inference:

```
python main.py infer \
  --checkpoint models/trained_models/best_picking_model_v2.pth \
  --batch-size 4 \
  --test-data  data/das-picking/labeled-data/test \
  --targets    data/das-picking/labeled-data/targets_masks \
  --das-test-dir data/das-picking/labeled-data/test \
  --epoch 100 --exp-id 0 --gpu 0 \
  --num-plot-images 16 --plot-original-das --dpi 150
```

This computes global P/S precision/recall/F1 (time-tolerance matching), saves CSV picks per event, and writes a few preview figures.

## Notebooks

Use [das-spp-example.ipynb](https://github.com/mirianacorsaro/das-spp/blob/master/das-spp_example.ipynb) for interactive inference. The notebook demonstrates how to:
- Load a trained checkpoint
- Run single-event or batch inference
- Visualize DAS with predicted P/S picks

## Citation & Contact

If you use this repository in academic work, please cite it.

Contact: miriana.corsaro@ingv.it
