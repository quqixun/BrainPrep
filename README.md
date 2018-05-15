# Preprocessing on Brain MRI Sequence

This is a pipeline to do preprocessing on brain MR images of **ADNI** dataset  
by using FMRIB Software Library (**FSL**) and Advanced Normalization Tools (**ANTs**).

## 1. Install FSL & ANTs

Download and install **FSL** as instructions [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation).  
Compile **ANTs** from source code in [Linux and macOS](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS), or in [Windows 10](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Windows-10).

## 2. Install Python Packages

All required libraries are listed as below:

- tqdm
- numpy
- scipy
- nipype
- nibabel
- matplotlib
- sciKit-fuzzy (optional)
- scikit-learn (optional)

## 3. Download Dataset

The dataset used in this repo is AD and NC screening images of ADNI1 and ADNI2.  
See [README.md in *data*](https://github.com/quqixun/BrainPrep/tree/master/data).

Here is one sample of original image.  

<img src="https://github.com/quqixun/BrainPrep/blob/master/imgs/original.png" alt="original image" width="250">

## 4. Reorgnization Files

Switch the working directory to *src*.
Run reorgnize.py, which merge ADNI1 and ADNI2 into one folder.
```
python reorgnize.py
```

## 5. Registration

Run registraion.py to transform images into the coordinate system of template by **FSL FLIRT**.
```
python registraion.py
```

The output of the above image from this step looks like:  

<img src="https://github.com/quqixun/BrainPrep/blob/master/imgs/registration.png" alt="registration" width="250">

## 6. Skull-Strpping

Run skull_stripping.py to remove skull from registrated images by **FSL BET**.
```
python skull_stripping.py
```

 Output:  
 
<img src="https://github.com/quqixun/BrainPrep/blob/master/imgs/skull_stripping.png" alt="skull stripping" width="250">

## 7. Bias Field Correction

Run bias_correction.py to remove bias-field signal from images by **ANTs**.
```
python bias_correction.py
```

Output:

<img src="https://github.com/quqixun/BrainPrep/blob/master/imgs/bias_correction.png" alt="bias field correction" width="250">

## 8. Enhancement (optional)

Based on outputs from step 7, run enhancement.py to enhance images by **histogram equalization**.
```
python enahncement.py
```

## 9. Tissue Segmentation (optional)

Based on outputs from step 7, run segment.py to segment brain into GM, WM and CSF  
by **KMeans** or **Fuzzy-CMeans** (you should change settings in script).
```
python segment.py
```
Or run fast_segment.py to do segmentation by **FSL FAST**.
```
python fast_segment.py
```
