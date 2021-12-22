# MLCyberProject
```bash
├── DataFile 
    └── clean_validation_data.h5 // all the datasets files can be obtained from the link below
    └── clean_test_data.h5
    └── sunglasses_poisoned_data.h5
    └── anonymous_1_poisoned_data.h5
    └── Multi-trigger Multi-target
        └── eyebrows_poisoned_data.h5
        └── lipstick_poisoned_data.h5
        └── sunglasses_poisoned_data.h5
├── Repaired_models
    └── fine_pruned_anonymous1_model.h5
    └── fine_pruned_anonymous1_weights.h5
    └── fine_pruned_multi_model.h5
    └── fine_pruned_multi_model_weights.h5
    └── fine_pruned_sunglasses_model.h5
    └── fine_pruned_sunglasses_model_weights.h5    
├── models
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_bd_weights.h5
    └── anonymous_1_bd_net.h5
    └── anonymous_1_bd_weights.h5
    └── anonymous_2_bd_net.h5
    └── anonymous_2_bd_weights.h5
├── entropy_files
    └── entropy_clean_multi.h5
    └── entropy_clean_anonymous2.h5
    └── entropy_clean_anonymous1.h5
    └── entropy_clean_sunglasses.h5
├── eval_anonymous_1.py
├── eval_multi_trigger_target.py
├── eval_sunglasses.py
├── eval.py // this is the evaluation fine_pruned script
├── eval02.py // this is the evaluation STRIP script

```

## I. Dependencies
   1. Python 3.6.9
   2. Keras 2.3.1
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. TensorFlow-gpu 1.15.2
   
## II. Validation Data
   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing) and store them under `DataFile/` directory.
   2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals each containing 9 images in the validation dataset.
   3. sunglasses_poisoned_data.h5 contains test images with sunglasses trigger that activates the backdoor for sunglasses_bd_net.h5. Similarly, there are other .h5 files with poisoned data that correspond to different BadNets under models directory.

## III. Evaluating the Repaired Backdoored Model
For Fine-Prune：
  1. For approach Find-Prune: We uploaded a pdf and a completed python code that used for obtained fine_pruned models.
  2. For input images that for testing: Please use the eval script and put the test image into the data floder(MLCyberProject/data).
  3. For evaluating the model, execute eval.py by running:
    python3 `eval_*.py </DataFile directory>`  for example python3 eval_anonymous_1.py DataFile/test_image.png
    
For Strip:  
  1.You need to download or use the code in strip.py to get the entropy_file (I recommend using the code), this is the key step.
  
  2.You need to open eval02, and then change the file path and file name according to the notes to meet your actual situation. 
  
  3.Run eval to get the result.
 
## IV. Evaluating the Submissions
The teams should submit a single eval.py script for each of the four BadNets provided to you. In other words, your submission should include four eval.py scripts, each corresponding to one of the four BadNets provided. YouTube face dataset has classes in range [0, 1282]. So, your eval.py script should output a class in range [0, 1283] for a test image w.r.t. a specific backdoored model. Here, output label 1283 corresponds to poisoned test image and output label in [0, 1282] corresponds to the model's prediction if the test image is not flagged as poisoned. Effectively, design your eval.py with input: a test image (in png or jpeg format), output: a class in range [0, 1283]. Output 1283 if the test image is poisoned, else, output the class in range [0,1282].

Teams should submit their solutions using GitHub. All your models (and datasets) should be uploaded to the GitHub repository. If your method relies on any dataset with large size, then upload the data to a shareable drive and provide the link to the drive in the GitHub repository. To efficiently evaluate your work, provide a README file with clear instructions on how to run the eval.py script with an example.
For example: `python3 eval_anonymous_2.py DataFile/test_image.png`. Here, eval_anonymous_2.py is designed for anonynous_2_bd_net.h5 model. Output should be either 1283 (if test_image.png is poisoned) or one class in range [0, 1282] (if test_image.png is not poisoned).
