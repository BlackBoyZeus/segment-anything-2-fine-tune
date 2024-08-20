from box import Box

config = {
    "num_devices": 1,
    "num_workers": 2,
    "num_epochs": 3,
    "eval_interval": 3,
    "out_checkpoint_dir": "out/training",
    "save_validation_images_result": False,
    "segmentated_validation_images_dir":"<segmentated_validation_images_dir path>",
    "prompt_type":"points", #points/bounding_box
    "save_image_embeddings":True,
    "save_embeddings_only_for_iterative_sampling": True,  # Temporarily save image embeddings only during the correction clicks loop. Useful for low storage.
    "image_features_embeddings_dir":"<image_features_embeddings_dir path>",
    "iterative_sampling":True,
    "correction_clicks":7,#(only for iterative sampling=True)
    "opt": {
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'sam2_hiera_l.yaml', #sam2 model type (sam2_hiera_l.yaml,sam2_hiera_t.yaml,sam2_hiera_t.yaml,sam2_hiera_b+.yaml)
        "base_model_checkpoint": "<base model checkpoint path>",
        "Train_from_fine_tuned_model": False,
        "fine_tuned_checkpoint": "<fine tuned model check point path (Add path if you want to continue the training from the fine tuned model) >",
        "freeze": {
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "image_resize": 256, #for iterative sampling use 256
        "train": {
            "root_dir": "<images path>",
            "annotation_file": "annotation file path (.json)"
        },
        "val": {
            "root_dir": "<images path>",
            "annotation_file": "annotation file path (.json)"
        }
    },
    # --- Loss Weights ---
    "loss": {
        "segmentation_weight": 1.0,  # Weight for segmentation loss (Focal + Dice)
        "temporal_weight": 0.5,    # Weight for temporal loss (MSE between frames)
        "tv_weight": 0.1,         # Weight for total variation loss
        "laplacian_weight": 0.05   # Weight for Laplacian smoothing loss 
    }
}

cfg = Box(config)

'''
Explanation of Changes:

Loss Weights Section: A new section called "loss" is added to the config dictionary.
Individual Loss Weights: Within the "loss" section, you now have:
"segmentation_weight": The weight for the combined Focal and Dice loss, which represents the core segmentation objective.
"temporal_weight": The weight for the temporal consistency loss (MSE between consecutive frames).
"tv_weight": The weight for the total variation loss, encouraging spatial smoothness.
"laplacian_weight": The weight for the Laplacian smoothing loss, further promoting smooth mask predictions.
How to Use:

Set Values: Replace the placeholder values (1.0, 0.5, 0.1, 0.05) with your desired weights for each loss component.
Access in Training Script: Access these loss weights in your train_sam() function using cfg.loss.segmentation_weight, cfg.loss.temporal_weight, cfg.loss.tv_weight, and cfg.loss.laplacian_weight.
Important Considerations:

Balance: Carefully choose the loss weights to balance the different objectives (segmentation accuracy, temporal consistency, spatial smoothness).
Tuning: Experiment with different weight combinations to find the best settings for your specific dataset and model. Start with small values for the regularization weights (temporal, total variation, Laplacian) and increase them gradually if needed.
Over-Regularization: Be mindful of over-regularization. If the regularization weights are too high, the model might prioritize smoothness over segmentation accuracy, leading to overly smooth masks that miss important details.
'''
