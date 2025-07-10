import os

# Define the datasets
datasets = [
    #"/home/alingold/pix2pix/datasets/combined_plant_0bp_noBlurryRemoval_pix2pix",
    "/home/alingold/pix2pix/datasets/cannula_only_combined"
]

# Define the base paths for the models
#base_path_nogan = "/home/alingold/probabilistic_pix2pix/checkpoints/checkpoints_chpc/combined_plant_0bp_noBlurryRemoval_probabilistic_new_1_layer_nogan_"
#base_path_gan = "/home/alingold/probabilistic_pix2pix/checkpoints/checkpoints_chpc/gan_combined_plant_0bp_noBlurryRemoval_probabilistic_new_1_layer_gan_over_100_"
base_path_nogan_shift = "/home/alingold/probabilistic_pix2pix/checkpoints/checkpoints_chpc/combined_plant_0bp_noBlurryRemoval_probabilistic_new_1_layer_nogan_shift_"
#base_path_gan_shift = "/home/alingold/probabilistic_pix2pix/checkpoints/checkpoints_chpc/gan_combined_plant_0bp_noBlurryRemoval_probabilistic_new_1_layer_gan_over_100_shift_"

# Iterate over the models and datasets
for i in range(5):
    for dataset in datasets:
        # Test nogan models
        #model_nogan = f"{base_path_nogan}{i}"
        #print(f"Testing nogan model {i} with dataset {dataset}")
        #os.system(f"python test.py --dataroot {dataset} --name checkpoints_chpc/combined_plant_0bp_noBlurryRemoval_probabilistic_new_1_layer_nogan_{i} --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 100")
        
        # Test gan models
        #model_gan = f"{base_path_gan}{i}"
        #print(f"Testing gan model {i} with dataset {dataset}")
        #os.system(f"python test.py --dataroot {dataset} --name checkpoints_chpc/gan_combined_plant_0bp_noBlurryRemoval_probabilistic_new_1_layer_gan_over_100_{i} --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 100")

        # Test nogan shift models
        model_gan_shift = f"{base_path_nogan_shift}{i}"
        print(f"Testing gan shift model {i} with dataset {dataset}")
        os.system(f"python test.py --dataroot {dataset} --name checkpoints_chpc/combined_plant_0bp_noBlurryRemoval_probabilistic_new_1_layer_nogan_shift_{i} --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 100")
        
        # Test gan shift models
        #model_gan_shift = f"{base_path_gan_shift}{i}"
        #print(f"Testing gan shift model {i} with dataset {dataset}")
        #os.system(f"python test.py --dataroot {dataset} --name checkpoints_chpc/gan_combined_plant_0bp_noBlurryRemoval_probabilistic_new_1_layer_gan_over_100_shift_{i} --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 100")
