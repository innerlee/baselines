#/bin/bash
cp ~/.mujoco/mjkey.txt ../rllab-curriculum/mjpro131/bin/
cp ~/.mujoco/mjkey.txt ../rllab-curriculum/vendor/mujoco/
# source activate cpu_rllab #rllab_goal_rl
source activate rllab_goal_rl
# python testArm3d.py

# --renderMode 'single'
# --renderMode 'multiple'

# 10.26
    # change arm3d_task12 to arm3d_task12_with and arm3d_task12_without

# 10.25

    # test code can run
    python -m baselines.run --record --ps _TfRunningMeanStd_TF_adam --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12_without --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=36864000 --seed 0 --num_env 3 --save_path ./result/test

    # # run tasks 1026Experiments 
    #     # task 1
    #         # sparse with different distance
            # python -m baselines.run --record  --rewardModeForArm3d sparse1 --sparse1_dis 0.1 --ps _TfRunningMeanStd_TF_adam --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=36864000 --seed 0 --num_env 2 --save_path ./result/1026Experiments   
            # python -m baselines.run --record  --rewardModeForArm3d sparse1 --sparse1_dis 0.2 --ps _TfRunningMeanStd_TF_adam --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=36864000 --seed 0 --num_env 2 --save_path ./result/1026Experiments   
            # python -m baselines.run --record  --rewardModeForArm3d sparse1 --sparse1_dis 0.3 --ps _TfRunningMeanStd_TF_adam --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=36864000 --seed 0 --num_env 2 --save_path ./result/1026Experiments   
    #         # density
            # python -m baselines.run --record --ps _TfRunningMeanStd_TF_adam --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=36864000 --seed 0 --num_env 2 --save_path ./result/1026Experiments   
    #     # task 12    
    #         # with total task
    #             # sparse
                # python -m baselines.run --record  --rewardModeForArm3d sparse1 --sparse1_dis 0.1 --ps _TfRunningMeanStd_TF_adam_withTotal --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=36864000 --seed 0 --num_env 3 --save_path ./result/1026Experiments   
    #             # density
                # python -m baselines.run --record  --ps _TfRunningMeanStd_TF_adam_withTotal --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=36864000 --seed 0 --num_env 3 --save_path ./result/1026Experiments   
              # without total task
    #         # !!! change cmd_util env mode about task12
    #             # sparse
                # python -m baselines.run --record  --rewardModeForArm3d sparse1 --sparse1_dis 0.1 --ps _TfRunningMeanStd_TF_adam_withoutTotal --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=36864000 --seed 0 --num_env 2 --save_path ./result/1026Experiments   
    #             # density
                # python -m baselines.run --record  --ps _TfRunningMeanStd_TF_adam_withoutTotal --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=36864000 --seed 0 --num_env 2 --save_path ./result/1026Experiments   
        
    # test sparse for task2 and save fix 2
    # python -m baselines.run --record  --rewardModeForArm3d sparse1 --sparse1_dis 0.1 --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_adam --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=36864000 --seed 0 --num_env 2 --save_path ./result/1026_task2

    # test sparse
    # python -m baselines.run --record  --rewardModeForArm3d sparse1 --sparse1_dis 0.1 --task2InitNoise 0.0 --ps _TfRunningMeanStd --ent_coef 0.00 --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=1228800 --seed 0 --num_env 2 --save_path ./result/1026_task2

    # test after fix save
    # python -m baselines.run --record --task2InitNoise 0.0 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --num_env 4 --save_path ./result/1025_fixSave

    # Test load & save
    # python -m baselines.run --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_RMSProp --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --num_env 2 --save_path ./result/testSave
    # python -m baselines.run --load_num_env 2 --load_num 00002 --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_RMSProp --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --load_num_env 2  --load_path ./result/testSave

    # success 
        # python -m baselines.run --load_num_env 2 --load_num 00002 --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_sgd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --load_num_env 2  --load_path ./result/testSave
        # python -m baselines.run  --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_sgd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --num_env 2  --save_path ./result/testSave
    # python -m baselines.run --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_adam --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --num_env 2 --save_path ./result/testSave
    # python -m baselines.run --load_num_env 2 --load_num 00002 --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_adam --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --load_num_env 2 --load_path ./result/testSave
    # python -m baselines.run --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_sgd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --load_num_env 2 --save_path ./result/testSave
    # python -m baselines.run --load_num_env 2 --load_num 00002 --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_adam --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --load_num_env 2 --load_path ./result/testSave
    # python -m baselines.run --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_adam --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --load_num_env 2 --save_path ./result/testSave
    # python -m baselines.run --load_num_env 2 --load_num 00002 --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_version_1.6 --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --load_num_env 2 --load_path ./result/testSave
    # python -m baselines.run --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_version_1.6 --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --load_num_env 2 --save_path ./result/testSave
    # python -m baselines.run --load_num_env 2 --load_num 00002 --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_version --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --load_num_env 2 --load_path ./result/testSave
    # python -m baselines.run --task2InitNoise 0.0 --ps _TfRunningMeanStd_TF_version --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --load_num_env 2 --save_path ./result/testSave
    # python -m baselines.run --load_num_env 2 --load_num 00010 --record --task2InitNoise 0.0 --ps _TfRunningMeanStd_4 --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --load_num_env 2 --load_path ./result/testSave
    # python -m baselines.run --record --task2InitNoise 0.0 --ps _TfRunningMeanStd_4 --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --num_env 2 --save_path ./result/testSave

    # render four experiments for task2
    # python -m baselines.run --record --task2InitNoise 0.0 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --num_env 4 --save_path ./result/1025_1
    # python -m baselines.run --record --task2InitNoise 0.1 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --num_env 4 --save_path ./result/1025_1
    # python -m baselines.run --record --task2InitNoise 0.2 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --num_env 4 --save_path ./result/1025_1
    # python -m baselines.run --record --task2InitNoise 0.3 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=12288000 --seed 0 --num_env 4 --save_path ./result/1025_1

    # render fixed, run all the  rest of the experiments
    # python -m baselines.run --record --task2InitNoise 0 --ps _TfRunningMeanStd_removeTotalTask --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 6 --save_path ./result/1024
    # python -m baselines.run --record --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 6 --save_path ./result/1024
    # python -m baselines.run --record --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 16 --save_path ./result/1024
    # python -m baselines.run --record --task2InitNoise 0 --ps _TfRunningMeanStd --ent_coef 0.01  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 6 --save_path ./result/1024
    # python -m baselines.run --record --task2InitNoise 0.1 --ps _TfRunningMeanStd_withTotalTask --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 9 --save_path ./result/1024
    
        
# 10.24
# python -m baselines.run --record --task2InitNoise 0 --ps _TfRunningMeanStd_removeTotalTask --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 6 --save_path ./result/1024
# python -m baselines.run --record --task2InitNoise 0.0 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 6 --save_path ./result/1024
# python -m baselines.run --record --task2InitNoise 0.1 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 6 --save_path ./result/1024
# python -m baselines.run --record --task2InitNoise 0.2 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 6 --save_path ./result/1024
# python -m baselines.run --record --task2InitNoise 0.3 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 6 --save_path ./result/1024
# python -m baselines.run --record --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 6 --save_path ./result/1024
# python -m baselines.run --record --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 16 --save_path ./result/1024
# python -m baselines.run --record   --task2InitNoise 0 --ps _TfRunningMeanStd --ent_coef 0.01  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 6 --save_path ./result/1024
# python -m baselines.run --record --task2InitNoise 0.1 --ps _TfRunningMeanStd_withTotalTask --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 9 --save_path ./result/1024


# run 10.23 final
# python -m baselines.run --task2InitNoise 0 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 2 --save_path ./result/1023Final
# python -m baselines.run --task2InitNoise 0 --ps _TfRunningMeanStd --ent_coef 0.01  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 2 --save_path ./result/1023Final
# python -m baselines.run --record --task2InitNoise 0.1 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 2 --save_path ./result/test

# run 10.23 
# python -m baselines.run --render  --task2InitNoise 0.1 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 2 --save_path ./result/1023
# python -m baselines.run --record  --task2InitNoise 0 --ps _TfRunningMeanStd --ent_coef 0.00  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 2 --save_path ./result/test

#10.23 
# add self.task2InitNoise  in arm3d, so we should use --task2InitNoise to add noise to task2
# python -m baselines.run --render  --ps _TfRunningMeanStd_ecoef0.01_removeTotalTask  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 6 --save_path ./result/1023Experiments \
# python -m baselines.run --render  --ps _TfRunningMeanStd_ecoef0.00_removeTotalTask_removeNoise  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 2 --save_path ./result/1023Experiments \

#play
# play will meet wrong when the load_num_env is not equal with num_env
# so if you want to play, please set the load_num_env and num_env same, and make num_timesteps
# python -m baselines.run --num_env_play 2  --play --render  --ps _TfRunningMeanStd_ecoef0.01_removeTotalTask  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=0 --seed 0 --num_env 2 --save_path ./result/test/2 \
# --load_num_env 6 --load_path ./result/1022Experiments --load_num 02400 

#10.22
# python -m baselines.run --num_env_play 2  --play --record  --ps _TfRunningMeanStd_ecoef0.01_removeTotalTask  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 6 --save_path ./result/test
# python -m baselines.run --num_env_play 1  --play --record  --ps _TfRunningMeanStd_ecoef0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 3 --save_path ./result/1022Experiments
# SET ENV
# python -m baselines.run --num_env_play 3  --play --record  --ps _TfRunningMeanStd_ecoef0.01  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=4e7 --seed 0 --num_env 9 --save_path ./result/1022Experiments

# change load_path and add load_num
# change load_num_env for the loaded model not the new model
#10.19
# python -m baselines.run \
# --stepNumMax 1111 --env arm3d_task1  --reward_scale 1 --normalize \
# --seed 0 --num_env 2 --save_path ./result/test \
# --num_env_play 2  --play --render \
# --alg=ppo2 --network=mlp --num_timesteps=1 \
# --ps _TfRunningMeanStd_ecoef0.00 #--render 
# --load_num_env 1 --load_path ./result/1019Experiments --load_num 03900 \

# python -m baselines.run --render --ps _TfRunningMeanStd_ecoef0.01  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 9 --save_path ./result/1019Experiments
# python -m baselines.run --render --ps _TfRunningMeanStd_ecoef0.01  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 9 --save_path ./result/1019Experiments
# python -m baselines.run --render --ps _TfRunningMeanStd_ecoef0.01_removeTotalTask  --stepNumMax 1111 --env arm3d_task12 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 6 --save_path ./result/1019Experiments
# python -m baselines.run --render --ps _TfRunningMeanStd_ecoef0.00  --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 3 --save_path ./result/1019Experiments

# 10.18
# python -m baselines.run --ps _TfRunningMeanStd_ecoef0.01 --sparse1_dis 0.3 maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1018Experiments  
# python -m baselines.run --load_num_env 2 --render --load_path 01700 --ps _TfRunningMeanStd_ecoef0.01 --sparse1_dis 0.2 maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1018Experiments  
# python -m baselines.run --ps _TfRunningMeanStd_ecoef0.01 --sparse1_dis 0.1 maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Exp1018Experimentseriments  
# python -m baselines.run --ps _TfRunningMeanStd_ecoef0.01  --stepNumMax 1500 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1018Experiments 

# 10.17
# set ent_coef
    # python -m baselines.run --load_num_env 2 --render --load_path 03900 --ps _TfRunningMeanStd_ecoef0.00  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 2 --save_path ./result/test 
    # ## sparse
        # python -m baselines.run --load_num_env 2 --render --load_path 03900  --ps _TfRunningMeanStd_ecoef0.00 --sparse1_dis 1 maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments  
        # python -m baselines.run --load_num_env 2 --render --load_path 01650 --ps _TfRunningMeanStd_ecoef0.01 --sparse1_dis 1 maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments  
        # python -m baselines.run --ps _TfRunningMeanStd_ecoef0.00 --sparse1_dis 0.7 maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments  
        # python -m baselines.run --load_num_env 2 --render --load_path 01400 --ps _TfRunningMeanStd_ecoef0.00 --sparse1_dis 0.5 maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments  
        # python -m baselines.run --load_num_env 2s --load_path 00250 --render --ps _TfRunningMeanStd_ecoef0.00 --sparse1_dis 0.3 maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments  
    # ## dense baseline
    # python -m baselines.run --load_num_env 2 --render --load_path 03850 --ps _TfRunningMeanStd_ecoef0.01  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments 
    # ## for less steps
        # python -m baselines.run --ps _TfRunningMeanStd_ecoef0.01  --stepNumMax 900 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments   
        # python -m baselines.run --load_num_env 2 --render --load_path 03700 --ps _TfRunningMeanStd_ecoef0.01  --stepNumMax 700 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments   
        # python -m baselines.run baselines.run --load_num_env 2 --render --load_path 03900 --ps _TfRunningMeanStd_ecoef0.01  --stepNumMax 500 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments   
    # # test 2 env
        # python -m baselines.run --ps _TfRunningMeanStd_ecoef0.01  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 2 --save_path ./result/1017Experiments 
    # python -m baselines.run --ps _TfRunningMeanStd_ecoef0.03  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments  
    # python -m baselines.run --ps _TfRunningMeanStd_ecoef0.05  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments  
    # python -m baselines.run --ps _TfRunningMeanStd_ecoef0.07  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments  
    # python -m baselines.run --ps _TfRunningMeanStd_ecoef0.1  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments  
    # python -m baselines.run --ps _TfRunningMeanStd_ecoef0.3  --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e9 --seed 0 --num_env 16 --save_path ./result/1017Experiments  

#play
# python -m baselines.run --ps _TfRunningMeanStd_test maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 16 --save_path ./result/1016Experiments_2  
# python -m baselines.run --load_path 00600 --render --ps _TfRunningMeanStd --sparse1_dis 0.1 maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 16 --save_path ./result/1016Experiments_2  
# 10.16 after meeting
# set noReNor
# python -m baselines.run --ps _TfRunningMeanStd_noReNor --sparse1_dis 0.1 maxSteps --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 2 --save_path ./result/1016Experiments_2  
# GPU 0
# python -m baselines.run --ps _TfRunningMeanStd_noReNor maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 16 --save_path ./result/1016Experiments_2  
# # restore noReNor
# GPU 1
# python -m baselines.run --ps _TfRunningMeanStd maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 16 --save_path ./result/1016Experiments_2  
# # set ent_coef
# GPU 2
# python -m baselines.run --ps _TfRunningMeanStd_ecoef0.01 maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 16 --save_path ./result/1016Experiments_2  
# python -m baselines.run --ps _TfRunningMeanStd_ecoef0.01 --sparse1_dis 0.1 maxSteps --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 2 --save_path ./result/1016Experiments_2  
# # restore ent_coef
# GPU 3
# python -m baselines.run --ps _TfRunningMeanStd --sparse1_dis 0.1 maxSteps --stepNumMax 1111 --env arm3d_task1 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 16 --save_path ./result/1016Experiments_2  

# 10.16-test
# python -m baselines.run --ps _TfRunningMeanStd --sparse1_dis 0.1 maxSteps --stepNumMax 1111 --env arm3d_task2 --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 2 --save_path ./result/1016test  

# 10.16-1
# recurrent sparse1 task2
# python -m baselines.run --sparse1_dis 0.1 maxSteps --stepNumMax 1111 --env arm3d_task2  --render --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 2 --save_path ./result/1016Experiments_1  
# python -m baselines.run --sparse1_dis 0.05 maxSteps --stepNumMax 1111 --env arm3d_task2  --render --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 2 --save_path ./result/test  

# recurrent task12
# python -m baselines.run --sparse1_dis 0.05 maxSteps --stepNumMax 1111 --env arm3d_task12  --render --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 2 --save_path ./result/1016Experiments_1  
# python -m baselines.run --sparse1_dis 0.1maxSteps --stepNumMax 1111 --env arm3d_task12  --render --normalize --reward_scale 1 --rewardModeForArm3d sparse1 --alg=ppo2 --network=mlp --num_timesteps=2e7 --seed 0 --num_env 2 --save_path ./result/1016Experiments_1  
#_NoReNormalizing
# _maxSteps_300 