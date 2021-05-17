import foolbox.attacks as attacks
import foolbox.distances as distances

distance_objects = [	
				distances.MeanSquaredDistance,
				distances.MeanAbsoluteDistance,
				distances.Linfinity,
				distances.L0
			]

distances_names = [
					'L2',
					'L1',
					'Linf',
					'L0'
				  ]

gradient_attacks = [
					attacks.GradientAttack ,
					attacks.GradientSignAttack ,
					attacks.FGSM ,
					attacks.LinfinityBasicIterativeAttack ,
					attacks.BasicIterativeMethod ,
					attacks.BIM ,
					attacks.L1BasicIterativeAttack ,
					attacks.L2BasicIterativeAttack ,
					attacks.ProjectedGradientDescentAttack ,
					attacks.ProjectedGradientDescent ,
					attacks.PGD ,
					attacks.RandomStartProjectedGradientDescentAttack ,
					attacks.RandomProjectedGradientDescent ,
					attacks.RandomPGD ,
					attacks.AdamL1BasicIterativeAttack ,
					attacks.AdamL2BasicIterativeAttack ,
					attacks.AdamProjectedGradientDescentAttack ,
					attacks.AdamProjectedGradientDescent ,
					attacks.AdamPGD ,
					attacks.AdamRandomStartProjectedGradientDescentAttack ,
					attacks.AdamRandomProjectedGradientDescent ,
					attacks.AdamRandomPGD ,
					attacks.MomentumIterativeAttack ,
					attacks.MomentumIterativeMethod ,
					attacks.DeepFoolAttack ,
					attacks.NewtonFoolAttack ,
					attacks.DeepFoolL2Attack ,
					attacks.DeepFoolLinfinityAttack ,
					attacks.ADefAttack ,
					attacks.SaliencyMapAttack ,
					attacks.IterativeGradientAttack ,
					attacks.IterativeGradientSignAttack ,
					attacks.CarliniWagnerL2Attack ,
					attacks.EADAttack ,
					attacks.DecoupledDirectionNormL2Attack ,
					attacks.SparseL1BasicIterativeAttack ,
					attacks.VirtualAdversarialAttack ,
					]


gradient_attacks_names = [
					'GradientAttack',
					'GradientSignAttack',
					'FGSM',
					'LinfinityBasicIterativeAttack',
					'BasicIterativeMethod',
					'BIM',
					'L1BasicIterativeAttack',
					'L2BasicIterativeAttack',
					'ProjectedGradientDescentAttack',
					'ProjectedGradientDescent',
					'PGD',
					'RandomStartProjectedGradientDescentAttack',
					'RandomProjectedGradientDescent',
					'RandomPGD',
					'AdamL1BasicIterativeAttack',
					'AdamL2BasicIterativeAttack',
					'AdamProjectedGradientDescentAttack',
					'AdamProjectedGradientDescent',
					'AdamPGD',
					'AdamRandomStartProjectedGradientDescentAttack',
					'AdamRandomProjectedGradientDescent',
					'AdamRandomPGD',
					'MomentumIterativeAttack',
					'MomentumIterativeMethod',
					'DeepFoolAttack',
					'NewtonFoolAttack',
					'DeepFoolL2Attack',
					'DeepFoolLinfinityAttack',
					'ADefAttack',
					'SaliencyMapAttack',
					'IterativeGradientAttack',
					'IterativeGradientSignAttack',
					'CarliniWagnerL2Attack',
					'EADAttack',
					'DecoupledDirectionNormL2Attack',
					'SparseL1BasicIterativeAttack',
					'VirtualAdversarialAttack',
					]

score_attacks = 	[
					attacks.SinglePixelAttack ,
					attacks.LocalSearchAttack ,
					]
	

score_attacks_names = 	[
					'SinglePixelAttack' ,
					'LocalSearchAttack' ,
					]


decision_attacks =	[
					attacks.BoundaryAttack ,
					attacks.SpatialAttack ,
					attacks.PointwiseAttack ,
					attacks.GaussianBlurAttack ,
					attacks.ContrastReductionAttack ,
					attacks.AdditiveUniformNoiseAttack ,
					attacks.AdditiveGaussianNoiseAttack ,
					attacks.SaltAndPepperNoiseAttack ,
					attacks.BlendedUniformNoiseAttack ,
					# attacks.BoundaryAttackPlusPlus ,
					attacks.GenAttack ,
					attacks.HopSkipJumpAttack ,
					]

decision_attacks_names = [
					'BoundaryAttack',
					'SpatialAttack',
					'PointwiseAttack',
					'GaussianBlurAttack',
					'ContrastReductionAttack',
					'AdditiveUniformNoiseAttack',
					'AdditiveGaussianNoiseAttack',
					'SaltAndPepperNoiseAttack',
					'BlendedUniformNoiseAttack',
					# 'BoundaryAttackPlusPlus',
					'GenAttack',
					'HopSkipJumpAttack',				
					]

all_attacks = gradient_attacks + score_attacks + decision_attacks
all_attacks_names = gradient_attacks_names + score_attacks_names + decision_attacks_names
