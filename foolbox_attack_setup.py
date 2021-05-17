import foolbox.attacks as attacks
import foolbox.distances as distances

distance_objects = [	
				distances.MeanAbsoluteDistance,
				distances.MeanSquaredDistance,
				distances.Linfinity,
				distances.L0
			]

distances_names = [
					'L1',
					'L2',
					'Linf',
					'L0'
				  ]

# use the following to setup each attack
# {
# 'name': ,
# 'object': ,
# 'params': ,
# }

l1_attack_dicts = [	

					{
					'name': 'Saliency Map Attack, theta=0.02',
					'object': attacks.SaliencyMapAttack,
					'params': {'theta': 0.02, 'max_perturbations_per_pixel':15} ,
					},
					{
					'name': 'Saliency Map Attack, theta=0.03',
					'object': attacks.SaliencyMapAttack,
					'params': {'theta': 0.03} ,
					},
					{
					'name': 'Saliency Map Attack, not fast',
					'object': attacks.SaliencyMapAttack,
					'params': {'fast': False} ,
					},
					{
					'name': 'Saliency Map Attack, theta=0.5',
					'object': attacks.SaliencyMapAttack,
					'params': {'theta': 0.5} ,
					},
					{
					'name': 'Saliency Map Attack, theta=1.',
					'object': attacks.SaliencyMapAttack,
					'params': {'theta': 1.} ,
					},
					{
					'name': 'Saliency Map Attack, default',
					'object': attacks.SaliencyMapAttack,
					'params': {} ,
					},
					{
					'name': 'EAD Attack, beta=0.1',
					'object': attacks.EADAttack,
					'params': {'regularization': 0.1} ,
					},
					{
					'name': 'EAD Attack, initial_lr=0.003',
					'object': attacks.EADAttack,
					'params': {'initial_learning_rate': 3e-3} ,
					},
					{
					'name': 'EAD Attack, beta=0.01',
					'object': attacks.EADAttack,
					'params': {'regularization': 0.01} ,
					},
					{
					'name': 'EAD Attack, default',
					'object': attacks.EADAttack,
					'params': {} ,
					},
					{
					'name': 'Pointwise Attack, default',
					'object': attacks.PointwiseAttack,
					'params': {} ,
					},
					{
					'name': 'Sparse L1 Basic Iterative Attack, default',
					'object': attacks.SparseL1BasicIterativeAttack,
					'params': {} ,
					},

				  ]

l2_attack_dicts = [	
					{
					'name': 'Carlini Wagner L2 Attack, default',
					'object': attacks.CarliniWagnerL2Attack,
					'params': {} ,
					},
					# {
					# 'name': 'Carlini Wagner L2 Attack, lr=5e-3',
					# 'object': attacks.CarliniWagnerL2Attack,
					# 'params': {'learning_rate': 5e-3, 'max_iterations':1000} ,
					# },
					{
					'name': 'Carlini Wagner L2 Attack, lr=1e-3',
					'object': attacks.CarliniWagnerL2Attack,
					'params': {'learning_rate': 1e-3, 'max_iterations':2000} ,
					},
					# {
					# 'name': 'Carlini Wagner L2 Attack, lr=5e-4',
					# 'object': attacks.CarliniWagnerL2Attack,
					# 'params': {'learning_rate': 5e-4, 'max_iterations':5000} ,
					# },
					{
					'name': 'Decoupled Direction Norm L2 Attack, levels=10000',
					'object': attacks.DecoupledDirectionNormL2Attack,
					'params': {'levels': 10000} ,
					},
					{
					'name': 'Decoupled Direction Norm L2 Attack, levels=1024',
					'object': attacks.DecoupledDirectionNormL2Attack,
					'params': {'levels': 1024} ,
					},
					{
					'name': 'Decoupled Direction Norm L2 Attack, levels=256',
					'object': attacks.DecoupledDirectionNormL2Attack,
					'params': {'levels': 256} ,
					},
					{
					'name': 'Decoupled Direction Norm L2 Attack, default',
					'object': attacks.DecoupledDirectionNormL2Attack,
					'params': {} ,
					},
					{
					'name': 'L2 Basic Iterative Attack, default',
					'object': attacks.L2BasicIterativeAttack,
					'params': {} ,
					},
					{
					'name': 'L1 Basic Iterative Attack, default',
					'object': attacks.L1BasicIterativeAttack,
					'params': {} ,
					},
				  ]




l0_attack_dicts = [	

					{
					'name': 'Saliency Map Attack, theta=1., max_per_pixel=15',
					'object': attacks.SaliencyMapAttack,
					'params': {'theta': 0.02, 'max_perturbations_per_pixel':15} ,
					},
					{
					'name': 'Saliency Map Attack, theta=1., max_per_pixel=30',
					'object': attacks.SaliencyMapAttack,
					'params': {'theta': 0.02, 'max_perturbations_per_pixel':30} ,
					},
					{
					'name': 'Saliency Map Attack, theta=1., max_per_pixel=60',
					'object': attacks.SaliencyMapAttack,
					'params': {'theta': 0.02, 'max_perturbations_per_pixel':60} ,
					},
					# {
					# 'name': 'Saliency Map Attack, theta=0.1',
					# 'object': attacks.SaliencyMapAttack,
					# 'params': {'theta': 0.1} ,
					# },
					{
					'name': 'Saliency Map Attack, theta=0.5',
					'object': attacks.SaliencyMapAttack,
					'params': {'theta': 0.5} ,
					},
					{
					'name': 'Saliency Map Attack, theta=1.',
					'object': attacks.SaliencyMapAttack,
					'params': {'theta': 1.} ,
					},
					# {
					# 'name': 'Saliency Map Attack, default',
					# 'object': attacks.SaliencyMapAttack,
					# 'params': {} ,
					# },					
					{
					'name': 'Pointwise Attack, default',
					'object': attacks.PointwiseAttack,
					'params': {} ,
					},
					{
					'name': 'Sparse L1 Basic Iterative Attack, default',
					'object': attacks.SparseL1BasicIterativeAttack,
					'params': {} ,
					},
					{
					'name': 'Local Search Attack, default',
					'object': attacks.LocalSearchAttack,
					'params': {} ,
					},
					{
					'name': 'Local Search Attack, p=1',
					'object': attacks.LocalSearchAttack,
					'params': {'p':1} ,
					},
					{
					'name': 'Local Search Attack, p=5',
					'object': attacks.LocalSearchAttack,
					'params': {'p':5} ,
					},
					{
					'name': 'Salt And Pepper Attack, default',
					'object': attacks.SaltAndPepperNoiseAttack,
					'params': {} ,
					},
				  ]


linf_attack_dicts = [	

					{
					'name': 'Linfinity Basic Iterative Attack, default',
					'object': attacks.LinfinityBasicIterativeAttack,
					'params': {} ,
					},
					{
					'name': 'Linfinity Basic Iterative Attack, stepsize=0.0003',
					'object': attacks.LinfinityBasicIterativeAttack,
					'params': {'stepsize': 0.0003} ,
					},
					{
					'name': 'Linfinity Basic Iterative Attack, stepsize=0.001',
					'object': attacks.LinfinityBasicIterativeAttack,
					'params': {'stepsize': 0.001} ,
					},
					{
					'name': 'Linfinity Basic Iterative Attack, stepsize=0.003',
					'object': attacks.LinfinityBasicIterativeAttack,
					'params': {'stepsize': 0.003} ,
					},
					{
					'name': 'Momentum Iterative Attack, default',
					'object': attacks.MomentumIterativeAttack,
					'params': {} ,
					},
					{
					'name': 'Momentum Iterative Attack, stepsize = 0.001',
					'object': attacks.MomentumIterativeAttack,
					'params': {'stepsize': 0.001} ,
					},
					{
					'name': 'Momentum Iterative Attack, stepsize = 0.003',
					'object': attacks.MomentumIterativeAttack,
					'params': {'stepsize': 0.003} ,
					},
					{
					'name': 'Adam Projected Gradient Descent, default',
					'object': attacks.AdamProjectedGradientDescentAttack,
					'params': {} ,
					},
					{
					'name': 'Adam Projected Gradient Descent, stepsize = 0.003',
					'object': attacks.AdamProjectedGradientDescentAttack,
					'params': {'stepsize': 0.003} ,
					},
					{
					'name': 'Adam Projected Gradient Descent, stepsize = 0.001',
					'object': attacks.AdamProjectedGradientDescentAttack,
					'params': {'stepsize': 0.001} ,
					},
				  ]

attack_dict_list = [
					l1_attack_dicts,
					l2_attack_dicts,
					linf_attack_dicts,
					l0_attack_dicts
				  ]