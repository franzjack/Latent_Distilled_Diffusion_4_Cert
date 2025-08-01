
def get_model_details(opt):

	if opt.model_name == "eSIRS":
	    opt.species_labels = ["S", "I"]
	    opt.x_dim = 2
	    opt.traj_len = 32
	elif opt.model_name == "SIR":
	    opt.species_labels = ["S", "I"]
	    opt.x_dim = 2
	    opt.traj_len = 16
	elif opt.model_name == "TS":
	    opt.species_labels = ["P1", "P2"]
	    opt.x_dim = 2
	    opt.traj_len = 32
	elif opt.model_name == "Toy" or opt.model_name == "Oscillator":
	    opt.species_labels = ["A", "B", "C"]
	    opt.x_dim = 3
	    opt.traj_len = 32
	elif opt.model_name == "MAPK":
	    opt.species_labels = ["M3K"]
	    opt.x_dim = 1
	    opt.p_dim = 1
	    opt.traj_len = 32
	elif opt.model_name == "EColi":
	    opt.species_labels = ["N", "C", "S"]
	    opt.x_dim = 3
	    opt.p_dim = 2
	    opt.traj_len = 32
	elif opt.model_name == "LV":
	    opt.species_labels = ["A", "B"]
	    opt.x_dim = 2
	    opt.traj_len = 63
	elif opt.model_name == "LV64":
	    opt.species_labels = ["A", "B"]
	    opt.x_dim = 2
	    opt.traj_len = 64
	else:
		opt.species_labels = []

	return opt