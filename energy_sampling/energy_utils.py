from energies import NineGaussianMixture, TwentyFiveGaussianMixture, HardFunnel, EasyFunnel, ManyWell

def get_energy(energy_name, device):
    if energy_name == '9gmm':
        energy = NineGaussianMixture(device=device)
    elif energy_name == '25gmm':
        energy = TwentyFiveGaussianMixture(device=device)
    elif energy_name == 'hard_funnel':
        energy = HardFunnel(device=device)
    elif energy_name == 'easy_funnel':
        energy = EasyFunnel(device=device)
    elif energy_name == 'many_well':
        energy = ManyWell(device=device)
    return energy