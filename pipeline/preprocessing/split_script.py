import random as rnd
rnd.seed(42)

def split_script(countries, num_groups):
    
    unique_countries = list(countries)
    rnd.shuffle(unique_countries)

    groups = [unique_countries[i::num_groups] for i in range(num_groups)]
    return groups
