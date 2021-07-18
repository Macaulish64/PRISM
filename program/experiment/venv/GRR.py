import numpy as np


def perturb(datas, domain):

    eps=5.0#自己设定


    ee = np.exp(eps)
    p = ee / (ee + domain - 1)
    q = 1 / (ee + domain - 1)
    var = q * (1 - q) / (p - q) ** 2


    n = len(datas)
    perturbed_datas = np.zeros(n, dtype=np.int)
    for i in range(n):
        y = x = datas[i]
        p_sample = np.random.random_sample()

        if p_sample > p:
            y = np.random.randint(0, domain - 1)
            if y >= x:
                y += 1
        perturbed_datas[i] = y
    return perturbed_datas

if __name__ == '__main__':
    vector=[0,0,1,0,0,0,0,0]
    domain=2
    epsilon=1
    perturbed_vector=perturb(vector,domain)
    print(perturbed_vector)