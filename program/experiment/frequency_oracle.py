from math import exp
import utils
import itertools
import numpy as np
import xxhash


class FrequencyOracle:
    def __init__(self):
        pass

    class UE():

        def __init__(self, eps, domain):
            self.ee = np.exp(eps)
            self.p = 0.5
            self.q = 1 / (self.ee + 1)
            self.var = 4 * self.ee / (self.ee - 1) ** 2

        def perturb(self, datas, domain):
            n = len(datas)
            perturbed_datas = np.zeros(n, dtype=object)
            samples_one = np.random.random_sample(n)
            for i in range(n):
                samples_zero = np.random.random_sample(domain)

                # enlarge domain to avoid overflow during aggregation
                y = np.zeros(domain, dtype=np.int32)

                for k in range(domain):
                    if samples_zero[k] < self.q:
                        y[k] = 1

                v = datas[i]
                y[v] = 1 if samples_one[i] < self.p else 0

                perturbed_datas[i] = y

            return perturbed_datas

        def aggregate(self, domain, perturbed_datas):
            ESTIMATE_DIST = np.sum(perturbed_datas, axis=0)
            return ESTIMATE_DIST

    class RR():

        def __init__(self, eps, domain):
            self.ee = np.exp(eps)
            self.p = self.ee / (self.ee + domain - 1)
            self.q = 1 / (self.ee + domain - 1)
            self.var = self.q * (1 - self.q) / (self.p - self.q) ** 2

        def perturb(self, datas, domain):
            n = len(datas)
            perturbed_datas = np.zeros(n, dtype=np.int)
            for i in range(n):
                y = x = datas[i]
                p_sample = np.random.random_sample()

                if p_sample > self.p:
                    y = np.random.randint(0, domain - 1)
                    if y >= x:
                        y += 1
                perturbed_datas[i] = y
            return perturbed_datas

        def support_sr(self, report, value):
            return report == value

        def aggregate(self, domain, perturbed_datas):
            ESTIMATE_DIST = np.zeros(domain)
            unique, counts = np.unique(perturbed_datas, return_counts=True)
            for i in range(len(unique)):
                ESTIMATE_DIST[unique[i]] = counts[i]

            return ESTIMATE_DIST

    def optimized_unary_encoding(self, epsilon, original_vector, num_users):
        domain_size = len(original_vector)
        ue = UE(epsilon, domain_size)
        perturbed_datas = ue.perturb(original_vector, domain_size)
        aggregated_datas = ue.aggregate(domain_size, perturbed_datas)
        return aggregated_datas

    def generalized_randomized_response(self, epsilon, original_vector):
        domain_size = len(original_vector)
        rr=FrequencyOracle.RR(epsilon,domain_size)

        perturbed_datas = rr.perturb(original_vector, domain_size)
        aggregated_datas = rr.aggregate(domain_size, perturbed_datas)
        return aggregated_datas
if __name__ == '__main__':
    vector=[0,0,1,0,0,0,0,0]
    epsilon=1
    F=(FrequencyOracle())
    print(F.generalized_randomized_response(epsilon,vector))