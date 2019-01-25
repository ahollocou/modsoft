from libcpp.vector cimport vector
from libcpp.map cimport map
from numpy cimport int_t, float_t, int32_t

cimport cython


@cython.cdivision(True)
cdef map[int_t, float_t] project_top_K(map[int_t, float_t] in_map, int_t K):
    cdef float_t cum_sum = 0
    cdef lamb, new_lamb, v, v_max = 0.
    cdef int_t i = 0
    cdef int_t j, j_max
    cdef map[int_t, float_t] out_map
    while in_map.size() > 0 and i < K:
        first = 1
        for it in in_map:
            j = it.first
            v = it.second
            if first==1 or v > v_max:
                v_max = v
                j_max = j
                first = 0
        in_map.erase(j_max)
        cum_sum += v_max
        new_lamb = (1. / (i + 1.)) * (1. - cum_sum)
        if v_max + new_lamb <= 0.:
            break
        else:
            out_map[j_max] = v_max
            lamb = new_lamb
            i += 1

    for it in out_map:
        i = it.first
        v = it.second
        out_map[i] = max(v + lamb, 0)

    return out_map

        
cdef class ModSoft(object):
    
    cdef int_t n_com
    cdef vector[map[int_t, float_t]] p
    cdef vector[float_t] avg_p
    
    cdef int_t n_nodes
    cdef vector[map[int_t, float_t]] graph_edges
    cdef vector[float_t] degree
    cdef float_t w, w_inv
    
    cdef float_t learning_rate, bias

    cdef float_t resolution
    
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def __init__(self, int_t n_nodes, int32_t[:] indices, int32_t[:] indptr, float_t[:] data,
                 int_t n_com, float_t learning_rate, bias, int_t[:] init_part, float_t resolution):
        self.n_com = n_com
        self.learning_rate = learning_rate
        self.bias = bias

        self.resolution = resolution

        self.n_nodes = n_nodes
        self.graph_edges.resize(self.n_nodes)
        self.degree.resize(self.n_nodes)
        self.w = 0
        cdef int_t node, i
        for node in range(n_nodes):
            for i in range(indptr[node], indptr[node + 1]):
                self.graph_edges[node][indices[i]] = data[i]
                if indices[i] == node:
                    self.degree[node] += 2. * data[i]
                    self.w += 2. * data[i]
                else:
                    self.degree[node] += data[i]
                    self.w += data[i]
        self.w_inv = 1. / self.w
        
        self.p.resize(self.n_nodes)
        self.avg_p.resize(self.n_nodes)
        for node in range(n_nodes):
            self.p[node][init_part[node]] = 1.
            self.avg_p[init_part[node]] += self.w_inv * self.degree[node]

    @cython.boundscheck(False)
    def modularity(self):
        cdef float_t Q = 0.
        for com in range(self.n_nodes):
            Q -= self.avg_p[com] * self.avg_p[com]
        cdef float_t weight
        for node in range(self.n_nodes):
            for it1 in self.graph_edges[node]:
                neighbor = it1.first
                weight = it1.second
                if neighbor == node:
                    weight *= 2.
                for it2 in self.p[node]:
                    com = it2.first
                    p_com = it2.second
                    if self.p[neighbor].count(com) > 0:
                        Q += self.resolution * self.w_inv * weight * p_com * self.p[neighbor][com]
        return Q
            
    def one_step(self, int_t n_com=0, float_t learning_rate=0., float_t bias=0.):
        if n_com == 0:
            n_com = self.n_com
        if learning_rate == 0.:
            learning_rate = self.learning_rate
        if bias == 0.:
            bias = self.bias
        
        cdef map[int_t, float_t] new_p
        
        cdef int_t neighbor, com
        cdef float_t weight, p_com, new_p_com
        
        for node in range(self.n_nodes):
            new_p.clear()
            for it1 in self.p[node]:
                com = it1.first
                p_com = it1.second
                new_p[com] = bias * p_com
            for it1 in self.graph_edges[node]:
                neighbor = it1.first
                weight = it1.second
                for it2 in self.p[neighbor]:
                    com = it2.first
                    p_com = it2.second
                    if new_p.count(com) == 0:
                        new_p[com] = 0
                    new_p[com] += self.resolution * learning_rate * weight * p_com
            for it1 in new_p:
                com = it1.first
                new_p_com = it1.second
                new_p[com] = new_p_com - learning_rate * self.degree[node] * self.avg_p[com]
            new_p = project_top_K(new_p, n_com)
            for it1 in self.p[node]:
                com = it1.first
                p_com = it1.second
                self.avg_p[com] -= self.w_inv * self.degree[node] * p_com
            self.p[node].clear()
            for it1 in new_p:
                com = it1.first
                new_p_com = it1.second
                self.avg_p[com] += self.w_inv * self.degree[node] * new_p_com
                self.p[node][com] = new_p_com

    def get_membership(self):
        return self.p