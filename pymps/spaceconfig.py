'''
SpaceConfiguration for hamiltonian.
'''
from numpy import *
from numpy.linalg import *
from scipy.misc import factorial
from itertools import combinations
import pdb

__all__ = ['SpaceConfig', 'SuperSpaceConfig', 'SpinSpaceConfig',
           'standard_order', 'get_quantum_number']

standard_order = ['nambu', 'atom', 'spin', 'orbit']


def parse_order(config, order):
    '''
    Parse config to standard order.

    Args:
        config (len-4 list): the configuration.
        order (len-4 list): the target order string.

    Returns:
        len-4 list, the configuration in target order.
    '''
    cfg = []
    for s in order:
        cfg.append(config[standard_order.index(s)])
    return cfg


class SpaceConfig(object):
    SPACE_TOKENS = ['nambu', 'atom', 'spin', 'orbit']
    '''
    Space configuration class.
    A typical hamiltonian space config consist of 4 part:

        1. Use Nambu space?
        2. How many spins?
        3. Number of sites.
            the sites in a unit cell for momentum being a good quantum number.
            the total sites for k not a good quantum number.
        4. How many orbital to be considerd.

    config:
        an len-4 array/list, the elements are:

        1. nnambu, 2 if use Nambu space else 1.
        2. nspin, number of spins.
        3. natom, number of atoms.
        4. norbit, number of orbits.
    kspace:
        True if k is a good quantum number and you decided to work in k space.
    '''

    def __init__(self, config, kspace=True):
        self.config = array(config)
        self.kspace = kspace

    def __str__(self):
        return ' x '.join(['%s(%s)' % (token, n) for token, n in zip(self.SPACE_TOKENS, self.config)])

    def __getattr__(self, name):
        if name[0] is 'n':
            substr = name[1:]
            if substr == 'flv':
                if 'nambu' in self.SPACE_TOKENS:
                    return self.ndim / self.nnambu
                else:
                    return self.ndim
            elif substr in self.SPACE_TOKENS:
                return self.config[self.SPACE_TOKENS.index(substr)]
        else:
            raise AttributeError('Can not find attibute %s' % name)

    @property
    def ndim(self):
        '''
        The dimension of space.
        '''
        return self.config.prod()

    @property
    def hndim(self):
        '''
        The dimension of hamiltonian.
        '''
        return self.ndim

    def get_axis(self, token):
        '''
        Get the axis index of specific token.
        '''
        assert(token in self.SPACE_TOKENS)
        return self.SPACE_TOKENS.index(token)

    def ind2c(self, index):
        '''
        Parse index into space len-4 config indices like (nambui,spini,atomi,orbiti).

        index:
            the index(0 ~ ndim-1).
        '''
        N = self.config
        dim = len(N)
        indl = ndarray(list(shape(index)) + [dim], dtype='int32')
        for i in range(dim):
            indl[..., -1 - i] = index % N[-1 - i]
            index = index / N[-1 - i]
        return indl

    def c2ind(self, indices=None, nambuindex=0, spinindex=0, atomindex=0, orbitindex=0):
        '''
        Parse space config index into index.

        indices:
            a len-4 array for all the above indices.
        nambuindex/spinindex/atomindex/orbitindex:
            index for nambuspace/spinspace/atomspace/orbitspace
        '''
        if self.nnambu == 2 and self.nspin == 1:
            spinindex = 0
        if indices is None:
            indices = parse_order(
                [nambuindex, spinindex, atomindex, orbitindex], order=self.SPACE_TOKENS)

        indices = asarray(indices)
        N = self.config[-indices.shape[-1]:]
        assert(indices.shape[-1] == len(N))
        n = indices.shape[-1]
        cc = indices[..., 0]
        for i in range(n - 1):
            cc = cc * N[i + 1] + indices[..., i + 1]
        return cc

    def subspace(self, nambuindex=None, spinindex=None, atomindex=None, orbitindex=None):
        '''
        Get the subspace mask. the single index version of subspace2.

        nambuindex/spinindex/atomindex/orbitindex:
            nambu/spin/atomindex/orbitindex index, default is all indices.
        '''
        if self.nnambu == 2 and self.nspin == 1:
            spinindex = 0
        mask = ones(self.ndim, dtype='bool')
        if nambuindex != None:
            mask = mask & (self._get_indexer('nambu') == nambuindex)
        if spinindex != None:
            mask = mask & (self._get_indexer('spin') == spinindex)
        if atomindex != None:
            mask = mask & (self._get_indexer('atom') == atomindex)
        if orbitindex != None:
            mask = mask & (self._get_indexer('orbit') == orbitindex)
        return mask

    def _get_indexer(self, which):
        '''Get the array for indexing'''
        s_axis = self.get_axis(which)
        nl = prod(self.config[:s_axis])
        nr = prod(self.config[s_axis + 1:])
        return kron(ones(nl, dtype='int32'), kron(arange(self.config[s_axis]), ones(nr, dtype='int32')))


class SuperSpaceConfig(SpaceConfig):
    '''
    Space config in the occupation representation.
    Note that the dimension of space here are different from dimension of hamiltonian.
    Notice, Notice, Notice -> we arange space as |0,1,2,3 ... >, here higher number means lower power.
    e.g.
        |up=0,dn=0> -> 0
        |up=1,dn=0> -> 1
        |up=0,dn=1> -> 2
        |up=1,dn=1> -> 3

    Attributes:
        ne (integer/None, number of electrons): None for unrestricted.
        ne_conserve (bool): True if electron numbers are conserved else False(readonly).
        hndim (integer): the dimension of hilbert space.
        :nsite: the number of sites(flavor).
    '''

    def __init__(self, config, ne=None):
        super(SuperSpaceConfig, self).__init__(
            [1] + list(config)[-3:], kspace=False)
        if ne is not None:
            self._setne(ne)
        else:
            self.ne = None
            self._table = None

        #_binaryparser is a parser from config to index, use '&' operation to parse.
        self._binaryparser = 1 << arange(self.nsite)

    def __str__(self):
        return super(SuperSpaceConfig, self).__str__() + (', with %s electrons.' % self.ne if self.ne_conserve else '')

    def __getattr__(self, name):
        return super(SuperSpaceConfig, self).__getattr__(name)

    @property
    def ne_conserve(self):
        '''
        Electron number conservative.
        '''
        return self.ne is not None

    @property
    def nsite(self):
        '''
        The number of sites, note that it equivalent to nflv, superconducting space is needless.
        '''
        res = self.ndim
        if self.nnambu == 2 and self.nspin >= 2:
            res /= 2
        return res

    @property
    def hndim(self):
        '''
        The dimension of hamiltonian.
        '''
        nsite = self.nsite
        ne = self.ne
        if self.ne_conserve:
            hndim = int(factorial(nsite) / factorial(ne) /
                        factorial(nsite - ne))
        else:
            hndim = pow(2, nsite)
        return hndim

    @property
    def basis(self):
        '''
        get the whole basis.
        '''
        return self.ind2config(arange(self.hndim)).astype(uint8)

    def _setne(self, N):
        '''
        Set eletron number.

        N: 
            the new eletron number.
        '''
        self.ne = N
        comb = fromiter(combinations(arange(self.nsite), N),
                        dtype=[('', int32)] * N).view((int32, N))
        self._table = sort(sum(2**comb, axis=-1))

    def ind2config(self, index):
        '''
        Parse index into eletron configuration.

        Args:
            index (integer/1D array): the index(0 ~ hndim-1).

        Returns:
            1D array/2D array(dtype bool), A electron configuration.
        '''
        if self.ne_conserve:
            id = self._table[index]
            return self.id2config(id)
        else:
            return self.id2config(index)

    def config2ind(self, config):
        '''
        Parse eletron configuration to index.

        Args:
            config (1D/2D array, A electron configuration is len-nsite array with items 0):1(state without/with electron).

        Returns: 
            a index(0 ~ hndim-1).
        '''
        if self.ne_conserve:
            # parse config to id - the number representation
            id = self.config2id(config)
            return searchsorted(self._table, id)
        else:
            return self.config2id(config)

    def config2id(self, config):
        '''
        Convert config to id

        Args:
            config (1D array/2D array): an array of 0 and 1 that indicates the config of electrons.

        Returns:
            integer/1D array, indices.
        '''
        return sum(config * self._binaryparser, axis=-1)

    def id2config(self, id):
        '''
        Convert id to config.

        Args:
            :id: a number indicate the whole space index.

        Returns:
            1D array/2D array, the configuration of electrons.
        '''
        if ndim(id) > 0:
            id = id[..., newaxis]
        return (id & self._binaryparser) > 0

    def indices_occ(self, occ=[], unocc=[], return_info=False):
        '''
        Find the indices with specific sites occupied.

        Args:
            occ (list): the sites with electron. default is no site.
            unocc (list): unoccupied sites. default is no site.
            return_info (bool): get informations.

                * 'rindex': 1d array, indices for reversed occ,unocc
                * 'e_between': list, # of electrons between sites.

        Returns:
            1d array, indices.
            (1d array, dict) if return_info.
        '''
        occ = unique(occ).astype('int32')
        unocc = unique(unocc).astype('int32')
        no = len(occ)
        nn = len(unocc)
        # get remained sites
        usedsites = concatenate([occ, unocc])
        order = argsort(usedsites)
        usedsites = usedsites[order]
        usedcounter = append(ones(len(occ), dtype='int32'),
                             zeros(len(unocc), dtype='int32'))[order]
        remainsites = delete(arange(self.nsite), usedsites)
        if no + nn != self.nsite - len(remainsites):
            raise ValueError(
                'no match for indices occ-%s,unocc-%s! @indices_occ' % (occ, unocc))
        # get possible combinations
        if self.ne_conserve:
            distri = array(list(combinations(remainsites, self.ne - no)))
            if return_info:
                ecounter = [(distri < site).sum(axis=-1) for site in usedsites]
            # turn into indices.
            ids0 = (2**distri).sum(axis=-1)
            ids = ids0 + (2**occ).sum()
            indices = searchsorted(self._table, ids)
            if return_info:
                rids = ids0 + (2**unocc).sum()
                rindex = searchsorted(self._table, rids)
                info = {'rindex': rindex, 'e_between': ecounter}
                return indices, info
            else:
                return indices
        else:
            distri = self.id2config(
                arange(2**(self.nsite - no - nn)))[..., :-no - nn]
            if return_info:
                # ecounter=distri[...,usedsites[0]:usedsites[1]-1].sum(axis=-1)
                ecounter = [distri[..., :site - i].sum(axis=-1) + sum(
                    usedcounter[:i]) for i, site in enumerate(usedsites)]
            parser = delete(self._binaryparser, usedsites)
            ids0 = (distri * parser).sum(axis=-1)
            ids = ids0 + (2**occ).sum()
            res = [ids]
            if return_info:
                info = {'rindex': ids0 +
                        (2**unocc).sum(), 'e_between': ecounter}
                return ids, info
            else:
                return ids

    def get_quantum_number(self, name):
        '''
        Args:
            name (str): name of good quantum number, it can be one of
                * 'M': nup-ndn, or 2*Sz
                * 'Q': nup+ndm,
                * 'P': charge parity, Q%2
                * 'R': spin parity, M%4

        Returns:
            1darray: a list of quantum number corresponding to each spaceconfig dimension.
        '''
        # check data
        if name not in ['P', 'Q', 'M', 'R']:
            raise ValueError

        # get configuration table
        hndim = self.hndim
        configs = self.ind2config(arange(hndim))

        qns = zeros([hndim, nstr], dtype='int32')
        if qtype == 'M' or qtype == 'R':
            # for quantum number nup-ndn
            nspin = self.nspin
            for i in range(nspin):
                spini = nspin - 2 * i - 1  # nspin/2.-i-0.5
                imask = self.subspace(spinindex=i)
                qns += configs[:, imask].sum(axis=-1) * spini
                if qtype == 'R':
                    qns %= 4
        else:
            qns = sum(configs, axis=-1)
            if qtype == 'P':
                qns %= self.per[istr]
        return qns


class SpinSpaceConfig(SpaceConfig):
    '''
    Space configuration for spin system.

    spin number and atom number are considered.
    '''
    SPACE_TOKENS = ['atom', 'spin']

    def __init__(self, config):
        assert(len(config) == 2 and config[1] > 1)
        super(SpinSpaceConfig, self).__init__(config, kspace=False)

    @property
    def hndim(self):
        '''Hamiltonian dimension.'''
        return self.config[1]**self.config[0]

    def config2ind(self, config):
        '''
        Convert config to id

        Args:
            config (1D array/2D array): an array of 0 and 1 that indicates the config of electrons.

        Returns:
            integer/1D array, indices.
        '''
        nspin = self.nspin
        config = asarray(config)
        return sum(config * self.nspin**arange(self.natom - 1, -1, -1), axis=-1)

    def ind2config(self, ind):
        '''
        Convert id to config.

        Args:
            ind (int): ind-th dimension.

        Returns:
            1D array/2D array, the configuration of spins.
        '''
        nspin = self.nspin
        ind = asarray(ind)
        config = []
        for i in range(self.natom):
            ci = ind % nspin
            ind = (ind - ci) / nspin
            config.append(ci[..., newaxis])
        return concatenate(config[::-1], axis=-1)

    def get_quantum_number(self, name):
        '''
        Args:
            name (str): name of good quantum number, it can be one of
                * 'M': nup-ndn, or 2*Sz
                * 'R': spin parity, M%4

        Returns:
            1darray: a list of quantum number corresponding to each spaceconfig dimension.
        '''
        # check data
        if name not in ['M', 'R']:
            raise ValueError

        # get configuration table
        hndim = self.hndim
        configs = self.ind2config(arange(hndim))

        qns = zeros(hndim, dtype='int32')
        # get quantum number nup-ndn
        nspin = self.nspin
        for i in range(nspin):
            spini = nspin - 2 * i - 1  # nspin/2.-i-0.5
            qns += sum(spini * (configs == i), axis=-1)
        if qtype == 'R':
            qns %= 4
        return qns
