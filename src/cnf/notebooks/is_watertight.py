import numpy as np

TOL_MERGE = 1e-8
ZERO = np.finfo(np.float64).resolution * 100


def decimal_to_digits(decimal, min_digits=None) -> int:
    """
    Return the number of digits to the first nonzero decimal.

    Parameters
    -----------
    decimal:    float
    min_digits: int, minimum number of digits to return

    Returns
    -----------

    digits: int, number of digits to the first nonzero decimal
    """
    digits = abs(int(np.log10(decimal)))
    if min_digits is not None:
        digits = np.clip(digits, min_digits, 20)
    return int(digits)


def float_to_int(data, digits=None):
    """
    Given a numpy array of float/bool/int, return as integers.

    Parameters
    -------------
    data :  (n, d) float, int, or bool
      Input data
    digits : float or int
      Precision for float conversion

    Returns
    -------------
    as_int : (n, d) int
      Data as integers
    """
    # convert to any numpy array
    data = np.asanyarray(data)

    # we can early-exit if we've been passed data that is already
    # an integer, unsigned integer, boolean, or empty
    if data.dtype.kind in "iub" or data.size == 0:
        return data
    elif data.dtype.kind != "f":
        # if it's not a floating point try to make it one
        data = data.astype(np.float64)

    if digits is None:
        # get digits from `tol.merge`
        digits = decimal_to_digits(TOL_MERGE)
    elif not isinstance(digits, (int, np.integer)):
        raise TypeError("Digits must be `None` or `int`, not `{type(digits)}`")

    # see if we can use a smaller integer
    d_extrema = max(abs(data.min()), abs(data.max())) * 10**digits

    # compare against `np.iinfo(np.int32).max`
    dtype = [np.int32, np.int64][int(d_extrema > 2147483646)]

    # multiply by requested power of ten
    # then subtract small epsilon to avoid "go either way" rounding
    # then do the rounding and convert to integer
    return np.round((data * 10**digits) - 1e-6).astype(dtype)


def hashable_rows(data, digits=None):
    """
    We turn our array into integers based on the precision
    given by digits and then put them in a hashable format.

    Parameters
    ---------
    data : (n, m) array
      Input data
    digits : int or None
      How many digits to add to hash if data is floating point
      If None, tol.merge will be used

    Returns
    ---------
    hashable : (n,)
      May return as a `np.void` or a `np.uint64`
    """
    # if there is no data return immediately
    if len(data) == 0:
        return np.array([])

    # get array as integer to precision we care about
    as_int = float_to_int(data, digits=digits)

    # if it is flat integers already return
    if len(as_int.shape) == 1:
        return as_int

    # if array is 2D and smallish, we can try bitbanging
    # this is significantly faster than the custom dtype
    if len(as_int.shape) == 2 and as_int.shape[1] <= 4:
        # can we pack the whole row into a single 64 bit integer
        precision = int(np.floor(64 / as_int.shape[1]))

        # get the extreme values of the data set
        d_min, d_max = as_int.min(), as_int.max()
        # since we are quantizing the data down we need every value
        # to fit in a partial integer so we have to check against extrema
        threshold = 2 ** (precision - 1)

        # if the data is within the range of our precision threshold
        if d_max < threshold and d_min > -threshold:
            # the resulting package
            hashable = np.zeros(len(as_int), dtype=np.uint64)
            # offset to the middle of the unsigned integer range
            # this array should contain only positive values
            bitbang = (as_int + threshold).astype(np.uint64).T
            # loop through each column and bitwise xor to combine
            # make sure as_int is int64 otherwise bit offset won't work
            for offset, column in enumerate(bitbang):
                # will modify hashable in place
                np.bitwise_xor(hashable, column << (offset * precision), out=hashable)
            return hashable

    # reshape array into magical data type that is weird but works with unique
    dtype = np.dtype((np.void, as_int.dtype.itemsize * as_int.shape[1]))
    # make sure result is contiguous and flat
    result = np.ascontiguousarray(as_int).view(dtype).reshape(-1)
    result.flags["WRITEABLE"] = False

    return result


def group(values, min_len=0, max_len=np.inf):
    """
    Return the indices of values that are identical

    Parameters
    ----------
    values : (n,) int
      Values to group
    min_len : int
      The shortest group allowed
      All groups will have len >= min_length
    max_len : int
      The longest group allowed
      All groups will have len <= max_length

    Returns
    ----------
    groups : sequence
      Contains indices to form groups
      IE [0,1,0,1] returns [[0,2], [1,3]]
    """
    original = np.asanyarray(values)

    # save the sorted order and then apply it
    order = original.argsort()
    values = original[order]

    # find the indexes which are duplicates
    if values.dtype.kind == "f":
        # for floats in a sorted array, neighbors are not duplicates
        # if the difference between them is greater than approximate zero
        nondupe = np.greater(np.abs(np.diff(values)), ZERO)
    else:
        # for ints and strings we can check exact non- equality
        # for all other types this will only work if they defined
        # an __eq__
        nondupe = values[1:] != values[:-1]

    dupe_idx = np.append(0, np.nonzero(nondupe)[0] + 1)
    dupe_len = np.diff(np.concatenate((dupe_idx, [len(values)])))
    dupe_ok = np.logical_and(
        np.greater_equal(dupe_len, min_len), np.less_equal(dupe_len, max_len)
    )
    groups = [order[i : (i + j)] for i, j in zip(dupe_idx[dupe_ok], dupe_len[dupe_ok])]
    return groups


def group_rows(data, require_count=None, digits=None):
    """
    Returns index groups of duplicate rows, for example:
    [[1,2], [3,4], [1,2]] will return [[0,2], [1]]


    Note that using require_count allows numpy advanced
    indexing to be used in place of looping and
    checking hashes and is ~10x faster.


    Parameters
    ----------
    data : (n, m) array
      Data to group
    require_count : None or int
      Only return groups of a specified length, eg:
      require_count =  2
      [[1,2], [3,4], [1,2]] will return [[0,2]]
    digits : None or int
    If data is floating point how many decimals
    to consider, or calculated from tol.merge

    Returns
    ----------
    groups : sequence (*,) int
      Indices from in indicating identical rows.
    """

    # start with getting a sortable format
    hashable = hashable_rows(data, digits=digits)

    # if there isn't a constant column size use more complex logic
    if require_count is None:
        return group(hashable)

    # record the order of the rows so we can get the original indices back
    order = hashable.argsort()
    # but for now, we want our hashes sorted
    hashable = hashable[order]
    # this is checking each neighbour for equality, example:
    # example: hashable = [1, 1, 1]; dupe = [0, 0]
    dupe = hashable[1:] != hashable[:-1]
    # we want the first index of a group, so we can slice from that location
    # example: hashable = [0 1 1]; dupe = [1,0]; dupe_idx = [0,1]
    dupe_idx = np.append(0, np.nonzero(dupe)[0] + 1)
    # if you wanted to use this one function to deal with non- regular groups
    # you could use: np.array_split(dupe_idx)
    # this is roughly 3x slower than using the group_dict method above.
    start_ok = np.diff(np.concatenate((dupe_idx, [len(hashable)]))) == require_count
    groups = np.tile(dupe_idx[start_ok].reshape((-1, 1)), require_count) + np.arange(
        require_count
    )
    groups_idx = order[groups]

    if require_count == 1:
        return groups_idx.reshape(-1)
    return groups_idx


def is_watertight(edges, edges_sorted=None):
    """
    Parameters
    -----------
    edges : (n, 2) int
      List of vertex indices
    edges_sorted : (n, 2) int
      Pass vertex indices sorted on axis 1 as a speedup

    Returns
    ---------
    watertight : boolean
      Whether every edge is shared by an even
      number of faces
    winding : boolean
      Whether every shared edge is reversed
    """
    # passing edges_sorted is a speedup only
    if edges_sorted is None:
        edges_sorted = np.sort(edges, axis=1)

    # group sorted edges
    groups = group_rows(edges_sorted, require_count=2)
    watertight = bool((len(groups) * 2) == len(edges))

    # are opposing edges reversed
    opposing = edges[groups].reshape((-1, 4))[:, 1:3].T
    # wrap the weird numpy bool
    winding = bool(np.equal(*opposing).all())

    return watertight, winding
