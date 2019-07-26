def linear_interpolation(l, r, alpha):
    # https://github.com/openai/EPG/blob/master/epg/utils.py#L154
    return l + alpha * (r - l)


def choose_left(l, r, alpha=None):
    return l


class PiecewiseLinearSchedule(object):
    # https://github.com/openai/EPG/blob/master/epg/utils.py#L158
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """
        Computes values for a piecewise schedule.
        :param endpoints: list of (iter, val) tuples that define interpolation points for the schedule
        :param interpolation: function that performs interpolation between endpoints
        :param outside_value: value for points outside the range defined in endpoints
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        assert self._outside_value is not None
        return self._outside_value


class PiecewiseSchedule(PiecewiseLinearSchedule):
    def __init__(self, endpoints, outside_value):
        super().__init__(endpoints, outside_value=outside_value, interpolation=choose_left)
