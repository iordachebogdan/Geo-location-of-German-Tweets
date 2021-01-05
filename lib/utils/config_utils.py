from collections import deque
import copy


def deep_get(d, path):
    """Dict deep get"""
    path = path.split("/")
    current = d
    for item in path:
        try:
            current = current[item]
        except Exception:
            return None
    return current


def deep_set(d, path, value):
    """Dict deep set"""
    current = d
    path = path.split("/")
    for item in path[:-1]:
        current = current[item]
    current[path[-1]] = value


def expand_config(config):
    """Given a configuration dictionary recursivly expand all array-like values and return
    a list of configuration dictionaries.

    For example:
        given a configuration dict with a key 'C' that has associated a value of
        [0, 1, 2], expand the current configuration into 3 new ones where every
        key and value is copied, but the 'C' key has value 0 in the first config,
        value 1 in the second, and value 2 in the third. Continue the process
        for each generated config as long as there exist an array-like value in its dict.

    Useful for defining configurations for hyperparameter grid search.
    """

    # use a queue for the generated configs
    q = deque([config])

    # check if there is any array-like value
    def find_list(d, path):
        if isinstance(d, list):
            return path
        if not isinstance(d, dict):
            return None
        for key in d:
            ret = find_list(d[key], path + ("/" if path != "" else "") + f"{key}")
            if ret:
                return ret
        return None

    # while there are configurations with array-like values expand them
    while True:
        current = q.popleft()
        path = find_list(current, "")
        if path:
            lst = deep_get(current, path)
            new_configs = [copy.deepcopy(current) for _ in range(len(lst))]
            for i in range(len(lst)):
                deep_set(new_configs[i], path, lst[i])
                q.append(new_configs[i])
        else:
            q.appendleft(current)
            break

    return list(q)
