from collections import deque
import copy


def deep_get(d, path):
    path = path.split("/")
    current = d
    for item in path:
        try:
            current = current[item]
        except Exception:
            return None
    return current


def deep_set(d, path, value):
    current = d
    path = path.split("/")
    for item in path[:-1]:
        current = current[item]
    current[path[-1]] = value


def expand_config(config):
    q = deque([config])

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
