def all_partitions(collection):
    if not collection:
        yield []
        return
    first, *rest = collection
    for partial in all_partitions(rest):
        yield [{first}] + partial
        for i in range(len(partial)):
            new_partition = [set(s) for s in partial]
            new_partition[i].add(first)
            yield new_partition

def partition_score(partition, judgments):
    set_index = {}
    for idx, group in enumerate(partition):
        for node in group:
            set_index[node] = idx
    score = 0
    for (i, j), answer in judgments.items():
        if answer == 'Yes':
            if set_index[i] == set_index[j]:
                score += 1
        elif answer == 'No' or answer == 'Not sure':
            if set_index[i] != set_index[j]:
                score += 1
    return score

def best_partition(items, judgments):
    best_part = None
    best_score = -1
    for partition in all_partitions(items):
        score = partition_score(partition, judgments)
        if score > best_score:
            best_score = score
            best_part = partition
    return best_part, best_score

def greedy_partition(items, judgments):
    clusters = [{item} for item in items]
    def score_for_merge(c1, c2):
        merged = c1 | c2
        score_delta = 0
        for (i, j), answer in judgments.items():
            if (i in c1 and j in c2) or (i in c2 and j in c1):
                if answer == 'Yes':
                    score_delta += 1
                elif answer == 'No' or answer == 'Not sure':
                    score_delta -= 1
        return score_delta
    improved = True
    while improved:
        improved = False
        best_increase = 0
        best_pair = None
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                delta = score_for_merge(clusters[i], clusters[j])
                if delta > best_increase:
                    best_increase = delta
                    best_pair = (i, j)
        if best_pair:
            i, j = best_pair
            clusters[i] = clusters[i] | clusters[j]
            del clusters[j]
            improved = True
    overall_score = partition_score(clusters, judgments)
    return clusters, overall_score

def merge_bboxes(p, persons_int):
    merged_bboxes = []
    for group in p:
        people_cnt = len(group)
        if  people_cnt > 1:
            x_min = min([persons_int[i][0] for i in group])
            y_min = min([persons_int[i][1] for i in group])
            x_max = max([persons_int[i][2] for i in group])
            y_max = max([persons_int[i][3] for i in group])
            merged_bboxes.append(([x_min, y_min, x_max, y_max], people_cnt))
        else:        
            merged_bboxes.append((persons_int[list(group)[0]], people_cnt))
    return merged_bboxes

__all__ = ["greedy_partition", "merge_bboxes"]